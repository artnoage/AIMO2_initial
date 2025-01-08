import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import *
from utils.benchmark_utils import *
from utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class ListGenerator:
    """Generates lists of solution components with best/worst variants"""
    
    def __init__(self, solver, best_of: int, completions: int):
        self.solver = solver
        self.best_of = best_of
        self.completions = completions
        self.analysis_agent = AnalysisAgent(solver)
        self.step_agent = NextStepAgent(solver)
        self.completion_agent = CompletionAgent(solver)
        self.verifier = NumericVerifier()

    async def _score_with_completions(
        self,
        problem: str,
        current_solution: str,
        correct_answer: str
    ) -> Tuple[float, Optional[str], Optional[str], Optional[str]]:
        """
        Score a partial solution by attempting completions.
        Returns (score, good_completion, bad_completion, completion_prompt).
        """
        successful = 0
        good_completion = None
        bad_completion = None
        worst_failure_score = 0  # Accumulate failures: +1 wrong answer, +2 invalid solution, +4 invalid completion
        
        for _ in range(self.completions):
            try:
                completion_result = await self.completion_agent.generate(
                    problem,
                    current_solution,
                    return_prompt=True
                )
                completion = completion_result[1]  # Get just the completion response
                completion_prompt = completion_result[0].content  # Store the prompt
                complete_solution = current_solution + completion
                
                failure_score = 0
                
                # First validate the completion itself
                is_valid_completion, _ = validate_completion(current_solution, completion)
                if not is_valid_completion:
                    failure_score += 4
                
                # Then validate the complete solution
                is_valid_solution, _ = validate_solution(complete_solution)
                if not is_valid_solution:
                    failure_score += 2
                
                # Finally verify the answer
                is_correct, answer = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    if failure_score == 0:  # Only store good completions if they pass all checks
                        successful += 1
                        if good_completion is None:
                            good_completion = completion
                else:
                    failure_score += 1
                
                # Update bad completion if this one is worse
                if failure_score > worst_failure_score:
                    worst_failure_score = failure_score
                    bad_completion = completion
                    
            except Exception:
                successful += 0  # Explicitly count failed attempts
        
        final_score = successful / self.completions

        # If both completions are None, create an additional result using completion agent prompt
        if good_completion is None and bad_completion is None:
            try:
                completion_prompt = await self.completion_agent.generate(
                    problem,
                    current_solution,
                    return_prompt=True
                )
                # Return the prompt as both good and bad completion with scores 1 and 0
                return final_score, completion_prompt[0].content, completion_prompt[0].content
            except Exception:
                pass

        return final_score, good_completion, bad_completion, completion_prompt

    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> List[Dict[str, Any]]:
        """
        Generate solution components with best/worst variants.
        Returns list of dicts with prompt/chosen/rejected/scores.
        """
        results = []
        current_solution = ""
        rejected_part_solution = ""  # Track rejected solution path
        logs = []
        logs.append("\n=== List Generation Details ===")
        
        # Generate and score analyses
        analyses = []
        analysis_prompt = None
        best_analysis = None
        worst_analysis = None
        best_score = 0.0
        worst_score = float('inf')
        
        for _ in range(self.best_of):
            try:
                if analysis_prompt is None:
                    prompt, analysis = await self.analysis_agent.generate(
                        problem,
                        return_prompt=True
                    )
                    analysis_prompt = prompt
                else:
                    analysis = await self.analysis_agent.generate(problem)
                
                is_valid, reason = validate_analysis(analysis)
                if is_valid:
                    score, good_completion, bad_completion, completion_prompt = await self._score_with_completions(
                        problem,
                        analysis,
                        correct_answer
                    )
                    analyses.append((analysis, score))
                    
                    # Update best and worst scores/solutions
                    if score > best_score:
                        best_score = score
                        best_analysis = analysis
                    if score < worst_score:
                        worst_score = score
                        worst_analysis = analysis
                    
                    # Check for perfect and zero scores
                    has_perfect = any(a[1] == 1.0 for a in analyses)
                    has_zero = any(a[1] == 0.0 for a in analyses)
                    
                    # Break out of for loop if we have both
                    if has_perfect and has_zero:
                        logs.append(f"\n✓ Early stop in analysis: Found both perfect (1.0) and zero scoring analyses")
                        continue
                        
            except Exception:
                continue
                
        if len(analyses) < 2:
            logs.append(f"\n❌ Less than two valid analyses")
            print("\n".join(logs))
            return []
            
        # Use tracked best/worst scores
        best_analysis_score = best_score
        worst_analysis_score = worst_score

        # Check if both scores are zero
        if best_analysis_score == 0 and worst_analysis_score == 0:
            logs.append(f"\n❌ ALL analysis paths received zero scores")
            print("\n".join(logs))
            return []
            
        logs.append(f"\n📊 Analysis Phase:")
        logs.append(f"├─ Best score: {best_analysis_score:.3f}")
        logs.append(f"├─ Worst score: {worst_analysis_score:.3f}")
        logs.append(f"└─ Score difference: {(best_analysis_score - worst_analysis_score):.3f}")
        
        # Append if score difference is more than 0.5 and analyses are different
        if best_analysis_score - worst_analysis_score > 0.5 and best_analysis != worst_analysis:
            rejected_part_solution = worst_analysis  # Track rejected analysis
            results.append({
                'problem': problem,
                'correct_answer': correct_answer,
                'prompt': {'content': analysis_prompt, 'role': 'user'},
                'chosen': {'content': best_analysis, 'role': 'assistant'},
                'rejected': {'content': worst_analysis, 'role': 'assistant'},
                'score_chosen': best_analysis_score,
                'score_rejected': worst_analysis_score,
                'rejected_part_solution': rejected_part_solution
            })
        
        # Use best analysis as starting point
        current_solution = analyses[-1][0]
        step_num = 1
        
        while True:
            steps = []
            step_prompt = None
            best_step = None
            worst_step = None
            best_step_score = 0.0
            worst_step_score = float('inf')
            
            # Generate and score steps
            has_perfect = False
            has_zero = False
            
            for _ in range(self.best_of):
                try:
                    if step_prompt is None:
                        prompt, step = await self.step_agent.generate(
                            problem,
                            current_solution,
                            return_prompt=True
                        )
                        step_prompt = prompt
                    else:
                        step = await self.step_agent.generate(
                            problem,
                            current_solution
                        )
                    
                    test_solution = current_solution + step
                    
                    # Check if step contains answer
                    answer = extract_answer_from_solution(test_solution)
                    if answer is not None:
                        # Verify the answer is correct
                        is_correct, _ = await self.verifier.verify(
                            test_solution,
                            correct_answer,
                            problem
                        )
                        # Also validate the complete solution
                        is_valid, _ = validate_solution(test_solution)
                        
                        if is_correct and is_valid:  # Both correct answer and valid solution
                            steps.append((step, 1.0))
                            has_perfect = True
                            break
                            
                    # Score step if no answer yet
                    is_valid = validate_step(step, expected_step=step_num)
                    if is_valid:
                        score, good_completion, bad_completion, completion_prompt = await self._score_with_completions(
                            problem,
                            test_solution,
                            correct_answer
                        )
                        steps.append((step, score))
                        
                        # Update best and worst steps/solutions
                        if score > best_step_score:
                            best_step_score = score
                            best_step = step
                        if score < worst_step_score:
                            worst_step_score = score
                            worst_step = step
                            
                        # Check for perfect or zero score
                        if score == 1.0:
                            has_perfect = True
                        elif score == 0.0:
                            has_zero = True
                            
                        # If we have both perfect and zero scoring steps at this level,
                        # we don't need to sample more steps at this level - we already 
                        # have good examples of what works and what doesn't work.
                        # Using break will exit only this for loop (sampling loop) but continue
                        # the outer while loop (solution steps)
                        if has_perfect and has_zero:
                            logs.append(f"\n✓ Early stop sampling step {step_num}: Found both perfect (1.0) and zero scoring steps")
                            break  # Exits only the for loop, continues with while loop
                            
                except Exception:
                    continue
                    
            if not steps:  # No valid steps generated
                break
                
            # Check if all steps have zero score or all steps have high scores (>0.7)
            all_zero = all(step[1] == 0 for step in steps)
            all_high = all(step[1] > 0.7 for step in steps)
            
            if all_zero:
                logs.append(f"\n❌ Stopping: All generated steps received zero score")
                break
            elif all_high:
                logs.append(f"\n✓ Stopping: All generated steps have high scores (>0.7)")
                break
            
            logs.append(f"\n📊 Step {step_num} Phase:")
            logs.append(f"├─ Best score: {best_step_score:.3f}")
            logs.append(f"├─ Worst score: {worst_step_score:.3f}")
            logs.append(f"└─ Score difference: {(best_step_score - worst_step_score):.3f}")
            
            # Only append if worst score is zero AND we have a valid best step with non-zero score
            # AND the steps are different
            if (worst_step_score == 0 and best_step is not None and 
                best_step_score > 0 and best_step != worst_step):
                # Update rejected part solution with current solution plus worst step
                rejected_part_solution = current_solution + worst_step
                results.append({
                    'problem': problem,
                    'correct_answer': correct_answer,
                    'prompt': {'content': step_prompt, 'role': 'user'},
                    'chosen': {'content': best_step, 'role': 'assistant'},
                    'rejected': {'content': worst_step, 'role': 'assistant'},
                    'score_chosen': best_step_score,
                    'score_rejected': worst_step_score,
                    'rejected_part_solution': rejected_part_solution
                })
            
            # Use best step and continue
            current_solution += steps[-1][0]
            
            # Check if we found a valid and correct solution
            answer = extract_answer_from_solution(current_solution)
            if answer is not None:
                is_correct, _ = await self.verifier.verify(current_solution, correct_answer, problem)
                is_valid, _ = validate_solution(current_solution)
                if is_correct and is_valid:
                    break
                
            step_num += 1
            
        # Print all logs at the end
        print("\n".join(logs))
        return results

async def main():
    """Main function for list generation approach"""
    config = BenchmarkConfig.from_args('List generation approach for creating training data')
    
    async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[List[Dict]]:
        """Process a single example using list generation"""
        try:
            # Initialize solver
            solver = get_model(config, role="solver")
            
            # Create list generator
            generator = ListGenerator(solver, config.best_of, config.completions)
            
            # Extract answer
            correct_answer = extract_answer_from_solution(example['solution'])
            if correct_answer is None:
                print(f"Warning: Could not extract answer from solution for example {running_id}")
                return None
                
            # Generate solution components
            results = await generator.generate(example['problem'], correct_answer)
            
            # Add example ID to results
            for result in results:
                result['id'] = example_id
                
            return results
            
        except Exception as e:
            print(f"Error processing example {running_id}: {str(e)}")
            return None

    await run_benchmark(
        config=config,
        process_example_func=process_example
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
