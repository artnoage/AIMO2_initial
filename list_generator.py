import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *

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
    ) -> float:
        """Score a partial solution by attempting completions"""
        successful = 0
        print("\n🔄 Starting completion attempts...")
        
        for attempt in range(self.completions):
            try:
                print(f"\nAttempt {attempt + 1}/{self.completions}:")
                complete_solution = current_solution + await self.completion_agent.generate(
                    problem,
                    current_solution
                )
                is_correct, answer = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                print(f"Verification result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
                if is_correct:
                    successful += 1
                    print("✅ Successful completion!")
                else:
                    print("❌ Failed verification")
            except Exception as e:
                print(f"❌ Error in completion: {str(e)}")
                continue
        
        final_score = successful / self.completions
        print(f"\n📊 Final score: {final_score} ({successful}/{self.completions} successful)")
        return final_score

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
                print(f"\n🔍 Analysis Validation: {'✅ Passed' if is_valid else '❌ Failed'}")
                if not is_valid:
                    print(f"Reason: {reason}")
                if is_valid:
                    score = await self._score_with_completions(
                        problem,
                        analysis,
                        correct_answer
                    )
                    analyses.append((analysis, score))
                    
                    print(f"\n🎯 Analysis Score: {score}")
                    if score > 0:
                        print("✅ Found successful completion!")
                        print(f"Analysis:\n{analysis[:200]}...")
                    
                    # Update best and worst scores
                    if score > best_score:
                        best_score = score
                        best_analysis = analysis
                        print(f"📈 New best score: {best_score}")
                    if score < worst_score:
                        worst_score = score
                        worst_analysis = analysis
                        print(f"📉 New worst score: {worst_score}")
                    
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
            return []
            
        # Use tracked best/worst scores
        best_analysis_score = best_score
        worst_analysis_score = worst_score
            
        logs.append(f"\n📊 Analysis Phase:")
        logs.append(f"├─ Best score: {best_analysis_score:.3f}")
        logs.append(f"├─ Worst score: {worst_analysis_score:.3f}")
        logs.append(f"└─ Score difference: {(best_analysis_score - worst_analysis_score):.3f}")
        
        results.append({
            'prompt': {'content': analysis_prompt, 'role': 'user'},
            'chosen': {'content': best_analysis, 'role': 'assistant'},
            'rejected': {'content': worst_analysis, 'role': 'assistant'},
            'score_chosen': best_analysis_score,
            'score_rejected': worst_analysis_score
        })
        
        # Use best analysis as starting point
        current_solution = analyses[-1][0]
        step_num = 1
        
        while True:
            steps = []
            step_prompt = None
            
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
                        score, _, _ = await self.verifier.verify(
                            test_solution,
                            correct_answer,
                            problem
                        )
                        if score:  # Valid answer found
                            steps.append((step, 1.0))
                            has_perfect = True
                            break
                            
                    # Score step if no answer yet
                    is_valid = validate_step(step)
                    print(f"\n🔍 Step Validation: {'✅ Passed' if is_valid else '❌ Failed'}")
                    if is_valid:
                        score = await self._score_with_completions(
                            problem,
                            test_solution,
                            correct_answer
                        )
                        steps.append((step, score))
                        
                        # Check for perfect or zero score
                        if score == 1.0:
                            has_perfect = True
                        elif score == 0.0:
                            has_zero = True
                            
                        # Break out of for loop if we have both
                        if has_perfect and has_zero:
                            logs.append(f"\n✓ Early stop in step {step_num}: Found both perfect (1.0) and zero scoring steps")
                            continue
                            
                except Exception:
                    continue
                    
            if not steps:  # No valid steps generated
                break
                
            # Sort and get best/worst step
            steps.sort(key=lambda x: x[1])
            best_step_score = steps[-1][1]
            worst_step_score = steps[0][1]
                
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
            
            results.append({
                'prompt': {'content': step_prompt, 'role': 'user'},
                'chosen': {'content': steps[-1][0], 'role': 'assistant'},
                'rejected': {'content': steps[0][0], 'role': 'assistant'},
                'score_chosen': best_step_score,
                'score_rejected': worst_step_score
            })
            
            # Use best step and continue
            current_solution += steps[-1][0]
            
            # Check if we found a valid answer
            if extract_answer_from_solution(current_solution) is not None:
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
            solver = get_model(ModelOption[config.solver], temp=config.temperature)
            
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
