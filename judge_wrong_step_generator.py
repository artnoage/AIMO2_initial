import os
import asyncio
import logging
import random
import re
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import *
from utils.benchmark_utils import *
from utils.agents import *
from utils.log_handler import MarkdownLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class JudgeWrongStepGenerator:
    """Generates wrong solution steps by finding valid but incorrect solutions using judge guidance"""
    
    def __init__(self, solver, judge, best_of: int, completions: int):
        self.solver = solver
        self.judge = judge
        self.best_of = best_of
        self.completions = completions
        self.solution_agent = FullSolutionAgent(solver)
        self.step_agent = NextStepAgent(solver)
        self.completion_agent = CompletionAgent(solver)
        self.judge_agent = JudgeAgent(judge)
        self.verifier = NumericVerifier()
        self.logs = []
        
    def _extract_step_number(self, judge_response: str) -> Optional[int]:
        """Extract step number from judge response"""
        # Look for "Step X" pattern
        step_match = re.search(r'Step\s+(\d+)', judge_response)
        if step_match:
            try:
                step_num = int(step_match.group(1))
                # Validate non-negative
                if step_num < 0:
                    self.logs.append(f"Judge predicted invalid negative step number: {step_num}")
                    return None
                return step_num
            except ValueError:
                return None
        return None
        
    def _extract_explanation(self, judge_response: str) -> Optional[str]:
        """Extract explanation from judge response"""
        # Look for content between <EXPLANATION> tags
        explanation_match = re.search(r'<EXPLANATION>(.*?)<EXPLANATION>', judge_response, re.DOTALL)
        if explanation_match:
            return explanation_match.group(1).strip()
        # If no tags, try to extract everything after the step number
        step_match = re.search(r'Step\s+\d+\.\s*(.*)', judge_response)
        if step_match:
            return step_match.group(1).strip()
        return None
        
    async def _verify_completions(
        self,
        problem: str,
        partial_solution: str,
        correct_answer: str,
        step_index: int
    ) -> Tuple[bool, bool, Optional[str]]:
        """Try multiple completions of a partial solution to check if any are correct.
        Returns:
            - found_verified: True if any solution verified correctly
            - found_valid: True if any solution both verified and validated
            - correct_step: The next correct step if found, None otherwise
            - correct_completion: The full valid completion if found, None otherwise
        """
        found_verified = False 
        found_valid = False
        correct_step = None
        good_completion = None
        completion_prompt = None

        for i in range(self.completions):
            try:
                if completion_prompt is None:
                    prompt, completion = await self.completion_agent.generate(
                        problem,
                        partial_solution,
                        return_prompt=True
                    )
                    completion_prompt = prompt
                else:
                    completion = await self.completion_agent.generate(
                        problem,
                        partial_solution
                    )
                complete_solution = partial_solution + completion
                
                # First verify if the answer is correct
                is_correct, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    found_verified = True
                    # Validate the completion itself
                    is_valid_completion, completion_reason = validate_completion(partial_solution, completion)
                    if not is_valid_completion:
                        self.logs.append(f"Invalid completion: {completion_reason}")
                        continue
                        
                    # Then check if the complete solution is valid
                    is_valid, validation_reason = validate_solution(complete_solution)
                    if is_valid:
                        found_valid = True
                        # Log the successful completion details
                        self.logs.append("\n=== Valid Completion Found ===")
                        self.logs.append("Partial solution up to this point:")
                        self.logs.append("```")
                        self.logs.append(partial_solution)
                        self.logs.append("```")
                        self.logs.append("\nValid completion:")
                        self.logs.append("```")
                        self.logs.append(completion)
                        self.logs.append("```")
                        # Extract the next step
                        completion_steps = split_into_steps(complete_solution)
                        next_step_index = step_index + 1
                        correct_step = completion_steps[next_step_index]
                        # Store the successful completion
                        good_completion = completion
                        break
                    else:
                        self.logs.append(f"Found verified but invalid solution: {validation_reason}")
                        continue
                    
            except Exception:
                continue

        # Log appropriate message based on what we found
        if step_index == 0:
            self.logs.append(f"Analysis section: Verified={found_verified}, Valid={found_valid}")
        else:
            self.logs.append(f"Step {step_index}: Verified={found_verified}, Valid={found_valid}")
            if not found_verified:
                self.logs.append(f"Step {step_index} is wrong: No verified solutions found")
            elif not found_valid:
                self.logs.append(f"Example dropped: Found verified but no valid solutions at step {step_index}")
                
        return found_verified, found_valid, correct_step, good_completion, completion_prompt

    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate both correct and wrong solutions, then validate judge's prediction of the first wrong step.
        Returns list of dicts with prompts and solutions/steps for ORPO training.
        """
        # Search for both correct and wrong solutions
        correct_solution = None
        wrong_solution = None
        solution_prompt = None
        attempts = 0
        
        while (correct_solution is None or wrong_solution is None) and attempts < self.best_of:
            try:
                attempts += 1
                if solution_prompt is None:
                    prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                    solution_prompt = prompt
                    solution = remove_inst_tokens(solution) if solution else None
                else:
                    solution = await self.solution_agent.generate(problem)
                    solution = remove_inst_tokens(solution) if solution else None
                
                if solution is None:
                    continue
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    continue
                    
                # Check if solution is wrong
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if is_correct and correct_solution is None:
                    correct_solution = solution
                    self.logs.append(f"✓ Found correct solution on attempt {attempts}")
                elif not is_correct and wrong_solution is None:
                    wrong_solution = solution
                    self.logs.append(f"✓ Found wrong solution on attempt {attempts}")
                    
            except Exception as e:
                self.logs.append(f"Error in attempt {attempts}: {str(e)}")
                continue
                
        if correct_solution is None:
            logging.error("❌ Failed to find correct solution")
            return None
        if wrong_solution is None:
            logging.error("❌ Failed to find wrong solution")
            return None
            
        # Split wrong solution into steps
        steps = split_into_steps(wrong_solution)
        if not steps:
            logging.error("❌ No steps found in solution - likely incorrect format")
            return None
            
        # Get partial solutions
        partial_solutions = get_partial_solutions(steps)
        num_steps = len(partial_solutions)
        
        if num_steps < 2:  # Need at least analysis + one step
            logging.error("❌ Not enough steps found (need at least analysis + one step)")
            return None
            
        # Ask judge to identify the first wrong step
        judge_response = await self.judge_agent.find_first_wrong_step(problem, wrong_solution)
        self.judge_prediction = self._extract_step_number(judge_response)
        self.logs.append("\n\nThis is the judges repsonse\n\n")
        self.logs.append(judge_response)
        self.logs.append("\n\nThis is the judges \n\n")
        self.logs.append(str(self.judge_prediction))
        # Store judge's prediction and use it as starting point
        
        # Return None if judge didn't identify a valid step number or predicted step 0
        if self.judge_prediction is None or self.judge_prediction == 0:
            self.logs.append("Judge didn't identify valid step number or predicted step 0, returning None")
            return None
            
        # Return None if judge predicted beyond solution length 
        if self.judge_prediction >= num_steps:
            self.logs.append("Judge predicted step beyond solution length, returning None")
            return None
            
        current_step = self.judge_prediction
        self.logs.append(f"Judge identified step {current_step} as first error")
            
        # For the judge to be correct:
        # 1. The previous step must be completable correctly
        # 2. The predicted step must not be completable correctly
        
        prev_step_valid = False
        last_good_step = None
        saved_good_completion = None
        saved_completion_prompt = None
        wrong_step_index = current_step
        step_is_wrong = False

        # First verify the previous step can be completed correctly
        prev_found_verified, prev_found_valid, prev_correct_step, prev_good_completion, prev_completion_prompt = (
            await self._verify_completions(
                problem,
                partial_solutions[current_step - 1],
                correct_answer,
                current_step - 1
            )
        )
            
        if prev_found_valid:
            # Previous step is valid, now we can check the current step
            prev_step_valid = True
            saved_good_completion = prev_good_completion
            saved_completion_prompt = prev_completion_prompt
            correct_step=prev_correct_step

            self.logs.append(f"✓ Previous step {current_step - 1} is valid")
            
            # Check if the current step is invalid
            found_verified, _, _, _, _ = await self._verify_completions(
                problem,
                partial_solutions[current_step],
                correct_answer,
                current_step
            )
            
            step_is_wrong= not found_verified
            if step_is_wrong:
                self.logs.append(f"✗ Step {current_step} cannot be completed correctly")
                self.logs.append("Judge was correct")
            else:
                self.logs.append(f"✓ Step {current_step} is valid - Judge was incorrect")
        else:
            self.logs.append(f"✗ Previous step {current_step - 1} cannot be completed correctly")
            self.logs.append("Judge was incorrect - error starts earlier")
            
        # The judge is correct only if both conditions are met
        self.judge_was_correct = step_is_wrong and prev_step_valid
        
        # Return None if judge was wrong
        if not self.judge_was_correct:
            return None
            
        # Extract judge's explanation
        judge_explanation = self._extract_explanation(judge_response)
        if not judge_explanation:
            self.logs.append("Could not extract judge explanation")
            return None
            
        # Get wrong solution up to the bad step
        wrong_solution_partial = '\n'.join(steps[:current_step + 1])
        
        # Split completion into steps
        completion_steps = saved_good_completion.split('\n')
        
        # Only remove step number from first step to avoid duplication
        if completion_steps:
            completion_steps[0] = re.sub(r'^Step\s+\d+[:.]\s*', '', completion_steps[0])
        
        # Create combined solution
        combined_solution = (
            f"{wrong_solution_partial}\n\n"
            "Error Explanation: " + f"{judge_explanation}\n\n"
            "Correct continuation:\n" + 
            '\n'.join(completion_steps)
        )
        
        # Create entries for ORPO training
        results = [
            # First entry: full solution comparison
            {
                'problem': problem,
                'correct_answer': correct_answer,
                'prompt': {'content': solution_prompt, 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(correct_solution), 'role': 'assistant'},
                'rejected': {'content': wrong_solution, 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            },
            # Second entry: wrong solution with explanation and correction
            {
                'problem': problem,
                'correct_answer': correct_answer,
                'prompt': {'content': solution_prompt, 'role': 'user'},
                'chosen': {'content': combined_solution, 'role': 'assistant'},
                'rejected': {'content': wrong_solution, 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            }
        ]
        
        return results

async def main():
    """Main function for wrong step generation"""
    config = BenchmarkConfig.from_args('Wrong step generation with judge guidance')
    logger = MarkdownLogger()
    
    async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
        """Process a single example"""
        try:
            # Initialize solver and judge
            solver = get_model(config, role="solver")
            judge = get_model(config, role="judge")
            
            # Create generator
            generator = JudgeWrongStepGenerator(solver, judge, config.best_of, config.completions)
            
            # Extract answer
            correct_answer = extract_answer_from_solution(example['solution'])
            if correct_answer is None:
                generator.logs.append(f"Warning: Could not extract answer from solution for example {running_id}")
                return None
                
            # Generate wrong step
            result = await generator.generate(example['problem'], correct_answer)
            if result is None:
                return None
                
            # Prepare comprehensive logs for this example
            all_logs = []
            all_logs.append("\n" + "="*80)
            all_logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
            all_logs.append("="*80)
            
            # Problem details
            all_logs.append(f"\n📋 Problem:")
            all_logs.append(f"{example['problem'][:200]}...")
            all_logs.append(f"\n✓ Expected Answer: {correct_answer}")
            
            # Add generator logs
            all_logs.extend(generator.logs)
            
            if result:
                # Add solution quality metrics
                all_logs.append("\n📊 Solution Quality:")
                
                # Add details about the generated solutions
                all_logs.append("\n🔍 Generated Solutions:")
                for entry in result:
                    all_logs.append(f"✓ Chosen: {entry['chosen']['content'][:200]}...")
                    all_logs.append(f"✗ Rejected: {entry['rejected']['content'][:200]}...")
            
            # Print logs for this example
            print("\n".join(all_logs))
            
            # Save comprehensive logs to markdown file
            log_file = logger.save_logs(all_logs, example_id)
            
            # Add example ID to results
            if result:
                for entry in result:
                    entry['id'] = example_id
                return result
            
        except Exception as e:
            logging.error(f"Error processing example {running_id}: {str(e)}")
            return None

    await run_benchmark(
        config=config,
        process_example_func=process_example
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nBenchmark interrupted by user")
    except Exception as e:
        logging.error(f"\nBenchmark failed with error: {e}")
