import os
import asyncio
import logging
import random
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

class WrongStepGenerator:
    """Generates wrong solution steps by finding valid but incorrect solutions"""
    
    def __init__(self, solver, best_of: int, completions: int):
        self.solver = solver
        self.best_of = best_of
        self.completions = completions
        self.solution_agent = FullSolutionAgent(solver)
        self.step_agent = NextStepAgent(solver)
        self.completion_agent = CompletionAgent(solver)
        self.verifier = NumericVerifier()
        self.logs = []
        
        
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
        Generate both correct and wrong solutions, identify which step causes wrong solution to fail.
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
            
        # Start with a random step
        current_step = random.randint(0, num_steps - 1)
        going_up = None  # Direction flag: None=initial, True=up, False=down
        last_bad_step = None
        last_good_step = None
        wrong_step_index = None
        saved_good_completion = None
        saved_completion_prompt = None
        
        self.logs.append("\n=== Analyzing solution steps ===")
        self.logs.append(f"Starting analysis at step {current_step}")
        
        while True:
            self.logs.append(f"\nChecking step {current_step}...")
            
            found_verified, found_valid, correct_step, good_completion, completion_prompt = await self._verify_completions(
                problem,
                partial_solutions[current_step],
                correct_answer,
                current_step
            )
            if found_verified and not found_valid:
                return None

            if found_valid:
                self.logs.append(f"✓ Step {current_step} is valid")
                last_good_step = correct_step
                saved_good_completion = good_completion
                saved_completion_prompt = completion_prompt
                
                if going_up is None:
                    # First check was good, go up to find potential wrong step
                    going_up = True
                elif not going_up:
                    # We were going down and found a good step
                    # The wrong step must be the last_bad_step we found
                    wrong_step_index = last_bad_step
                    break
                    
                # Move up one step
                if current_step + 1 >= num_steps:
                    # We reached the top without finding a wrong step
                    logging.error("❌ Reached end without finding wrong step")
                    return None
                current_step += 1
                
            else:
                self.logs.append(f"✗ Step {current_step} cannot be completed correctly")
                last_bad_step = current_step
                
                if going_up is None:
                    # First check was bad, go down to find last good step
                    going_up = False
                elif going_up:
                    # We were going up and found a bad step
                    # The wrong step must be here
                    wrong_step_index = current_step
                    break
                    
                # Move down one step
                if current_step - 1 < 0:
                    # We reached the bottom without finding good step
                    logging.error("❌ Reached start without finding good step")
                    return None
                current_step -= 1

        if wrong_step_index is None:
            logging.error("❌ Failed to identify wrong step")
            return None
            
        # Create two entries for ORPO training
        results = []
        
        # First entry: full solution comparison
        results.append({
            'problem': problem,
            'correct_answer': correct_answer,
            'prompt': {'content': solution_prompt, 'role': 'user'},
            'chosen': {'content': remove_inst_tokens(correct_solution), 'role': 'assistant'},
            'rejected': {'content': wrong_solution, 'role': 'assistant'},
            'score_chosen': 1.0,
            'score_rejected': 0.0
        })
        
        # Second entry: step comparison
        step_prompt = await self.step_agent.generate(
            problem,
            partial_solutions[max(0, wrong_step_index - 1)],
            return_prompt=True
        )
        results.append({
            'problem': problem,
            'correct_answer': correct_answer,
            'prompt': {'content': step_prompt[0], 'role': 'user'},
            'chosen': {'content': remove_inst_tokens(last_good_step), 'role': 'assistant'},
            'rejected': {'content': steps[wrong_step_index], 'role': 'assistant'},
            'score_chosen': 1.0,
            'score_rejected': 0.0
        })

        # Third entry: completion comparison
        results.append({
            'problem': problem,
            'correct_answer': correct_answer,
            'prompt': {'content': saved_completion_prompt, 'role': 'user'},
            'chosen': {'content': remove_inst_tokens(saved_good_completion), 'role': 'assistant'},
            'rejected': {'content': ''.join(steps[wrong_step_index:]), 'role': 'assistant'},
            'score_chosen': 1.0,
            'score_rejected': 0.0
        })
        
        return results

async def main():
    """Main function for wrong step generation"""
    config = BenchmarkConfig.from_args('Wrong step generation approach')
    logger = MarkdownLogger()  # Create single logger instance for all examples
    
    async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
        """Process a single example"""
        try:
            # Initialize solver
            solver = get_model(config, role="solver")
            
            # Create generator
            generator = WrongStepGenerator(solver, config.best_of, config.completions)
            
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
