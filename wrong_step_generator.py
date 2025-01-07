import os
import asyncio
import logging
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
        """
        found_verified = False
        found_valid = False
        correct_step = None

        for i in range(self.completions):
            try:
                completion = await self.completion_agent.generate(
                    problem,
                    partial_solution
                )
                complete_solution = partial_solution + completion
                
                # First verify the completed solution
                is_correct, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    found_verified = True
                    # If verified, check if it's valid
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
                
        return found_verified, found_valid, correct_step

    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate both correct and wrong solutions, identify which step causes wrong solution to fail.
        Returns dict with problem, answer, both solutions and wrong step info.
        """
        # Search for both correct and wrong solutions
        correct_solution = None
        wrong_solution = None
        attempts = 0
        
        while (correct_solution is None or wrong_solution is None) and attempts < self.best_of:
            try:
                attempts += 1
                solution = await self.solution_agent.generate(problem)
                
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
            
        if len(steps) < 2:  # Need at least analysis + one step
            logging.error("❌ Not enough steps found (need at least analysis + one step)")
            return None
            
        # Get partial solutions
        partial_solutions = get_partial_solutions(steps)
        
        # Track the last known good step
        last_good_step = None
        
        self.logs.append("\n=== Analyzing solution steps ===")
        
        # Check each step sequentially
        for i in range(len(partial_solutions)):
            self.logs.append(f"\nChecking step {i}...")
            
            # Special handling for analysis section (i=0)
            if i == 0:
                found_verified, found_valid, correct_step = await self._verify_completions(
                    problem,
                    partial_solutions[i],
                    correct_answer,
                    i
                )
                
                if not found_valid:
                    self.logs.append("✗ Analysis section is wrong - dropping entry")
                    return None
                    
                self.logs.append("✓ Analysis section is valid")
                last_good_step = correct_step
                continue
                
            # For other steps, try completions from this partial solution    
            found_verified, found_valid, correct_step = await self._verify_completions(
                problem,
                partial_solutions[i],
                correct_answer,
                i
            )
            
            # Print verification results and correct step
            if correct_step:
                print(f"Step {i} verification: verified={found_verified}, valid={found_valid}, correct_step={correct_step}")
            else:
                print(f"Step {i} verification: verified={found_verified}, valid={found_valid}, but correct step is null")
            
            if found_valid:
                self.logs.append(f"✓ Step {i} is valid")
                last_good_step = correct_step
            elif not found_verified:
                self.logs.append(f"✗ Found wrong step at step {i}")
                return {
                    'problem': problem,
                    'correct_answer': correct_answer,
                    'correct_solution': correct_solution,
                    'wrong_solution': wrong_solution,
                    'wrong_step_index': i,
                    'wrong_step': steps[i],
                    'partial_solution': partial_solutions[max(0, i - 1)],
                    'correct_step': last_good_step}
                
        # If we get here, all steps were valid (shouldn't happen with a wrong solution)
        self.logs.append("✓ All steps are valid")
        return None

async def main():
    """Main function for wrong step generation"""
    config = BenchmarkConfig.from_args('Wrong step generation approach')
    logger = MarkdownLogger()  # Create single logger instance for all examples
    
    async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
        """Process a single example"""
        try:
            # Initialize solver
            solver = get_model(ModelOption[config.solver], temp=config.temperature)
            
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
                
                
                # Add wrong step details
                all_logs.append("\n🔍 Wrong Step Details:")
                all_logs.append(f"✓ Found at step: {result['wrong_step_index']}")
                all_logs.append(f"✓ Wrong step content:")
                all_logs.append(result['wrong_step'])
            
            # Print logs for this example
            print("\n".join(all_logs))
            
            # Save comprehensive logs to markdown file
            log_file = logger.save_logs(all_logs, example_id)
            
            # Add example ID to result
            if result:
                result['id'] = example_id
                return [result]
            
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
