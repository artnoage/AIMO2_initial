import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.log_handler import MarkdownLogger

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
    ) -> Tuple[bool, Optional[str]]:
        """Try multiple completions of a partial solution to check if any are correct"""
        successful = 0
        correct_completion = None

        # First try to get a successful completion
        for i in range(self.completions):
            try:
                completion = await self.completion_agent.generate(
                    problem,
                    partial_solution
                )
                complete_solution = partial_solution + completion
                
                # Verify the completed solution
                is_correct, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    successful += 1
                    correct_completion = completion
                    break
                    
            except Exception:
                continue

        # If we found a successful completion, extract the correct step
        if successful > 0 and correct_completion:
            self.logs.append("\nDEBUG: Found successful completion:")
            self.logs.append("=" * 50)
            self.logs.append(correct_completion)
            self.logs.append("=" * 50)
            
            # Get all steps from the completion
            completion_steps = split_into_steps(correct_completion)
            self.logs.append(f"\nDEBUG: Split into {len(completion_steps)} steps:")
            for i, step in enumerate(completion_steps):
                self.logs.append(f"\nStep {i}:")
                self.logs.append("-" * 30)
                self.logs.append(step)
                self.logs.append("-" * 30)
            
            # Extract the step at the current index
            if step_index < len(completion_steps):
                correct_step = completion_steps[step_index]
                self.logs.append(f"\nDEBUG: Extracted step {step_index}:")
                self.logs.append("-" * 30)
                self.logs.append(correct_step)
                self.logs.append("-" * 30)
            else:
                correct_step = None
                self.logs.append(f"\nDEBUG: Step index {step_index} out of bounds (max: {len(completion_steps)-1})")
                
            if step_index == 0:
                self.logs.append(f"Analysis section: {successful}/{self.completions} completions successful")
            else:
                self.logs.append(f"Step {step_index}: {successful}/{self.completions} completions successful")
            return True, correct_step

        # No successful completion found
        if step_index == 0:
            self.logs.append(f"Analysis section: {successful}/{self.completions} completions successful")
        else:
            self.logs.append(f"Step {step_index}: {successful}/{self.completions} completions successful")
        return False, None

    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a wrong solution and identify which step causes it to go wrong.
        Returns dict with problem, answer, wrong solution and wrong step info.
        """
        # Try to generate both correct and wrong solutions
        wrong_solution = None
        correct_solution = None
        attempts = 0
        
        while (wrong_solution is None or correct_solution is None) and attempts < self.best_of:
            try:
                attempts += 1
                solution = await self.solution_agent.generate(problem)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    continue
                    
                # Check if solution is correct/wrong
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
                
        if wrong_solution is None or correct_solution is None:
            logging.error("❌ Failed to find both correct and wrong solutions")
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
        
        # Find first step that makes all completions wrong
        wrong_step_index = None
        correct_step = None
        
        self.logs.append("\n=== Analyzing solution steps ===")
        
        # First check the analysis section (index 0)
        self.logs.append("\nChecking analysis section...")
        has_correct, current_step = await self._verify_completions(
            problem,
            partial_solutions[0],
            correct_answer,
            0
        )
        
        if not has_correct:
            self.logs.append("✗ Found wrong analysis section - skipping example")
            return None
            
        self.logs.append("✓ Analysis section is valid")
        correct_step = current_step
        
        # Then check numbered steps
        for i in range(1, len(partial_solutions)):
            self.logs.append(f"\nChecking step {i}...")
            # Try completions
            has_correct, current_step = await self._verify_completions(
                problem,
                partial_solutions[i],
                correct_answer,
                i
            )
            
            if not has_correct:
                wrong_step_index = i
                self.logs.append(f"✗ Found wrong step at step {i}")
                
                # Use the last known correct step from our iterative verification
                self.logs.append(f"✓ Using last known correct step from verification process:")
                self.logs.append("=" * 50)
                self.logs.append(correct_step)
                self.logs.append("=" * 50)
                
                return {
                    'problem': problem,
                    'correct_answer': correct_answer,
                    'wrong_solution': wrong_solution,
                    'wrong_step_index': wrong_step_index,
                    'wrong_step': steps[wrong_step_index],
                    'partial_solution': partial_solutions[max(0, wrong_step_index - 1)],
                    'correct_step': correct_step,
                    'correct_solution': correct_solution
                }
            
            self.logs.append(f"✓ Step {i} is valid")
            correct_step = current_step  # Update the last known correct step
            # Continue checking next step since this one is valid
                
        # If we get here, all steps were valid
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
                wrong_quality = analyze_solution_quality(result['wrong_solution'])
                all_logs.append(f"✓ Wrong solution:")
                all_logs.append(f"  ├─ Length: {wrong_quality['length']} words")
                all_logs.append(f"  ├─ Steps: {wrong_quality['step_count']}")
                all_logs.append(f"  ├─ Has equations: {'Yes' if wrong_quality['has_equations'] else 'No'}")
                all_logs.append(f"  └─ Format score: {wrong_quality['formatting_quality']}/5")
                
                # Add wrong step details
                all_logs.append("\n🔍 Wrong Step Details:")
                all_logs.append(f"✓ Found at step: {result['wrong_step_index']}")
                all_logs.append(f"✓ Wrong step content:")
                all_logs.append(result['wrong_step'])
            
            # Print logs for this example
            print("\n".join(all_logs))
            
            # Save comprehensive logs to markdown file
            log_file = logger.save_logs(all_logs, example_id)
            
            # Add example ID and logs to result
            if result:
                result['id'] = example_id
                result['logs'] = "\n".join(all_logs)
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
