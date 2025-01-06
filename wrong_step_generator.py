import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *

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
            print("\nDEBUG: Found successful completion:")
            print("=" * 50)
            print(correct_completion)
            print("=" * 50)
            
            # Get all steps from the completion
            completion_steps = self._split_into_steps(correct_completion)
            print(f"\nDEBUG: Split into {len(completion_steps)} steps:")
            for i, step in enumerate(completion_steps):
                print(f"\nStep {i}:")
                print("-" * 30)
                print(step)
                print("-" * 30)
            
            # Extract the step at the current index
            if step_index < len(completion_steps):
                correct_step = completion_steps[step_index]
                print(f"\nDEBUG: Extracted step {step_index}:")
                print("-" * 30)
                print(correct_step)
                print("-" * 30)
            else:
                correct_step = None
                print(f"\nDEBUG: Step index {step_index} out of bounds (max: {len(completion_steps)-1})")
                
            if step_index == 0:
                print(f"Analysis section: {successful}/{self.completions} completions successful")
            else:
                print(f"Step {step_index}: {successful}/{self.completions} completions successful")
            return True, correct_step

        # No successful completion found
        if step_index == 0:
            print(f"Analysis section: {successful}/{self.completions} completions successful")
        else:
            print(f"Step {step_index}: {successful}/{self.completions} completions successful")
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
        # Try to generate wrong but valid solutions
        wrong_solution = None
        attempts = 0
        
        while attempts < self.best_of:
            try:
                attempts += 1
                solution = await self.solution_agent.generate(problem)
                
                # Validate solution structure
                is_valid, _ = validate_solution(solution)
                if not is_valid:
                    continue
                    
                # Check if solution is wrong
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if not is_correct:
                    wrong_solution = solution
                    break
                    
            except Exception:
                continue
                
        if wrong_solution is None:
            return None
            
        # Split wrong solution into steps
        steps = split_into_steps(wrong_solution)
        if not steps:
            print("❌ No steps found in solution - likely incorrect format")
            return None
            
        if len(steps) < 2:  # Need at least analysis + one step
            print("❌ Not enough steps found (need at least analysis + one step)")
            return None
            
        # Get partial solutions
        partial_solutions = get_partial_solutions(steps)
        
        # Find first step that makes all completions wrong
        wrong_step_index = None
        correct_completion = None
        
        print("\n=== Analyzing solution steps ===")
        
        # First check the analysis section (index 0)
        print("\nChecking analysis section...")
        has_correct, correct_step = await self._verify_completions(
            problem,
            partial_solutions[0],
            correct_answer,
            0
        )
        
        if not has_correct:
            print("✗ Found wrong analysis section - skipping example")
            return None
            
        print("✓ Analysis section is valid")
        
        # Then check numbered steps
        for i in range(1, len(partial_solutions)):
            print(f"\nChecking step {i}...")
            # Try completions
            has_correct, correct_step = await self._verify_completions(
                problem,
                partial_solutions[i],
                correct_answer,
                i
            )
            
            if not has_correct:
                wrong_step_index = i
                print(f"✗ Found wrong step at step {i}")
                return {
                    'problem': problem,
                    'correct_answer': correct_answer,
                    'wrong_solution': wrong_solution,
                    'wrong_step_index': wrong_step_index,
                    'wrong_step': steps[wrong_step_index],
                    'partial_solution': partial_solutions[max(0, wrong_step_index - 1)],
                    'correct_step': correct_step  # Save the correct step from the successful completion
                }
            
            print(f"✓ Step {i} is valid")
            # Continue checking next step since this one is valid
                
        # If we get here, all steps were valid
        print("✓ All steps are valid")
        return None

async def main():
    """Main function for wrong step generation"""
    config = BenchmarkConfig.from_args('Wrong step generation approach')
    
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
                print(f"Warning: Could not extract answer from solution for example {running_id}")
                return None
                
            # Generate wrong step
            result = await generator.generate(example['problem'], correct_answer)
            if result is None:
                return None
                
            # Add example ID
            result['id'] = example_id
            return [result]
            
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
