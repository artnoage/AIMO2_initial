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
        
    def _split_into_steps(self, solution: str) -> List[str]:
        """Split a solution into analysis and numbered steps"""
        # First split into analysis and steps sections
        parts = solution.lower().split("step")
        if not parts:
            return []
            
        steps = []
        # Add analysis section if present
        analysis = parts[0]
        if "analysis" in analysis.lower():
            steps.append(parts[0])
            
        # Process numbered steps
        for step in parts[1:]:
            if step.strip():  # Skip empty steps
                # Reconstruct the step with its prefix
                full_step = "Step" + step
                steps.append(full_step)
                
        return steps
        
    def _get_partial_solutions(self, steps: List[str]) -> List[str]:
        """Generate partial solutions ending at each step"""
        partial_solutions = []
        current = ""
        
        for step in steps:
            current += step + "\n"
            partial_solutions.append(current)
            
        return partial_solutions
        
    async def _verify_completions(
        self,
        problem: str,
        partial_solution: str,
        correct_answer: str,
        step_index: int
    ) -> Tuple[bool, Optional[str]]:
        """Try multiple completions of a partial solution to check if any are correct"""
        successful = 0
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
                    # Extract the correct step at the given index
                    completion_steps = self._split_into_steps(completion)
                    if len(completion_steps) > 0:
                        correct_step = completion_steps[0]
                        if i == self.completions - 1:  # Last attempt
                            self.logs.append(f"Step {step_index}: {successful}/{self.completions} completions successful")
                            return True, correct_step
                    
            except Exception:
                continue
        
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
        steps = self._split_into_steps(wrong_solution)
        if len(steps) < 2:  # Need at least analysis + one step
            return None
            
        # Get partial solutions
        partial_solutions = self._get_partial_solutions(steps)
        
        # Find first step that makes all completions wrong
        wrong_step_index = None
        correct_completion = None
        
        self.logs.append("\n=== Analyzing solution steps ===")
        for i, partial in enumerate(partial_solutions):
            self.logs.append(f"\nChecking step {i}...")
            # Try completions
            has_correct, correct_step = await self._verify_completions(
                problem,
                partial,
                correct_answer,
                i
            )
            
            if not has_correct:
                wrong_step_index = i
                self.logs.append(f"✗ Found wrong step at index {i}")
                break
            else:
                correct_step = correct_step
                self.logs.append(f"✓ Step {i} is valid")
                
        if wrong_step_index is None or correct_step is None:
            self.logs.append("❌ Could not identify wrong step")
            return None
            
        # Print collected logs
        print("\n".join(self.logs))
            
        return {
            'problem': problem,
            'correct_answer': correct_answer,
            'wrong_solution': wrong_solution,
            'wrong_step_index': wrong_step_index,
            'wrong_step': steps[wrong_step_index],
            'partial_solution': partial_solutions[wrong_step_index - 1],
            'correct_step': correct_step
        }

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
