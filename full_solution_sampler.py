import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class FullSolutionSampler:
    """Samples full solutions until finding correct answer or hitting max attempts"""
    
    def __init__(self, solver, best_of: int):
        self.solver = solver
        self.best_of = best_of
        self.solution_agent = FullSolutionAgent(solver)
        self.verifier = NumericVerifier()

    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate and validate full solutions until finding correct one.
        Returns dict with prompt/chosen/rejected/scores if found, None otherwise.
        """
        solution_prompt = None
        best_solution = None
        worst_solution = None
        best_score = 0.0
        worst_score = float('inf')
        
        for attempt in range(self.best_of):
            try:
                # Generate solution
                if solution_prompt is None:
                    prompt, solution = await self.solution_agent.generate(
                        problem,
                        return_prompt=True
                    )
                    solution_prompt = prompt
                else:
                    solution = await self.solution_agent.generate(problem)

                # Validate solution format
                is_valid, reason = validate_solution(solution)
                if not is_valid:
                    print(f"✗ Solution validation failed: {reason}")
                    continue

                # Extract and verify answer
                answer = extract_answer_from_solution(solution)
                if answer is None:
                    continue
                    
                is_correct, reason = await self.verifier.verify(
                    solution,
                    correct_answer, 
                    problem
                )
                
                # Score: 1.0 for correct, 0.0 for incorrect
                score = float(is_correct)
                
                # Update best/worst tracking
                if score > best_score:
                    best_score = score
                    best_solution = solution
                if score < worst_score:
                    worst_score = score 
                    worst_solution = solution
                    
                # Early exit if we found a correct solution
                if is_correct:
                    break
                    
            except Exception:
                continue
                
        # Return None if we didn't find both a best and worst solution
        if best_solution is None or worst_solution is None:
            return None
            
        # Only return results if we found a correct solution
        if best_score == 1.0:
            return {
                'prompt': {'content': solution_prompt, 'role': 'user'},
                'chosen': {'content': best_solution, 'role': 'assistant'},
                'rejected': {'content': worst_solution, 'role': 'assistant'},
                'score_chosen': best_score,
                'score_rejected': worst_score
            }
            
        return None

async def main():
    """Main function for full solution sampling"""
    config = BenchmarkConfig.from_args('Full solution sampling for training data')
    
    async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
        """Process a single example using full solution sampling"""
        try:
            # Initialize solver
            solver = get_model(ModelOption[config.solver], temp=config.temperature)
            
            # Create sampler
            sampler = FullSolutionSampler(solver, config.best_of)
            
            # Extract answer
            correct_answer = extract_answer_from_solution(example['solution'])
            if correct_answer is None:
                print(f"Warning: Could not extract answer from solution for example {running_id}")
                return None
                
            # Generate and validate solutions
            result = await sampler.generate(example['problem'], correct_answer)
            
            # Add example ID if we got a result
            if result is not None:
                result['id'] = example_id
                
            return result
            
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
