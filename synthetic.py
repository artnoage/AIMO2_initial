import os
import asyncio
from typing import Optional, Dict, List
from datetime import datetime
from dotenv import load_dotenv
from utils.progress_tracker import ProgressTracker
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import run_benchmark
from utils.agents import FullSolutionAgent, AnswerVerifierAgent, SolutionVerifierAgent

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

from utils.verification import verify_solution_with_model

def check_format(response: str, full_solution: str) -> bool:
    """Check if response contains required words and is sufficiently detailed"""
    lower_response = response.lower()
    required_words = ['analysis', 'problem', 'step']
    has_required_words = all(word in lower_response for word in required_words)
    is_long_enough = len(response) >= len(full_solution) * 1.03
    has_no_links = 'http' not in lower_response
    return has_required_words and is_long_enough and has_no_links

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, best_of: int) -> Optional[Dict]:
    """Process a single example with multiple attempts and verification levels"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None

        solution_agent = FullSolutionAgent(solver_model)
        solutions = []
        verification_levels = []
        best_solution = None
        
        for attempt in range(best_of):
            try:
                current_solution = await solution_agent.generate(example["problem"], running_id, attempt)
                
                # Verify solution
                level, current_answer = await verify_solution_with_model(
                    current_solution,
                    example['solution'],
                    example['problem'],
                    verifier_model,
                    second_verifier_model
                )
                
                solutions.append(current_solution)
                verification_levels.append(level)
                
                # Store best solution
                if level == 4 and best_solution is None:
                    best_solution = current_solution
                    
                # If we have a perfect solution and at least one other attempt, we can stop
                if level == 4 and attempt > 0:
                    break
                    
            except Exception as e:
                print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
                solutions.append("Error occurred")
                verification_levels.append(0)
        
        # Calculate statistics
        level_counts = {i: verification_levels.count(i) for i in range(5)}
        
        # Print results
        print(f"\nExample {running_id + 1}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Verification levels: {verification_levels}")
        print(f"Level counts: {level_counts}")
        print(f"Success: {'Yes' if 4 in verification_levels else 'No'}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'model_responses': solutions,
            'verification_levels': verification_levels,
            'level_counts': level_counts,
            'best_solution': best_solution,
            'solved': 4 in verification_levels,
            'solution_type': 'synthetic'
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for synthetic solution generation and verification."""
    config = BenchmarkConfig.from_args('Synthetic Model Benchmark')
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(
        total_examples=config.split_slice.stop if config.split_slice else 0,
        best_of=config.best_of
    )
    
    await run_benchmark(
        config,
        lambda example, running_id, example_id, solver_model, verifier_model, best_of:
            process_example(
                example=example,
                running_id=running_id,
                example_id=example_id,
                solver_model=solver_model,
                verifier_model=verifier_model,
                second_verifier_model=get_model(ModelOption[config.second_verifier], temp=config.verifier_temp),
                best_of=best_of
            ),
        progress_tracker=progress_tracker
    )
    
    # Save final results
    progress_tracker.save_results(config.solver, config.split)
    progress_tracker.print_final_stats()

if __name__ == "__main__":
    asyncio.run(main())
