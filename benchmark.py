import os
import asyncio
from typing import Optional, Dict
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *
from bench_utils.progress_tracker import ProgressTracker
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, best_of: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {str(running_id)}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {str(running_id)}")
            return None

        solution_agent = FullSolutionAgent(solver_model)
        solutions = []
        correct_count = 0
        best_solution = None
        
        for attempt in range(best_of):
            try:
                current_solution = await solution_agent.generate(example["problem"], running_id, attempt)
                
                # Create and use appropriate verifier
                verifier = create_verifier(
                    config.verification_type,
                    verifier_model=verifier_model,
                    second_verifier_model=second_verifier_model,
                    tolerance=config.tolerance
                )
                score, total_steps, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer,
                    example["problem"]
                )
                if score == total_steps:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
                        
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'verification_score': score,
                    'verification_steps': total_steps,
                    'is_correct': score == total_steps
                })
            except Exception as e:
                print(f"Error in attempt {str(attempt + 1)} for example {str(running_id)}: {str(e)}")
                # Handle error case
                solution_info = {
                    'solution': "Error occurred",
                    'answer': None,
                    'verification_score': 0,
                    'verification_steps': 1,
                    'is_correct': False
                }
                solutions.append(solution_info)
        
        
        # Print statistics
        print(f"\nExample {str(running_id + 1)}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{best_of}")
        print(f"Success rate: {(correct_count/best_of)*100:.1f}%")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'correct_answer': correct_answer,
            'model_solutions': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'verification_scores': [s['verification_score'] for s in solutions],
            'verification_steps': [s['verification_steps'] for s in solutions]
        }
        
    except Exception as e:
        print(f"Error processing example {str(running_id)}: {e}")
        return None

async def main():
    """Main function for benchmarking mathematical problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems')
    verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
    
    # Initialize models
    solver_model = get_model(ModelOption[config.solver], temp=config.temperature)
    
    global progress_tracker
    
    await run_benchmark(
        config=config,
        process_example_func=process_example,
        solver_model=solver_model,
        verifier_model=verifier_model,
        second_verifier_model=second_verifier_model
    )
    
    if progress_tracker:
        progress_tracker.print_final_stats()

if __name__ == "__main__":
    progress_tracker = None
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if progress_tracker:
            progress_tracker.print_final_stats()
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        if progress_tracker:
            progress_tracker.print_final_stats()
