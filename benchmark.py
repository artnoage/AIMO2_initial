import os
import asyncio
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()



    
    
async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, best_of: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        solution_agent = FullSolutionAgent(solver_model)
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
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
                level, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer,
                    example["problem"]
                )
                
                if level == 4:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
                        best_answer = current_answer
                        
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'verification_level': level,
                    'is_correct': level == 4
                })
            except Exception as e:
                print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
                current_solution = "Error occurred"
                current_answer = None
                is_correct = False
            
            solutions.append({
                'solution': current_solution,
                'answer': current_answer,
                'is_correct': is_correct
            })
        
    
        model_answer = best_answer if best_answer is not None else solutions[0]['answer']
        
        # Print statistics
        print(f"\nExample {running_id + 1}:")
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
            'correct_answer': correct_answer,
            'model_responses': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'correct_binary': [1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions],
            'model_answer_raw': model_answer,
            'correct_answer_raw': correct_answer,
            'attempts': {
                'total': len(solutions),
                'correct_count': correct_count
            },
            'solution_type': 'complete'  # Indicate this is a complete solution, not step-by-step
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for benchmarking mathematical problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems')
    verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
    
    await run_benchmark(
        config,
        lambda example, running_id, example_id, solver_model, verifier_model, best_of:
            process_example(
                example=example,
                running_id=running_id,
                example_id=example_id,
                solver_model=solver_model,
                verifier_model=verifier_model,
                second_verifier_model=second_verifier_model,
                best_of=best_of,
                config=config
            )
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
