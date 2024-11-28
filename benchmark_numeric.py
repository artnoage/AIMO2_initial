import os
import json
import asyncio
import argparse
from asyncio import TimeoutError
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from utils.augmented_data_handler import handle_augmented_data_file, save_augmented_data, get_existing_ids
from utils.utils import ModelOption, get_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from utils.utils import extract_answer_from_solution
import re

SYSTEM_PROMPT="""You are a precise mathematical problem solver. You will be given a problem to solve.

DO:
▪ List applicable theorems/techniques upfront
▪ If possible each step must contain a justification
▪ Use LaTeX notation
▪ Your final answer MUST be a single number in a LaTeX box

FORMAT:

**Problem Analysis and Approach**:
1. Start by categorizing the problem
2. List specific tools or theorems that will guide your solution

**PROOF**:
Show your work step by step with clear justifications in brackets.

**ANSWER**:
\\(\\boxed{n}\\) where n is your final numeric answer"""

def extract_numeric_answer(solution: str) -> Optional[float]:
    """
    Extract numeric answer from a solution string.
    Looks for a number inside a LaTeX boxed environment.
    Returns float if found, None otherwise.
    """
    # Look for \boxed{<number>} pattern
    pattern = r'\\boxed\{([+-]?\d*\.?\d+)\}'
    match = re.search(pattern, solution)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def is_answer_correct(model_answer: Optional[float], correct_answer: Optional[float], tolerance: float = 0.001) -> bool:
    """Compare two numeric answers within tolerance"""
    if model_answer is None or correct_answer is None:
        return False
    return abs(model_answer - correct_answer) <= tolerance

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, best_of: int = 1) -> Optional[Dict]:
    """Process a single example and return results"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        # Extract the correct answer
        try:
            correct_answer = extract_numeric_answer(example['solution'])
            if correct_answer is None:
                print(f"Warning: Could not extract numeric answer from solution for example {running_id}")
                return None
        except Exception as e:
            print(f"Error extracting answer from solution for example {running_id}: {str(e)}")
            return None

        prompt = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
        for attempt in range(best_of):
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = await asyncio.wait_for(
                        solver_model.ainvoke(prompt),
                        timeout=300
                    )
                    current_solution = response.content
                    break
                except (Exception, TimeoutError) as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed after {max_retries} attempts for problem {running_id + 1}, attempt {attempt + 1}")
                        if isinstance(e, TimeoutError):
                            print(f"Timeout error: Model took longer than 5 minutes to respond")
                        raise e
                    print(f"{'Timeout' if isinstance(e, TimeoutError) else 'Connection'} error for problem {running_id + 1}, attempt {attempt + 1}. Retrying... ({retry_count}/{max_retries})")
                    await asyncio.sleep(1)

            current_answer = extract_numeric_answer(current_solution)
            is_correct = is_answer_correct(current_answer, correct_answer)
            
            if is_correct:
                correct_count += 1
                if best_solution is None:
                    best_solution = current_solution
                    best_answer = current_answer
            
            solutions.append({
                'solution': current_solution,
                'answer': current_answer,
                'is_correct': is_correct
            })
            
            if attempt >= best_of - 1:
                break
        
        solution = best_solution if best_solution is not None else solutions[0]['solution']
        model_answer = best_answer if best_answer is not None else solutions[0]['answer']
        
        success_ratio = f"{correct_count}/{best_of}"
        success_percentage = (correct_count / best_of) * 100
        print(f"\nProblem {running_id + 1}: {success_ratio} ({success_percentage:.1f}%)")
        print(f"Correct Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'model_answer_raw': model_answer,
            'correct_answer_raw': correct_answer,
            'attempts': {
                'total': len(solutions),
                'correct_count': correct_count
            }
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Benchmark model on numeric problems')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL', help='Model to use for solving problems')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--max-concurrent', type=int, default=16,
                       help='Maximum number of concurrent problems (default: 16)')
    parser.add_argument('--best-of', type=int, default=5,
                       help='Number of attempts per problem (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for model generation (default: 0.7)')
    parser.add_argument('--tolerance', type=float, default=0.001,
                       help='Tolerance for numeric comparison (default: 0.001)')
    args = parser.parse_args()

    if args.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    dataset = dataset.shuffle(seed=42)

    print("\nDataset Information:")
    num_examples = len(dataset)
    print(f"Number of examples: {num_examples}")

    if num_examples == 0:
        print("Error: Dataset is empty! Check your source filter and split arguments.")
        return

    try:
        solver_model = get_model(ModelOption[args.solver], temp=args.temperature)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    print(f"\nBenchmarking solver: {args.solver} on {args.split} split...")

    example_data = []
    for example in dataset:
        processed = {
            'id': example['id'],
            'problem': example['problem'],
            'solution': example['solution']
        }
        example_data.append(processed)
    
    if not example_data:
        print("No valid examples to process after initial filtering.")
        return

    def calculate_error_rate(results):
        if not results:
            return 0.0
        correct_count = sum(1 for r in results if any(r['is_correct_list']))
        return 1.0 - (correct_count / len(results))

    results = []
    error_rate_points = []
    total_examples = len(example_data)
    print(f"\nStarting processing of {total_examples} examples...")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, example['id'], solver_model, args.best_of)

    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    augmented_filename = os.path.join('augmented_datasets', 
                                    f"benchmark_numeric_{timestamp}.json")
    
    existing_ids = get_existing_ids(augmented_filename)
    if existing_ids:
        print(f"\nFound {len(existing_ids)} existing examples - will skip these IDs")
    
    example_data = [ex for ex in example_data if ex['id'] not in existing_ids]
    if not example_data:
        print("All examples have already been processed!")
        return
        
    print(f"\nWill process {len(example_data)} new examples")
    
    if not handle_augmented_data_file(augmented_filename):
        print("Operation cancelled by user.")
        return
        
    progress_bar = tqdm(total=total_examples, desc="Processing examples")
    current_batch = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            augmented_example = {
                'id': result['id'],
                'problem': result['problem'],
                'solution': str(result['correct_answer']),
                'model_responses': result['model_responses'],
                'is_correct_list': result['is_correct_list'],
                'solver': args.solver,
                'total_attempts': len(result['model_responses']),
                'correct_answer': result['correct_answer']
            }
            current_batch.append(augmented_example)
            
            if len(results) % 100 == 0:
                last_hundred = results[-100:]
                batch_error_rate = calculate_error_rate(last_hundred)
                cumulative_error_rate = calculate_error_rate(results)
                error_rate_points.append({
                    'examples_processed': len(results),
                    'batch_error_rate': batch_error_rate,
                    'cumulative_error_rate': cumulative_error_rate
                })
                print(f"\nAt {len(results)} examples:")
                print(f"Batch Error Rate (last 100): {batch_error_rate:.4f}")
                print(f"Cumulative Error Rate: {cumulative_error_rate:.4f}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                intermediate_filename = os.path.join('results', 
                    f"benchmark_numeric_{args.solver}_{timestamp}.json")
                output_data = {
                    'error_rate_points': error_rate_points,
                    'current_batch_error_rate': batch_error_rate,
                    'current_cumulative_error_rate': cumulative_error_rate
                }
                os.makedirs('results', exist_ok=True)
                with open(intermediate_filename, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nSaved intermediate results after {len(results)} examples")
            
            if current_batch:
                save_augmented_data(current_batch, augmented_filename, len(results))
                current_batch = []
            
        progress_bar.update(1)
    progress_bar.close()

    if not results:
        print("\nNo examples were successfully processed.")
        return

    results.sort(key=lambda x: x['id'])

    any_correct_count = sum(1 for r in results if any(r['is_correct_list']))
    majority_correct_count = sum(1 for r in results if sum(r['is_correct_list']) > len(r['is_correct_list'])/2)

    progress_bar.close()
    print("\n\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    
    if len(results) > 0:
        any_accuracy = (any_correct_count / len(results)) * 100
        majority_accuracy = (majority_correct_count / len(results)) * 100
        print(f"Any-Correct Accuracy: {any_correct_count}/{len(results)} = {any_accuracy:.2f}%")
        print(f"Majority-Correct Accuracy: {majority_correct_count}/{len(results)} = {majority_accuracy:.2f}%")
        
        at_least_one_correct = sum(1 for r in results if r['attempts']['correct_count'] > 0)
        majority_correct = sum(1 for r in results if r['attempts']['correct_count'] > args.best_of // 2)
        
        print(f"\nBest-of-{args.best_of} Statistics:")
        print(f"Problems with at least one correct solution: {at_least_one_correct}/{len(results)} = {(at_least_one_correct/len(results))*100:.2f}%")
        print(f"Problems with majority correct solutions: {majority_correct}/{len(results)} = {(majority_correct/len(results))*100:.2f}%")
    else:
        print("No examples were successfully processed.")

    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join('results', 
                                  f"benchmark_numeric_{args.solver}_{timestamp}.json")
    
    final_batch_error_rate = calculate_error_rate(results[-100:] if len(results) >= 100 else results)
    final_cumulative_error_rate = calculate_error_rate(results)
    error_rate_points.append({
        'examples_processed': len(results),
        'batch_error_rate': final_batch_error_rate,
        'cumulative_error_rate': final_cumulative_error_rate
    })
    
    run_parameters = {
        'solver': args.solver,
        'split': args.split,
        'source': args.source,
        'max_concurrent': args.max_concurrent,
        'best_of': args.best_of,
        'temperature': args.temperature,
        'tolerance': args.tolerance
    }

    end_time = datetime.now()
    total_duration = end_time - start_time

    final_statistics = {
        'total_examples': len(results),
        'any_correct': any_correct_count,
        'any_correct_accuracy': any_accuracy,
        'majority_correct': majority_correct_count,
        'majority_correct_accuracy': majority_accuracy,
        'best_of_stats': {
            'at_least_one_correct': at_least_one_correct,
            'at_least_one_correct_percentage': (at_least_one_correct/len(results))*100,
            'majority_correct': majority_correct,
            'majority_correct_percentage': (majority_correct/len(results))*100
        },
        'timing': {
            'total_duration_seconds': total_duration.total_seconds(),
            'average_time_per_example': total_duration.total_seconds() / len(results)
        }
    }

    output_data = {
        'run_parameters': run_parameters,
        'error_rate_points': error_rate_points,
        'final_batch_error_rate': final_batch_error_rate,
        'final_cumulative_error_rate': final_cumulative_error_rate,
        'final_statistics': final_statistics
    }
    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {results_filename}")
    
    if current_batch:
        save_augmented_data(current_batch, augmented_filename, len(results))
    
    print("\nError Rate Progression:")
    for point in error_rate_points:
        print(f"After {point['examples_processed']} examples:")
        print(f"  Batch Error Rate (last 100): {point['batch_error_rate']:.4f}")
        print(f"  Cumulative Error Rate: {point['cumulative_error_rate']:.4f}")

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
