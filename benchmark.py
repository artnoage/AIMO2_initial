import os
import json
import asyncio
import argparse
from asyncio import TimeoutError
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from utils.utils import *
from langchain_core.messages import  HumanMessage, SystemMessage
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from utils.utils import extract_answer_from_solution


os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Load environment variables from .env file
load_dotenv()


def save_results(results: list, model_name: str):
    """
    Save the benchmarking results to a JSON file within the benchmark_results directory.
    The filename includes the model name and a timestamp for uniqueness.
    """
    os.makedirs('../benchmark_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join('benchmark_results', f"benchmark_results_{model_name}_{timestamp}.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, best_of: int = 1) -> Optional[Dict]:
    """
    Process a single example and print its results immediately:
    - Count input tokens
    - Extract the correct answer from the solution
    - Generate the solution using the model
    - Extract the model's answer
    - Count output tokens
    - Determine correctness
    - Print results
    """
    try:
        # Validate input data
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        # Extract the correct answer
        try:
            correct_answer = extract_answer_from_solution(example['solution'])
            if correct_answer is None:
                print(f"Warning: Could not extract answer from solution for example {running_id}")
                print(f"Solution text: {example['solution']}...")
                return None
        except Exception as e:
            print(f"Error extracting answer from solution for example {running_id}: {str(e)}")
            return None
        # Create the chat prompt
        prompt = [SystemMessage(content=BENCHMARK_SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        # Make multiple attempts
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
        for attempt in range(best_of):
            # Try up to 3 times for each attempt in case of connection errors
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Add 5 minute timeout
                    response = await asyncio.wait_for(
                        solver_model.ainvoke(prompt),
                        timeout=300  # 5 minutes in seconds
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
                    await asyncio.sleep(1)  # Wait a second before retrying
            current_answer = extract_answer_from_solution(current_solution)
            
            # Verify the solution
            is_correct = await compare_math_answers(current_answer, correct_answer, example["problem"], verifier_model)
            
            if is_correct:
                correct_count += 1
                if best_solution is None:  # Keep the first correct solution
                    best_solution = current_solution
                    best_answer = current_answer
            
            solutions.append({
                'solution': current_solution,
                'answer': current_answer,
                'is_correct': is_correct
            })
            
            # Always collect all attempts up to best_of
            if attempt >= best_of - 1:
                break
        
        # Use the best solution if we found one, otherwise use the first attempt
        solution = best_solution if best_solution is not None else solutions[0]['solution']
        model_answer = best_answer if best_answer is not None else solutions[0]['answer']
        
        # First check if solution contains required keywords
        solution_lower = solution.lower()
        has_problem = 'problem' in solution_lower
        has_analysis = 'analysis' in solution_lower
        has_step = 'step' in solution_lower
        
        # Only verify if all required words are present
        is_correct = False
        if has_problem and has_analysis and has_step:
            is_correct = await compare_math_answers(model_answer, correct_answer, example["problem"], verifier_model)
        
        # Print results immediately
        success_ratio = f"{correct_count}/{best_of}"
        success_percentage = (correct_count / best_of) * 100
        print(f"\nProblem {running_id + 1}: {success_ratio} ({success_percentage:.1f}%)")
        print(f"Extracted Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        # Return the result
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'model_answer_raw': model_answer,  # Keep the best/last answer for compatibility
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
    # Start timing the entire process
    start_time = datetime.now()
    
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Benchmark model on NuminaMath-CoT dataset')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL', help='Model to use for solving problems')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='GEMINI_FLASH', help='Model to use for verifying answers')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--dataset', type=str, default='filtered',
                       choices=['original', 'filtered', 'aime'],
                       help='Dataset to use: original (NuminaMath-CoT), filtered (Numina-Numerics), or aime (AIME validation)')
    parser.add_argument('--max-concurrent', type=int, default=16,
                       help='Maximum number of concurrent problems (default: 32)')
    parser.add_argument('--best-of', type=int, default=5,
                       help='Number of attempts per problem (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for model generation (default: 0.5)')
    args = parser.parse_args()

    # Validate max concurrent
    if args.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return
    args = parser.parse_args()

    # Load the dataset based on selection
    try:
        if args.dataset == 'original':
            dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
        elif args.dataset == 'aime':
            dataset = load_dataset("AI-MO/aimo-validation-aime", split=args.split)
        else:  # filtered
            username = HfApi().whoami()["name"]
            dataset = load_dataset(f"{username}/Numina", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter by source if specified
    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    # Shuffle the dataset for randomness
    dataset = dataset.shuffle(seed=42)

    # Print dataset information
    print("\nDataset Information:")
    num_examples = len(dataset)
    print(f"Number of examples: {num_examples}")

    if num_examples == 0:
        print("Error: Dataset is empty! Check your source filter and split arguments.")
        return

    # Initialize the models
    try:
        solver_model = get_model(ModelOption[args.solver], temp=args.temperature)
        verifier_model = get_model(ModelOption[args.verifier])
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    print(f"\nBenchmarking solver: {args.solver}, verifier: {args.verifier} on {args.split} split...")


    # Prepare the list of examples to process
    example_data = []
    for example in dataset:
        processed = {
            'id': example['id'],  # Use dataset ID
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
        # Count results where at least one attempt was correct
        correct_count = sum(1 for r in results if any(r['is_correct_list']))
        return 1.0 - (correct_count / len(results))

    # Process examples with controlled concurrency
    results = []
    error_rate_points = []
    total_examples = len(example_data)
    print(f"\nStarting processing of {total_examples} examples...")

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, example['id'], solver_model, verifier_model, args.best_of)

    # Create tasks for all examples with best_of parameter
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    print(f"\nWill process {len(example_data)} examples")
        
    # Process all examples with progress bar
    progress_bar = tqdm(total=total_examples, desc="Processing examples")
    current_batch = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            # Add to current batch using data we already have
            augmented_example = {
                'id': result['id'],
                'problem': result['problem'],
                'solution': result['correct_answer'],
                'model_responses': result['model_responses'],
                'is_correct_list': result['is_correct_list'],
                'solver': args.solver,
                'verifier': args.verifier,
                'total_attempts': len(result['model_responses']),
                'correct_answer': result['correct_answer']
            }
            current_batch.append(augmented_example)
            
            # Save error rate every 100 examples
            if len(results) % 100 == 0:
                # Calculate error rate for the last 100 results
                last_hundred = results[-100:]
                batch_error_rate = calculate_error_rate(last_hundred)
                # Also calculate cumulative error rate
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
                    f"benchmark_intermediate_{args.solver}_{args.verifier}_{timestamp}.json")
                output_data = {
                    'error_rate_points': error_rate_points,
                    'current_batch_error_rate': batch_error_rate,
                    'current_cumulative_error_rate': cumulative_error_rate
                }
                os.makedirs('results', exist_ok=True)
                with open(intermediate_filename, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nSaved intermediate results after {len(results)} examples")
            
            # Save current batch of augmented data
            if current_batch:
                current_batch = []
            
        progress_bar.update(1)
    progress_bar.close()

    if not results:
        print("\nNo examples were successfully processed.")
        return

    # Sort results by ID to maintain the original order
    results.sort(key=lambda x: x['id'])

    # Calculate final statistics
    any_correct_count = sum(1 for r in results if any(r['is_correct_list']))
    majority_correct_count = sum(1 for r in results if sum(r['is_correct_list']) > len(r['is_correct_list'])/2)

    # Print final results
    progress_bar.close()
    print("\n\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    
    if len(results) > 0:
        any_accuracy = (any_correct_count / len(results)) * 100
        majority_accuracy = (majority_correct_count / len(results)) * 100
        print(f"Any-Correct Accuracy: {any_correct_count}/{len(results)} = {any_accuracy:.2f}%")
        print(f"Majority-Correct Accuracy: {majority_correct_count}/{len(results)} = {majority_accuracy:.2f}%")
        
        # Calculate best-of-N statistics
        at_least_one_correct = sum(1 for r in results if r['attempts']['correct_count'] > 0)
        majority_correct = sum(1 for r in results if r['attempts']['correct_count'] > args.best_of // 2)
        
        print(f"\nBest-of-{args.best_of} Statistics:")
        print(f"Problems with at least one correct solution: {at_least_one_correct}/{len(results)} = {(at_least_one_correct/len(results))*100:.2f}%")
        print(f"Problems with majority correct solutions: {majority_correct}/{len(results)} = {(majority_correct/len(results))*100:.2f}%")
    else:
        print("No examples were successfully processed.")

    # Save final results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join('results', 
                                  f"benchmark_results_{args.solver}_{args.verifier}_{timestamp}.json")
    
    # Save final error rate and points
    final_batch_error_rate = calculate_error_rate(results[-100:] if len(results) >= 100 else results)
    final_cumulative_error_rate = calculate_error_rate(results)
    error_rate_points.append({
        'examples_processed': len(results),
        'batch_error_rate': final_batch_error_rate,
        'cumulative_error_rate': final_cumulative_error_rate
    })
    
    # Collect all command line parameters
    run_parameters = {
        'solver': args.solver,
        'verifier': args.verifier,
        'split': args.split,
        'source': args.source,
        'dataset': args.dataset,
        'max_concurrent': args.max_concurrent,
        'best_of': args.best_of,
        'temperature': args.temperature
    }

    # Calculate timing information
    end_time = datetime.now()
    total_duration = end_time - start_time

    # Collect final statistics
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
    
    # Save final augmented data batch
    if current_batch:
        current_batch = []
    
    # Print error rate progression
    print("\nError Rate Progression:")
    for point in error_rate_points:
        print(f"After {point['examples_processed']} examples:")
        print(f"  Batch Error Rate (last 100): {point['batch_error_rate']:.4f}")
        print(f"  Cumulative Error Rate: {point['cumulative_error_rate']:.4f}")

    # Calculate and print timing information
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
