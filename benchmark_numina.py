import os
import re
import json
import asyncio
import argparse
from enum import Enum
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from utils.augmented_data_handler import handle_augmented_data_file, save_augmented_data, get_existing_ids
from utils.utils import ModelOption, get_model
from typing import List, Dict, Optional
from itertools import islice
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import HfApi
from tqdm import tqdm
import time
from utils.utils import extract_answer_from_solution
from utils.utils import ModelOption

def load_intermediate_results(solver_model: ModelOption, verifier_model: ModelOption) -> Tuple[Optional[List[int]], Optional[List[str]], Optional[List[float]]]:
    """Load intermediate results from saved JSON files"""
    intermediate_files = [f for f in os.listdir('benchmark_results') 
                        if f.startswith(f'benchmark_intermediate_{solver_model.name}_{verifier_model.name}')]
    if not intermediate_files:
        return None, None, None
    
    # Load the intermediate results in chronological order
    intermediate_results = []
    intermediate_timestamps = []
    intermediate_accuracies = []
    for filename in sorted(intermediate_files):
        filepath = os.path.join('benchmark_results', filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            examples_processed = len(data)  # Count examples in the augmented data
            correct_count = sum(1 for ex in data if ex['is_correct'])
            accuracy = (correct_count / examples_processed) * 100 if examples_processed > 0 else 0
            
            intermediate_results.append(examples_processed)
            intermediate_timestamps.append(datetime.now().isoformat())
            intermediate_accuracies.append(accuracy)
    
    return intermediate_results, intermediate_timestamps, intermediate_accuracies

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT="""You are a precise mathematical problem solver. You will be given a problem to solve.

DO:
▪ List applicable theorems/techniques upfront
▪ If possible each step must contain a justification. 
▪ Use LaTeX notation

FORMAT:

**Problem Analysis and Approach**:
1. Start by categorizing the problem (e.g., "This is an inequality problem involving algebraic identities" or "This is a combinatorial proof").
2. List specific tools or theorems that will guide your solution (e.g., "AM-GM inequality," "Basic algebraic manipulations").

**PROOF**:
Example format for each step:
Given: \\( a, b, c > 0 \\) and \\( a + b + c = 3 \\). Prove that \\( abc \\leq 1 \\).

Step 1. By the AM-GM inequality, \\( \\frac{a + b + c}{3} \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Apply AM-GM inequality to \\( a, b, c \\)]  
Step 2. Substituting \\( a + b + c = 3 \\), we get \\( 1 \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Replace with given sum condition]  
Step 3. Cube both sides to eliminate the root: \\( 1 \\geq abc \\) \\hspace{10pt} [Cube both sides to solve for \\( abc \\)]  
Step 4. Thus, \\( abc \\leq 1 \\), as required.  

For each step, clearly state the action, use concise LaTeX notation, and provide a justification in brackets.

**ANSWER**:
\\(\\boxed{\\text{result}}\\) """





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

async def compare_math_answers(model_answer: Optional[str], correct_answer: Optional[str], problem: str, model) -> bool:
    """Use the model to compare two mathematical answers"""
    if model_answer is None or correct_answer is None:
        return False
        
    comparison_prompt = [
        SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
        HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
    ]
    
    try:
        response = await model.ainvoke(comparison_prompt)
        return response.content.strip().lower() == 'yes'
    except Exception:
        return False

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model) -> Optional[Dict]:
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
        prompt = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        # Generate the solution using the model
        response = await solver_model.ainvoke(prompt)  # Await the async response
        solution = response.content
        
        # Extract the model's answer from the solution
        model_answer = extract_answer_from_solution(solution)
        # Compare answers using model verification
        is_correct = await compare_math_answers(model_answer, correct_answer, example["problem"], verifier_model)
        
        # Print results immediately
        status = '✓' if is_correct else '✗'
        print(f"\nProblem {running_id + 1}: {status}")
        print(f"Extracted Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        # Return the result
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_solution': solution,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'model_answer_raw': model_answer,
            'correct_answer_raw': correct_answer
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
                       default='NEMOTRON', help='Model to use for solving problems')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='GEMINI_FLASH', help='Model to use for verifying answers')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--dataset', type=str, default='filtered',
                       choices=['original', 'filtered', 'aime'],
                       help='Dataset to use: original (NuminaMath-CoT), filtered (Numina-Olympiads), or aime (AIME validation)')
    parser.add_argument('--max-concurrent', type=int, default=4,
                       help='Maximum number of concurrent problems (default: 4)')
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
            dataset = load_dataset(f"{username}/Numina-Olympiads", split=args.split)
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
        solver_model = get_model(ModelOption[args.solver])
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
        correct_count = sum(1 for r in results if r['is_correct'])
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
            return await process_example(example, running_id, example['id'], solver_model, verifier_model)

    # Create tasks for all examples
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    # Initialize augmented dataset filename
    augmented_filename = os.path.join('augmented_datasets', 
                                    "benchmark_augmented.json")
    
    # Get existing IDs to skip
    existing_ids = get_existing_ids(augmented_filename)
    if existing_ids:
        print(f"\nFound {len(existing_ids)} existing examples - will skip these IDs")
    
    # Filter out examples with existing IDs
    example_data = [ex for ex in example_data if ex['id'] not in existing_ids]
    if not example_data:
        print("All examples have already been processed!")
        return
        
    print(f"\nWill process {len(example_data)} new examples")
    
    # Check if user wants to proceed with augmented data handling
    if not handle_augmented_data_file(augmented_filename):
        print("Operation cancelled by user.")
        return
        
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
                'solution': next((ex['solution'] for ex in example_data if ex['id'] == result['id']), None),
                'model_response': result['model_solution'],
                'is_correct': result['is_correct'],
                'solver': args.solver,
                'verifier': args.verifier
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
                
                intermediate_filename = os.path.join('results', 
                    f"benchmark_intermediate_{args.solver}_{args.verifier}.json")
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
                save_augmented_data(current_batch, augmented_filename, len(results))
                current_batch = []
            
        progress_bar.update(1)
    progress_bar.close()

    if not results:
        print("\nNo examples were successfully processed.")
        return

    # Sort results by ID to maintain the original order
    results.sort(key=lambda x: x['id'])

    # Calculate final statistics
    correct_count = sum(1 for r in results if r['is_correct'])

    # Print final results
    progress_bar.close()
    print("\n\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    
    if len(results) > 0:
        accuracy = (correct_count / len(results)) * 100
        print(f"Final Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")
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
    
    output_data = {
        'error_rate_points': error_rate_points,
        'final_batch_error_rate': final_batch_error_rate,
        'final_cumulative_error_rate': final_cumulative_error_rate
    }
    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {results_filename}")
    
    # Save final augmented data batch
    if current_batch:
        save_augmented_data(current_batch, augmented_filename, len(results))
    
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
