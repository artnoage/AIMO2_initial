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

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT="""You are a mathematical problem solver who sometimes makes mistakes. You will be given a problem to solve.

DO:
▪ List applicable theorems/techniques upfront
▪ If possible each step must contain a justification. 
▪ Use LaTeX notation
▪ Feel free to make reasonable mistakes in your reasoning

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

def calculate_error_rate(results):
    """Calculate error rate from results"""
    if not results:
        return 0.0
    correct_count = sum(1 for r in results if r['is_correct'])
    return correct_count / len(results)

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

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, max_attempts: int) -> Optional[Dict]:
    """Process a single example and keep sampling until we get a wrong answer"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        prompt = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        # Multiple attempts until we get a wrong answer or hit max attempts
        attempts = 0
        is_correct = True  # Start with True to enter the loop
        solution = None
        model_answer = None
        
        while attempts < max_attempts and is_correct:  # Keep trying while answers are correct
            attempts += 1
            response = await solver_model.ainvoke(prompt)
            solution = response.content
            model_answer = extract_answer_from_solution(solution)
            is_correct = await compare_math_answers(model_answer, correct_answer, example["problem"], verifier_model)
            if not is_correct:  # Found a wrong answer, break
                break
                
        # Print results for this example
        attempts_str = f" (after {attempts} attempts)" if attempts > 1 else ""
        status = '✗' if not is_correct else '✓'  # Reversed from normal - we want wrong answers
        print(f"\nProblem {running_id + 1}: {status}{attempts_str}")
        print(f"Expected Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_solution': solution,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'model_answer_raw': model_answer,
            'correct_answer_raw': correct_answer,
            'attempts': attempts
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    # Start timing the entire process
    start_time = datetime.now()
    
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Benchmark model on NuminaMath-CoT dataset - Wrong Answer Generation')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL_ORIGINAL', help='Model to use for solving problems')
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
    parser.add_argument('--max-attempts', type=int, default=5,
                       help='Maximum attempts to get a wrong answer (default: 5)')
    args = parser.parse_args()

    # Validate max concurrent
    if args.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

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

    # Initialize the models with higher temperature for solver to encourage mistakes
    try:
        solver_model = get_model(ModelOption[args.solver], temp=0.9)  # Higher temperature
        verifier_model = get_model(ModelOption[args.verifier])
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    print(f"\nBenchmarking solver: {args.solver}, verifier: {args.verifier} on {args.split} split...")

    # Process examples with controlled concurrency
    results = []
    error_rate_points = []
    total_examples = len(dataset)
    print(f"\nStarting processing of {total_examples} examples...")

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, example['id'], solver_model, verifier_model, args.max_attempts)

    # Create tasks for all examples
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(dataset)]
    
    # Initialize augmented dataset filename
    augmented_filename = os.path.join('augmented_datasets', 
                                    "reject_augmented.json")
    
    # Get existing IDs to skip
    existing_ids = get_existing_ids(augmented_filename)
    if existing_ids:
        print(f"\nFound {len(existing_ids)} existing examples - will skip these IDs")
    
    # Filter out examples with existing IDs
    example_data = [ex for ex in dataset if ex['id'] not in existing_ids]
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
                'attempts': result['attempts'],
                'solver': args.solver,
                'verifier': args.verifier
            }
            current_batch.append(augmented_example)
            
            # Save error rate every 100 examples
            if len(results) % 100 == 0:
                # Calculate error rate for the last 100 results
                last_hundred = results[-100:]
                batch_error_rate = 1 - calculate_error_rate(last_hundred)  # Invert since we want wrong answers
                # Also calculate cumulative error rate
                cumulative_error_rate = 1 - calculate_error_rate(results)  # Invert since we want wrong answers
                error_rate_points.append({
                    'examples_processed': len(results),
                    'batch_error_rate': batch_error_rate,
                    'cumulative_error_rate': cumulative_error_rate
                })
                print(f"\nAt {len(results)} examples:")
                print(f"Batch Wrong Answer Rate (last 100): {batch_error_rate:.4f}")
                print(f"Cumulative Wrong Answer Rate: {cumulative_error_rate:.4f}")
                
                intermediate_filename = os.path.join('results', 
                    f"reject_intermediate_{args.solver}_{args.verifier}.json")
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

    # Calculate final statistics - note we want wrong answers here
    wrong_count = sum(1 for r in results if not r['is_correct'])
    total_attempts = sum(r['attempts'] for r in results)

    # Print final results
    print("\n\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    print(f"Total wrong answers generated: {wrong_count}")
    print(f"Average attempts per example: {total_attempts/len(results):.2f}")
    
    if len(results) > 0:
        wrong_rate = (wrong_count / len(results)) * 100
        print(f"Wrong Answer Rate: {wrong_count}/{len(results)} = {wrong_rate:.2f}%")
    else:
        print("No examples were successfully processed.")

    # Save final results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join('results', 
                                  f"reject_results_{args.solver}_{args.verifier}_{timestamp}.json")
    
    # Save final error rate and points
    final_batch_error_rate = 1 - calculate_error_rate(results[-100:] if len(results) >= 100 else results)
    final_cumulative_error_rate = 1 - calculate_error_rate(results)
    error_rate_points.append({
        'examples_processed': len(results),
        'batch_error_rate': final_batch_error_rate,
        'cumulative_error_rate': final_cumulative_error_rate
    })
    
    output_data = {
        'error_rate_points': error_rate_points,
        'final_batch_error_rate': final_batch_error_rate,
        'final_cumulative_error_rate': final_cumulative_error_rate,
        'total_attempts': total_attempts,
        'average_attempts': total_attempts/len(results)
    }
    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {results_filename}")
    
    # Save final augmented data batch
    if current_batch:
        save_augmented_data(current_batch, augmented_filename, len(results))
    
    # Print error rate progression
    print("\nWrong Answer Rate Progression:")
    for point in error_rate_points:
        print(f"After {point['examples_processed']} examples:")
        print(f"  Batch Wrong Answer Rate (last 100): {point['batch_error_rate']:.4f}")
        print(f"  Cumulative Wrong Answer Rate: {point['cumulative_error_rate']:.4f}")

    # Calculate and print timing information
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
import json
import argparse
import os

def add_model_to_file(filename: str, model_name: str) -> None:
    """Add model name to all entries in a JSON file."""
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return
        
    try:
        # Read the file
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Check if it's a list of dictionaries
        if not isinstance(data, list):
            print(f"Error: File {filename} does not contain a list of examples")
            return
            
        # Add model name to each entry
        modified = False
        for entry in data:
            if isinstance(entry, dict) and 'model' not in entry:
                entry['model'] = model_name
                modified = True
                
        if not modified:
            print("No changes needed - all entries already have 'model' field")
            return
            
        # Write back to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Successfully added model name to {filename}")
        
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not valid JSON")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Add model name to JSON files')
    parser.add_argument('--file', type=str, required=True,
                       help='JSON file to process')
    parser.add_argument('--solver', type=str, required=True,
                       help='Solver model name to add')
    parser.add_argument('--verifier', type=str, required=True,
                       help='Verifier model name to add')
    
    args = parser.parse_args()
    add_model_to_file(args.file, args.model)

if __name__ == "__main__":
    main()
import json
import argparse
import os

def add_model_to_file(filename: str, model_name: str) -> None:
    """Add model name to all entries in a JSON file."""
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return
        
    try:
        # Read the file
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Check if it's a list of dictionaries
        if not isinstance(data, list):
            print(f"Error: File {filename} does not contain a list of examples")
            return
            
        # Add model name to each entry
        modified = False
        for entry in data:
            if isinstance(entry, dict) and 'model' not in entry:
                entry['model'] = model_name
                modified = True
                
        if not modified:
            print("No changes needed - all entries already have 'model' field")
            return
            
        # Write back to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Successfully added model name to {filename}")
        
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not valid JSON")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Add model name to JSON files')
    parser.add_argument('--file', type=str, required=True,
                       help='JSON file to process')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to add')
    
    args = parser.parse_args()
    add_model_to_file(args.file, args.model)

if __name__ == "__main__":
    main()
