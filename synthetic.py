import os
import re
import json
import asyncio
import argparse
from enum import Enum
from typing import Optional, Dict
from utils.augmented_data_handler import handle_augmented_data_file, save_augmented_data, get_existing_ids
from utils.utils import ModelOption, get_model, extract_answer_from_solution
from datetime import datetime
from typing import  Dict, Optional, Tuple
from langchain_core.messages import  HumanMessage, SystemMessage
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """You are a precise mathematical problem solver. You receive problems with partial solutions as hints.

PROCESS:
▪ Silently analyze the given hint for relevant techniques and insights.
▪ Develop a complete, independent solution from scratch.

REQUIRED:
▪ Begin by listing applicable theorems, definitions, or techniques you will use.
▪ For each proof step, include a justification in brackets. Use clear LaTeX notation for all mathematical expressions.

PROHIBITED:
▪ Avoid restating the problem.
▪ Do not reference or rely on the partial solution.

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
\\(\\boxed{\\text{final answer}}\\) 
"""

async def compare_math_solutions(
    model_solution: Optional[str], 
    correct_solution: Optional[str], 
    problem: str, 
    verifier_model, 
    second_verifier_model
) -> Tuple[bool, bool, bool]:
    model_answer = extract_answer_from_solution(model_solution)
    correct_answer = extract_answer_from_solution(correct_solution)
    
    if model_answer is None or correct_answer is None or model_solution is None:
        return False, False, False

    # First verification: check if answers are equivalent
    comparison_prompt = [
        SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
        HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
    ]
    
    try:
        first_response = await verifier_model.ainvoke(comparison_prompt)
        first_result = first_response.content.strip().lower() == 'yes'
        
        # Second verification: check if the full solution is correct
        second_prompt = [
            SystemMessage(content="You are a mathematical solution validator. Given a problem and a proposed solution, respond ONLY with 'yes' if the solution is mathematically correct and complete, or 'no' if it contains any errors or is incomplete. Just one word, no explanation."),
            HumanMessage(content=f"Problem:\n{problem}\n\nProposed solution:\n{model_solution}\n\nIs this solution mathematically correct and complete?")
        ]
        
        second_response = await second_verifier_model.ainvoke(second_prompt)
        second_result = second_response.content.strip().lower() == 'yes'
        
        return first_result and second_result, first_result, second_result

    except Exception:
        return False, False, False


def get_partial_solution(solution: str) -> str:
    """Get partial solution by removing last three lines if more than 3 lines,
    otherwise return first line"""
    lines = solution.strip().split('\n\n')
    if len(lines) <= 4:
        return lines[0]
    return '\n\n'.join(lines[:-4])

def check_required_words(response: str) -> bool:
    """Check if response contains required words (case insensitive)"""
    lower_response = response.lower()
    required_words = ['analysis', 'problem', 'step']
    return all(word in lower_response for word in required_words)

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, max_attempts: int) -> Optional[Dict]:
    """Process a single example with multiple attempts"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
        correct_solution=example['solution']    
        correct_answer = extract_answer_from_solution(correct_solution)
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Combine problem with partial solution
        partial_solution = get_partial_solution(correct_solution)
        combined_prompt = f"{example['problem']}\n\nPartial solution:\n{partial_solution}"
        
        prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=combined_prompt)
        ]
        
        # Multiple attempts until correct or max attempts reached
        attempts = 0
        is_correct = False
        verifier_disagreements = 0
        
        while attempts < max_attempts and not is_correct:
            attempts += 1
            response = await solver_model.ainvoke(prompt)
            model_solution = response.content
            
            # Check for required words before proceeding with verification
            if not check_required_words(model_solution):
                is_correct, first_verify, second_verify = False, False, False
                continue
                
            
            is_correct, first_verify, second_verify = await compare_math_solutions(model_solution, correct_solution, example["problem"], verifier_model, second_verifier_model)
            if first_verify != second_verify:
                verifier_disagreements += 1
            if is_correct:
                break
        model_answer = extract_answer_from_solution(model_solution)    
        # Print results for this example
        attempts_str = f" (after {attempts} attempts)" if attempts > 1 else ""
        disagreement_str = f" [Verifiers disagreed {verifier_disagreements} times]" if verifier_disagreements > 0 else ""
        status = '✓' if is_correct else '✗'
        print(f"\nProblem {running_id + 1}: {status}{attempts_str}{disagreement_str}")
        print(f"Expected Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'partial_solution': partial_solution,
            'correct_answer': correct_answer,
            'model_response': model_solution,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'verifier_disagreements': verifier_disagreements,
            'attempts': attempts
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Synthetic Model Benchmark')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL_ORIGINAL', help='Model to use for solving problems')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='GEMINI_FLASH', help='Model to use for verifying answers')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--max-concurrent', type=int, default=100,
                       help='Maximum number of concurrent problems (default: 4)')
    parser.add_argument('--max-attempts', type=int, default=30,
                       help='Maximum number of attempts to get correct solution (default: 5)')
    args = parser.parse_args()

    if args.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina-Olympiads", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    dataset = dataset.shuffle(seed=42)
    num_examples = len(dataset)
    
    print("\nDataset Information:")
    print(f"Number of examples: {num_examples}")

    if num_examples == 0:
        print("Error: Dataset is empty!")
        return

    solver_model = get_model(ModelOption[args.solver], temp=0.2)
    verifier_model = get_model(ModelOption[args.verifier], temp=0.05)
    second_verifier_model = get_model(ModelOption[args.verifier], temp=0.1)  # Same model type as first verifier
    print(f"\nBenchmarking solver: {args.solver}, verifier: {args.verifier} on {args.split} split...")

    # Create example data with dataset IDs and build lookup map
    example_data = []
    example_map = {}
    for ex in dataset:
        example = {
            'id': ex['id'],
            'problem': ex['problem'],
            'solution': ex['solution']
        }
        example_data.append(example)
        example_map[ex['id']] = example
    
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
            return await process_example(example, running_id, example['id'], solver_model, verifier_model, second_verifier_model, args.max_attempts)

    # Create tasks for all examples
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    # Initialize augmented dataset filename
    os.makedirs('augmented_datasets', exist_ok=True)
    augmented_filename = os.path.join('augmented_datasets', 
                                    "synthetic_augmented.json")
    
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
                'solution': example_map[result['id']]['solution'],
                'partial_solution': result['partial_solution'],
                'model_response': result['model_response'],
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
                    f"synthetic_intermediate_{args.solver}_{args.verifier}.json")
                output_data = {
                    'error_rate_points': error_rate_points
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

    results.sort(key=lambda x: x['id'])

    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_count / len(results)) * 100 if results else 0

    print("\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    print(f"Final Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")
    total_disagreements = sum(r['verifier_disagreements'] for r in results)
    print(f"Total verifier disagreements: {total_disagreements}")
    print(f"Average disagreements per example: {total_disagreements/len(results):.2f}")

    # Save final results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join('results', 
                                  f"synthetic_results_{args.solver}_{args.verifier}_{timestamp}.json")
    
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

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
