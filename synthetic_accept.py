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

SYSTEM_PROMPT = """You are a precise mathematical problem solver. You will be given a problem to solve.

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
\\(\\boxed{\\text{final answer}}\\) """

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

def check_required_words(response: str, full_solution: str) -> bool:
    """
    Check if response contains required words and is sufficiently longer than full solution
    """
    # Check for required words (case insensitive)
    lower_response = response.lower()
    required_words = ['analysis', 'problem', 'step']
    has_required_words = all(word in lower_response for word in required_words)
    
    # Check length requirement (at least 8% longer than full solution)
    is_long_enough = len(response) >= len(full_solution) * 1.03
    
    return has_required_words and is_long_enough

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, max_format_attempts: int, max_verification_attempts: int) -> Optional[Dict]:
    """Process a single example with multiple attempts for both formatting and verification"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
        correct_solution = example['solution']    
        correct_answer = extract_answer_from_solution(correct_solution)
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=example['problem'])
        ]
        
        # Track both format and verification attempts
        format_attempts = 0
        verification_attempts = 0
        is_correct = False
        verifier_disagreements = 0
        
        while format_attempts < max_format_attempts:
            format_attempts += 1
            response = await solver_model.ainvoke(prompt)
            model_solution = response.content
            
            # Check for required words and length before proceeding with verification
            if not check_required_words(model_solution, correct_solution):
                continue
                
            # If format check passes, try verification up to max_verification_attempts times
            while verification_attempts < max_verification_attempts and not is_correct:
                verification_attempts += 1
                is_correct, first_verify, second_verify = await compare_math_solutions(model_solution, correct_solution, example["problem"], verifier_model, second_verifier_model)
                if first_verify != second_verify:
                    verifier_disagreements += 1
                if is_correct:
                    break
            
            # If we got a correct answer or used all verification attempts, stop trying new formats
            if is_correct or verification_attempts >= max_verification_attempts:
                break
        model_answer = extract_answer_from_solution(model_solution)    
        # Print results for this example
        attempts_str = f" (format: {format_attempts}, verify: {verification_attempts})"
        disagreement_str = f" [Verifiers disagreed {verifier_disagreements} times]" if verifier_disagreements > 0 else ""
        status = '✓' if is_correct else '✗'
        print(f"\nProblem {running_id + 1}: {status}{attempts_str}{disagreement_str}")
        print(f"Expected Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_response': model_solution,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'verifier_disagreements': verifier_disagreements,
            'format_attempts': format_attempts,
            'verification_attempts': verification_attempts
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
    parser.add_argument('--max-concurrent', type=int, default=512,
                       help='Maximum number of concurrent problems (default: 4)')
    parser.add_argument('--max-format-attempts', type=int, default=10,
                       help='Maximum attempts to get properly formatted solution (default: 3)')
    parser.add_argument('--max-verification-attempts', type=int, default=3,
                       help='Maximum attempts to get correct solution after format check (default: 1)')
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

    solver_model = get_model(ModelOption[args.solver], temp=0.1)
    verifier_model = get_model(ModelOption[args.verifier], temp=0)
    second_verifier_model = get_model(ModelOption[args.verifier], temp=0)  # Same model type as first verifier
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
            return await process_example(example, running_id, example['id'], solver_model, verifier_model, second_verifier_model, args.max_format_attempts, args.max_verification_attempts)

    # Initialize augmented dataset filename
    os.makedirs('augmented_datasets', exist_ok=True)
    augmented_filename = os.path.join('augmented_datasets', 
                                    "accept_augmented.json")
    
    # Get existing IDs to skip
    existing_ids = get_existing_ids(augmented_filename)
    if existing_ids:
        print(f"\nFound {len(existing_ids)} existing examples - will skip these IDs")
    
    # Filter out examples with existing IDs
    example_data = [ex for ex in example_data if ex['id'] not in existing_ids]
    if not example_data:
        print("All examples have already been processed!")
        return
        
    filtered_count = len(example_data)
    print(f"\nWill process {filtered_count} new examples ({total_examples - filtered_count} skipped)")
    
    # Create tasks only for new examples
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    # Check if user wants to proceed with augmented data handling
    if not handle_augmented_data_file(augmented_filename):
        print("Operation cancelled by user.")
        return
        
    # Process new examples with progress bar
    progress_bar = tqdm(total=len(example_data), desc="Processing examples")
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
                'model_response': result['model_response'],
                'is_correct': result['is_correct'],
                'verifier_disagreements': result['verifier_disagreements'],
                'format_attempts': result['format_attempts'],
                'verification_attempts': result['verification_attempts']
            }
            current_batch.append(augmented_example)
            
            # Save metrics and data every 100 examples
            if len(current_batch) >= 100:
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
                
                # Save current batch of augmented data
                save_augmented_data(current_batch, augmented_filename, len(results))
                current_batch = []

                # Save intermediate results less frequently (every 500 examples)
                if len(results) % 500 == 0:
                    intermediate_filename = os.path.join('results', 
                        f"accept_intermediate_{args.solver}_{args.verifier}.json")
                    output_data = {
                        'error_rate_points': error_rate_points
                    }
                    os.makedirs('results', exist_ok=True)
                    with open(intermediate_filename, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    print(f"\nSaved intermediate results after {len(results)} examples")
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
                                  f"accept_results_{args.solver}_{args.verifier}_{timestamp}.json")
    
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
