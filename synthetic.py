import os
import json
import asyncio
import argparse
from typing import Optional, Dict
from utils.augmented_data_handler import handle_augmented_data_file, save_augmented_data, get_existing_ids
from utils.utils import ModelOption, get_model, extract_answer_from_solution
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

from utils.agents import FullSolutionAgent, AnswerVerifierAgent, SolutionVerifierAgent

async def verify_solution(
    model_solution: Optional[str],
    correct_solution: Optional[str],
    problem: str,
    verifier_model,
    second_verifier_model
) -> int:
    """
    Returns verification_level where:
    0 - Failed format check
    1 - Failed answer verification
    2 - Failed first solution verification
    3 - Failed second solution verification
    4 - Passed all checks
    """
    
    model_answer = extract_answer_from_solution(model_solution)
    correct_answer = extract_answer_from_solution(correct_solution)
    
    if model_answer is None or correct_answer is None or model_solution is None:
        return 0

    try:
        # Check answer equivalence using AnswerVerifierAgent
        answer_verifier = AnswerVerifierAgent(verifier_model)
        if not await answer_verifier.verify(problem, model_solution, correct_answer):
            return 1

        # Check solution completeness with first verifier
        solution_verifier = SolutionVerifierAgent(verifier_model)
        if not await solution_verifier.verify(problem, model_solution):
            return 2
            
        # Only check second verifier if first one passed
        second_solution_verifier = SolutionVerifierAgent(second_verifier_model)
        if not await second_solution_verifier.verify(problem, model_solution):
            return 3
            
        return 4

    except Exception as e:
        return 0

def check_format(response: str, full_solution: str) -> bool:
    """Check if response contains required words and is sufficiently detailed"""
    lower_response = response.lower()
    required_words = ['analysis', 'problem', 'step']
    has_required_words = all(word in lower_response for word in required_words)
    is_long_enough = len(response) >= len(full_solution) * 1.03
    has_no_links = 'http' not in lower_response
    return has_required_words and is_long_enough and has_no_links

async def process_example(
    example: Dict,
    running_id: int,
    example_id: int,
    solver_model,
    verifier_model,
    second_verifier_model,
    max_attempts: int
) -> Optional[Dict]:
    """Process a single example with multiple attempts, keeping all responses"""
    
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None

        prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=example['problem'])
        ]
        
        model_responses = []
        verification_results = []
        
        for attempt in range(max_attempts):
            response = await solver_model.ainvoke(prompt)
            model_solution = response.content
            model_responses.append(model_solution)
            
            # Check format and verify solution
            if not check_format(model_solution, example['solution']):
                verification_results.append(0)
            else:
                level = await verify_solution(
                    model_solution,
                    example['solution'],
                    example['problem'],
                    verifier_model,
                    second_verifier_model
                )
                verification_results.append(level)
                
                # If we get a valid solution (level 4) on first attempt,
                # try one more time to get a negative example for DPO
                if level == 4:
                    if attempt == 0:
                        continue  # Get one more attempt for a negative example
                    break  # Already have a good and bad example, stop here
                
                # Break if we've hit max attempts
                if len(verification_results) >= max_attempts:
                    break
        
        # Count occurrences of each verification level
        level_counts = {i: verification_results.count(i) for i in range(5)}
        
        # Print verification results for this problem
        print(f"\nProblem {running_id + 1}:")
        print(f"Format Check Failed: {level_counts[0]}/{len(verification_results)}")
        print(f"Answer Check Failed: {level_counts[1]}/{len(verification_results)}")
        print(f"First Verifier Failed: {level_counts[2]}/{len(verification_results)}")
        print(f"Second Verifier Failed: {level_counts[3]}/{len(verification_results)}")
        print(f"All Checks Passed: {level_counts[4]}/{len(verification_results)}")
        print(f"Final Status: {'Solved' if 4 in verification_results else 'Failed'}")
        print("-" * 80)
                
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'model_responses': model_responses,
            'verification_results': verification_results,
            'best_response': model_responses[verification_results.index(4)] if 4 in verification_results else None,
            'solved': 4 in verification_results
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Synthetic Model Benchmark')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL', help='Model to use for solving problems')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='GEMINI_FLASH', help='Model to use for first verifier')
    parser.add_argument('--second-verifier', type=str, choices=[model.name for model in ModelOption],
                       default='CODER', help='Model to use for second verifier')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--max-concurrent', type=int, default=512,
                       help='Maximum number of concurrent problems')
    parser.add_argument('--max-attempts', type=int, default=200,
                       help='Maximum attempts per problem')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for model generation')
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
    
    dataset = dataset.shuffle(seed=24)
    print(f"\nDataset size: {len(dataset)} examples")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    solver_model = get_model(ModelOption[args.solver], temp=args.temperature)
    verifier_model = get_model(ModelOption[args.verifier], temp=0)
    second_verifier_model = get_model(ModelOption[args.second_verifier], temp=0)

    # Initialize directories and files
    os.makedirs('results', exist_ok=True)
    os.makedirs('augmented_datasets', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join('results', f"synthetic_results_{timestamp}.json")
    augmented_filename = os.path.join('augmented_datasets', "synthetic_augmented.json")
    
    # Get existing IDs to skip
    existing_ids = get_existing_ids(augmented_filename)
    if existing_ids:
        print(f"\nFound {len(existing_ids)} existing examples - will skip these IDs")
    
    # Filter out examples with existing IDs
    dataset = dataset.filter(lambda x: x['id'] not in existing_ids)
    print(f"\nWill process {len(dataset)} new examples")
    
    if len(dataset) == 0:
        print("All examples have already been processed!")
        return
        
    # Check if user wants to proceed with augmented data handling
    if not handle_augmented_data_file(augmented_filename):
        print("Operation cancelled by user.")
        return

    # Process examples with controlled concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(
                example, running_id, example['id'],
                solver_model, verifier_model, second_verifier_model,
                args.max_attempts
            )

    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(dataset)]
    
    # Process examples with progress bar
    results = []
    current_batch = []
    progress_bar = tqdm(total=len(dataset), desc="Processing examples")
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            
            # Add to current batch
            augmented_example = {
                'id': result['id'],
                'problem': result['problem'],
                'correct_solution': result['correct_solution'],
                'model_responses': result['model_responses'],
                'verification_results': result['verification_results'],
                'solved': result['solved']
            }
            current_batch.append(augmented_example)
            
            # Save intermediate results every 100 examples
            if len(results) % 100 == 0:
                solved_count = sum(1 for r in results if r['solved'])
                print(f"\nProcessed {len(results)} examples:")
                print(f"Current success rate: {solved_count}/{len(results)} = {(solved_count/len(results))*100:.2f}%")
                
                # Count verification levels
                level_counts = {i: 0 for i in range(5)}  # 0-4 levels
                for r in results:
                    for level in r['verification_results']:
                        level_counts[level] += 1
                        
                print("\nVerification Level Statistics:")
                print(f"Level 0 (Format Check Failed): {level_counts[0]} times")
                print(f"Level 1 (Answer Verification Failed): {level_counts[1]} times")
                print(f"Level 2 (First Solution Verification Failed): {level_counts[2]} times")
                print(f"Level 3 (Second Solution Verification Failed): {level_counts[3]} times")
                print(f"Level 4 (All Verifications Passed): {level_counts[4]} times")
                
                # Save current batch of augmented data
                save_augmented_data(current_batch, augmented_filename, len(results))
                current_batch = []
                
                # Calculate detailed statistics
                total_attempts = sum(len(r['verification_results']) for r in results)
                avg_attempts = total_attempts / len(results)
                successful_attempts = [len(r['verification_results']) for r in results if r['solved']]
                avg_successful_attempts = sum(successful_attempts) / len(successful_attempts) if successful_attempts else 0
                
                # Calculate level ratios
                total_verifications = sum(len(r['verification_results']) for r in results)
                level_ratios = {i: level_counts[i] / total_verifications * 100 for i in range(5)}
                
                print("\nDetailed Statistics:")
                print(f"Average attempts per problem: {avg_attempts:.2f}")
                print(f"Average attempts for successful solutions: {avg_successful_attempts:.2f}")
                print("\nVerification Level Ratios:")
                print(f"Format Check Failed: {level_ratios[0]:.2f}%")
                print(f"Answer Check Failed: {level_ratios[1]:.2f}%")
                print(f"First Verifier Failed: {level_ratios[2]:.2f}%")
                print(f"Second Verifier Failed: {level_ratios[3]:.2f}%")
                print(f"All Checks Passed: {level_ratios[4]:.2f}%")
                
                # Save intermediate results with detailed statistics
                with open(results_file, 'w') as f:
                    json.dump({
                        'scores': [{
                            'id': r['id'],
                            'solved': r['solved'],
                            'attempts_used': len(r['verification_results'])
                        } for r in results],
                        'metadata': {
                            'solver': args.solver,
                            'verifier': args.verifier,
                            'second_verifier': args.second_verifier,
                            'temperature': args.temperature,
                            'max_attempts': args.max_attempts,
                            'examples_processed': len(results),
                            'success_rate': (solved_count/len(results)) * 100,
                            'statistics': {
                                'average_attempts': avg_attempts,
                                'average_successful_attempts': avg_successful_attempts,
                                'level_ratios': level_ratios
                            }
                        }
                    }, f, indent=2)
        
        progress_bar.update(1)
    
    progress_bar.close()

    if not results:
        print("\nNo examples were successfully processed.")
        return

    # Calculate and display final statistics
    solved_count = sum(1 for r in results if r['solved'])
    success_rate = (solved_count / len(results)) * 100

    print("\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    print(f"Successfully solved: {solved_count}/{len(results)} = {success_rate:.2f}%")
    
    # Count final verification levels
    level_counts = {i: 0 for i in range(5)}  # 0-4 levels
    for r in results:
        for level in r['verification_results']:
            level_counts[level] += 1
            
    print("\nFinal Verification Level Statistics:")
    print(f"Level 0 (Format Check Failed): {level_counts[0]} times")
    print(f"Level 1 (Answer Verification Failed): {level_counts[1]} times")
    print(f"Level 2 (First Solution Verification Failed): {level_counts[2]} times")
    print(f"Level 3 (Second Solution Verification Failed): {level_counts[3]} times")
    print(f"Level 4 (All Verifications Passed): {level_counts[4]} times")
    
    # Calculate average attempts needed for successful solutions
    successful_attempts = [len(r['verification_results']) for r in results if r['solved']]
    if successful_attempts:
        avg_attempts = sum(successful_attempts) / len(successful_attempts)
        print(f"Average attempts for successful solutions: {avg_attempts:.2f}")

    # Calculate final detailed statistics
    total_attempts = sum(len(r['verification_results']) for r in results)
    avg_attempts = total_attempts / len(results)
    successful_attempts = [len(r['verification_results']) for r in results if r['solved']]
    avg_successful_attempts = sum(successful_attempts) / len(successful_attempts) if successful_attempts else 0
    
    # Calculate final level ratios
    total_verifications = sum(len(r['verification_results']) for r in results)
    level_ratios = {i: level_counts[i] / total_verifications * 100 for i in range(5)}
    
    print("\nFinal Detailed Statistics:")
    print(f"Average attempts per problem: {avg_attempts:.2f}")
    print(f"Average attempts for successful solutions: {avg_successful_attempts:.2f}")
    print("\nFinal Verification Level Ratios:")
    print(f"Format Check Failed: {level_ratios[0]:.2f}%")
    print(f"Answer Check Failed: {level_ratios[1]:.2f}%")
    print(f"First Verifier Failed: {level_ratios[2]:.2f}%")
    print(f"Second Verifier Failed: {level_ratios[3]:.2f}%")
    print(f"All Checks Passed: {level_ratios[4]:.2f}%")
    
    # Save final results with detailed statistics
    with open(results_file, 'w') as f:
        json.dump({
            'scores': [{
                'id': r['id'],
                'solved': r['solved'],
                'attempts_used': len(r['verification_results'])
            } for r in results],
            'metadata': {
                'solver': args.solver,
                'verifier': args.verifier,
                'second_verifier': args.second_verifier,
                'temperature': args.temperature,
                'max_attempts': args.max_attempts,
                'final_success_rate': success_rate,
                'total_duration_seconds': (datetime.now() - start_time).total_seconds(),
                'statistics': {
                    'average_attempts': avg_attempts,
                    'average_successful_attempts': avg_successful_attempts,
                    'level_ratios': level_ratios
                }
            }
        }, f, indent=2)
    
    # Save any remaining augmented data
    if current_batch:
        save_augmented_data(current_batch, augmented_filename, len(results))
        
    print(f"\nResults saved to {results_file}")
    print(f"Augmented data saved to {augmented_filename}")
    print(f"Total execution time: {datetime.now() - start_time}")

if __name__ == "__main__":
    asyncio.run(main())
