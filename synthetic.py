import os
import json
import asyncio
from typing import Optional, Dict, List
from utils.progress_tracker import ProgressTracker
from utils.benchmark_config import BenchmarkConfig
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

        solution_agent = FullSolutionAgent(solver_model)
        model_responses = []
        verification_results = []
        
        for attempt in range(max_attempts):
            model_solution = await solution_agent.generate(example['problem'], running_id, attempt)
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
    
    config = BenchmarkConfig.from_args('Synthetic Model Benchmark')
    
    if config.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina", split=config.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if config.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == config.source)
    
    dataset = dataset.shuffle(seed=24)
    print(f"\nDataset size: {len(dataset)} examples")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    solver_model = get_model(ModelOption[config.solver], temp=config.temperature)
    verifier_model = get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    second_verifier_model = get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)

    # Initialize directories and files
    os.makedirs('results', exist_ok=True)
    os.makedirs('augmented_datasets', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join('results', f"synthetic_results_{timestamp}.json")
    augmented_filename = os.path.join('augmented_datasets', "synthetic_augmented.json")
    

    # Process examples with controlled concurrency
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(
                example, running_id, example['id'],
                solver_model, verifier_model, second_verifier_model,
                config.best_of
            )

    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(dataset)]
    
    # Initialize progress tracker
    tracker = ProgressTracker(len(dataset), config.best_of)
    progress_bar = tqdm(total=len(dataset), desc="Processing examples")
    current_batch = []
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            # Add result to tracker
            tracker.add_result({
                **result,
                'metadata': {
                    'verification_levels': result['verification_results'],
                    'attempts_to_solve': len(result['verification_results']),
                    'best_verification_level': max(result['verification_results']),
                    'format_check_fails': result['verification_results'].count(0),
                    'answer_check_fails': result['verification_results'].count(1),
                    'first_verifier_fails': result['verification_results'].count(2),
                    'second_verifier_fails': result['verification_results'].count(3),
                    'full_success': result['verification_results'].count(4)
                }
            })
            
            # Print progress and save data every 100 examples
            if len(tracker.results) % 100 == 0:
                tracker.print_progress()
                
                # Count verification levels
                level_counts = {i: 0 for i in range(5)}  # 0-4 levels
                for r in tracker.results:
                    for level in r['verification_results']:
                        level_counts[level] += 1
                        
                print("\nVerification Level Statistics:")
                print(f"Level 0 (Format Check Failed): {level_counts[0]} times")
                print(f"Level 1 (Answer Verification Failed): {level_counts[1]} times")
                print(f"Level 2 (First Solution Verification Failed): {level_counts[2]} times")
                print(f"Level 3 (Second Solution Verification Failed): {level_counts[3]} times")
                print(f"Level 4 (All Verifications Passed): {level_counts[4]} times")
                
                # Calculate detailed statistics
                total_attempts = sum(len(r['verification_results']) for r in tracker.results)
                avg_attempts = total_attempts / len(tracker.results)
                successful_attempts = [len(r['verification_results']) for r in tracker.results if r['solved']]
                avg_successful_attempts = sum(successful_attempts) / len(successful_attempts) if successful_attempts else 0
                
                # Calculate level ratios
                total_verifications = sum(len(r['verification_results']) for r in tracker.results)
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
                        } for r in tracker.results],
                        'metadata': {
                            'solver': config.solver,
                            'verifier': config.verifier,
                            'second_verifier': config.second_verifier,
                            'temperature': config.temperature,
                            'max_attempts': config.best_of,
                            'examples_processed': len(tracker.results),
                            'success_rate': (sum(1 for r in tracker.results if r['solved'])/len(tracker.results)) * 100,
                            'statistics': {
                                'average_attempts': avg_attempts,
                                'average_successful_attempts': avg_successful_attempts,
                                'level_ratios': level_ratios
                            }
                        }
                    }, f, indent=2)
        
        progress_bar.update(1)
    
    progress_bar.close()

    # Print final statistics
    tracker.print_final_stats()
    
    # Save final results with all metadata
    tracker.save_results(config.solver, config.split)
    
    print(f"\nResults saved to {results_file}")
    print(f"Augmented data saved to {augmented_filename}")
    print(f"Total execution time: {datetime.now() - start_time}")

if __name__ == "__main__":
    asyncio.run(main())
