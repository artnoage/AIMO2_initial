import os
import json
import asyncio
from asyncio import TimeoutError
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from langchain_core.messages import  HumanMessage, SystemMessage
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from utils.utils import *
from utils.benchmark_config import *
from utils.progress_tracker import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Load environment variables from .env file
load_dotenv()


async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, config: BenchmarkConfig) -> Optional[Dict]:
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
    """Main function for benchmarking mathematical problem solving.
    
    Loads dataset, initializes models, and processes examples concurrently
    while tracking progress and saving results.
    """
    config = BenchmarkConfig.from_args('Benchmark model on NuminaMath-CoT dataset')
    
    # Add verifier configuration to BenchmarkConfig
    parser = ArgumentParser(description='Additional verifier configuration')
    parser.add_argument('--verifier', type=str, 
                       choices=[model.name for model in ModelOption],
                       default='GEMINI_FLASH', 
                       help='Model to use for verifying answers')
    parser.add_argument('--dataset', type=str, default='filtered',
                       choices=['original', 'filtered', 'aime'],
                       help='Dataset to use: original (NuminaMath-CoT), filtered (Numina-Numerics), or aime (AIME validation)')
    extra_args = parser.parse_args()

    # Validate max concurrent
    if config.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return
    
    # Load the dataset based on selection
    try:
        if extra_args.dataset == 'original':
            dataset = load_dataset("AI-MO/NuminaMath-CoT", split=config.split)
            if config.split_slice:
                dataset = dataset.select(range(*config.split_slice.indices(len(dataset))))
        elif extra_args.dataset == 'aime':
            dataset = load_dataset("AI-MO/aimo-validation-aime", split=config.split)
            if config.split_slice:
                dataset = dataset.select(range(*config.split_slice.indices(len(dataset))))
        else:  # filtered
            username = HfApi().whoami()["name"]
            dataset = load_dataset(f"{username}/Numina", split=config.split)
            if config.split_slice:
                dataset = dataset.select(range(*config.split_slice.indices(len(dataset))))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter by source if specified
    if config.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == config.source)
    
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
        solver_model = get_model(ModelOption[config.solver], temp=config.temperature)
        verifier_model = get_model(ModelOption[extra_args.verifier])
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    print(f"\nBenchmarking solver: {config.solver}, verifier: {extra_args.verifier} on {config.split} split...")


    progress_tracker = ProgressTracker(total_examples=len(dataset), best_of=args.best_of)
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

    print(f"\nStarting processing of {progress_tracker.total_examples} examples...")

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, example['id'], solver_model, verifier_model, config)

    # Create tasks for all examples with best_of parameter
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    print(f"\nWill process {len(example_data)} examples")
        
    # Process all examples with progress bar
    progress_bar = tqdm(total=len(example_data), desc="Processing examples")
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            progress_tracker.add_result(result)
            progress_tracker.print_progress()
        progress_bar.update(1)
    progress_bar.close()
    progress_tracker.print_final_stats()
    progress_tracker.save_results(config.solver, config.split)

async def cleanup_resources(solver_model=None, verifier_model=None):
    """Cleanup any resources used by the models"""
    try:
        if solver_model:
            await solver_model.aclose()
        if verifier_model:
            await verifier_model.aclose()
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
    finally:
        # Ensure resources are cleaned up
        asyncio.run(cleanup_resources())
