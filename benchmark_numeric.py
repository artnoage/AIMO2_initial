import asyncio
from typing import Optional, Dict
from utils.utils import *
from utils.progress_tracker import ProgressTracker
from langchain_core.messages import HumanMessage, SystemMessage
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from utils.benchmark_config import *



async def process_example(example: Dict, running_id: int, example_id: int, solver_model, best_of: int = 1, tolerance: float = 0.01) -> Optional[Dict]:
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

        prompt = [SystemMessage(content=NUMERIC_SOLVER_SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
        for attempt in range(best_of):
            try:
                current_solution = await get_model_response(solver_model, prompt, running_id, attempt)
                current_answer = extract_numeric_answer(current_solution)
                is_correct = is_answer_correct(current_answer, correct_answer, tolerance)
                
                if is_correct:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
                        best_answer = current_answer
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
            
            if attempt >= best_of - 1:
                break
        
        solution = best_solution if best_solution is not None else solutions[0]['solution']
        model_answer = best_answer if best_answer is not None else solutions[0]['answer']
        
        # Print statistics for this example
        print(f"\nExample {running_id + 1}:")
        print(f"Problem: {example['problem'][:200]}...")  # First 200 chars of problem
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
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
    """Main function for benchmarking numeric problem solving.
    
    Loads dataset, initializes model, and processes examples concurrently
    while tracking progress and saving results.
    """
    config = BenchmarkConfig.from_args('Benchmark model on numeric problems')

    if config.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina", split=config.split)
        if config.split_slice:
            dataset = dataset.select(range(*config.split_slice.indices(len(dataset))))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if config.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == config.source)
    
    dataset = dataset.shuffle(seed=42)

    print("\nDataset Information:")
    num_examples = len(dataset)
    print(f"Number of examples: {num_examples}")

    if num_examples == 0:
        print("Error: Dataset is empty! Check your source filter and split arguments.")
        return

    try:
        solver_model = get_model(ModelOption[config.solver], temp=config.temperature)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    print(f"\nBenchmarking solver: {config.solver} on {config.split} split...")

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

    progress_tracker = ProgressTracker(total_examples=len(example_data), best_of=config.best_of)
    print(f"\nStarting processing of {progress_tracker.total_examples} examples...")

    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, example['id'], solver_model, config.best_of, tolerance=0.01)

    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    if not example_data:
        print("No examples to process!")
        return
        
    print(f"\nWill process {len(example_data)} examples")
    
    progress_bar = tqdm(total=len(example_data), desc="Processing examples")
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            progress_tracker.add_result(result)
            progress_tracker.print_progress()
            # Save progress every 100 examples
            if len(progress_tracker.results) % 100 == 0:
                progress_tracker.save_results(config.solver, config.split)
        progress_bar.update(1)
    progress_bar.close()
    
    progress_tracker.print_final_stats()
    progress_tracker.save_results(config.solver, config.split)

async def cleanup_resources(solver_model=None):
    """Cleanup any resources used by the model"""
    try:
        if solver_model:
            await solver_model.aclose()
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
