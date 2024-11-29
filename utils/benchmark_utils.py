import asyncio
from typing import Optional, Dict, List, Callable, Any, Tuple
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi
from utils.progress_tracker import ProgressTracker
from utils.benchmark_config import BenchmarkConfig
from utils.utils import get_model, ModelOption, get_model_response

async def process_attempts(
    solver_model,
    prompt: List,
    best_of: int,
    running_id: int,
    verify_func: Callable
) -> Tuple[List[Dict], Optional[str], Optional[Any], int]:
    """
    Make multiple solution attempts and track results
    Returns: (solutions, best_solution, best_answer, correct_count)
    """
    solutions = []
    correct_count = 0
    best_solution = None
    best_answer = None
    
    for attempt in range(best_of):
        try:
            current_solution = await get_model_response(solver_model, prompt, running_id, attempt)
            current_answer = verify_func(current_solution)
            is_correct = current_answer[1] if isinstance(current_answer, tuple) else current_answer
            
            if is_correct:
                correct_count += 1
                if best_solution is None:
                    best_solution = current_solution
                    best_answer = current_answer[0] if isinstance(current_answer, tuple) else current_answer
        except Exception as e:
            print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
            current_solution = "Error occurred"
            current_answer = None
            is_correct = False
        
        solutions.append({
            'solution': current_solution,
            'answer': current_answer[0] if isinstance(current_answer, tuple) else current_answer,
            'is_correct': is_correct
        })
        
        if attempt >= best_of - 1:
            break
            
    return solutions, best_solution, best_answer, correct_count

async def run_benchmark(
    config: BenchmarkConfig,
    process_example_func: Callable,
    system_prompt: str
) -> None:
    """Generic benchmark runner that handles dataset loading and example processing"""
    if config.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    try:
        if config.dataset == 'original':
            dataset = load_dataset("AI-MO/NuminaMath-CoT", split=config.split)
        elif config.dataset == 'aime':
            dataset = load_dataset("AI-MO/aimo-validation-aime", split=config.split)
        else:  # filtered
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

    async def process_with_semaphore(example: Dict, running_id: int) -> Optional[Dict]:
        async with semaphore:
            return await process_example_func(
                example, running_id, example['id'], 
                solver_model, config.best_of
            )

    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
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

    try:
        if solver_model:
            await solver_model.aclose()
    except Exception as e:
        print(f"Error during cleanup: {e}")
