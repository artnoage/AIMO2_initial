import re
import os
import asyncio
import json
from functools import wraps
from typing import Optional, Dict, List, Callable, Tuple, TypeVar, Any
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi, whoami
from bench_utils.benchmark_config import BenchmarkConfig, ModelOption
from bench_utils.progress_tracker import *
T = TypeVar('T')

def get_model(model: ModelOption, temp: float = 0.1):
    """
    Initialize the ChatOpenAI model based on the selected ModelOption.
    For LOCAL models, it connects to a local endpoint.
    For other models, it uses the OpenRouter API.
    """
    if model == ModelOption.LOCAL:
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key="EMPTY",
            base_url="http://localhost:8000/v1")
    else:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=openrouter_api_key)


def async_retry(max_retries: int = 3, timeout: int = 300):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    #print(f"Timeout error. Retrying... ({retry_count}/{max_retries})")
                    await asyncio.sleep(1)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    #print(f"Error: Other error. Retrying... ({retry_count}/{max_retries})")
                    await asyncio.sleep(1)
            raise Exception(f"Failed after {max_retries} retries")
        return wrapper
    return decorator

def extract_numeric_answer(solution: str) -> Optional[float]:
    """
    Extract numeric answer from a solution string.
    Looks for a number inside a LaTeX boxed environment.
    Returns float if found, None otherwise.
    """
    if not solution:
        return None
    
    # First extract the raw boxed content
    raw_answer = extract_answer_from_solution(solution)
    if raw_answer is None:
        return None
        
    # Clean the answer string
    clean_answer = raw_answer.strip()
    if not clean_answer:
        return None
        
    # Remove any LaTeX formatting that might interfere with number parsing
    clean_answer = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', clean_answer)
    clean_answer = clean_answer.replace('\\', '')
    
    try:
        # Handle fractions like "1/2"
        if '/' in clean_answer:
            num, denom = clean_answer.split('/')
            return float(num.strip()) / float(denom.strip())
        return float(clean_answer)
    except (ValueError, ZeroDivisionError) as e:
        return None

def is_answer_correct(model_answer: Optional[float], correct_answer: Optional[float], tolerance: float) -> bool:
    """Compare two numeric answers within tolerance"""
    if model_answer is None or correct_answer is None:
        return False
    return abs(model_answer - correct_answer) <= tolerance

@async_retry(max_retries=10, timeout=120)
async def get_model_response(solver_model, prompt,max_tokens=None) -> str:
    """Get response from model with retry logic"""
    if max_tokens==None:
        response = await solver_model.ainvoke(prompt)
    else:
        response = await solver_model.ainvoke(prompt,max_tokens)
    return response.content

def count_manual_steps(solution: str) -> int:
    """Count steps in a solution by looking for step indicators"""
    # Look for common step patterns
    step_patterns = [
        r'Step\s+\d+',  # "Step 1", "Step 2", etc.
        r'\d+\)\s',     # "1)", "2)", etc.
        r'\d+\.\s',     # "1.", "2.", etc.
        r'First,',      # Common word indicators
        r'Second,',
        r'Third,',
        r'Finally,'
    ]
    
    total_steps = 0
    for pattern in step_patterns:
        steps = re.findall(pattern, solution, re.IGNORECASE)
        total_steps += len(steps)
    
    return max(1, total_steps)  # Return at least 1 step

def extract_answer_from_solution(solution: str) -> Optional[str]:
    """
    Extract the first boxed answer from the solution text by searching for LaTeX boxed answers: \boxed{X}.
    Returns the raw answer string with LaTeX notation preserved, or None if no boxed answer is found.
    """
    def find_matching_brace(s: str, start: int) -> int:
        """
        Find the index of the matching closing brace for an opening brace at the given start position.
        
        Args:
            s (str): The string to search.
            start (int): The index of the opening brace '{'.
        
        Returns:
            int: The index of the matching closing brace '}', or -1 if not found.
        """
        count = 1  # Initialize brace count
        i = start + 1  # Start searching after the opening brace
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Pattern to find all occurrences of \boxed{ with proper escaping
    pattern = re.compile(r'\\boxed\{')
    for match in pattern.finditer(solution):
        start = match.end() - 1  # Position of the opening brace '{'
        end = find_matching_brace(solution, start)
        if end != -1:
            # Extract content between the braces
            content = solution[start + 1:end].strip()
            return content  # Return the first found boxed content

    return None  # Return None if no boxed content is found


async def run_benchmark(
    config: BenchmarkConfig,
    process_example_func: Callable
) -> None:
    """Generic benchmark runner that handles dataset loading and example processing"""
    if config.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    # Load exclude list if provided
    excluded_ids = set()
    if config.exclude and os.path.exists(config.exclude):
        try:
            with open(config.exclude, 'r') as f:
                exclude_data = json.load(f)
                excluded_ids = {item['id'] for item in exclude_data if 'id' in item}
            print(f"Loaded {len(excluded_ids)} problems to exclude")
        except Exception as e:
            print(f"Error loading exclude file: {e}")
            return

    try:
        if config.dataset == 'original':
            dataset = load_dataset("AI-MO/NuminaMath-CoT", split=config.split)
        elif config.dataset == 'aime':
            dataset = load_dataset("AI-MO/aimo-validation-aime", split=config.split)
        else:  # filtered
            username = whoami()["name"]
            dataset = load_dataset(f"{username}/Numina", split=config.split)
        
        # Filter out excluded problems
        if excluded_ids:
            dataset = dataset.filter(lambda x: x['id'] not in excluded_ids)
            print(f"Filtered dataset to exclude {len(excluded_ids)} problems")
            
        if config.split_slice:
            dataset = dataset.select(range(*config.split_slice.indices(len(dataset))))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if config.split_slice:
        dataset_length = min(config.split_slice.stop, len(dataset))
    else:
        dataset_length = len(dataset)

    progress_tracker = ProgressTracker(
        total_examples=dataset_length,
        best_of=config.best_of,
        config=config
    )

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

    print(f"\nStarting processing of {progress_tracker.total_examples} examples...")
    try:
        semaphore = asyncio.Semaphore(config.max_concurrent)

        async def process_with_semaphore(example: Dict, running_id: int) -> Optional[Dict]:
            async with semaphore:
                result = await process_example_func(
                    example=example,
                    running_id=running_id,
                    example_id=example['id'],
                    config=config
                )
            if result:
                progress_tracker.add_result(result)
                progress_tracker.print_progress(config.solver, config.split)
            return result

        tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
        
        print(f"\nWill process {len(example_data)} examples")
        
        progress_bar = tqdm(total=len(example_data), desc="Processing examples")
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
            progress_bar.update(1)
        progress_bar.close()
    
    finally:
        progress_tracker.print_final_stats()
        progress_tracker.save_results(config.solver, config.split)
