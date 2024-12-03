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

def filter_by_token_ranges(examples: List[Dict], tokenizer, max_tokens: int = 4096) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Filter examples by token count and track distribution.
    
    Args:
        examples: List of conversation examples
        tokenizer: The tokenizer to use for counting
        max_tokens: Maximum allowed tokens per example
        
    Returns:
        Tuple of (filtered_examples, token_ranges)
    """
    token_ranges = {
        "0-1024": 0,
        "1024-2048": 0,
        "2048-4096": 0
    }
    
    filtered_examples = []
    for example in examples:
        total_tokens = sum(len(tokenizer.encode(msg["content"])) 
                         for msg in example["conversations"])
        if total_tokens <= 1024:
            token_ranges["0-1024"] += 1
            filtered_examples.append(example)
        elif total_tokens <= 2048:
            token_ranges["1024-2048"] += 1
            filtered_examples.append(example)
        elif total_tokens <= 4096:
            token_ranges["2048-4096"] += 1
            filtered_examples.append(example)
            
    return filtered_examples, token_ranges

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
                    print(f"Timeout error. Retrying... ({retry_count}/{max_retries})")
                    await asyncio.sleep(1)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    print(f"Error: {str(e)}. Retrying... ({retry_count}/{max_retries})")
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

@async_retry(max_retries=3, timeout=120)
async def get_model_response(solver_model, prompt) -> str:
    """Get response from model with retry logic"""
    response = await solver_model.ainvoke(prompt, max_tokens=2048)
    return response.content

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
    process_example_func: Callable,
    system_prompt=None,
    verifier_model=None,
    second_verifier_model=None
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
            username = whoami()["name"]
            dataset = load_dataset(f"{username}/Numina", split=config.split)
            
        if config.split_slice:
            dataset = dataset.select(range(*config.split_slice.indices(len(dataset))))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if config.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == config.source)
    
    # Load problems to exclude
    exclude_problems = set()
    if config.exclude:
        try:
            with open(config.exclude, 'r') as f:
                exclude_data = json.load(f)
                # Handle both old format (with metadata) and new format (direct list)
                if isinstance(exclude_data, dict) and "results" in exclude_data:
                    exclude_problems = {item['problem'] for item in exclude_data["results"]}
                else:
                    exclude_problems = {item['problem'] for item in exclude_data}
            print(f"\nLoaded {len(exclude_problems)} problems to exclude from {config.exclude}")
        except Exception as e:
            print(f"Warning: Could not load exclude file {config.exclude}: {e}")
    
    dataset = dataset.shuffle(seed=42)
    
    # Filter out excluded examples
    original_len = len(dataset)
    if exclude_problems:
        dataset = dataset.filter(lambda x: x['problem'] not in exclude_problems)
        excluded_count = original_len - len(dataset)
        print(f"Excluded {excluded_count} examples based on problem text")

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
                example=example,
                running_id=running_id,
                example_id=example['id'],
                solver_model=solver_model,
                verifier_model=verifier_model,
                second_verifier_model=second_verifier_model,
                best_of=config.best_of,
                config=config
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

    # Cleanup is handled by the context managers in get_model()
    pass
