import re
import os
import asyncio
import json
from glob import glob
from functools import wraps
import aiohttp
from typing import Optional, Dict, List, Callable, Tuple, TypeVar, Any
from pathlib import Path
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi, whoami
from bench_utils.benchmark_config import BenchmarkConfig, ModelOption
from bench_utils.progress_tracker import *
T = TypeVar('T')

def get_model(model: ModelOption, temp: float = 0.1, model_name: Optional[str] = None):
    """
    Initialize the ChatOpenAI model based on the selected ModelOption.
    For LOCAL models, it connects to a local endpoint and uses provided model name if any.
    For other models, it uses the OpenRouter API.
    
    Args:
        model: The model option to use
        temp: Temperature for generation
        model_name: Optional model name to use instead of model.value
    """
    if model == ModelOption.LOCAL:
        name = model_name if model_name else model.value
        return ChatOpenAI(
            model=name,
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
    clean_answer = re.sub(r'\\textbf{([^}]*)}', r'\1', clean_answer)  # Remove \textbf{} first
    clean_answer = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', clean_answer)  # Remove other LaTeX commands
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

@async_retry(max_retries=5, timeout=180)
async def get_model_response(solver_model, prompt,max_tokens=None) -> str:
    """Get response from model with retry logic"""
    if max_tokens==None:
        response = await solver_model.ainvoke(prompt)
    else:
        response = await solver_model.ainvoke(prompt,max_tokens=max_tokens)
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

def is_multiple_choice(problem: str) -> bool:
    """Check if the problem contains multiple choice indicators (A,B,C,D)"""
    # Look for patterns like "(A)", "A)", "A.", etc followed by another option
    pattern = r'(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*'
    return bool(re.search(pattern, problem))

async def load_lora_adapter(lora_name: str, lora_path: str):
    """Send request to load LoRA adapter to local LLM server"""
    print(f"\nAttempting to load LoRA adapter:")
    print(f"Name: {lora_name}")
    print(f"Path: {lora_path}")
    
    try:
        async with aiohttp.ClientSession() as session:
            print("Sending request to server...")
            async with session.post(
                "http://localhost:8000/v1/load_lora_adapter",
                json={
                    "lora_name": lora_name,
                    "lora_path": lora_path
                }
            ) as response:
                response_text = await response.text()
                print(f"Server response status: {response.status}")
                print(f"Server response text: {response_text}")
                
                if response.status != 200:
                    if "already been loaded" in response_text:
                        print("LoRA adapter already loaded, continuing...")
                    else:
                        raise Exception(f"Failed to load LoRA adapter: {response_text}")
                else:
                    print("LoRA adapter loaded successfully")
    except Exception as e:
        print(f"Error during LoRA loading: {str(e)}")
        raise

def get_latest_lora_path():
    """Get the path of the most recent lora folder"""
    lora_folders = glob('loras/*/')
    if not lora_folders:
        return None
    return max(lora_folders, key=os.path.getctime)

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
    print("hello")
    # Handle LoRA loading based on config
    if config.lora_dir:
        # If specific directory provided, use that
        lora_dir = Path(config.lora_dir)
        if not lora_dir.exists():
            print(f"Warning: LoRA directory {lora_dir} does not exist")
        else:
            try:
                lora_name = lora_dir.name
                print(f"Loading LoRA adapter {lora_name} from: {lora_dir}")
                await load_lora_adapter(lora_name, str(lora_dir.absolute()))
            except Exception as e:
                print(f"Warning: Failed to load LoRA adapter {lora_name}: {e}")
    elif config.upload_lora:
        # Only try latest if no specific directory provided
        lora_path = get_latest_lora_path()
        if lora_path:
            try:
                print(f"Using latest LoRA adapter from: {lora_path}")
                lora_name = Path(lora_path).name
                await load_lora_adapter(lora_name, str(Path(lora_path).absolute()))
            except Exception as e:
                print(f"Warning: Failed to load latest LoRA adapter: {e}")
                print("Continuing benchmark without LoRA adapter...")

    # Load exclude list if provided
    excluded_problems = set()
    if config.exclude and os.path.exists(config.exclude):
        try:
            with open(config.exclude, 'r') as f:
                exclude_data = json.load(f)
                excluded_problems = {item['problem'] for item in exclude_data if 'problem' in item}
            print(f"Loaded {len(excluded_problems)} problems to exclude")
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
            
        # First sort by ID to ensure consistent ordering
        dataset = dataset.sort('id')

        # Filter out multiple choice problems if configured
        if hasattr(config, 'exclude_multiple_choice') and config.exclude_multiple_choice:
            dataset = dataset.filter(lambda x: not is_multiple_choice(x['problem']))
            print(f"Filtered out multiple choice problems")
            
        # Filter out excluded problems
        if excluded_problems:
            dataset = dataset.filter(lambda x: x['problem'] not in excluded_problems)
            print(f"Filtered dataset to exclude {len(excluded_problems)} problems")
            
        # Shuffle dataset with seed if specified
        if config.seed is not None:
            dataset = dataset.shuffle(seed=config.seed)
            
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
                progress_tracker.print_progress()
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
        progress_tracker.save_results()
