import re
import os
import asyncio
import json
import signal
from glob import glob
import sympy
from functools import wraps
from contextlib import contextmanager
import aiohttp
from typing import Optional, Dict, List, Callable, Tuple, TypeVar, Any
from langchain_openai import ChatOpenAI
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi, whoami
from bench_utils.benchmark_config import BenchmarkConfig, ModelOption
from bench_utils.progress_tracker import *
T = TypeVar('T')
import logging
from latex2sympy2 import latex2sympy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hybrid_creator.log', mode='w')
    ]
)

# Ensure all handlers use the same formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

# Set logging level for specific loggers
logging.getLogger('hybrid_creator').setLevel(logging.DEBUG)

# Compile regex patterns once



class TimeoutException(Exception): pass

class CustomChat:
    """Simple chat model that makes direct requests to server"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        temperature: float = 0,
        api_key: str = "EMPTY"
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        """Async call to completion endpoint"""
        max_tokens = kwargs.get("max_tokens", None)
        
        # Handle different prompt types
        if hasattr(prompt, 'content'):  # LangChain message object
            prompt_text = f"[INST]{prompt.content}[/INST]"
        elif isinstance(prompt, list):  # List of messages
            # Take the last message's content if it's a list
            prompt_text = f"[INST]{prompt[-1].content}[/INST]" if prompt else ""
        else:  # String or other
            prompt_text = f"[INST]{str(prompt)}[/INST]"
            
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "temperature": self.temperature,
            "stream": False
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                ) as response:
                    response_text = await response.text() 
                    if response.status != 200:
                        raise ValueError(f"Error from API: {response_text}")
                    
                    result = await response.json()
                    return type('Response', (), {
                        'content': result.get("choices", [{}])[0].get("text", "")
                    })()
            except Exception as e:
                print(f"Exception in CustomChat.ainvoke: {str(e)}")
                raise

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def get_model(model: ModelOption, temp: float, model_name: Optional[str] = None):
    """
    Initialize the ChatOpenAI model based on the selected ModelOption.
    For LOCAL models, it connects to a local endpoint and uses provided model name if any.
    For other models, it uses the OpenRouter API.
    
    Args:
        model: The model option to use
        temp: Temperature for generation
        model_name: Optional model name to use instead of model.value
    """
    name = model_name if model_name else model.value
    if model == ModelOption.LOCAL:
        return CustomChat(
            model=name,
            temperature=temp,
            api_key="EMPTY",
            base_url="http://localhost:8000/v1")
    else:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        
        return ChatOpenAI(
            model=name,
            temperature=temp,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1")


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

def extract_numeric_answer(answer: str, debug: bool = False) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value from a LaTeX answer string.
    First tries to evaluate using sympy, then falls back to direct float conversion.
    Returns float if found, None otherwise.
    """
    if not answer:
        return None, "No answer provided" if debug else (None, None)
        
    # Clean the answer string
    clean_answer = answer.strip()
    clean_answer = re.sub(r'\\textbf{([^}]*)}', r'\1', clean_answer)  # Remove \textbf{} first   
    clean_answer = re.sub(r'\\text{[^}]*}', '', clean_answer)
    clean_answer = clean_answer.replace('\\,', '')
    
    # Keep only what comes after the last = or \approx if present
    if '=' in clean_answer or '\\approx' in clean_answer:
        last_eq = clean_answer.rfind('=')
        last_approx = clean_answer.rfind('\\approx')
        split_point = max(last_eq, last_approx)
        if split_point != -1:
            clean_answer = clean_answer[split_point + (2 if last_eq > last_approx else 8):].strip()
    if not clean_answer:
        return None, "Empty answer after cleaning" if debug else (None, None)
    try:
        with time_limit(10):  # 10 second timeout
            # Parse LaTeX to sympy-compatible format
            latex_expr = latex2sympy(clean_answer)
            # Convert to sympy expression and evaluate
            expr = sympy.sympify(latex_expr)
            # Handle both single values and lists/matrices
            if hasattr(expr, 'evalf'):
                result = float(expr.evalf())
            elif isinstance(expr, list):
                # Take first element if it's a list/matrix
                result = float(expr[0].evalf())
            else:
                result = float(expr)
            return (result, f"Sympy success: {clean_answer} -> {latex_expr} -> {expr} -> {result}") if debug else (result, None)
    except TimeoutException:
        return (None, f"Timeout error: Processing took more than 10 seconds for input: {clean_answer}") if debug else (None, None)
    except (sympy.SympifyError, TypeError, ValueError) as e:
        return (None, f"Sympy error: {str(e)} on input: {clean_answer}") if debug else (None, None)

def is_answer_correct(model_answer: Optional[float], correct_answer: Optional[float], tolerance: float) -> bool:
    """Compare two numeric answers within tolerance"""
    if model_answer is None or correct_answer is None:
        return False
    return abs(model_answer - correct_answer) <= tolerance

@async_retry(max_retries=3, timeout=120)
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

STEP_NUMBER_PATTERNS = [
    re.compile(r'^.{0,2}(\d+)[.:\)]'),
    re.compile(r'^.{0,2}\((\d+)\)'),
    re.compile(r'^.{0,2}(\d+)\s')
]

def validate_analysis(resp: str) -> Tuple[bool, str]:
    """Validate an analysis response"""
    if "[/INST]" in resp:
        return False, "Contains [/INST] token"
        
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 20:
        return False, f"Too short: only {word_count} words (minimum 20)"
        
    # Analysis should mention problem and analysis
    if "problem" not in resp.lower():
        return False, "Missing 'problem' keyword"
    if "analysis" not in resp.lower():
        return False, "Missing 'analysis' keyword"
        
    return True, "Analysis valid"

def validate_solution(solution: str) -> Tuple[bool, str]:
    """
    Validate a complete solution against all required criteria.
    Returns (is_valid, reason) tuple.
    """
    # Check for analysis section
    if "analysis" not in solution.lower():
        return False, "Missing analysis section"
    
    # Check analysis length
    analysis_parts = [p for p in solution.lower().split("step") if "analysis" in p.lower()]
    if analysis_parts and len(analysis_parts[0].split()) < 20:
        return False, "Analysis section too short (< 20 words)"
        
    # Check for boxed answer
    if "\\boxed{" not in solution:
        return False, "Missing boxed answer"
        
    # Split into steps and validate each
    steps = solution.lower().split("step")[1:]  # Skip text before first "step"
    if not steps:
        return False, "No numbered steps found"
        
    # Track step numbers found
    found_numbers = []
    
    for i, step in enumerate(steps, 1):
        # Check step length
        step_words = len(step.split())
        if step_words < 18:
            return False, f"Step {i} too short ({step_words} words)"
        if step_words > 100:
            return False, f"Step {i} too long ({step_words} words)"
            
        # Check step numbering
        number_found = False
        for pattern in STEP_NUMBER_PATTERNS:
            match = pattern.search(step)
            if match:
                found_numbers.append(int(match.group(1)))
                number_found = True
                break
        if not number_found:
            return False, f"Missing number for step {i}"
            
    # Verify sequential step numbers
    expected_numbers = list(range(1, len(steps) + 1))
    if found_numbers != expected_numbers:
        return False, f"Steps not properly numbered. Found {found_numbers}, expected {expected_numbers}"
        
    return True, "Solution valid"

def validate_step(resp: str, expected_step: Optional[int] = None) -> bool:
    """Validate a solution step"""
    if "[/INST]" in resp:
        return False
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 18 or word_count > 100:
        return False
        
    # Check step numbering if expected step is provided
    if expected_step is not None:
        step_mentions = [
            f"step {expected_step}",
            f"step{expected_step}",
            f"({expected_step})",
            f"{expected_step}."
        ]
        if not any(mention.lower() in resp.lower() for mention in step_mentions):
            return False
            
    # Steps should not have multiple step mentions
    step_count = resp.lower().count("step")
    return step_count <= 1

def analyze_solution_quality(solution: str) -> Dict[str, Any]:
    """Analyze various quality metrics of a solution"""
    explanation_patterns = r'because|since|as\s+|explain|due\s+to|results?\s+in|leads?\s+to'
    logical_patterns = r'therefore|thus|hence|consequently|so|accordingly'
    
    return {
        'length': len(solution.split()),
        'has_analysis': bool(re.search(r'analysis|approach|strategy', solution.lower())),
        'step_count': len(re.findall(r'step\s+\d+', solution.lower())),
        'has_boxed': '\\boxed{' in solution,
        'has_equations': bool(re.search(r'\$.*\$', solution)),
        'has_therefore': bool(re.search(logical_patterns, solution.lower())),
        'has_explanation': bool(re.search(explanation_patterns, solution.lower())),
        'formatting_quality': sum([
            '\\boxed{' in solution,
            bool(re.search(r'\$.*\$', solution)),
            bool(re.findall(r'step\s+\d+', solution.lower())),
            bool(re.search(logical_patterns, solution.lower())),
            bool(re.search(explanation_patterns, solution.lower()))
        ])
    }

def calculate_rejected_score(solution: str) -> float:
    """Calculate rejected solution score starting from 0.4 and applying penalties"""
    score = 0.4
    
    # Penalty for no boxed answer
    if '\\boxed{' not in solution:
        score -= 0.2
        
    # Penalty for short solutions
    if len(solution.split()) < 80:
        score -= 0.1
        
    # Penalty for invalid analysis
    if not validate_analysis(solution):
        score -= 0.1
        
    # Penalty for incorrect step numbering
    steps = solution.lower().split("step")
    if len(steps) > 1:  # Only check if there are steps
        found_numbers = []
        missing_numbers = 0
        
        for step in steps[1:]:  # Skip text before first "step"
            number_found = False
            for pattern in STEP_NUMBER_PATTERNS:
                match = pattern.search(step)
                if match:
                    found_numbers.append(int(match.group(1)))
                    number_found = True
                    break
            if not number_found:
                missing_numbers += 1
                logging.debug(f"Missing step number in: {step[:50]}...")
        
        # Check if numbers are sequential starting from 1
        expected_sequence = list(range(1, len(steps)))
        
        # Calculate penalties
        if missing_numbers > 0:
            penalty = min(0.1, 0.02 * missing_numbers)
            score -= penalty
            logging.debug(f"Applied penalty {penalty} for {missing_numbers} missing step numbers")
            
        if found_numbers:
            # Check sequence correctness
            wrong_numbers = sum(1 for a, b in zip(found_numbers, expected_sequence) if a != b)
            if wrong_numbers > 0:
                penalty = min(0.1, 0.02 * wrong_numbers)
                score -= penalty
                logging.debug(f"Applied penalty {penalty} for {wrong_numbers} incorrect step numbers")
            
    return max(0.0, score)  # Ensure non-negative score


async def run_benchmark(
    config: BenchmarkConfig,
    process_example_func: Callable
) -> None:
    """Generic benchmark runner that handles dataset loading and example processing"""
    if config.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

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
            dataset = load_dataset(f"{username}/Olympiads", split=config.split)
            
        # First sort by ID to ensure consistent ordering
        dataset = dataset.sort('id')
            
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
        all_logs = []
        
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if result:
                    results.append(result)
                    if 'logs' in result:
                        all_logs.append(result['logs'])
                    if 'total_solution_attempts' in result:
                        all_logs.append(f"\nTotal solution attempts for example {len(results)}: {result['total_solution_attempts']}")
                    progress_bar.update(1)
            except Exception as e:
                all_logs.append(f"Error processing example: {str(e)}")
        progress_bar.close()
    
    finally:
        # Print all collected logs
        print("\n" + "="*80)
        print("COMPLETE LOG OUTPUT")
        print("="*80)
        for log in all_logs:
            print("\n" + log)
        print("\n" + "="*80)
        
        progress_tracker.print_final_stats()
        progress_tracker.save_results()
