import re
import os
import asyncio
import json
import signal
import sympy
from functools import wraps
from contextlib import contextmanager
import aiohttp
from typing import Optional, Dict, List, Callable, Tuple, TypeVar, Any
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import whoami
from utils.benchmark_config import *
from utils.progress_tracker import *
T = TypeVar('T')
from latex2sympy2 import latex2sympy





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

def get_model(config: BenchmarkConfig, role: str = "solver"):
    """
    Initialize the ChatOpenAI model based on configuration.
    For LOCAL models, it connects to a local endpoint.
    For other models, it uses the OpenRouter API.
    
    Args:
        config: The benchmark configuration
        role: The role of the model (e.g. "solver", "verifier", etc.)
    """
    model = ModelOption[getattr(config, role)]
    name = model.value
    temp = getattr(config, "temperature", 0.9)
    
    if model == ModelOption.LOCAL:
        return CustomChat(
            model=name,
            temperature=temp,
            api_key="EMPTY",
            base_url=f"http://localhost:{config.port}/v1")
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
                    await asyncio.sleep(1)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
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
    clean_answer = clean_answer.replace('^\\circ', '')  # Remove degree symbol
    
    def has_single_term(text: str) -> bool:
        """Check if text has only a single term (no operators outside brackets)"""
        bracket_level = 0
        for char in text:
            if char == '{':
                bracket_level += 1
            elif char == '}':
                bracket_level -= 1
            elif bracket_level == 0 and char in '+-*/^':
                return False
        return True

    # Handle = and \approx separately
    if '=' in clean_answer:
        eq_pos = clean_answer.rfind('=')
        before_eq = clean_answer[:eq_pos].strip()
        if has_single_term(before_eq):
            clean_answer = clean_answer[eq_pos + 1:].strip()
    
    if '\\approx' in clean_answer:
        approx_pos = clean_answer.rfind('\\approx')
        before_approx = clean_answer[:approx_pos].strip()
        if has_single_term(before_approx):
            clean_answer = clean_answer[approx_pos + 8:].strip()
    
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
    re.compile(r'^.{0,2}(\d+)[:\)]'),  # Removed dot from pattern
    re.compile(r'^.{0,2}\((\d+)\)'),
    re.compile(r'^.{0,2}(\d+)\s')
]

def validate_analysis(resp: str) -> Tuple[bool, str]:
    """Validate an analysis response"""
    #if "[/INST]" in resp:
    #    return False, "Contains [/INST] token"
        
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 20:
        return False, f"Too short: only {word_count} words (minimum 20)"
        
    # Analysis should mention problem and analysis
    if "problem" not in resp.lower():
        return False, "Missing 'problem' keyword"
    if "analysis" not in resp.lower():
        return False, "Missing 'analysis' keyword"
        
    # Analysis should not contain steps or boxed answers
    if "step" in resp.lower():
        return False, "Contains step(s)"
    if "\\boxed{" in resp:
        return False, "Contains boxed answer"
        
    return True, "Analysis valid"

def validate_solution(solution: str) -> Tuple[bool, str]:
    """Validate a complete solution"""
    # Check for analysis section
    if "analysis" not in solution.lower():
        return False, "Missing analysis section"
    #if "[/INST]" in solution.lower():
    #    return False, "Contains [/INST] token"
    #if "INST]" in solution.lower():
    #    return False, "Contains [/INST] token"
    if "[…]" in solution.lower():   
        return False, "Skips steps"
    # Check analysis length
    analysis_parts = [p for p in solution.lower().split("step") if "analysis" in p.lower()]
    if analysis_parts and len(analysis_parts[0].split()) < 20:
        return False, "Analysis section too short (< 20 words)"
        
    # Check for boxed answer
    if "\\boxed{" not in solution:
        return False, "Missing boxed answer"
        
    # Split into steps
    steps = solution.lower().split("step")[1:]  # Skip text before first "step"
    if not steps:
        return False, "No steps found"

    # Validate each step
    for i, step in enumerate(steps, 1):
        full_step = "Step" + step
        if not validate_step(full_step, expected_step=i):
            return False, f"Invalid step {i}"
        
    return True, "Solution valid"

def validate_completion(partial_solution: str, completion: str) -> Tuple[bool, str]:
    """
    Validate if a completion properly continues from a partial solution.
    
    Args:
        partial_solution: The solution up to a certain point
        completion: The proposed completion of the solution
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    # Check for invalid tokens
    if "[...]" in completion:
        return False, "Contains invalid tokens ([...])"
        
    # Get the last step number from partial solution
    parts = partial_solution.split("Step")
    last_step = 0
    for part in parts[1:]:  # Skip first split which is before "Step"
        for pattern in STEP_NUMBER_PATTERNS:
            match = pattern.search(part)
            if match:
                try:
                    num = int(match.group(1))
                    last_step = max(last_step, num)
                except ValueError:
                    continue
                    
    # Split completion into steps
    completion_steps = completion.split("Step")[1:]  # Skip text before first "Step"
    if not completion_steps:
        return False, "Completion contains no steps"
        
    # Validate each step in completion
    for i, step in enumerate(completion_steps, 1):
        expected_step_num = last_step + i
        full_step = "Step" + step
        if not validate_step(full_step, expected_step=expected_step_num):
            return False, f"Invalid step {expected_step_num} in completion"
            
    return True, "Valid completion"

def validate_step(resp: str, expected_step: Optional[int] = None) -> bool:
    """Validate a solution step"""
    #if "[/INST]" in resp:
    #    return False
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 22 or word_count > 100:
        return False
        
    # Check step numbering if expected step is provided
    if expected_step is not None:
        # First check for any step numbers in the text
        found_numbers = []
        
        # Reject if there are any decimal numbers in steps (e.g. 2.1)
        if re.search(r'step\s*\d+\.\d+', resp.lower()):
            return False
            
        for pattern in STEP_NUMBER_PATTERNS:
            match = pattern.search(resp)
            if match:
                try:
                    num = int(match.group(1))
                    # Reject decimal numbers
                    if '.' in match.group(1):
                        return False
                    found_numbers.append(num)
                except ValueError:
                    return False
                
        # If we found any numbers, they must match the expected step
        if found_numbers:
            if not any(num == expected_step for num in found_numbers):
                return False
        else:
            # No explicit numbers found, check for text mentions
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

class NumericVerifier:
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    async def verify(self, solution: str, correct_answer: str, problem: str) -> Tuple[bool, Optional[str]]:
        """Verify if solution's answer matches correct_answer within tolerance"""
        if not solution or not correct_answer:
            return False, None
            
        model_answer = extract_answer_from_solution(solution)
        if model_answer is None:
            return False, None
        
        # Extract and convert answers to numeric values
        numeric_answer, _ = extract_numeric_answer(model_answer, debug=False)
        correct_numeric, _ = extract_numeric_answer(correct_answer, debug=False)
        
        if numeric_answer is None or correct_numeric is None:
            return False, model_answer
            
        # Compare the numeric values
        is_correct = abs(numeric_answer - correct_numeric) <= self.tolerance
            
        return is_correct, model_answer

def split_into_steps(solution: str) -> List[str]:
    """
    Split a solution into analysis and numbered steps.
    Returns a list where first element is analysis (if present) followed by steps.
    """
    # First split on "Step" keyword
    parts = solution.split("Step")
    if not parts:
        return []
        
    steps = []
    # Process first part (potential analysis)
    if "analysis" in parts[0].lower():
        steps.append(parts[0].strip())
        
    # Process numbered steps
    for step in parts[1:]:
        if step.strip():  # Skip empty steps
            # Reconstruct the step with its prefix
            full_step = "Step" + step
            steps.append(full_step.strip())
            
    return steps

def get_partial_solutions(steps: List[str]) -> List[str]:
    """
    Generate partial solutions ending at each step.
    Each partial solution includes all previous steps.
    """
    partial_solutions = []
    current = ""
    
    for step in steps:
        if current:
            current += "\n\n"  # Add spacing between steps
        current += step
        partial_solutions.append(current)
        
    return partial_solutions

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
        if config.dataset == 'Metaskepsis/Numina':  # Default option
            dataset = load_dataset("Metaskepsis/Numina", split=config.split)
        else:  # Custom dataset name
            dataset = load_dataset(config.dataset, split=config.split)
            
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
