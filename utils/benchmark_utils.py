import re
import os
import asyncio
import signal
import sympy
from functools import wraps
from contextlib import contextmanager
import aiohttp
from typing import Optional, Dict, List, Callable, Tuple, TypeVar, Any
from utils.benchmark_config import *
T = TypeVar('T')
from latex2sympy2 import latex2sympy
from langchain_core.messages import HumanMessage
class TimeoutException(Exception): pass

class OpenRouterChat:
    """Chat model that makes direct requests to OpenRouter API"""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0,
        api_key: str = None
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        """Async call to OpenRouter chat completion endpoint"""
        max_tokens = kwargs.get("max_tokens", None)
        # Handle different prompt types
        if hasattr(prompt, 'content'):  # LangChain message object
            messages = [{"role": "user", "content": prompt.content}]
        elif isinstance(prompt, list):  # List of messages
            messages = [{"role": "user", "content": prompt[-1].content}] if prompt else []
        else:  # String or other
            messages = [{"role": "user", "content": str(prompt)}]
            
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Create a new session for each request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"Error from OpenRouter API: {await response.text()}")
                    
                    result = await response.json()
                    return type('Response', (), {
                        'content': result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    })()
            except Exception as e:
                print(f"Exception in OpenRouterChat.ainvoke: {str(e)}")
                raise



class CustomChat:
    """Chat model that makes requests using OpenAI chat format"""
    
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
        """Async call to chat completion endpoint"""
        max_tokens = kwargs.get("max_tokens", None)
        
        # Convert prompt to messages format
        if hasattr(prompt, 'content'):  # LangChain message object
            messages = [{"role": "user", "content": prompt.content}]
        elif isinstance(prompt, list):  # List of messages
            messages = [{"role": "user", "content": prompt[-1].content}] if prompt else []
        else:  # String or other
            messages = [{"role": "user", "content": str(prompt)}]
            
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"Error from API: {await response.text()}")
                    
                    result = await response.json()
                    return type('Response', (), {
                        'content': result.get("choices", [{}])[0].get("message", {}).get("content", "")
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

def get_model(config: BenchmarkConfig, role: str = "main"):
    """
    Initialize the ChatOpenAI model based on configuration.
    For LOCAL models, it connects to a local endpoint.
    For other models, it uses the OpenRouter API.
    
    Args:
        config: The benchmark configuration
        role: The role of the model (e.g. "main", "auxiliary", etc.)
    """
    model = ModelOption[getattr(config, role)]
    
    name = model.value
    
    if role=="main":
        temp=config.main_temp
    elif role=="auxiliary":
        temp = config.auxiliary_temp
    else:
        temp=config.auxiliary2_temp

    if (model == ModelOption.LOCAL) or (model == ModelOption.LOCAL_2):
        port = {
            "main": config.main_port,
            "auxiliary": config.auxiliary_port,
            "auxiliary2": config.auxiliary2_port
        }.get(role, config.main_port)
        
        return CustomChat(
            model=name,
            temperature=temp,
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1")
    else:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        
        return OpenRouterChat(
            model=name,
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
                    await asyncio.sleep(1)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    await asyncio.sleep(1)
            raise Exception(f"Failed after {max_retries} retries")
        return wrapper
    return decorator

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

def extract_numeric_answer(answer: str, debug: bool = False) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value from a LaTeX answer string.
    First tries to evaluate using sympy, then falls back to direct float conversion.
    Returns float if found, None otherwise.
    """
    if not answer:
        return None, "No answer provided" if debug else (None, None)
        
    # Check for logical operators that indicate multiple answers
    if "\\text{or}" in answer or "\\text{and}" in answer:
        return None, "Answer contains 'or'/'and' operators" if debug else (None, None)
        
    # Clean the answer string
    clean_answer = answer.strip()
    clean_answer = re.sub(r'\\textbf{([^}]*)}', r'\1', clean_answer)  # Remove \textbf{} first   
    clean_answer = re.sub(r'\\text{[^}]*}', '', clean_answer)
    clean_answer = clean_answer.replace('\\pm', '')
    clean_answer = clean_answer.replace('\\ ', '')
    clean_answer = clean_answer.replace('\\,', '')
    clean_answer = clean_answer.replace('\\%', '')
    clean_answer = clean_answer.replace('^{\\circ}', '')  # Remove degree symbol
    clean_answer = clean_answer.replace('^\\circ', '')  # Remove degree symbol
    
    # Only split on = or \approx if there's a single term before it
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
            elif isinstance(expr, list) or isinstance(expr, tuple) or (
                hasattr(expr, 'is_Matrix') and expr.is_Matrix
            ):
                return (None, f"Rejected list/matrix answer: {expr}") if debug else (None, None)
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

@async_retry(max_retries=3, timeout=240)
async def get_model_response(model, prompt, max_tokens=None) -> str:
    """Get response from model with retry logic"""
    try:
        if max_tokens==None:
            response = await model.ainvoke(prompt)
        else:
            response = await model.ainvoke(prompt, max_tokens=max_tokens)
        return response.content
    except Exception as e:
        # Add small delay before retry to prevent overwhelming API
        await asyncio.sleep(0.1)
        raise

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
    Extract the answer from the solution text by searching for either:
    1. LaTeX boxed answers: \boxed{X}
    2. Hash-marked answers: #### X
    Returns the raw answer string with LaTeX notation preserved, or None if no answer is found.
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

    # First try to find boxed answer
    pattern = re.compile(r'\\boxed\{')
    for match in pattern.finditer(solution):
        start = match.end() - 1  # Position of the opening brace '{'
        end = find_matching_brace(solution, start)
        if end != -1:
            # Extract content between the braces
            content = solution[start + 1:end].strip()
            return content  # Return the first found boxed content

    # If no boxed answer found, try hash format
    if "####" in solution:
        return solution.split("####")[1].strip()

    return None  # Return None if no answer format is found

STEP_NUMBER_PATTERNS = [
    re.compile(r'^.*?Step\s*(\d+)[:.)\s]'),  # Match "Step N" with various separators
    re.compile(r'^.*?(\d+)[:.)](?:\s|$)'),   # Match "N." or "N)" at start
    re.compile(r'^\s*(\d+)\.\s'),            # Match "N. " at start
    re.compile(r'^\s*\((\d+)\)\s'),          # Match "(N) " at start
    re.compile(r'^\s*Step\s*(\d+)$')         # Match just "Step N" at end of line
]

def validate_analysis(resp: str) -> Tuple[bool, str]:
    """Validate an analysis response"""
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
        
    if "[…]" in solution.lower():   
        return False, "Skips steps"

def validate_solution2(solution: str) -> Tuple[bool, str]:
    """Validate a solution with thinking section and steps"""
    # Check for thinking section
    thinking_parts = re.findall(r'<thinking>(.*?)</thinking>', solution, re.DOTALL)
    if not thinking_parts:
        return False, "Missing thinking section"
    
    # Check thinking section content
    thinking = thinking_parts[0].strip()
    if len(thinking.split()) < 20:
        return False, "Thinking section too short"
        
    # Check for steps after thinking
    after_thinking = solution.split('</thinking>')[-1].strip()
    if not after_thinking:
        return False, "No solution steps after thinking section"
        
    if "[…]" in solution.lower():   
        return False, "Skips steps"
        
    # Check for links/URLs
    if any(x in solution.lower() for x in ['http://', 'https://', '.com', '.org', '.net', '.edu']):
        return False, "Contains URLs/links"
        
    # Check analysis section
    analysis_parts = [p for p in solution.lower().split("step") if "analysis" in p.lower()]
    if analysis_parts:
        is_valid, reason = validate_analysis(analysis_parts[0])
        if not is_valid:
            return False, f"Analysis section: {reason}"
        
    # Check for invalid phrases
    invalid_phrases = ["Could you help finish this solution?",
        "Remember to put the final answer",
        "Could you help finish this calculation"]
    for phrase in invalid_phrases:
        if phrase in solution:
            return False, f"Contains invalid phrase: '{phrase}'"

    # Check for boxed answer
    if "\\boxed{" not in solution:
        return False, "Missing boxed answer"
        
    # Split solution into steps and validate each one
    steps = split_into_steps(solution)
    if len(steps) <= 1:  # Only analysis or no steps
        return False, "No steps found"
        
    # Skip analysis section if present
    start_idx = 1 if "analysis" in steps[0].lower() else 0
    
    # Validate each step
    for i, step in enumerate(steps[start_idx:], 1):
        is_valid, reason = validate_step(step, expected_step=i)
        if not is_valid:
            return False, f"Step {i}: {reason}"
    
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
    # Check if completion starts with "Step" in first 5 chars
    if not completion[:5].strip().startswith("Step"):
        return False, "Completion must start with 'Step'"
        
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
        
    # Track found step numbers to ensure no duplicates or gaps
    found_steps = set()
    
    # Validate each step in completion
    for i, step in enumerate(completion_steps, 1):
        expected_step_num = last_step + i
        full_step = "Step" + step
        
        # Find actual step number in the completion
        actual_step = None
        for pattern in STEP_NUMBER_PATTERNS:
            match = pattern.search(full_step)
            if match:
                try:
                    actual_step = int(match.group(1))
                    break
                except ValueError:
                    continue
                    
        if actual_step is None:
            return False, f"Could not find step number in completion step {i}"
            
        if actual_step != expected_step_num:
            return False, f"Expected step {expected_step_num}, found step {actual_step}"
            
        if actual_step in found_steps:
            return False, f"Duplicate step number {actual_step}"
            
        found_steps.add(actual_step)
        
        # Validate step format and content
        is_valid, reason = validate_step(full_step, expected_step=expected_step_num)
        if not is_valid:
            return False, f"Step {expected_step_num}: {reason}"
            
    # Check for gaps in step numbers
    expected_steps = set(range(last_step + 1, last_step + len(completion_steps) + 1))
    if found_steps != expected_steps:
        return False, f"Missing or out of order steps. Expected {expected_steps}, found {found_steps}"
            
    return True, "Valid completion"
                    

def validate_step(resp: str, expected_step: Optional[int] = None) -> Tuple[bool, str]:
    """Validate a solution step"""
    # Check if response has less than 20 words or more than 120
    word_count = len(resp.split())
    if word_count < 20:
        return False, f"Step too short ({word_count} words < 20)"
    if word_count > 220:
        return False, f"Step too long ({word_count} words > 220)"
        
    # Check step numbering if expected step is provided
    if expected_step is not None:
        # First check for any step numbers in the text
        found_numbers = []
        
        # Reject if there are any decimal numbers in steps (e.g. 2.1)
        if re.search(r'step\s*\d+\.\d+', resp.lower()):
            return False, "Contains decimal step numbers"
            
        for pattern in STEP_NUMBER_PATTERNS:
            match = pattern.search(resp)
            if match:
                try:
                    num = int(match.group(1))
                    # Reject decimal numbers
                    if '.' in match.group(1):
                        return False, "Contains decimal step numbers"
                    found_numbers.append(num)
                except ValueError:
                    return False, "Invalid step number format"
                
        # If we found any numbers, they must match the expected step
        if found_numbers:
            if not any(num == expected_step for num in found_numbers):
                return False, f"Step number mismatch: expected {expected_step}"
        else:
            # No explicit numbers found, check for text mentions
            step_mentions = [
                f"step {expected_step}",
                f"step{expected_step}",
                f"({expected_step})",
                f"{expected_step}."
            ]
            if not any(mention.lower() in resp.lower() for mention in step_mentions):
                return False, f"Missing step number {expected_step}"
            
    # Steps should not have multiple step mentions
    step_count = resp.lower().count("step")
    if step_count > 1:
        return False, "Multiple step mentions"
        
    return True, "Step valid"

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

def remove_inst_tokens(text: str) -> str:
    """Remove all occurrences of [/INST] and [control_655] from the text"""
    text = text.replace("[/INST]", "")
    text = text.replace("[control_655]", "")
    return text

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
    First element is analysis (if present), followed by steps.
    """
    if not steps:
        return []
        
    partial_solutions = []
    current = ""
    
    # Handle analysis section if present
    if "analysis" in steps[0].lower():
        current = steps[0]
        steps = steps[1:]  # Remove analysis from steps to process
        partial_solutions.append(current)
        current += "\n\n"  # Add spacing after analysis
    
    # Process remaining steps
    for step in steps:
        current += step
        partial_solutions.append(current)
        current += "\n\n"  # Add spacing between steps
        
    return partial_solutions

