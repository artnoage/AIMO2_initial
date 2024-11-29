import re
import os
import asyncio
from enum import Enum
from functools import wraps
from typing import Optional, List, Dict, Tuple, TypeVar, Callable, Any
from langchain_core.messages import SystemMessage, HumanMessage

T = TypeVar('T')

NUMERIC_SOLVER_SYSTEM_PROMPT = """You are a precise mathematical problem solver. You will be given a problem to solve.

DO:
▪ List applicable theorems/techniques upfront
▪ If possible each step must contain a justification
▪ Use LaTeX notation
▪ Your final answer MUST be a single number in a LaTeX box

FORMAT:

**Problem Analysis and Approach**:
1. Start by categorizing the problem
2. List specific tools or theorems that will guide your solution

**PROOF**:
Show your work step by step with clear justifications in brackets.

**ANSWER**:
\\(\\boxed{n}\\) where n is your final numeric answer"""
from langchain_openai import ChatOpenAI

class ModelOption(Enum):
    """Enum class representing different model options for chat completion.
    
    Each enum value corresponds to a specific model endpoint that can be used
    with either OpenRouter API, SambaNova API, or local deployment.
    """
    CLAUDE = "anthropic/claude-3.5-sonnet:beta"
    GEMINI_PRO_FREE = "google/gemini-pro-1.5-exp"
    GEMINI_FLASH_FREE="google/gemini-flash-1.5-exp"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GEMINI_FLASH="google/gemini-flash-1.5"
    GPT = "openai/gpt-4o"
    GPT_MINI="openai/gpt-4o-mini"
    MASTER = "openai/o1-preview-2024-09-12"
    MASTER_MINI="openai/o1-mini"
    LOCAL = "artnoage/metastral"
    NEMOTRON= "nvidia/llama-3.1-nemotron-70b-instruct"
    CODER="qwen/qwen-2.5-coder-32b-instruct"

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
    
    # First extract the raw boxed content
    raw_answer = extract_answer_from_solution(solution)
    if raw_answer is None:
        return None
        
    # Clean the answer and try to convert to float
    clean_answer = raw_answer.strip()
    try:
        return float(clean_answer)
    except ValueError:
        return None

def is_answer_correct(model_answer: Optional[float], correct_answer: Optional[float], tolerance: float = 0.01) -> bool:
    """Compare two numeric answers within tolerance"""
    if model_answer is None or correct_answer is None:
        return False
    return abs(model_answer - correct_answer) <= tolerance

@async_retry(max_retries=3, timeout=300)
async def get_model_response(solver_model, prompt, running_id: int, attempt: int) -> str:
    """Get response from model with retry logic"""
    response = await solver_model.ainvoke(prompt)
    return response.content

BENCHMARK_SYSTEM_PROMPT = """You are a precise mathematical problem solver. You will be given a problem to solve.

DO:
▪ List applicable theorems/techniques upfront
▪ If possible each step must contain a justification
▪ Use LaTeX notation
▪ Your final answer MUST be a single number in a LaTeX box

FORMAT:

**Problem Analysis and Approach**:
1. Start by categorizing the problem
2. List specific tools or theorems that will guide your solution

**PROOF**:
Show your work step by step with clear justifications in brackets.

**ANSWER**:
\\(\\boxed{n}\\) where n is your final answer"""

async def compare_math_answers(model_answer: Optional[str], correct_answer: Optional[str], problem: str, model) -> bool:
    """Use the model to compare two mathematical answers"""
    if model_answer is None or correct_answer is None:
        return False
        
    comparison_prompt = [
        SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
        HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
    ]
    
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await asyncio.wait_for(
                model.ainvoke(comparison_prompt),
                timeout=300
            )
            return response.content.strip().lower() == 'yes'
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Verification failed after {max_retries} attempts")
                return False
            print(f"Connection error during verification. Retrying... ({retry_count}/{max_retries})")
            await asyncio.sleep(1)
    return False

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
