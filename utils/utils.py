import re
import os
from enum import Enum
from typing import Optional
from langchain_openai import ChatOpenAI

class ModelOption(Enum):
    CLAUDE = "anthropic/claude-3.5-sonnet:beta"
    GEMINI_PRO_FREE = "google/gemini-pro-1.5-exp"
    GEMINI_FLASH_FREE="google/gemini-flash-1.5-exp"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GEMINI_FLASH="google/gemini-flash-1.5"
    GPT = "openai/gpt-4o"
    GPT_MINI="openai/gpt-4o-mini"
    MASTER = "openai/o1-preview-2024-09-12"
    MASTER_MINI="openai/o1-mini"
    LOCAL = "models"
    GROQ = "llama-3.1-70b-versatile"
    NOUS ="nousresearch/hermes-3-llama-3.1-405b:free"
    NEMOTRON= "nvidia/llama-3.1-nemotron-70b-instruct"
    SAMBA_BIG= "Meta-Llama-3.1-405B-Instruct"
    SAMBA_SMALL= "Meta-Llama-3.1-70B-Instruct"
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
    elif model == ModelOption.SAMBA_BIG or model==ModelOption.SAMBA_SMALL :
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=os.getenv("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1")
    else:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=openrouter_api_key)

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
