import re
import sympy
import asyncio
import tempfile
import subprocess
import os
import sys
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple, Any
from latex2sympy2 import latex2sympy

from utils.model_utils import TimeoutException, time_limit

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

def count_manual_steps(solution: str) -> int:
    """
    Count steps in a solution using XML tags.
    Steps must be properly enclosed in <step>...</step> tags.
    Returns the number of valid step sections found.
    """
    # Extract all step sections
    step_sections = re.findall(r'<step>(.*?)</step>', solution, re.DOTALL)
    
    # Count only valid step sections
    valid_steps = 0
    for section in step_sections:
        # Step must have a number indicator
        if re.search(r'Step\s*\d+[:.)\s]', section, re.IGNORECASE):
            valid_steps += 1
            
    return max(1, valid_steps)  # Return at least 1 step

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

def has_thinking_section(solution: str) -> bool:
    """Check if solution has a thinking section"""
    thinking_parts = re.findall(r'<thinking>(.*?)</thinking>', solution, re.DOTALL)
    return bool(thinking_parts)

def extract_thinking_section(solution: str) -> Optional[str]:
    """Extract content from <thinking> or <reasoning> tags"""
    thinking_pattern = r'<(?:thinking|reasoning)>(.*?)</(?:thinking|reasoning)>'
    match = re.search(thinking_pattern, solution, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
    
def extract_response_section(solution: str) -> Optional[str]:
    """Extract content from <response> tags"""
    response_pattern = r'<response>(.*?)</response>'
    match = re.search(response_pattern, solution, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def has_response_section(solution: str) -> bool:
    """Check if solution has a response section"""
    response_parts = re.findall(r'<response>(.*?)</response>', solution, re.DOTALL)
    return bool(response_parts)

def extract_code_from_response(response: str) -> str:
    """Extract code from the model's response"""
    # First try to extract code from ```python blocks
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # If no code blocks, try to extract from <response> section
    response_match = re.search(r'<response>\s*(.*?)\s*</response>', response, re.DOTALL)
    if response_match:
        response_content = response_match.group(1)
        # Check if there are code blocks within the response section
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response_content, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        # If no code blocks in response section, assume the entire response section is code
        return response_content
    
    # If no structured format, assume the entire response is code
    return response

def check_code_quality(code: str) -> Tuple[bool, str]:
    """Check code for syntax errors and basic linting issues"""
    # First check for syntax errors
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    
    # Check for basic issues without requiring pylint
    issues = []
    
    # Check for potentially dangerous operations
    dangerous_patterns = [
        (r'os\.system', 'Contains potentially unsafe system call'),
        (r'subprocess\.', 'Contains potentially unsafe subprocess call'),
        (r'exec\s*\(', 'Contains potentially unsafe exec call'),
        (r'eval\s*\(', 'Contains potentially unsafe eval call'),
        (r'__import__', 'Contains potentially unsafe dynamic import'),
        (r'open\s*\(.+,\s*[\'"]w', 'Contains file write operation'),
        (r'import\s+requests', 'Contains network request library'),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, code):
            issues.append(message)
    
    # If there are issues, return them
    if issues:
        return False, "Linting issues: " + "; ".join(issues)
    
    return True, "Code passed quality checks"

def run_code_safely(code: str, timeout: int = 5) -> Tuple[bool, Optional[float], str]:
    """Run the code in a safe environment with timeout and capture the output"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))
    
    try:
        # Run the code with timeout
        with time_limit(timeout):
            # Use subprocess to run the code
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return False, None, f"Execution error: {result.stderr}"
            
            # Try to parse the output as a float
            output = result.stdout.strip()
            try:
                answer = float(output)
                return True, answer, "Success"
            except ValueError:
                return False, None, f"Output is not a valid number: '{output}'"
    
    except TimeoutException:
        return False, None, "Code execution timed out"
    except Exception as e:
        return False, None, f"Error running code: {str(e)}"
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

def check_steps_status(solution: str) -> Tuple[bool, bool]:
    """
    Check if solution has steps and if they are ordered.
    Returns (has_steps, is_ordered)
    """
    response_parts = re.findall(r'<response>(.*?)</response>', solution, re.DOTALL)
    if not response_parts:
        return False, False
        
    response = response_parts[0].strip()
    steps = []
    
    # Look for numbered steps (1., 2., etc)
    step_matches = re.finditer(r'(\d+)\.\s', response)
    for match in step_matches:
        step_num = int(match.group(1))
        steps.append(step_num)
    
    has_steps = bool(steps)
    is_ordered = has_steps and all(steps[i] < steps[i+1] for i in range(len(steps)-1))
    
    return has_steps, is_ordered

def has_boxed_answer(solution: str) -> bool:
    """Check if solution has a boxed answer"""
    return "\\boxed{" in solution

def validate_step(step: str, expected_step: int = None) -> Tuple[bool, str]:
    """
    Validate if a step has proper formatting and content.
    
    Args:
        step: The step content to validate
        expected_step: The expected step number, if known
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    # Check if step is empty
    if not step.strip():
        return False, "Step is empty"
    
    # Check if step has a number indicator
    step_match = None
    for pattern in STEP_NUMBER_PATTERNS:
        match = pattern.search(step)
        if match:
            step_match = match
            break
    
    if not step_match:
        return False, "Step does not have a valid step number format"
    
    # If expected step is provided, check if it matches
    if expected_step is not None:
        try:
            actual_step = int(step_match.group(1))
            if actual_step != expected_step:
                return False, f"Expected step {expected_step}, found step {actual_step}"
        except (ValueError, IndexError):
            return False, "Could not parse step number"
    
    # Check if step has meaningful content (more than just the step indicator)
    content_after_number = step[step_match.end():].strip()
    if len(content_after_number) < 10:  # Arbitrary minimum length
        return False, "Step has insufficient content"
    
    return True, "Valid step"

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
    
    # Check if we're using <step> tags format
    using_tags = "<step>" in partial_solution or "<step>" in completion
    
    if using_tags:
        # Extract steps from completion
        completion_steps = re.findall(r'<step>(.*?)</step>', completion, re.DOTALL)
        if not completion_steps:
            return False, "Completion contains no steps"
            
        # Extract steps from partial solution
        partial_steps = re.findall(r'<step>(.*?)</step>', partial_solution, re.DOTALL)
        
        # Get the last step number from partial solution
        last_step = 0
        for step in partial_steps:
            for pattern in STEP_NUMBER_PATTERNS:
                match = pattern.search(step)
                if match:
                    try:
                        num = int(match.group(1))
                        last_step = max(last_step, num)
                    except ValueError:
                        continue
        
        # Track found step numbers to ensure no duplicates or gaps
        found_steps = set()
        
        # Validate each step in completion
        for i, step in enumerate(completion_steps, 1):
            expected_step_num = last_step + i
            
            # Find actual step number in the completion
            actual_step = None
            for pattern in STEP_NUMBER_PATTERNS:
                match = pattern.search(step)
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
            is_valid, reason = validate_step(step, expected_step=expected_step_num)
            if not is_valid:
                return False, f"Step {expected_step_num}: {reason}"
        
        # Check for gaps in step numbers
        expected_steps = set(range(last_step + 1, last_step + len(completion_steps) + 1))
        if found_steps != expected_steps:
            return False, f"Missing or out of order steps. Expected {expected_steps}, found {found_steps}"
                
        return True, "Valid completion"
    else:
        # Traditional format (without tags)
        # Check if completion starts with "Step" in first 5 chars
        if not completion[:5].strip().startswith("Step"):
            return False, "Completion must start with 'Step'"
            
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
                    

class NumericVerifier:
    def __init__(self, tolerance: float = 1e-2):
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
    Split a solution into steps.
    Handles both:
    1. Steps enclosed in <step> tags
    2. Traditional "Step N" format
    
    Returns a list of steps from the response section only.
    """
    # First check for <step> tags
    step_tags = re.findall(r'<step>(.*?)</step>', solution, re.DOTALL)
    if step_tags:
        # Just return the steps without thinking section
        return step_tags
    
    # Fall back to traditional "Step" keyword splitting
    parts = solution.split("Step")
    if not parts:
        return []
        
    steps = []
    # Process first part (potential analysis)
    if parts[0].strip() and ("analysis" in parts[0].lower() or "<thinking>" in parts[0]):
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
    Handles both traditional steps and <step> tag format.
    Does NOT wrap in <response> tags to match completion_grpo.py behavior.
    """
    if not steps:
        return []
        
    partial_solutions = []
    current = ""
    
    # Check if we're using <step> tags format
    using_tags = any("<step>" in step or "</step>" in step for step in steps)
    
    # Process steps
    for step in steps:
        # For tagged format, wrap step in tags if not already wrapped
        if using_tags and not (step.strip().startswith("<step>") and step.strip().endswith("</step>")):
            step = f"<step>{step}</step>"
            
        if current:
            current += "\n\n"  # Add spacing between steps
        current += step
        partial_solutions.append(current)
        
    return partial_solutions
