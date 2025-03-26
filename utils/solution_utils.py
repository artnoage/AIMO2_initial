import re
import sympy
import tempfile
import subprocess
import os
import sys
import json
import random
import math
from math import inf
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
    if "\\text{ or }" in answer or "\\text{ and }" in answer:
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
    """
    Extract code from the model's response.
    Handles various formats including code blocks, response tags, and raw code.
    
    Args:
        response: The text to extract code from
        
    Returns:
        str: The extracted code or empty string if no code found
    """
    if not response or not response.strip():
        return ""
        
    # First try to extract code from ```python blocks (most reliable)
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # Also try other code block formats
    code_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # If we're already inside a response section, don't look for nested ones
    if not re.search(r'<response>', response):
        # If no code blocks, try to extract from <response> section
        response_match = re.search(r'<response>\s*(.*?)\s*</response>', response, re.DOTALL)
        if response_match:
            response_content = response_match.group(1)
            # Check if there are code blocks within the response section
            code_blocks = re.findall(r'```python\s*(.*?)\s*```', response_content, re.DOTALL)
            if code_blocks:
                return code_blocks[0]
            # Also try other code block formats
            code_blocks = re.findall(r'```\s*(.*?)\s*```', response_content, re.DOTALL)
            if code_blocks:
                return code_blocks[0]
            # If no code blocks in response section, assume the entire response section is code
            # Check if it looks like Python code (has def, import, print, etc.)
            if (re.search(r'\bdef\b|\bimport\b|\bprint\b|\bfor\b|\bif\b|\breturn\b', response_content) and 
                not re.search(r'<[a-z]+>', response_content)):  # Avoid HTML-like content
                return response_content
    
    # Look for Python-like code patterns in the entire response
    if (re.search(r'\bdef\b|\bimport\b|\bprint\b|\bfor\b|\bif\b|\breturn\b', response) and 
        not re.search(r'<[a-z]+>', response)):  # Avoid HTML-like content
        return response
        
    # If no structured format and no Python-like patterns, return empty string
    # This is more conservative than before to avoid treating non-code as code
    return ""

def check_code_quality(code: str) -> Tuple[bool, str]:
    """Check code for syntax errors and basic linting issues"""
    if not code.strip():
        return False, "Empty code"
        
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
    
    # Check if code has at least one function definition or meaningful computation
    if not re.search(r'\bdef\b|\bprint\b|\breturn\b|=\s*[a-zA-Z0-9_]+', code):
        return False, "Code lacks meaningful computation or function definitions"
    
    return True, "Code passed quality checks"

def run_code_safely(code: str, timeout: int = 300) -> Tuple[bool, Optional[float], str]:
    """Run the code in a safe environment with timeout and capture the output"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))
    
    try:
        # Use process group to ensure all child processes are terminated
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Use process group
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            
            if process.returncode != 0:
                return False, None, f"Execution error: {stderr}"
            
            # Try to parse the output as a float
            output = stdout.strip()
            try:
                answer = float(output)
                return True, answer, "Success"
            except ValueError:
                return False, None, f"Output is not a valid number: '{output}'"
                
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            import signal
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.communicate()  # Clean up
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

# Function removed - functionality merged into validate_solution with start_step parameter

def run_test_function(code: str, test_cases: List[float], correct_answer: float, timeout: int = 30) -> Tuple[bool, Dict[float, bool], str]:
    """
    Run the test function on multiple test cases
    
    Returns:
    - success: Whether the test function works correctly
    - results: Dictionary mapping test values to test results
    - error_message: Error message if any
    """
    # Print debug info about the input
    print(f"DEBUG - run_test_function received: type={type(correct_answer)}, value={correct_answer}")
    
    # Handle infinity values in test cases
    safe_test_cases = []
    for case in test_cases:
        if math.isinf(case) if hasattr(case, "__float__") else False:
            # Skip infinity values as they can cause issues in test functions
            continue
        safe_test_cases.append(case)
    
    # If we filtered out all test cases (unlikely), add some safe values
    if not safe_test_cases:
        safe_test_cases = [0.0, 1.0, -1.0, 1000.0, -1000.0]
        # If correct_answer is not infinity, add it
        if not (math.isinf(correct_answer) if hasattr(correct_answer, "__float__") else False):
            safe_test_cases.append(correct_answer)
    
    # Initialize results dictionary
    results = {}
    
    # Test each case individually in separate processes
    for test_case in safe_test_cases:
        # Create a simple test script for this specific test case
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file_path = temp_file.name
            
            # Create a minimal test script that just returns True/False
            test_script = f"""
{code}
# Run test on a single value and print result
try:
    result = test_solution({test_case})
    print("TRUE" if result else "FALSE")
except Exception as e:
    print(f"ERROR: {{str(e)}}")
"""
            temp_file.write(test_script.encode('utf-8'))
        
        try:
            # Run the test script in a separate process with timeout
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Use process group
            )
            
            try:
                # Use a shorter timeout for communicate to ensure we don't get stuck
                stdout, stderr = process.communicate(timeout=timeout/len(safe_test_cases))  # 5 second timeout per test case
                
                if process.returncode != 0:
                    results[test_case] = f"Error: {stderr}"
                else:
                    output = stdout.strip()
                    if output == "TRUE":
                        results[test_case] = True
                    elif output == "FALSE":
                        results[test_case] = False
                    elif output.startswith("ERROR:"):
                        results[test_case] = output
                    else:
                        results[test_case] = f"Unexpected output: {output}"
                    
            except subprocess.TimeoutExpired:
                # Kill the entire process group forcefully
                import signal
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass  # Process might already be gone
                
                # Try to clean up without waiting too long
                try:
                    process.communicate(timeout=1)  # Short timeout for cleanup
                except subprocess.TimeoutExpired:
                    pass  # If still hanging, just move on
                
                results[test_case] = "Timeout"
                
        except Exception as e:
            results[test_case] = f"Error: {str(e)}"
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            # Make sure the process is really gone
            try:
                if process.poll() is None:  # Process still running
                    import signal
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (NameError, ProcessLookupError, OSError):
                pass  # Process variable might not exist or process already gone
    
    # Check if the test function correctly identifies the correct answer
    correct_result = results.get(correct_answer, None)
    if correct_result is not True:
        return False, results, f"Test function failed to identify correct answer: {correct_result}"
    
    # Check if incorrect answers are identified as False
    incorrect_results = [v for k, v in results.items() if k != correct_answer]
    
    # Count different types of results
    incorrect_count = sum(1 for r in incorrect_results if r is True)  # Should be False
    timeout_count = sum(1 for r in incorrect_results if r == "Timeout")
    error_count = sum(1 for r in incorrect_results if isinstance(r, str) and r != "Timeout")
    
    if incorrect_count > 0:
        return False, results, f"Test function incorrectly accepted {incorrect_count} wrong answers"
    elif timeout_count > 0 or error_count > 0:
        # If we only have timeouts/errors but no incorrect acceptances, consider it a pass
        # but note the issues in the message
        message = ""
        if timeout_count > 0:
            message += f"{timeout_count} test cases timed out. "
        if error_count > 0:
            message += f"{error_count} test cases had errors. "
        return True, results, message.strip()
    
    return True, results, ""

def validate_solution(solution: str, start_step: int = 0) -> Tuple[bool, str]:
    """
    Validate if a solution has properly formatted steps with correct numbering.
    
    Args:
        solution: The complete solution to validate
        start_step: The step number to start from (for finalization validation)
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    # Extract steps from solution
    solution_steps = re.findall(r'<step>(.*?)</step>', solution, re.DOTALL)
    
    # If no step tags, try to extract steps based on "Step N" pattern
    if not solution_steps:
        # Look for "Step N" pattern in the solution
        step_matches = re.findall(r'Step\s*(\d+)[:.)\s]', solution)
        if not step_matches:
            return False, "Solution contains no steps"
            
        # We found step markers but they're not in tags
        # This is enough to validate the presence of steps
        # We'll extract the step numbers for validation
        step_numbers = []
        for match in step_matches:
            try:
                step_numbers.append(int(match))
            except (ValueError, TypeError):
                pass
                
        # Check if we have the expected sequence of steps
        expected_steps = set(range(start_step + 1, start_step + len(step_numbers) + 1))
        found_steps = set(step_numbers)
        
        if found_steps != expected_steps:
            return False, f"Missing or out of order steps. Expected {expected_steps}, found {found_steps}"
            
        # If we have the right steps in the right order, consider it valid
        return True, "Valid solution with step markers"
    
    if not solution_steps:
        return False, "Solution contains no steps"
    
    # Track found step numbers to ensure no duplicates or gaps
    found_steps = set()
    
    # Validate each step in solution
    for i, step in enumerate(solution_steps, 1):
        expected_step_num = start_step + i
        
        # Check if step starts with "Step N" - only accept this format
        step_match = re.search(r'Step\s*(\d+)[:.)\s]', step)
        
        if not step_match:
            return False, f"Step {i} does not have proper 'Step N:' format"
        
        # Extract the step number
        try:
            actual_step = int(step_match.group(1))
        except (ValueError, IndexError):
            return False, f"Could not parse step number in step {i}"
        
        # Validate step number
        if actual_step != expected_step_num:
            return False, f"Expected step {expected_step_num}, found step {actual_step}"
        
        if actual_step in found_steps:
            return False, f"Duplicate step number {actual_step}"
        
        found_steps.add(actual_step)
        
        # Check if step has sufficient content
        content_after_number = step[step_match.end():].strip()
        if len(content_after_number) < 10:  # Minimum content length
            return False, f"Step {actual_step} has insufficient content"
    
    # Check for gaps in step numbers
    expected_steps = set(range(start_step + 1, start_step + len(solution_steps) + 1))
    if found_steps != expected_steps:
        return False, f"Missing or out of order steps. Expected {expected_steps}, found {found_steps}"
    
    return True, "Valid solution"
                    

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

def extract_test_function(solution: str) -> str:
    """Extract the test_solution function from the model's response"""
    # First try to extract from response section
    response_match = re.search(r'<response>(.*?)</response>', solution, re.DOTALL)
    if response_match:
        response_content = response_match.group(1)
        # Extract code block from response
        code_match = re.search(r'```python(.*?)```', response_content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
    
    # If that fails, try to extract from the whole solution
    code_match = re.search(r'```python(.*?)```', solution, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no code blocks found, look for function definition directly
    func_match = re.search(r'def test_solution\(.*?\):(.*?)(?=\n\S|\Z)', solution, re.DOTALL)
    if func_match:
        return "def test_solution" + func_match.group(0)
    
    return ""




def generate_test_cases(correct_answer: float, num_cases: int = 50) -> List[float]:
    """
    Generate test cases including the correct answer and many incorrect answers.
    The test cases should be sufficiently different from the correct answer
    to ensure the test function properly discriminates between correct and incorrect answers.
    
    Args:
        correct_answer: The correct answer to the problem
        num_cases: Number of test cases to generate (default: 50)
        
    Returns:
        List of test values including the correct answer and incorrect answers
    """
    # Handle special cases first
    is_infinity = math.isinf(correct_answer) if hasattr(correct_answer, "__float__") else False
    
    # For infinity, use a large finite number instead
    if is_infinity:
        if correct_answer > 0:  # positive infinity
            correct_answer_for_tests = 1e10  # Use a very large number
        else:  # negative infinity
            correct_answer_for_tests = -1e10  # Use a very negative number
    else:
        correct_answer_for_tests = correct_answer
    
    test_cases = [correct_answer]  # Always include the actual correct answer
    
    # Generate values that are significantly different from the correct answer
    # to ensure the test function can discriminate between correct and incorrect answers
    
    # Fixed multipliers for predictable test cases
    multipliers = [0.5, 2.0, -1.0, 10.0, 0.1, 5.0, 0.01, 20.0, -0.5, -2.0, -5.0, -10.0, 100.0, 0.001]
    
    # Add some fixed offsets for values close to 0
    offsets = [0.1, 1.0, -0.1, -1.0, 100.0, 10.0, -10.0, 1000.0, -1000.0, 0.01, -0.01]
    
    # Add specific edge cases (avoid using inf directly)
    edge_cases = [0.0, 1.0, -1.0, 1e10 if correct_answer != inf else 1e9, -1e10 if correct_answer != -inf else -1e9]
    for case in edge_cases:
        # Ensure the test case is at least 1e-2 away from the correct answer
        if abs(case - correct_answer) > 1e-2:
            test_cases.append(case)
    
    # Skip multiplier-based cases for infinity
    if not is_infinity:
        # Add multiplier-based test cases
        for multiplier in multipliers:
            test_value = correct_answer_for_tests * multiplier
            # Make sure the test value is at least 1e-2 away from the correct answer
            if abs(test_value - correct_answer) > 1e-2:
                test_cases.append(test_value)
        
        # Add offset-based test cases (especially important when correct_answer is close to 0)
        if abs(correct_answer_for_tests) < 1.0:
            for offset in offsets:
                test_value = correct_answer_for_tests + offset
                if abs(test_value - correct_answer) > 1e-2:
                    test_cases.append(test_value)
        
        # Add random test cases to reach the desired number
        while len(test_cases) < num_cases + 1:  # +1 because we already have the correct answer
            # Mix of strategies for generating diverse test values
            strategy = random.randint(1, 3)
            
            if strategy == 1:
                # Random multiplier approach
                multiplier = random.uniform(0.001, 100.0) * random.choice([-1, 1])
                test_value = correct_answer_for_tests * multiplier
            elif strategy == 2:
                # Random offset approach
                magnitude = max(1.0, abs(correct_answer_for_tests) * 10)
                offset = random.uniform(-magnitude, magnitude)
                test_value = correct_answer_for_tests + offset
            else:
                # Completely random value within a reasonable range
                magnitude = max(100.0, abs(correct_answer_for_tests) * 100)
                test_value = random.uniform(-magnitude, magnitude)
            
            # Ensure the test value is at least 1e-2 away from the correct answer and not already in the list
            if abs(test_value - correct_answer) > 1e-2 and test_value not in test_cases:
                test_cases.append(test_value)
    else:
        # For infinity, generate a range of large finite values
        large_values = [1e6, 1e7, 1e8, 1e9, -1e6, -1e7, -1e8, -1e9]
        for val in large_values:
            if val not in test_cases:
                test_cases.append(val)
                
        # Add more random large values to reach the desired number
        while len(test_cases) < num_cases + 1:
            magnitude = random.randint(4, 9)  # 10^4 to 10^9
            sign = random.choice([-1, 1])
            test_value = sign * (10 ** magnitude)
            if test_value not in test_cases:
                test_cases.append(test_value)
    
    # Shuffle the test cases to avoid patterns
    random.shuffle(test_cases)
    
    # Ensure the correct answer is included
    if correct_answer not in test_cases:
        test_cases[0] = correct_answer
    
    return test_cases[:num_cases + 1]  # Limit to requested number of cases + correct answer


def split_into_steps(solution: str) -> List[str]:
    """
    Split a solution into steps.
    Handles:
    1. Steps enclosed in <step> tags
    2. Traditional "Step N" format
    3. Multiple steps inside a single <step> tag
    
    Returns a list of steps from the response section only, with each step wrapped in <step> tags.
    """
    # First check for <step> tags
    step_tags_with_content = re.findall(r'(<step>.*?</step>)', solution, re.DOTALL)
    if step_tags_with_content:
        # If we have step tags, extract them directly with tags included
        return step_tags_with_content
        
    # Extract content from step tags to check for multiple steps in one tag
    step_contents = re.findall(r'<step>(.*?)</step>', solution, re.DOTALL)
    if step_contents:
        # If we have only one step tag but it contains multiple steps
        if len(step_contents) == 1 and re.search(r'Step\s+\d+[\.:]', step_contents[0], re.IGNORECASE) and re.search(r'Step\s+\d+[\.:].*?Step\s+\d+[\.:]', step_contents[0], re.IGNORECASE | re.DOTALL):
            # Split the content of the single step tag by "Step"
            parts = step_contents[0].split("Step")
            steps = []
            for step in parts[1:]:  # Skip the first part before "Step"
                if step.strip():
                    full_step = "Step" + step
                    # Wrap in <step> tags
                    steps.append(f"<step>{full_step.strip()}</step>")
            return steps
        # Otherwise wrap each step in <step> tags
        return [f"<step>{step}</step>" for step in step_contents]
    
    # Fall back to traditional "Step" keyword splitting
    parts = solution.split("Step")
    if not parts:
        return []
        
    steps = []
    # Process first part (potential analysis)
    if parts[0].strip() and ("analysis" in parts[0].lower() or "<thinking>" in parts[0]):
        # Wrap in <step> tags if it's an analysis step
        steps.append(f"<step>{parts[0].strip()}</step>")
        
    # Process numbered steps
    for step in parts[1:]:
        if step.strip():  # Skip empty steps
            # Reconstruct the step with its prefix
            full_step = "Step" + step
            # Wrap in <step> tags
            steps.append(f"<step>{full_step.strip()}</step>")
            
    return steps

def get_partial_solutions(steps: List[str]) -> List[str]:
    """
    Generate partial solutions ending at each step.
    Each partial solution includes all previous steps.
    Expects steps to be wrapped in <step> tags and preserves them.
    Does NOT wrap in <response> tags to match finalization_grpo.py behavior.
    """
    if not steps:
        return []
        
    partial_solutions = []
    current = ""
    
    # Process steps (all steps should already have <step> tags)
    for step in steps:
        # Ensure step has proper tags
        if not (step.strip().startswith("<step>") and step.strip().endswith("</step>")):
            step = f"<step>{step}</step>"
            
        if current:
            current += "\n\n"  # Add spacing between steps
        current += step
        partial_solutions.append(current)
        
    return partial_solutions
