import os
import asyncio
import logging
import tempfile
import subprocess
import sys
import re
import random
from io import StringIO
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, List, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.model_utils import *
from utils.solution_utils import *
from utils.agents import *
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time of a block of code"""
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out")
    
    import signal
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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


def generate_test_cases(correct_answer: float, num_cases: int = 5) -> List[float]:
    """Generate test cases including the correct answer and some incorrect answers"""
    test_cases = [correct_answer]
    
    # Generate some values close to the correct answer
    for _ in range(num_cases // 2):
        # Add small perturbations to the correct answer
        perturbation = random.uniform(-0.1, 0.1) * abs(correct_answer)
        if perturbation == 0:  # Avoid adding exactly 0
            perturbation = 0.01
        test_cases.append(correct_answer + perturbation)
    
    # Generate some completely different values
    for _ in range(num_cases - len(test_cases) + 1):
        # Generate values that are significantly different
        multiplier = random.choice([0.5, 2, 10, -1])
        test_cases.append(correct_answer * multiplier)
    
    return test_cases


def run_test_function(code: str, test_cases: List[float], correct_answer: float, timeout: int = 30) -> Tuple[bool, Dict[float, bool], str]:
    """
    Run the test function on multiple test cases
    
    Returns:
    - success: Whether the test function works correctly
    - results: Dictionary mapping test values to test results
    - error_message: Error message if any
    """
    # Create a temporary file with the test function
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_path = temp_file.name
        
        # Write the test function to the file
        test_code = code + "\n\n"
        
        # Add code to run the test function on all test cases
        test_code += "import sys\n"
        test_code += "import json\n\n"
        test_code += "def run_tests():\n"
        test_code += "    results = {}\n"
        test_code += "    test_cases = " + str(test_cases) + "\n"
        test_code += "    for case in test_cases:\n"
        test_code += "        try:\n"
        test_code += "            result = test_solution(case)\n"
        test_code += "            results[str(case)] = bool(result)\n"
        test_code += "        except Exception as e:\n"
        test_code += "            results[str(case)] = f'Error: {str(e)}'\n"
        test_code += "    print(json.dumps(results))\n\n"
        test_code += "run_tests()\n"
        
        temp_file.write(test_code.encode('utf-8'))
    
    try:
        # Run the test function with a timeout
        with time_limit(timeout):
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        if result.returncode != 0:
            return False, {}, f"Execution error: {result.stderr}"
        
        # Parse the results
        try:
            results_dict = json.loads(result.stdout.strip())
            
            # Convert string keys back to floats
            parsed_results = {}
            for key, value in results_dict.items():
                try:
                    float_key = float(key)
                    parsed_results[float_key] = value
                except ValueError:
                    parsed_results[key] = value
            
            # Check if the test function correctly identifies the correct answer
            correct_result = parsed_results.get(correct_answer, None)
            if correct_result is not True:
                return False, parsed_results, f"Test function failed to identify correct answer: {correct_result}"
            
            # Check if at least one incorrect answer is identified as False
            incorrect_results = [v for k, v in parsed_results.items() if k != correct_answer]
            if not any(result is False for result in incorrect_results):
                return False, parsed_results, "Test function fails to reject any incorrect answers"
            
            return True, parsed_results, ""
            
        except json.JSONDecodeError:
            return False, {}, f"Failed to parse results: {result.stdout}"
            
    except TimeoutException:
        # Clean up the temporary file
        os.unlink(temp_file_path)
        return False, {}, "Code execution timed out"
    except Exception as e:
        # Clean up the temporary file
        os.unlink(temp_file_path)
        return False, {}, f"Error running test function: {str(e)}"


async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with test function verification"""
    logger = BenchmarkLogger()
    try:
        if not isinstance(example, dict) or 'problem' not in example or (('solution' not in example) and ('answer' not in example)):
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None
        
        # Extract the correct answer
        correct_answer = None
        if 'answer' in example and example['answer']:
            correct_answer = example['answer']
        else:
            correct_answer = extract_answer_from_solution(example['solution'])
        
        if correct_answer is None:
            logger.append(f"❌ Warning: Could not extract answer from solution for example {str(running_id)}")
            logger.print()
            return None

        # Convert correct_answer to float if possible
        try:
            numeric_answer, _ = extract_numeric_answer(correct_answer)
            if numeric_answer is not None:
                correct_answer = numeric_answer
            else:
                logger.append(f"❌ Warning: Could not convert answer to numeric value for example {str(running_id)}")
                logger.print()
                return None
        except:
            logger.append(f"❌ Warning: Error converting answer to numeric value for example {str(running_id)}")
            logger.print()
            return None

        main = get_model(config, role="main")
        testing_agent = TestingAgent(main)
        solutions = []
        correct_count = 0
        
        for attempt in range(config.best_of):
            try:
                prompt, full_solution = await testing_agent.generate(
                    example["problem"], 
                    correct_answer=str(correct_answer),
                    return_prompt=True
                )
                
                # Extract the test function from the solution
                test_function = extract_test_function(full_solution)
                
                if not test_function:
                    logger.append(f"❌ No test function found in solution for attempt {attempt+1}")
                    solutions.append({
                        'solution': full_solution,
                        'test_function': "",
                        'is_correct': False,
                        'error_message': "No test function found in solution"
                    })
                    continue
                
                # Check code quality first to save time
                code_quality_passed, quality_message = check_code_quality(test_function)
                
                if not code_quality_passed:
                    logger.append(f"❌ Code quality check failed for attempt {attempt+1}: {quality_message}")
                    solutions.append({
                        'solution': full_solution,
                        'test_function': test_function,
                        'is_correct': False,
                        'error_message': f"Code quality check failed: {quality_message}"
                    })
                    continue
                
                # Generate test cases
                test_cases = generate_test_cases(correct_answer)
                
                # Run the test function on all test cases
                success, results, error_message = run_test_function(
                    test_function, 
                    test_cases, 
                    correct_answer,
                    timeout=config.timeout
                )
                
                solutions.append({
                    'solution': full_solution,
                    'test_function': test_function,
                    'test_results': results,
                    'is_correct': success,
                    'error_message': error_message if not success else None
                })
                
                # Update statistics if correct
                if success:
                    correct_count += 1
                
            except Exception as e:
                logger.append(f"❌ Error in attempt {str(attempt + 1)} for example {str(running_id)}:")
                logger.append(f"Exception type: {type(e).__name__}")
                logger.append(f"Exception message: {str(e)}")
                import traceback
                logger.append(f"Traceback:\n{traceback.format_exc()}")
                
                # In case of error, we should still try to save any partial solution
                error_message = f"Error occurred: {type(e).__name__} - {str(e)}"
                solutions.append({
                    'solution': full_solution if 'full_solution' in locals() else error_message,
                    'test_function': "",
                    'is_correct': False,
                    'error_message': str(e)
                })
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Correct test functions: {correct_count}/{config.best_of}")
        logger.append(f"└─ Success rate: {(correct_count/config.best_of)*100:.1f}%")
        
        # Add test function details
        for i, s in enumerate(solutions):
            logger.append(f"\n📝 Test Function {i+1}:")
            if s['error_message']:
                logger.append(f"❌ Error: {s['error_message']}")
            else:
                logger.append(f"✓ Correct: {'Yes' if s['is_correct'] else 'No'}")
                if 'test_results' in s:
                    logger.append(f"Test Results:")
                    for value, result in s['test_results'].items():
                        if value == correct_answer:
                            logger.append(f"  ✓ Correct answer ({value}): {result}")
                        else:
                            logger.append(f"  ✗ Incorrect answer ({value}): {result}")
            
            # Show a snippet of the test function
            if s['test_function']:
                test_lines = s['test_function'].split('\n')
                test_preview = '\n'.join(test_lines[:10])
                if len(test_lines) > 10:
                    test_preview += f"\n... ({len(test_lines) - 10} more lines)"
                logger.append(f"Test function snippet:\n{test_preview}")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create individual entries for each solution
        result_entries = []
        
        # Add individual solution entries
        for i, s in enumerate(solutions):
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'model_solution': s['solution'],
                'test_function': s['test_function'],
                'test_results': s.get('test_results', {}),
                'is_correct': s['is_correct'],
                'error_message': s['error_message'],
                'attempt_number': i + 1,
                'total_attempts': len(solutions)
            })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [s['is_correct'] for s in solutions],
            'success_rate': (correct_count/config.best_of)*100 if config.best_of > 0 else 0,
            'total_solutions': len(solutions),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(solutions) - correct_count,
            'all_solutions_correct': all(s['is_correct'] for s in solutions)
        })
        
        return result_entries
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'is_correct_list': [],
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'all_solutions_correct': None
        }]


async def main():
    """Main function for benchmarking mathematical problem solving with test function verification."""
    config = BenchmarkConfig.from_args('Benchmark model on creating test functions for mathematical problems')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Benchmark interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Benchmark failed with error: {e}")
        logger.print()
