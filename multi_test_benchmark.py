import os
import asyncio
import logging
import re
import argparse
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, List, Any, Set
from collections import Counter
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.model_utils import *
from utils.solution_utils import *
from utils.agents import *
from utils.logger import BenchmarkLogger
# Import functions from test_benchmark.py
from test_benchmark import extract_test_function

# Import functions from programming_benchmark.py and solution_utils
from utils.solution_utils import extract_code_from_response, run_code_safely, check_code_quality, run_test_function


def calculate_answer_majority(answers, tolerance=1e-2):
    """
    Calculate the most common answer by counting how many answers are within tolerance
    of each unique answer.
    
    Args:
        answers: List of numeric answers
        tolerance: Numeric tolerance for grouping similar answers
        
    Returns:
        Tuple of (majority_answer, count_dict) where count_dict maps each answer to its count
    """
    if not answers:
        return None, {}
    
    # Count how many answers are within tolerance of each answer
    count_dict = {}
    for i, val in enumerate(answers):
        # Initialize count for this answer
        if val not in count_dict:
            count_dict[val] = 0
        
        # Count all answers within tolerance of this one
        for other_val in answers:
            if abs(val - other_val) <= tolerance:
                count_dict[val] += 1
    
    # Find the answer with the highest count
    if count_dict:
        majority_answer = max(count_dict.items(), key=lambda x: x[1])[0]
        return majority_answer, count_dict
    else:
        return None, {}


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


def test_result_with_function(test_code: str, result: float, timeout: int = 30) -> Tuple[bool, str]:
    """
    Test a result with a test function
    
    Args:
        test_code: The test function code
        result: The numeric result to test
        timeout: Maximum execution time in seconds
    
    Returns:
        - success: Whether the result passes the test
        - error_message: Error message if any
    """
    # Create test cases including the result and some incorrect values
    test_cases = [result]  # The result we want to test
    
    # Generate values that are significantly different from the result
    # to ensure the test function can discriminate between correct and incorrect answers
    multipliers = [0.5, 2.0, -1.0, 10.0, 0.1, 5.0]
    offsets = [0.1, 1.0, -0.1, -1.0, 100.0]
    
    for i, multiplier in enumerate(multipliers):
        # Ensure the test value is different enough from the result
        test_value = result * multiplier
        
        # For values close to zero, use offsets instead of multipliers
        if abs(test_value - result) <= 1e-6:
            test_value = result + offsets[i % len(offsets)]
            
        # Make sure we don't accidentally generate the same value
        if abs(test_value - result) > 1e-6:
            test_cases.append(test_value)
    
    # Run the test function on all test cases
    success, results, test_error = run_test_function(
        test_code,
        test_cases,
        result,  # We're testing if the test function accepts the result
        timeout=timeout
    )
    
    # If the test function accepts the result and rejects ALL incorrect answers, it's valid
    return success, test_error if not success else ""


async def generate_test_functions(
    problem: str,
    correct_answer: float,
    num_tests: int,
    config: BenchmarkConfig,
    logger: BenchmarkLogger
) -> List[str]:
    """
    Generate multiple test functions for a problem
    
    Returns:
        List of valid test function code strings
    """
    main_model = get_model(config, role="main")
    testing_agent = TestingAgent(main_model)
    
    valid_test_functions = []
    
    for i in range(num_tests):
        logger.append(f"\n🧪 Generating test function {i+1}/{num_tests}...")
        try:
            _, test_solution = await testing_agent.generate(
                problem, 
                return_prompt=True
            )
            
            # Extract the test function from the solution
            test_function = extract_test_function(test_solution)
            
            if not test_function:
                logger.append(f"❌ No test function found in solution {i+1}")
                continue
            
            # Check code quality for test function
            code_quality_passed, quality_message = check_code_quality(test_function)
            
            if not code_quality_passed:
                logger.append(f"❌ Test function {i+1} quality check failed: {quality_message}")
                continue
            
            logger.append(f"✓ Test function {i+1} generated successfully")
            valid_test_functions.append(test_function)
            
        except Exception as e:
            logger.append(f"❌ Error generating test function {i+1}: {str(e)}")
    
    return valid_test_functions


async def generate_solutions(
    problem: str,
    num_solutions: int,
    config: BenchmarkConfig,
    logger: BenchmarkLogger
) -> List[Dict]:
    """
    Generate multiple solutions for a problem
    
    Returns:
        List of solution dictionaries with code and result
    """
    main_model = get_model(config, role="main")
    programming_agent = ProgrammingAgent(main_model)
    
    solutions = []
    
    for i in range(num_solutions):
        logger.append(f"\n📝 Generating solution {i+1}/{num_solutions}...")
        try:
            _, full_solution = await programming_agent.generate(problem, return_prompt=True)
            
            # Extract code from solution
            response_match = re.search(r'<response>(.*?)</response>', full_solution, re.DOTALL)
            if response_match:
                response_content = response_match.group(1)
                solution_code = extract_code_from_response(response_content)
                if not solution_code:
                    logger.append(f"No code found in response section, trying whole solution")
                    solution_code = extract_code_from_response(full_solution)
            else:
                solution_code = extract_code_from_response(full_solution)
            
            logger.append(f"Extracted code length: {len(solution_code)} characters")
            if not solution_code:
                logger.append(f"❌ No code found in solution {i+1}")
                continue
            
            # Check code quality
            code_quality_passed, quality_message = check_code_quality(solution_code)
            
            if not code_quality_passed:
                logger.append(f"❌ Solution {i+1} quality check failed: {quality_message}")
                continue
            
            # Run the solution code to get a result
            execution_success, result, execution_error = run_code_safely(
                solution_code, 
                timeout=config.timeout
            )
            
            if execution_success:
                logger.append(f"✓ Solution {i+1} execution successful, result: {result}")
                solutions.append({
                    'solution_id': i+1,
                    'code': solution_code,
                    'result': result,
                    'full_solution': full_solution
                })
            else:
                logger.append(f"❌ Solution {i+1} execution failed: {execution_error}")
            
        except Exception as e:
            logger.append(f"❌ Error generating solution {i+1}: {str(e)}")
    
    return solutions


async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with multiple test functions approach"""
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
        
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Configuration:")
        logger.append(f"├─ Number of test functions: {config.best_of}")
        logger.append(f"└─ Number of solutions: {config.solutions_per_group}")
        
        # Step 1: Generate multiple test functions
        test_functions = await generate_test_functions(
            example["problem"],
            correct_answer,
            config.best_of,
            config,
            logger
        )
        
        if not test_functions:
            logger.append(f"\n❌ No valid test functions generated")
            logger.print()
            return [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'is_correct_list': [],
                'is_most_common_correct': None,
                'success_rate': 0,
                'total_results': 0,
                'correct_results': 0,
                'final_answer': None,
                'ensemble_correct': None
            }]
        
        logger.append(f"\n✓ Generated {len(test_functions)} valid test functions")
        
        # Step 2: Generate multiple solutions
        solutions = await generate_solutions(
            example["problem"],
            config.solutions_per_group,
            config,
            logger
        )
        
        if not solutions:
            logger.append(f"\n❌ No valid solutions generated")
            logger.print()
            return [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'is_correct_list': [],
                'is_most_common_correct': None,
                'success_rate': 0,
                'total_results': 0,
                'correct_results': 0,
                'final_answer': None,
                'ensemble_correct': None
            }]
        
        logger.append(f"\n✓ Generated {len(solutions)} valid solutions")
        
        # Step 3: Test each solution against each test function
        logger.append(f"\n🔍 Testing solutions against test functions...")
        
        # Track which solutions pass at least one test
        valid_solutions = []
        solution_test_results = {}
        
        for solution in solutions:
            solution_id = solution['solution_id']
            result = solution['result']
            passed_tests = []
            
            for i, test_function in enumerate(test_functions):
                test_id = i + 1
                success, error_message = test_result_with_function(
                    test_function,
                    result,
                    timeout=config.timeout
                )
                
                if success:
                    passed_tests.append(test_id)
                    logger.append(f"✓ Solution {solution_id} passed test {test_id}")
                else:
                    logger.append(f"❌ Solution {solution_id} failed test {test_id}: {error_message}")
            
            # If the solution passed at least one test, consider it valid
            if passed_tests:
                valid_solutions.append(solution)
                solution_test_results[solution_id] = passed_tests
                logger.append(f"✓ Solution {solution_id} passed {len(passed_tests)}/{len(test_functions)} tests")
            else:
                logger.append(f"❌ Solution {solution_id} failed all tests")
        
        # Step 4: Perform majority voting on valid solutions
        if not valid_solutions:
            logger.append(f"\n❌ No solutions passed any tests")
            is_correct = False
            final_answer = None
        else:
            # Extract results from valid solutions
            valid_results = [s['result'] for s in valid_solutions]
            
            # Use tolerance-based grouping for majority voting
            final_answer, answer_counts = calculate_answer_majority(valid_results, tolerance=1e-2)
            
            # Check if the final answer is correct
            is_correct = abs(correct_answer - final_answer) <= config.tolerance
            
            # Format the answer counts for display
            formatted_counts = {f"{k:.6f}": v for k, v in answer_counts.items()}
            
            logger.append(f"\n📊 Multi-Test Results:")
            logger.append(f"├─ Total valid solutions: {len(valid_solutions)}/{len(solutions)}")
            logger.append(f"├─ Answer distribution (with tolerance 1e-2): {formatted_counts}")
            logger.append(f"├─ Final answer: {final_answer}")
            logger.append(f"├─ Correct answer: {correct_answer}")
            logger.append(f"└─ Final answer correct: {'Yes' if is_correct else 'No'}")
            
            # Show which tests each solution passed
            logger.append(f"\n📊 Solution Test Results:")
            for solution_id, passed_tests in solution_test_results.items():
                solution_result = next(s['result'] for s in solutions if s['solution_id'] == solution_id)
                is_solution_correct = abs(solution_result - correct_answer) <= config.tolerance
                logger.append(f"Solution {solution_id} (result: {solution_result}, correct: {'Yes' if is_solution_correct else 'No'}):")
                logger.append(f"└─ Passed tests: {passed_tests}")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add individual solution entries
        for solution in solutions:
            solution_id = solution['solution_id']
            result = solution['result']
            passed_tests = solution_test_results.get(solution_id, [])
            
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'solution_id': solution_id,
                'numerical_result': result,
                'is_correct': abs(result - correct_answer) <= config.tolerance,
                'passed_any_test': len(passed_tests) > 0,
                'passed_tests': passed_tests,
                'tests_passed_count': len(passed_tests)
            })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [abs(s['result'] - correct_answer) <= config.tolerance for s in valid_solutions],
            'is_most_common_correct': is_correct,
            'success_rate': (sum(1 for s in valid_solutions if abs(s['result'] - correct_answer) <= config.tolerance) / len(valid_solutions) * 100) if valid_solutions else 0,
            'total_solutions': len(solutions),
            'valid_solutions': len(valid_solutions),
            'correct_solutions': sum(1 for s in valid_solutions if abs(s['result'] - correct_answer) <= config.tolerance),
            'final_answer': final_answer,
            'ensemble_correct': is_correct
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
            'is_most_common_correct': None,
            'success_rate': 0,
            'total_solutions': 0,
            'valid_solutions': 0,
            'correct_solutions': 0,
            'final_answer': None,
            'ensemble_correct': None
        }]


async def main():
    """Main function for multi-test benchmarking of mathematical problem solving."""
    # Get the base config first
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems using multiple test functions')
    
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
