import os
import asyncio
import logging
import re
import sys
from typing import Optional, Dict, List, Tuple, Any
from collections import Counter
from contextlib import contextmanager
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
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

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """
    Process a single example using a programmer-test approach with a shared test:
    1. Programmer implements multiple solutions
    2. A SINGLE test function is generated and used for all solutions
    3. Majority voting on all solutions and on solutions that pass the shared test
    """
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

        # Get models
        main_model = get_model(config, role="main")
        
        # Initialize agents
        programming_agent = ProgrammingAgent(main_model)
        testing_agent = TestingAgent(main_model)
        
        # Generate multiple programming solutions
        programming_solutions = []
        programming_codes = []
        programming_results = []
        programming_correctness = []
        
        # First phase: Generate programming solutions and check their correctness
        for attempt in range(config.best_of):
            try:
                logger.append(f"Generating programming solution {attempt+1}...")
                prompt, current_solution = await programming_agent.generate(example["problem"], return_prompt=True)
                
                # Extract code from solution
                response_match = re.search(r'<response>(.*?)</response>', current_solution, re.DOTALL)
                if response_match:
                    response_content = response_match.group(1)
                    code = extract_code_from_response(response_content)
                    if not code:
                        # If no code in response section, try the whole solution
                        logger.append(f"No code found in response section, trying whole solution")
                        code = extract_code_from_response(current_solution)
                else:
                    # If no response tags, extract from the whole solution
                    code = extract_code_from_response(current_solution)
                
                if not code:
                    logger.append(f"❌ No code found in programming solution {attempt+1}")
                    programming_solutions.append(current_solution)
                    programming_codes.append("")
                    programming_results.append(None)
                    programming_correctness.append(False)
                    continue
                
                # Check code quality
                code_quality_passed, quality_message = check_code_quality(code)
                
                if not code_quality_passed:
                    logger.append(f"❌ Code quality check failed for attempt {attempt+1}: {quality_message}")
                    programming_solutions.append(current_solution)
                    programming_codes.append(code)
                    programming_results.append(None)
                    programming_correctness.append(False)
                    continue
                
                # Run code safely
                execution_success, result, error_message = run_code_safely(code, timeout=config.timeout)
                
                if not execution_success:
                    logger.append(f"❌ Code execution failed for attempt {attempt+1}: {error_message}")
                    programming_solutions.append(current_solution)
                    programming_codes.append(code)
                    programming_results.append(None)
                    programming_correctness.append(False)
                    continue
                
                # Compare with correct answer
                is_correct = False
                try:
                    if isinstance(correct_answer, (int, float)) and isinstance(result, (int, float)):
                        # Use tolerance for numeric comparison
                        is_correct = abs(correct_answer - result) <= config.tolerance
                    else:
                        # Try string comparison as fallback
                        is_correct = str(correct_answer).strip() == str(result).strip()
                except Exception as e:
                    logger.append(f"Error comparing answers: {str(e)}")
                
                programming_solutions.append(current_solution)
                programming_codes.append(code)
                programming_results.append(result)
                programming_correctness.append(is_correct)
                
            except Exception as e:
                logger.append(f"❌ Error in programming attempt {attempt+1}: {str(e)}")
                programming_solutions.append(f"Error: {str(e)}")
                programming_codes.append("")
                programming_results.append(None)
                programming_correctness.append(False)
        
        # Initial majority vote on ALL programming solutions (before test validation)
        initial_answer_counts = Counter([str(ans) for ans in programming_results if ans is not None])
        initial_majority = initial_answer_counts.most_common(1)
        initial_majority_answer = initial_majority[0][0] if initial_majority else None
        initial_majority_count = initial_majority[0][1] if initial_majority else 0
        initial_majority_percentage = (initial_majority_count / len([r for r in programming_results if r is not None])) * 100 if initial_majority else 0
        
        # Check if the initial majority answer is correct
        initial_majority_correct = False
        if initial_majority_answer:
            try:
                if isinstance(correct_answer, (int, float)):
                    initial_majority_correct = abs(float(initial_majority_answer) - correct_answer) <= config.tolerance
                else:
                    initial_majority_correct = str(initial_majority_answer).strip() == str(correct_answer).strip()
            except:
                pass
        
        logger.append(f"Initial majority answer: {initial_majority_answer} ({initial_majority_count} votes, {initial_majority_percentage:.1f}% of valid results)")
        logger.append(f"Initial majority answer correct: {'✓' if initial_majority_correct else '✗'}")
        
        # Second phase: Generate FOUR test functions to validate all solutions
        # We'll use the problem and the expected answer to create generic tests
        test_functions = []
        test_successes = []
        
        for test_attempt in range(4):  # Generate 4 different test functions
            try:
                logger.append(f"Generating test function {test_attempt+1} of 4...")
                # Create a prompt that includes the problem and the expected answer
                test_prompt = f"Problem:\n{example['problem']}\n\nExpected Answer: {correct_answer}"
                
                _, test_solution = await testing_agent.generate(test_prompt, return_prompt=True)
                
                # Extract the test function
                test_function = extract_code_from_response(test_solution)
                
                if not test_function:
                    logger.append(f"❌ No test function found in test attempt {test_attempt+1}")
                    test_functions.append("")
                    test_successes.append(False)
                    continue
                
                # Check test function quality
                code_quality_passed, quality_message = check_code_quality(test_function)
                
                if not code_quality_passed:
                    logger.append(f"❌ Test function {test_attempt+1} quality check failed: {quality_message}")
                    test_functions.append(test_function)
                    test_successes.append(False)
                    continue
                
                # Generate test cases
                try:
                    test_cases = generate_test_cases(correct_answer, num_cases=50)
                except Exception as e:
                    logger.append(f"❌ Error generating test cases for test {test_attempt+1}: {str(e)}")
                    # Simple fallback if generate_test_cases fails
                    test_cases = [correct_answer, correct_answer + 1, correct_answer - 1, 
                                 0, 1, -1, 10, -10, 100, -100]
                
                # Run the test function on a sample test case to validate it
                try:
                    with time_limit(config.timeout + 30):
                        success, results, error_message = run_test_function(
                            test_function, 
                            [correct_answer],  # Just test with the correct answer
                            correct_answer,
                            timeout=config.timeout
                        )
                        test_successes.append(success)
                except TimeoutException:
                    logger.append(f"❌ Global timeout exceeded when validating test function {test_attempt+1}")
                    test_successes.append(False)
                    error_message = "Global timeout exceeded when validating test function"
                
                test_functions.append(test_function)
                
                if not test_successes[-1]:
                    logger.append(f"❌ Test function {test_attempt+1} validation failed: {error_message}")
                else:
                    logger.append(f"✓ Test function {test_attempt+1} validation successful")
                    
            except Exception as e:
                logger.append(f"❌ Error generating test function {test_attempt+1}: {str(e)}")
                test_functions.append("")
                test_successes.append(False)
        
        # Apply the four test functions to all programming solutions
        # A solution passes if it passes at least one of the four tests
        test_results = []  # List of lists, each inner list contains results for one solution against all tests
        test_passed = []   # List of booleans, True if solution passed at least one test
        
        # Check if we have any valid test functions
        valid_test_functions = [(func, idx) for idx, (func, success) in enumerate(zip(test_functions, test_successes)) if func and success]
        
        if valid_test_functions:
            for i, (code, result) in enumerate(zip(programming_codes, programming_results)):
                if not code or result is None:
                    test_results.append([{} for _ in range(4)])
                    test_passed.append(False)
                    continue
                
                solution_test_results = []
                solution_test_passed = False
                
                for test_idx, (test_function, test_num) in enumerate(valid_test_functions):
                    try:
                        logger.append(f"Testing solution {i+1} with test function {test_num+1}...")
                        
                        # Generate test cases
                        try:
                            test_cases = generate_test_cases(correct_answer, num_cases=50)
                        except Exception as e:
                            logger.append(f"❌ Error generating test cases: {str(e)}")
                            # Simple fallback if generate_test_cases fails
                            test_cases = [correct_answer, correct_answer + 1, correct_answer - 1, 
                                         0, 1, -1, 10, -10, 100, -100]
                        
                        # Run the test function on all test cases for this solution
                        try:
                            with time_limit(config.timeout + 30):
                                success, results, error_message = run_test_function(
                                    test_function, 
                                    test_cases, 
                                    correct_answer,
                                    timeout=config.timeout,
                                    solution_code=code  # Pass the solution code to be tested
                                )
                        except TimeoutException:
                            logger.append(f"❌ Global timeout exceeded when testing solution {i+1} with test {test_num+1}")
                            success = False
                            results = {}
                            error_message = "Global timeout exceeded when testing solution"
                        
                        solution_test_results.append(results)
                        
                        if success:
                            logger.append(f"✓ Solution {i+1} passed test function {test_num+1}")
                            solution_test_passed = True  # Mark as passed if at least one test passes
                        else:
                            logger.append(f"✗ Solution {i+1} failed test function {test_num+1}: {error_message}")
                        
                    except Exception as e:
                        logger.append(f"❌ Error testing solution {i+1} with test {test_num+1}: {str(e)}")
                        solution_test_results.append({})
                
                # Fill in any missing results if we have fewer than 4 valid test functions
                while len(solution_test_results) < 4:
                    solution_test_results.append({})
                
                test_results.append(solution_test_results)
                test_passed.append(solution_test_passed)
                
                if solution_test_passed:
                    logger.append(f"✓ Solution {i+1} passed at least one test")
                else:
                    logger.append(f"✗ Solution {i+1} failed all tests")
                
        else:
            # If we couldn't generate any valid test functions, mark all solutions as not tested
            test_results = [[{} for _ in range(4)] for _ in programming_solutions]
            test_passed = [False for _ in programming_solutions]
            logger.append("❌ No valid test functions available, skipping test validation")
        
        # Final majority vote only on solutions that passed the shared test (regardless of correctness)
        verified_results = [
            result for result, passed in zip(programming_results, test_passed)
            if passed and result is not None
        ]
        
        final_answer_counts = Counter([str(ans) for ans in verified_results])
        final_majority = final_answer_counts.most_common(1)
        final_majority_answer = final_majority[0][0] if final_majority else None
        final_majority_count = final_majority[0][1] if final_majority else 0
        final_majority_percentage = (final_majority_count / len(verified_results)) * 100 if verified_results else 0
        
        # Check if the final majority answer is correct
        final_majority_correct = False
        if final_majority_answer:
            try:
                if isinstance(correct_answer, (int, float)):
                    final_majority_correct = abs(float(final_majority_answer) - correct_answer) <= config.tolerance
                else:
                    final_majority_correct = str(final_majority_answer).strip() == str(correct_answer).strip()
            except:
                pass
        
        # If no verified results, fall back to initial majority
        if not verified_results:
            final_majority_answer = initial_majority_answer
            final_majority_count = initial_majority_count
            final_majority_percentage = initial_majority_percentage
            final_majority_correct = initial_majority_correct
        
        # Calculate statistics
        initial_success_rate = sum(programming_correctness) / len(programming_correctness) * 100 if programming_correctness else 0
        test_success_rate = sum(test_passed) / len(test_passed) * 100 if test_passed else 0
        
        # Count solutions that are both correct and pass the shared test
        verified_correct_count = sum(
            1 for is_correct, passed in zip(programming_correctness, test_passed)
            if is_correct and passed
        )
        verified_success_rate = verified_correct_count / len(programming_correctness) * 100 if programming_correctness else 0
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Test functions information
        logger.append(f"\n🧪 Shared Test Functions:")
        valid_test_count = sum(test_successes)
        logger.append(f"├─ Valid test functions: {valid_test_count}/4")
        
        for i, (test_func, is_valid) in enumerate(zip(test_functions, test_successes)):
            if test_func:
                test_lines = test_func.split('\n')
                test_preview = '\n'.join(test_lines[:3])  # Show fewer lines to keep output compact
                if len(test_lines) > 3:
                    test_preview += f"\n... ({len(test_lines) - 3} more lines)"
                logger.append(f"├─ Test {i+1}: {'✓' if is_valid else '✗'}")
                logger.append(f"│  └─ Preview: {test_preview}")
            else:
                logger.append(f"├─ Test {i+1}: ❌ No test function generated")
        
        # Programming solutions statistics
        logger.append(f"\n💻 Programming Solutions:")
        for i, (is_correct, result, passed_test) in enumerate(zip(programming_correctness, programming_results, test_passed)):
            status = "✓" if is_correct else "✗"
            test_status = "✓" if passed_test else "✗"
            logger.append(f"├─ Solution {i+1}: {status} (Answer: {result}) | Test: {test_status}")
        
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Initial success rate: {initial_success_rate:.1f}%")
        logger.append(f"├─ Test success rate: {test_success_rate:.1f}%")
        logger.append(f"├─ Verified success rate: {verified_success_rate:.1f}%")
        logger.append(f"├─ Initial majority vote:")
        logger.append(f"│  ├─ Answer: {initial_majority_answer} ({initial_majority_count} votes, {initial_majority_percentage:.1f}%)")
        logger.append(f"│  └─ Correct: {'✓' if initial_majority_correct else '✗'}")
        logger.append(f"└─ Final majority vote (verified):")
        logger.append(f"   ├─ Answer: {final_majority_answer} ({final_majority_count} votes, {final_majority_percentage:.1f}%)")
        logger.append(f"   └─ Correct: {'✓' if final_majority_correct else '✗'}")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add test entries (four test functions)
        for i, (test_function, is_valid) in enumerate(zip(test_functions, test_successes)):
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'role': 'tester',
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'test_function': test_function,
                'is_valid': is_valid,
                'test_number': i + 1
            })
        
        # Add programming entries
        for i, (prog_solution, prog_code, prog_result, prog_correct, test_result, test_pass) in enumerate(zip(
            programming_solutions, programming_codes, programming_results, programming_correctness,
            test_results, test_passed
        )):
            # Add programming entry
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'role': 'programmer',
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'model_solution': prog_solution,
                'model_code': prog_code,
                'model_result': prog_result,
                'is_correct': prog_correct,
                'test_passed': test_pass,  # True if passed at least one test
                'verified_correct': prog_correct and test_pass,
                'attempt_number': i + 1
            })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            
            # Test functions statistics
            'test_functions': test_functions,
            'test_functions_valid': test_successes,
            'valid_test_count': sum(test_successes),
            
            # Programming solutions statistics
            'programming_solutions_count': len(programming_solutions),
            'programming_correctness': programming_correctness,
            'programming_results': programming_results,
            'initial_success_rate': initial_success_rate,
            
            # Initial majority vote (before test validation)
            'initial_majority_answer': initial_majority_answer,
            'initial_majority_count': initial_majority_count,
            'initial_majority_percentage': initial_majority_percentage,
            'initial_majority_correct': initial_majority_correct,
            
            # Test statistics
            'test_passed': test_passed,
            'test_success_rate': test_success_rate,
            
            # Verified statistics (solutions that are correct AND pass their tests)
            'verified_correct': [c and t for c, t in zip(programming_correctness, test_passed)],
            'verified_success_rate': verified_success_rate,
            'verified_results': verified_results,
            
            # Final majority vote (after test validation)
            'final_majority_answer': final_majority_answer,
            'final_majority_count': final_majority_count,
            'final_majority_percentage': final_majority_percentage,
            'final_majority_correct': final_majority_correct,
            
            # Compatibility fields for ProgressTracker statistics - using verified results
            'is_correct_list': [c and t for c, t in zip(programming_correctness, test_passed)],  # Verified correctness (after test validation)
            'is_most_common_correct': final_majority_correct,  # Final majority correctness
            
            # Additional fields for final validation statistics
            'verified_correct_list': [c and t for c, t in zip(programming_correctness, test_passed)],  # After test validation
            'verified_most_common_correct': final_majority_correct,  # Final majority correctness
            
            'total_solutions': len(programming_solutions),
            'correct_solutions': verified_correct_count,  # Verified correct solutions
            'incorrect_solutions': len(programming_solutions) - verified_correct_count,  # Verified incorrect solutions
            'verified_correct_solutions': verified_correct_count,  # Final correct solutions (after test validation)
            'verified_incorrect_solutions': len(programming_solutions) - verified_correct_count  # Final incorrect solutions
        })
        
        return result_entries
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        import traceback
        logger.append(traceback.format_exc())
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'initial_success_rate': 0,
            'test_success_rate': 0,
            'verified_success_rate': 0,
            'initial_majority_answer': None,
            'final_majority_answer': None
        }]


async def main():
    """Main function for benchmarking with the Programmer-Shared-Test approach."""
    config = BenchmarkConfig.from_args('Benchmark Programmer-Test approach with a single shared test function')
    
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
