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
    Process a single example using a test-driven programmer approach:
    1. Generate both test suite and implementation in a single response
    2. Evaluate both components independently
    3. Check if they work together
    4. Generate multiple solutions if best_of > 1 and select the best one
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
        
        # Initialize agent
        test_driven_programmer_agent = TestDrivenProgrammerAgent(main_model)
        
        # Generate multiple test-driven programmer solutions
        solutions = []
        test_contents = []
        implementations = []
        implementation_results = []
        implementation_correctness = []
        test_correctness = []
        combined_correctness = []
        final_answers = []
        answer_sources = []
        
        # First phase: Generate multiple solutions
        for attempt in range(config.best_of):
            try:
                logger.append(f"Generating test-driven programmer solution {attempt+1}...")
                prompt, solution = await test_driven_programmer_agent.generate(example["problem"], return_prompt=True)
                
                # Extract test and implementation sections
                test_match = re.search(r'<test>(.*?)</test>', solution, re.DOTALL)
                implementation_match = re.search(r'<implementation>(.*?)</implementation>', solution, re.DOTALL)
                
                if not test_match or not implementation_match:
                    logger.append(f"❌ Missing test or implementation section in solution {attempt+1}")
                    solutions.append(solution)
                    test_contents.append("")
                    implementations.append("")
                    implementation_results.append(None)
                    implementation_correctness.append(False)
                    test_correctness.append(False)
                    combined_correctness.append(False)
                    final_answers.append(None)
                    answer_sources.append("invalid_solution")
                    continue
                
                test_content = test_match.group(1)
                implementation = implementation_match.group(1)
        
                # Evaluate implementation
                implementation_quality_passed, quality_message = check_code_quality(implementation)
                implementation_correct = False
                implementation_result = None
                
                if implementation_quality_passed:
                    logger.append(f"Implementation syntax check passed for attempt {attempt+1}")
                    execution_success, result, error_message = run_code_safely(implementation, timeout=config.timeout)
                    
                    if execution_success and result is not None:
                        implementation_result = result
                        implementation_correct = abs(correct_answer - result) <= config.tolerance
                        logger.append(f"Implementation result {attempt+1}: {result} (expected: {correct_answer})")
                        logger.append(f"Implementation correct: {'✓' if implementation_correct else '✗'}")
                    else:
                        logger.append(f"❌ Implementation execution failed for attempt {attempt+1}: {error_message}")
                else:
                    logger.append(f"❌ Implementation syntax check failed for attempt {attempt+1}: {quality_message}")
                
                # Evaluate test suite
                test_quality_passed, test_quality_message = check_code_quality(test_content)
                test_correct = False
        
                if test_quality_passed:
                    logger.append(f"Test syntax check passed for attempt {attempt+1}")
                    
                    # Test if the test_solution function works correctly
                    test_only_code = f"""
{test_content}

# Test if the test_solution function works with the correct answer
try:
    result = test_solution({correct_answer})
    print(f"Test function result with correct answer: {{result}}")
    success = result == True
except Exception as e:
    print(f"Error testing with correct answer: {{e}}")
    success = False
    
# Also try with an obviously wrong answer
try:
    wrong_result = test_solution({correct_answer + 100})
    print(f"Test function result with wrong answer: {{wrong_result}}")
    # A good test function should return False for a wrong answer
    success = success and (wrong_result == False)
except Exception as e:
    print(f"Error testing with wrong answer: {{e}}")
    success = False

print(f"Overall test function success: {{success}}")
"""
                    
                    test_syntax_success, test_output, test_syntax_error = run_code_safely(test_only_code, timeout=config.timeout)
                    
                    if test_syntax_success and "Overall test function success: True" in test_output:
                        logger.append(f"Test function works correctly with correct and incorrect answers for attempt {attempt+1}")
                        test_correct = True
                    else:
                        logger.append(f"❌ Test function validation failed for attempt {attempt+1}: {test_syntax_error}")
                        if test_output:
                            logger.append(f"Test output: {test_output}")
                else:
                    logger.append(f"❌ Test syntax check failed for attempt {attempt+1}: {test_quality_message}")
                
                # Evaluate combined solution
                combined_correct = False
        
                if implementation_quality_passed and test_quality_passed:
                    # Combine test and implementation
                    combined_code = f"""
{test_content}

# Implementation
{implementation}

# Get the result from the implementation
result = None
try:
    # Try to find the final result in the implementation
    # This is a bit hacky but should work for most cases
    exec_globals = {{'print': lambda x: None}}  # Suppress prints during execution
    exec(implementation, exec_globals)
    
    # Now use the test_solution function to verify the result
    if 'result' in exec_globals:
        result = exec_globals['result']
        test_result = test_solution(result)
        print(f"Implementation result: {{result}}")
        print(f"Test result: {{test_result}}")
        combined_correct = test_result
    else:
        # Try to find any numeric value that might be the result
        import re
        code_lines = implementation.strip().split('\\n')
        for line in reversed(code_lines):
            if 'print(' in line and not line.strip().startswith('#'):
                match = re.search('print\\\\s*\\\\(\\\\s*([0-9.+-]+)\\\\s*\\\\)', line)
                if match:
                    try:
                        result = float(match.group(1))
                        test_result = test_solution(result)
                        print(f"Found result in print statement: {{result}}")
                        print(f"Test result: {{test_result}}")
                        combined_correct = test_result
                        break
                    except:
                        pass
    
    # If we couldn't find a result, run the implementation directly
    if result is None:
        implementation_result, output = None, ""
        exec_locals = {{}}
        exec(implementation, globals(), exec_locals)
        # The implementation should print the result
        # We'll capture that in a separate run
        implementation_success, implementation_output, _ = run_code_safely(implementation, timeout=config.timeout)
        if implementation_success:
            # Try to extract a number from the output
            import re
            numbers = re.findall(r'[-+]?\\d*\\.\\d+|\\d+', implementation_output)
            if numbers:
                try:
                    result = float(numbers[-1])  # Take the last number as the result
                    test_result = test_solution(result)
                    print(f"Extracted result from output: {{result}}")
                    print(f"Test result: {{test_result}}")
                    combined_correct = test_result
                except:
                    pass
except Exception as e:
    print(f"Error in combined execution: {{e}}")
    combined_correct = False
"""
                    
                    combined_success, combined_output, combined_error = run_code_safely(combined_code, timeout=config.timeout)
                    
                    if combined_success:
                        logger.append(f"Combined solution execution successful for attempt {attempt+1}")
                        if "Test result: True" in combined_output:
                            logger.append(f"Implementation passes the test function for attempt {attempt+1}")
                            combined_correct = True
                        else:
                            logger.append(f"Implementation fails the test function for attempt {attempt+1}")
                            # If implementation is correct but tests fail, that's a test issue
                            if implementation_correct:
                                logger.append(f"Tests incorrectly fail on a correct implementation for attempt {attempt+1}")
                            # If implementation is wrong and tests fail, that could be good (tests catching errors)
                            elif not implementation_correct:
                                logger.append(f"Tests correctly identify an incorrect implementation for attempt {attempt+1}")
                                # This is actually good test behavior
                                test_correct = True
                    else:
                        logger.append(f"❌ Combined solution failed for attempt {attempt+1}: {combined_error}")
        
                # Determine final answer using fallback logic
                final_answer = None
                answer_source = None
                
                if implementation_correct and test_correct:
                    # Both implementation is correct and passes tests
                    final_answer = implementation_result
                    answer_source = "verified_implementation"
                    logger.append(f"✅ Using verified implementation answer for attempt {attempt+1}: {final_answer}")
                elif implementation_correct:
                    # Implementation is correct but tests fail or are incorrect
                    final_answer = implementation_result
                    answer_source = "implementation_fallback"
                    logger.append(f"⚠️ Fallback to correct implementation answer for attempt {attempt+1}: {final_answer}")
                elif implementation_result is not None:
                    # Implementation produces a result but it's incorrect
                    final_answer = implementation_result
                    answer_source = "implementation_fallback_incorrect"
                    logger.append(f"⚠️ Fallback to incorrect implementation answer for attempt {attempt+1}: {final_answer}")
                else:
                    # No usable answer
                    logger.append(f"❌ No usable answer found for attempt {attempt+1}")
                
                # Store all results
                solutions.append(solution)
                test_contents.append(test_content)
                implementations.append(implementation)
                implementation_results.append(implementation_result)
                implementation_correctness.append(implementation_correct)
                test_correctness.append(test_correct)
                combined_correctness.append(combined_correct)
                final_answers.append(final_answer)
                answer_sources.append(answer_source)
                
            except Exception as e:
                logger.append(f"❌ Error in test-driven programmer attempt {attempt+1}: {str(e)}")
                solutions.append(f"Error: {str(e)}")
                test_contents.append("")
                implementations.append("")
                implementation_results.append(None)
                implementation_correctness.append(False)
                test_correctness.append(False)
                combined_correctness.append(False)
                final_answers.append(None)
                answer_sources.append("error")
        
        # Select the best solution based on priority:
        # 1. Solutions with correct implementation and correct tests
        # 2. Solutions with correct implementation
        # 3. Solutions with correct tests
        # 4. Any solution with a final answer
        
        best_index = -1
        
        # Priority 1: Correct implementation and correct tests
        for i, (impl_correct, test_correct) in enumerate(zip(implementation_correctness, test_correctness)):
            if impl_correct and test_correct:
                best_index = i
                logger.append(f"Selected solution {i+1} with correct implementation and tests")
                break
        
        # Priority 2: Correct implementation
        if best_index == -1:
            for i, correct in enumerate(implementation_correctness):
                if correct:
                    best_index = i
                    logger.append(f"Selected solution {i+1} with correct implementation")
                    break
        
        # Priority 3: Correct tests
        if best_index == -1:
            for i, correct in enumerate(test_correctness):
                if correct:
                    best_index = i
                    logger.append(f"Selected solution {i+1} with correct tests")
                    break
        
        # Priority 4: Any solution with a final answer
        if best_index == -1:
            for i, answer in enumerate(final_answers):
                if answer is not None:
                    best_index = i
                    logger.append(f"Selected solution {i+1} with a final answer")
                    break
        
        # If no good solution found, use the first one
        if best_index == -1 and len(solutions) > 0:
            best_index = 0
            logger.append(f"No good solution found, defaulting to first solution")
        
        # Use the best solution
        if best_index != -1:
            solution = solutions[best_index]
            test_content = test_contents[best_index]
            implementation = implementations[best_index]
            implementation_result = implementation_results[best_index]
            implementation_correct = implementation_correctness[best_index]
            test_correct = test_correctness[best_index]
            combined_correct = combined_correctness[best_index]
            final_answer = final_answers[best_index]
            answer_source = answer_sources[best_index]
            
            logger.append(f"Using solution {best_index+1} as the final solution")
        else:
            logger.append(f"❌ No valid solutions found")
            logger.print()
            return [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'test_success': False,
                'implementation_success': False,
                'combined_success': False
            }]
        
        # Count correct solutions
        total_solutions = len(solutions)
        correct_solutions = sum(1 for ans in final_answers 
                               if ans is not None and abs(ans - correct_answer) <= config.tolerance)
        verified_correct_solutions = sum(1 for impl_corr, test_corr in 
                                        zip(implementation_correctness, test_correctness)
                                        if impl_corr and test_corr)
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        logger.append(f"\n📊 Statistics:")
        
        # Solution statistics
        logger.append(f"\n💻 Test-Driven Programmer Solutions:")
        for i, (impl_corr, test_corr, comb_corr, final_ans, source) in enumerate(zip(
            implementation_correctness, test_correctness, combined_correctness, final_answers, answer_sources)):
            
            impl_status = "✓" if impl_corr else "✗"
            test_status = "✓" if test_corr else "✗"
            comb_status = "✓" if comb_corr else "✗"
            best_marker = " [BEST]" if i == best_index else ""
            
            logger.append(f"├─ Solution {i+1}{best_marker}:")
            logger.append(f"│  ├─ Implementation: {impl_status} (Result: {implementation_results[i]})")
            logger.append(f"│  ├─ Test: {test_status}")
            logger.append(f"│  ├─ Combined: {comb_status}")
            logger.append(f"│  └─ Final: {final_ans} (source: {source})")
        
        # Overall statistics
        logger.append(f"\n📈 Overall Statistics:")
        logger.append(f"├─ Total solutions: {total_solutions}")
        logger.append(f"├─ Correct solutions: {correct_solutions} ({(correct_solutions/total_solutions)*100:.1f}%)")
        logger.append(f"├─ Verified correct solutions: {verified_correct_solutions} ({(verified_correct_solutions/total_solutions)*100:.1f}%)")
        
        # Best solution statistics
        logger.append(f"\n🏆 Best Solution (#{best_index+1}):")
        logger.append(f"├─ Implementation correct: {'✓' if implementation_correct else '✗'}")
        logger.append(f"├─ Test correct: {'✓' if test_correct else '✗'}")
        logger.append(f"├─ Combined works: {'✓' if combined_correct else '✗'}")
        if final_answer is not None:
            logger.append(f"└─ Final answer: {final_answer} (source: {answer_source})")
        else:
            logger.append(f"└─ Final answer: None")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add all test-driven programmer entries
        for i, (solution_i, test_i, impl_i, impl_res_i, impl_corr_i, 
                test_corr_i, comb_corr_i, final_ans_i, source_i) in enumerate(zip(
                    solutions, test_contents, implementations, implementation_results, 
                    implementation_correctness, test_correctness, combined_correctness, 
                    final_answers, answer_sources)):
            
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'role': 'test_driven_programmer',
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'model_solution': solution_i,
                'test_content': test_i,
                'implementation': impl_i,
                'implementation_result': impl_res_i,
                'implementation_correct': impl_corr_i,
                'test_correct': test_corr_i,
                'combined_correct': comb_corr_i,
                'final_answer': final_ans_i,
                'answer_source': source_i,
                'attempt_number': i + 1,
                'is_best_solution': i == best_index
            })
        
        # Calculate statistics exactly as in dual_proof_benchmark.py
        
        # Initial majority vote on ALL programming solutions (before test validation)
        initial_answer_counts = Counter([str(ans) for ans in final_answers if ans is not None])
        initial_majority = initial_answer_counts.most_common(1)
        initial_majority_answer = initial_majority[0][0] if initial_majority else None
        initial_majority_count = initial_majority[0][1] if initial_majority else 0
        initial_majority_percentage = (initial_majority_count / len([r for r in final_answers if r is not None])) * 100 if initial_majority else 0
        
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
        
        # For test-driven programmer, we don't have a separate test phase, so we use the same values for final
        final_majority_answer = initial_majority_answer
        final_majority_count = initial_majority_count
        final_majority_percentage = initial_majority_percentage
        final_majority_correct = initial_majority_correct
        
        # Create is_correct_list for compatibility with ProgressTracker
        is_correct_list = [
            ans is not None and abs(ans - correct_answer) <= config.tolerance 
            for ans in final_answers
        ]
        
        # Calculate success rates
        initial_success_rate = sum(is_correct_list) / len(is_correct_list) * 100 if is_correct_list else 0
        verified_correct_count = sum(1 for impl_corr, test_corr in 
                                    zip(implementation_correctness, test_correctness)
                                    if impl_corr and test_corr)
        verified_success_rate = verified_correct_count / len(solutions) * 100 if solutions else 0
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'test_success': test_correct,
            'implementation_success': implementation_correct,
            'combined_success': combined_correct,
            'final_answer': final_answer,
            'answer_source': answer_source,
            'final_answer_correct': final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance,
            'best_solution_index': best_index,
            
            # Solution statistics
            'all_implementation_correctness': implementation_correctness,
            'all_test_correctness': test_correctness,
            'all_combined_correctness': combined_correctness,
            'all_final_answers': final_answers,
            
            # Use exactly the same structure as in dual_proof_benchmark.py
            'example_processed_successfully': True,
            
            # Programming solutions statistics
            'programming_solutions_count': len(solutions),
            'programming_correctness': is_correct_list,
            'programming_results': final_answers,
            'initial_success_rate': initial_success_rate,
            
            # Initial majority vote (before test validation)
            'initial_majority_answer': initial_majority_answer,
            'initial_majority_count': initial_majority_count,
            'initial_majority_percentage': initial_majority_percentage,
            'initial_majority_correct': initial_majority_correct,
            
            # Test statistics
            'test_passed': test_correctness,  # Using test correctness as a proxy for test passing
            'test_success_rate': sum(test_correctness) / len(test_correctness) * 100 if test_correctness else 0,
            
            # Verified statistics (solutions that are correct AND pass their tests)
            'verified_correct': [c and t for c, t in zip(is_correct_list, test_correctness)],
            'verified_success_rate': verified_success_rate,
            'verified_results': [ans for ans, test_corr in zip(final_answers, test_correctness) if test_corr and ans is not None],
            
            # Final majority vote (after test validation)
            'final_majority_answer': final_majority_answer,
            'final_majority_count': final_majority_count,
            'final_majority_percentage': final_majority_percentage,
            'final_majority_correct': final_majority_correct,
            
            # Initial correctness statistics (before test validation)
            'initial_correctness': is_correct_list,
            'initial_majority_correct': initial_majority_correct,
            'initial_success_rate': initial_success_rate,
            
            # Test validation statistics
            'test_passed': test_correctness,
            'test_success_rate': sum(test_correctness) / len(test_correctness) * 100 if test_correctness else 0,
            
            # Final correctness statistics (after test validation)
            'final_correctness': [c and t for c, t in zip(is_correct_list, test_correctness)],
            'final_majority_correct': final_majority_correct,
            'verified_success_rate': verified_success_rate,
            
            # Compatibility fields for ProgressTracker statistics
            'is_correct_list': is_correct_list,
            'is_correct': final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance,
            'is_most_common_correct': initial_majority_correct,
            
            'total_solutions': total_solutions,
            'correct_solutions': correct_solutions,
            'incorrect_solutions': total_solutions - correct_solutions,
            'verified_correct_solutions': verified_correct_count,
            'verified_incorrect_solutions': total_solutions - verified_correct_count
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
            'test_success': False,
            'implementation_success': False,
            'combined_success': False
        }]


async def main():
    """Main function for benchmarking with the Test-Driven Programmer approach."""
    config = BenchmarkConfig.from_args('Benchmark Test-Driven Programmer approach')
    
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
