import os
import asyncio
import logging
import re
import math
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, List, Any
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




def ensemble_run_test_function(test_code: str, result: float, timeout: int = 30) -> Tuple[bool, str]:
    """
    Test a result with the test function
    
    Args:
        test_code: The test function code
        result: The numeric result to test
        timeout: Maximum execution time in seconds
    
    Returns:
        - success: Whether the result passes the test
        - error_message: Error message if any
    """
    # Create a simple test case with just the result
    test_cases = [result]
    
    # Run the test function on the result
    success, results, test_error = run_test_function(
        test_code,
        test_cases,
        result,  # We're testing if the test function accepts the result
        timeout=timeout
    )
    
    # If the test function accepts the result, it's a valid solution
    return success, test_error if not success else ""
        

async def process_group(
    problem: str, 
    correct_answer: float, 
    group_id: int, 
    config: BenchmarkConfig, 
    logger: BenchmarkLogger
) -> List[float]:
    """
    Process a group of solutions with a single test function
    
    Returns:
        List of numerical values that passed the test function
    """
    main_model = get_model(config, role="main")
    programming_agent = ProgrammingAgent(main_model)
    testing_agent = TestingAgent(main_model)
    
    # Generate test function first
    logger.append(f"\n🧪 Generating test function for group {group_id}...")
    try:
        _, test_solution = await testing_agent.generate(
            problem, 
            correct_answer=str(correct_answer),
            return_prompt=True
        )
        
        # Extract the test function
        test_function = extract_test_function(test_solution)
        
        if not test_function:
            logger.append(f"❌ No test function found in solution for group {group_id}")
            return []
        
        # Check code quality for test function
        code_quality_passed, quality_message = check_code_quality(test_function)
        
        if not code_quality_passed:
            logger.append(f"❌ Test function quality check failed: {quality_message}")
            return []
        
        logger.append(f"✓ Test function generated successfully")
        
        # Generate solutions for this group
        numerical_results = []
        for i in range(config.solutions_per_group):
            logger.append(f"\n📝 Generating solution {i+1} for group {group_id}...")
            try:
                _, full_solution = await programming_agent.generate(problem, return_prompt=True)
                
                # Extract code from solution using the same method as programming_benchmark
                # First check if response section exists
                response_match = re.search(r'<response>(.*?)</response>', full_solution, re.DOTALL)
                if response_match:
                    response_content = response_match.group(1)
                    solution_code = extract_code_from_response(response_content)
                    if not solution_code:
                        # If no code in response section, try the whole solution
                        logger.append(f"No code found in response section, trying whole solution")
                        solution_code = extract_code_from_response(full_solution)
                else:
                    # If no response tags, extract from the whole solution
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
                else:
                    logger.append(f"❌ Solution {i+1} execution failed: {execution_error}")
                    continue
                
                # Test the result against the test function
                success, error_message = ensemble_run_test_function(
                    test_function, 
                    result, 
                    timeout=config.timeout
                )
                
                if success:
                    logger.append(f"✓ Solution {i+1} passed the test with result: {result}")
                    numerical_results.append(result)
                else:
                    logger.append(f"❌ Solution {i+1} failed the test: {error_message}")
                    logger.append(f"  (Solution produced result: {result}, correct answer: {correct_answer})")
            
            except Exception as e:
                logger.append(f"❌ Error generating solution {i+1}: {str(e)}")
        
        return numerical_results
        
    except Exception as e:
        logger.append(f"❌ Error in group {group_id}: {str(e)}")
        return []


async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with ensemble approach"""
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

        # Calculate number of groups
        num_groups = math.ceil(config.best_of / config.solutions_per_group)
        solutions_per_group = config.solutions_per_group
        
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Configuration:")
        logger.append(f"├─ Number of groups: {num_groups}")
        logger.append(f"└─ Solutions per group: {solutions_per_group}")
        
        # Process each group
        all_numerical_results = []
        group_results = []
        
        for group_id in range(num_groups):
            logger.append(f"\n\n🔍 Processing Group {group_id + 1}/{num_groups}")
            numerical_values = await process_group(
                example["problem"],
                correct_answer,
                group_id + 1,
                config,
                logger
            )
            
            all_numerical_results.extend(numerical_values)
            group_results.append(numerical_values)
            logger.append(f"Group {group_id + 1} results: {len(numerical_values)} valid numerical values")
            if numerical_values:
                logger.append(f"Group {group_id + 1} values: {numerical_values}")
        
        # Perform majority voting on all numerical values that passed their tests
        if not all_numerical_results:
            logger.append(f"\n❌ No valid numerical results found across all groups")
            is_correct = False
            final_answer = None
        else:
            # Perform majority voting directly on the numerical values
            answer_counts = Counter(all_numerical_results)
            final_answer, count = answer_counts.most_common(1)[0]
            
            # Check if the final answer is correct
            is_correct = abs(correct_answer - final_answer) <= config.tolerance
            
            logger.append(f"\n📊 Ensemble Results:")
            logger.append(f"├─ Total valid numerical results: {len(all_numerical_results)}")
            logger.append(f"├─ Answer distribution: {dict(answer_counts)}")
            logger.append(f"├─ Final answer: {final_answer}")
            logger.append(f"├─ Correct answer: {correct_answer}")
            logger.append(f"└─ Final answer correct: {'Yes' if is_correct else 'No'}")
            
            # Show results by group
            logger.append(f"\n📊 Group Results:")
            for i, group_vals in enumerate(group_results):
                if group_vals:
                    group_counter = Counter(group_vals)
                    group_most_common, group_count = group_counter.most_common(1)[0]
                    group_correct = abs(correct_answer - group_most_common) <= config.tolerance
                    logger.append(f"Group {i+1}: {len(group_vals)} values, most common: {group_most_common} (correct: {'Yes' if group_correct else 'No'})")
                else:
                    logger.append(f"Group {i+1}: No valid results")
        
        # No need to show detailed solution information since we're only working with numerical values
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add individual numerical result entries
        for i, result in enumerate(all_numerical_results):
            group_id = i // config.solutions_per_group + 1
            solution_id = i % config.solutions_per_group + 1
            
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'numerical_result': result,
                'is_correct': abs(result - correct_answer) <= config.tolerance,
                'group_id': group_id,
                'solution_id': solution_id
            })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [abs(result - correct_answer) <= config.tolerance for result in all_numerical_results],
            'is_most_common_correct': is_correct,
            'success_rate': (sum(1 for result in all_numerical_results if abs(result - correct_answer) <= config.tolerance) / len(all_numerical_results) * 100) if all_numerical_results else 0,
            'total_results': len(all_numerical_results),
            'correct_results': sum(1 for result in all_numerical_results if abs(result - correct_answer) <= config.tolerance),
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
            'total_results': 0,
            'correct_results': 0,
            'final_answer': None,
            'ensemble_correct': None
        }]


async def main():
    """Main function for ensemble benchmarking of mathematical problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems using ensemble approach')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    import argparse
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Benchmark interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Benchmark failed with error: {e}")
        logger.print()
