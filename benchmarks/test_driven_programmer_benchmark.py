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
        
        # Generate test-driven programmer solution
        logger.append(f"Generating test-driven programmer solution...")
        prompt, solution = await test_driven_programmer_agent.generate(example["problem"], return_prompt=True)
        
        # Extract test and implementation sections
        test_match = re.search(r'<test>(.*?)</test>', solution, re.DOTALL)
        implementation_match = re.search(r'<implementation>(.*?)</implementation>', solution, re.DOTALL)
        
        if not test_match or not implementation_match:
            logger.append(f"❌ Missing test or implementation section in solution")
            logger.print()
            return [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'test_success': False,
                'implementation_success': False,
                'combined_success': False
            }]
        
        test_content = test_match.group(1)
        implementation = implementation_match.group(1)
        
        # Evaluate implementation
        implementation_quality_passed, quality_message = check_code_quality(implementation)
        implementation_correct = False
        implementation_result = None
        
        if implementation_quality_passed:
            logger.append(f"Implementation syntax check passed")
            execution_success, result, error_message = run_code_safely(implementation, timeout=config.timeout)
            
            if execution_success and result is not None:
                implementation_result = result
                implementation_correct = abs(correct_answer - result) <= config.tolerance
                logger.append(f"Implementation result: {result} (expected: {correct_answer})")
                logger.append(f"Implementation correct: {'✓' if implementation_correct else '✗'}")
            else:
                logger.append(f"❌ Implementation execution failed: {error_message}")
        else:
            logger.append(f"❌ Implementation syntax check failed: {quality_message}")
        
        # Evaluate test suite
        test_quality_passed, test_quality_message = check_code_quality(test_content)
        test_correct = False
        
        if test_quality_passed:
            logger.append(f"Test syntax check passed")
            
            # Create a dummy implementation that returns the correct answer
            test_only_code = f"""
{test_content}

# Dummy implementation that returns the correct answer
def solution():
    return {correct_answer}

# Run tests if this file is executed directly
if __name__ == '__main__':
    import unittest
    unittest.main()
"""
            
            test_syntax_success, _, test_syntax_error = run_code_safely(test_only_code, timeout=config.timeout)
            
            if test_syntax_success:
                logger.append(f"Test suite runs successfully with a dummy implementation")
                test_correct = True
            else:
                logger.append(f"❌ Test suite has runtime errors: {test_syntax_error}")
        else:
            logger.append(f"❌ Test syntax check failed: {test_quality_message}")
        
        # Evaluate combined solution
        combined_correct = False
        
        if implementation_quality_passed and test_quality_passed:
            # Combine test and implementation
            combined_code = f"""
{test_content}

# Implementation
{implementation}

# Run tests if this file is executed directly
if __name__ == '__main__':
    import unittest
    unittest.main()
"""
            
            combined_success, _, combined_error = run_code_safely(combined_code, timeout=config.timeout)
            
            if combined_success:
                logger.append(f"Combined solution runs successfully")
                combined_correct = True
            else:
                logger.append(f"❌ Combined solution failed: {combined_error}")
                
                # If implementation is correct but tests fail, that's a test issue
                if implementation_correct:
                    logger.append(f"Tests incorrectly fail on a correct implementation")
                # If implementation is wrong and tests fail, that could be good (tests catching errors)
                elif not implementation_correct:
                    logger.append(f"Tests correctly identify an incorrect implementation")
                    # This is actually good test behavior
                    test_correct = True
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Implementation correct: {'✓' if implementation_correct else '✗'}")
        logger.append(f"├─ Test suite correct: {'✓' if test_correct else '✗'}")
        logger.append(f"└─ Combined solution works: {'✓' if combined_correct else '✗'}")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add test-driven programmer entry
        result_entries.append({
            'id': example_id,
            'data_type': 'training',
            'role': 'test_driven_programmer',
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_solution': solution,
            'test_content': test_content,
            'implementation': implementation,
            'implementation_result': implementation_result,
            'implementation_correct': implementation_correct,
            'test_correct': test_correct,
            'combined_correct': combined_correct
        })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'test_success': test_correct,
            'implementation_success': implementation_correct,
            'combined_success': combined_correct,
            
            # Compatibility fields for ProgressTracker statistics
            'is_correct': implementation_correct or test_correct,  # Consider success if either component is correct
            'is_most_common_correct': implementation_correct,  # For backward compatibility
            
            'total_solutions': 1,
            'correct_solutions': 1 if implementation_correct or test_correct else 0,
            'incorrect_solutions': 0 if implementation_correct or test_correct else 1,
            'verified_correct_solutions': 1 if implementation_correct and test_correct else 0,
            'verified_incorrect_solutions': 1 if not (implementation_correct and test_correct) else 0
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
