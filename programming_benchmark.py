import os
import asyncio
import logging
import tempfile
import subprocess
import sys
import re
from io import StringIO
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, List
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



async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with programming solution verification"""
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
        except:
            pass

        main = get_model(config, role="main")
        programming_agent = ProgrammingAgent(main)
        solutions = []
        correct_count = 0
        best_solution = None
        
        for attempt in range(config.best_of):
            try:
                prompt, full_solution = await programming_agent.generate(example["problem"], return_prompt=True)
                
                # Store the full solution but extract code for execution
                # The full_solution contains the complete model output
                
                # First check if response section exists
                response_match = re.search(r'<response>(.*?)</response>', full_solution, re.DOTALL)
                if response_match:
                    response_content = response_match.group(1)
                    code = extract_code_from_response(response_content)
                    if not code:
                        # If no code in response section, try the whole solution
                        logger.append(f"No code found in response section, trying whole solution")
                        code = extract_code_from_response(full_solution)
                else:
                    # If no response tags, extract from the whole solution
                    code = extract_code_from_response(full_solution)
                
                logger.append(f"Extracted code length: {len(code)} characters")
                if not code:
                    logger.append(f"❌ No code found in solution")
                    solutions.append({
                        'solution': full_solution,
                        'code': "",
                        'answer': None,
                        'is_correct': False,
                        'error_message': "No code found in solution"
                    })
                    continue
                
                # Check code quality first to save time
                code_quality_passed, quality_message = check_code_quality(code)
                
                if not code_quality_passed:
                    logger.append(f"❌ Code quality check failed for attempt {attempt+1}: {quality_message}")
                    solutions.append({
                        'solution': full_solution,  # Store the complete model output
                        'code': code,
                        'answer': None,
                        'is_correct': False,
                        'error_message': f"Code quality check failed: {quality_message}"
                    })
                    continue
                
                # Only run code if it passes quality checks
                execution_success, result, error_message = run_code_safely(code, timeout=config.timeout)
                
                if not execution_success:
                    logger.append(f"❌ Code execution failed for attempt {attempt+1}: {error_message}")
                    solutions.append({
                        'solution': full_solution,
                        'code': code,
                        'answer': None,
                        'is_correct': False,
                        'error_message': error_message
                    })
                    continue
                
                # Compare with correct answer
                is_correct = False
                if isinstance(correct_answer, (int, float)) and isinstance(result, (int, float)):
                    # Use tolerance for numeric comparison
                    is_correct = abs(correct_answer - result) <= config.tolerance
                else:
                    # Try string comparison as fallback
                    is_correct = str(correct_answer).strip() == str(result).strip()
                
                solutions.append({
                    'solution': full_solution,  # This is the complete model output
                    'code': code,               # This is just the extracted code for execution
                    'answer': result,
                    'is_correct': is_correct,
                    'error_message': None
                })
                
                # Update statistics if correct
                if is_correct:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = full_solution
                
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
                    'code': "",
                    'answer': None,
                    'is_correct': False,
                    'error_message': str(e)
                })
        
        # Calculate most common answer statistics
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        most_common_answer = None
        is_most_common_correct = False
        if model_answers:
            from collections import Counter
            most_common_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_most_common_correct = any(str(s['answer']) == most_common_answer and s['is_correct'] for s in solutions)

        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Model answers: {[s['answer'] for s in solutions]}")
        logger.append(f"├─ Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        logger.append(f"├─ Correct solutions: {correct_count}/{config.best_of}")
        logger.append(f"├─ Success rate: {(correct_count/config.best_of)*100:.1f}%")
        logger.append(f"├─ Most common answer: {most_common_answer}")
        logger.append(f"└─ Most common answer correct? {'Yes' if is_most_common_correct else 'No'}")
        
        # Add code quality and execution details
        for i, s in enumerate(solutions):
            logger.append(f"\n📝 Solution {i+1}:")
            if s['error_message']:
                logger.append(f"❌ Error: {s['error_message']}")
                # Categorize the error
                if "Code quality check failed" in s['error_message']:
                    logger.append(f"   └─ Quality issue detected - skipped execution")
                elif "Execution error" in s['error_message'] or "timed out" in s['error_message']:
                    logger.append(f"   └─ Runtime error - code failed during execution")
            else:
                logger.append(f"✓ Answer: {s['answer']}")
                logger.append(f"✓ Correct: {'Yes' if s['is_correct'] else 'No'}")
            
            # Show a snippet of the code
            code_lines = s['code'].split('\n')
            code_preview = '\n'.join(code_lines[:10])
            if len(code_lines) > 10:
                code_preview += f"\n... ({len(code_lines) - 10} more lines)"
            logger.append(f"Code snippet:\n{code_preview}")
        
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
                'correct_solution': example.get('solution', ''),
                'correct_answer': correct_answer,
                'model_solution': s['solution'],
                'model_code': s['code'],
                'model_answer': s['answer'],
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
            'is_most_common_correct': is_most_common_correct,
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
            'is_most_common_correct': None,
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'all_solutions_correct': None
        }]


async def main():
    """Main function for benchmarking mathematical problem solving with programming solutions."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems using programming solutions')
    
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
