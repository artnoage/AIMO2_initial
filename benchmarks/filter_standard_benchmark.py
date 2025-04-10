import os
import asyncio
import logging
import sys
from collections import Counter
import numpy as np
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, List
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

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with programming solution verification and filtering"""
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
        solution_agent = FullSolutionAgent(main)
        solutions = []
        correct_count = 0
        best_solution = None
        
        # Force best_of to be 8 regardless of config
        best_of = 8
        
        for attempt in range(best_of):
            try:
                prompt, full_solution = await solution_agent.generate(example["problem"], return_prompt=True)
                
                # Create numeric verifier
                verifier = NumericVerifier(tolerance=config.tolerance)
                
                # Make sure correct_answer is a string before passing to verify
                correct_answer_str = str(correct_answer) if correct_answer is not None else ""
                
                is_correct, current_answer = await verifier.verify(
                    full_solution,
                    correct_answer_str,
                    example["problem"]
                )
                
                # Always append the solution, regardless of correctness
                solutions.append({
                    'solution': full_solution,
                    'answer': current_answer,
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

        # Calculate thinking length statistics
        thinking_lengths = [get_thinking_length(s['solution']) for s in solutions]
        correct_thinking_lengths = [length for length, s in zip(thinking_lengths, solutions) if s['is_correct']]
        incorrect_thinking_lengths = [length for length, s in zip(thinking_lengths, solutions) if not s['is_correct']]
        
        avg_thinking_length = sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0
        avg_correct_thinking = sum(correct_thinking_lengths) / len(correct_thinking_lengths) if correct_thinking_lengths else 0
        avg_incorrect_thinking = sum(incorrect_thinking_lengths) / len(incorrect_thinking_lengths) if incorrect_thinking_lengths else 0
        
        # Create thinking length distribution visualization
        if thinking_lengths:
            # Create a simple ASCII histogram
            correct_hist = create_ascii_histogram(correct_thinking_lengths, "Correct solutions thinking length")
            incorrect_hist = create_ascii_histogram(incorrect_thinking_lengths, "Incorrect solutions thinking length")
        
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
        logger.append(f"├─ Correct solutions: {correct_count}/{best_of}")
        logger.append(f"├─ Success rate: {(correct_count/best_of)*100:.1f}%")
        
        logger.append(f"├─ Answer distribution: {Counter(str(ans) for ans in model_answers).most_common()}")
        logger.append(f"├─ Most common answer: {most_common_answer}")
        logger.append(f"├─ Most common answer correct? {'Yes' if is_most_common_correct else 'No'}")
        logger.append(f"├─ Avg thinking length: {avg_thinking_length:.1f} chars")
        logger.append(f"├─ Avg correct thinking length: {avg_correct_thinking:.1f} chars")
        logger.append(f"└─ Avg incorrect thinking length: {avg_incorrect_thinking:.1f} chars")
        
        # Add thinking length distributions
        if thinking_lengths:
            logger.append("\n📊 Thinking Length Distributions:")
            logger.append(correct_hist)
            logger.append(incorrect_hist)
        
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
            
            # Show a snippet of the solution
            solution_lines = s['solution'].split('\n')
            solution_preview = '\n'.join(solution_lines[:5])
            if len(solution_lines) > 5:
                solution_preview += f"\n... ({len(solution_lines) - 5} more lines)"
            logger.append(f"Solution snippet:\n{solution_preview}")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create individual entries for each solution
        result_entries = []
        
        # Filter solutions based on correct count
        filtered_solutions = []
        if correct_count == 1 or correct_count == 2:
            # Keep only one correct solution
            for s in solutions:
                if s['is_correct']:
                    filtered_solutions.append(s)
                    break  # Only keep the first correct one
        else:
            # Keep all solutions
            filtered_solutions = solutions
        
        # Add individual solution entries for filtered solutions
        for i, s in enumerate(filtered_solutions):
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'problem': example['problem'],
                'correct_solution': example.get('solution', '') if 'solution' in example else '',
                'correct_answer': correct_answer,
                'model_solution': s['solution'],
                'model_answer': s['answer'],
                'is_correct': s['is_correct'],
                'error_message': s['error_message'],
                'attempt_number': i + 1,
                'total_attempts': len(filtered_solutions)
            })
        
        # Add statistics entry - this includes stats for all 8 solutions, not just filtered ones
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [s['is_correct'] for s in solutions],
            'is_most_common_correct': is_most_common_correct,
            'success_rate': (correct_count/best_of)*100 if best_of > 0 else 0,
            'total_solutions': len(solutions),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(solutions) - correct_count,
            'all_solutions_correct': all(s['is_correct'] for s in solutions),
            'filtered_solutions_count': len(filtered_solutions)
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
            'all_solutions_correct': None,
            'filtered_solutions_count': 0
        }]


def create_ascii_histogram(data: List[int], title: str) -> str:
    """Create a simple ASCII histogram for the given data"""
    if not data:
        return f"{title}:\n  No data available"
    
    # Create bins
    min_val = min(data) if data else 0
    max_val = max(data) if data else 0
    
    if min_val == max_val:
        return f"{title}:\n  All values are {min_val}"
    
    # Create 5 bins
    bin_width = max(1, (max_val - min_val) // 5)
    bins = list(range(min_val, max_val + bin_width, bin_width))
    
    # Count values in each bin
    hist = [0] * (len(bins) - 1)
    for val in data:
        for i in range(len(bins) - 1):
            if bins[i] <= val < bins[i+1]:
                hist[i] += 1
                break
        # Handle the last bin edge case
        if val == bins[-1]:
            hist[-1] += 1
    
    # Create ASCII representation
    result = [f"{title} (n={len(data)}):\n"]
    max_count = max(hist) if hist else 0
    scale = min(40, max_count)  # Scale to fit in console
    
    for i in range(len(hist)):
        bin_label = f"{bins[i]}-{bins[i+1]-1}" if bins[i+1]-1 > bins[i] else f"{bins[i]}"
        bar_length = int((hist[i] / max_count) * scale) if max_count > 0 else 0
        bar = "█" * bar_length
        result.append(f"  {bin_label.rjust(10)}: {bar} ({hist[i]})")
    
    return "\n".join(result)

async def main():
    """Main function for benchmarking mathematical problem solving with standard solutions
    and filtering results to keep only one correct solution when there are 1-2 correct solutions."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems using programming solutions with filtering')
    
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
