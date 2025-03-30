import os
import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
import sys
from collections import Counter
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

def calculate_length_weighted_majority(solutions: List[Dict], weight_function: str = "quadratic") -> Tuple[Optional[str], float]:
    """
    Calculate the majority answer weighted by thinking length.
    Uses different weight functions to penalize longer thinking.
    
    Args:
        solutions: List of solution dictionaries with 'answer' and 'solution' keys
        weight_function: Type of weight function to use:
                         - "linear": 1/length (standard inverse)
                         - "quadratic": 1/length² (stronger penalty)
                         - "sqrt": 1/sqrt(length) (milder penalty)
                         - "log": 1/log(length+1) (very mild penalty)
        
    Returns:
        Tuple of (majority_answer, confidence_score)
    """
    # Filter out None answers
    valid_solutions = [s for s in solutions if s['answer'] is not None]
    if not valid_solutions:
        return None, 0.0
    
    # Calculate thinking length for each solution
    thinking_lengths = [max(1, get_thinking_length(s['solution'])) for s in valid_solutions]
    
    # Calculate weights based on the selected weight function
    if weight_function == "linear":
        # Standard inverse weighting: 1/length
        weights = [1.0 / length for length in thinking_lengths]
    elif weight_function == "quadratic":
        # Quadratic inverse weighting: 1/length²
        weights = [1.0 / (length * length) for length in thinking_lengths]
    elif weight_function == "sqrt":
        # Square root inverse weighting: 1/sqrt(length)
        weights = [1.0 / np.sqrt(length) for length in thinking_lengths]
    elif weight_function == "log":
        # Logarithmic inverse weighting: 1/log(length+1)
        weights = [1.0 / np.log(length + 1) for length in thinking_lengths]
    else:
        # Default to linear inverse weighting
        weights = [1.0 / length for length in thinking_lengths]
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        # Fallback to equal weights if all weights are zero
        weights = [1.0 / len(valid_solutions)] * len(valid_solutions)
    else:
        weights = [w / total_weight for w in weights]
    
    # Count weighted votes for each answer
    answer_weights = {}
    for solution, weight in zip(valid_solutions, weights):
        answer_str = str(solution['answer'])
        if answer_str in answer_weights:
            answer_weights[answer_str] += weight
        else:
            answer_weights[answer_str] = weight
    
    # Find the answer with the highest weight
    if not answer_weights:
        return None, 0.0
    
    majority_answer, confidence = max(answer_weights.items(), key=lambda x: x[1])
    return majority_answer, confidence

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification and length-weighted majority voting"""
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

        main = get_model(config, role="main")
        solution_agent = FullSolutionAgent(main)
        solutions = []
        correct_count = 0
        best_solution = None
        
        for attempt in range(config.best_of):
            try:
                prompt, current_solution = await solution_agent.generate(example["problem"], return_prompt=True)
                # Create numeric verifier
                verifier = NumericVerifier(tolerance=config.tolerance)
                is_correct, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer,
                    example["problem"]
                )
                # Always append the solution, regardless of correctness
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'is_correct': is_correct,
                    'thinking_length': get_thinking_length(current_solution)
                })
                
                # Update statistics if correct
                if is_correct:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
            except Exception as e:
                logger.append(f"❌ Error in attempt {str(attempt + 1)} for example {str(running_id)}:")
                logger.append(f"Exception type: {type(e).__name__}")
                logger.append(f"Exception message: {str(e)}")
                import traceback
                logger.append(f"Traceback:\n{traceback.format_exc()}")
                
                # Retry this attempt up to 3 times
                for retry in range(3):
                    try:
                        logger.append(f"Retrying attempt {attempt + 1} (retry {retry + 1}/3)...")
                        current_solution = await solution_agent.generate(example["problem"])
                        
                        # Create numeric verifier
                        verifier = NumericVerifier(tolerance=config.tolerance)
                        is_correct, current_answer = await verifier.verify(
                            current_solution,
                            correct_answer,
                            example["problem"]
                        )
                        
                        if is_correct:
                            correct_count += 1
                            if best_solution is None:
                                best_solution = current_solution
                                
                        solutions.append({
                            'solution': current_solution,
                            'answer': current_answer,
                            'is_correct': is_correct,
                            'thinking_length': get_thinking_length(current_solution)
                        })
                        break  # Success, exit retry loop
                        
                    except Exception as retry_e:
                        logger.append(f"Retry {retry + 1} failed: {str(retry_e)}")
                        if retry == 2:  # Last retry failed
                            solution_info = {
                                'solution': f"Error occurred after 3 retries: {type(e).__name__} - {str(e)}",
                                'answer': None,
                                'is_correct': False,
                                'thinking_length': 0
                            }
                            solutions.append(solution_info)
                continue  # Move to next attempt
        
        
        # Calculate length-weighted majority answers with different weight functions
        linear_weighted_answer, linear_confidence = calculate_length_weighted_majority(solutions, weight_function="linear")
        quadratic_weighted_answer, quadratic_confidence = calculate_length_weighted_majority(solutions, weight_function="quadratic")
        sqrt_weighted_answer, sqrt_confidence = calculate_length_weighted_majority(solutions, weight_function="sqrt")
        log_weighted_answer, log_confidence = calculate_length_weighted_majority(solutions, weight_function="log")
        
        # Check if the weighted answers are correct
        is_linear_weighted_correct = False
        is_quadratic_weighted_correct = False
        is_sqrt_weighted_correct = False
        is_log_weighted_correct = False
        
        verifier = NumericVerifier(tolerance=config.tolerance)
        
        # Check linear weighted answer
        if linear_weighted_answer is not None:
            for s in solutions:
                if s['answer'] is not None and str(s['answer']) == linear_weighted_answer:
                    is_linear_weighted_correct = s['is_correct']
                    break
        
        # Check quadratic weighted answer
        if quadratic_weighted_answer is not None:
            for s in solutions:
                if s['answer'] is not None and str(s['answer']) == quadratic_weighted_answer:
                    is_quadratic_weighted_correct = s['is_correct']
                    break
        
        # Check sqrt weighted answer
        if sqrt_weighted_answer is not None:
            for s in solutions:
                if s['answer'] is not None and str(s['answer']) == sqrt_weighted_answer:
                    is_sqrt_weighted_correct = s['is_correct']
                    break
        
        # Check log weighted answer
        if log_weighted_answer is not None:
            for s in solutions:
                if s['answer'] is not None and str(s['answer']) == log_weighted_answer:
                    is_log_weighted_correct = s['is_correct']
                    break
        
        # Calculate standard majority answer for comparison
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        most_common_answer = None
        is_most_common_correct = False
        if model_answers:
            most_common_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_most_common_correct = any(str(s['answer']) == most_common_answer and s['is_correct'] for s in solutions)

        # Calculate thinking length statistics
        thinking_lengths = [s['thinking_length'] for s in solutions]
        correct_thinking_lengths = [s['thinking_length'] for s in solutions if s['is_correct']]
        incorrect_thinking_lengths = [s['thinking_length'] for s in solutions if not s['is_correct']]
        
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
        logger.append(f"├─ Thinking lengths: {thinking_lengths}")
        logger.append(f"├─ Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        logger.append(f"├─ Correct solutions: {correct_count}/{config.best_of}")
        logger.append(f"├─ Success rate: {(correct_count/config.best_of)*100:.1f}%")
        logger.append(f"├─ Standard majority answer: {most_common_answer}")
        logger.append(f"├─ Standard majority correct? {'Yes' if is_most_common_correct else 'No'}")
        logger.append(f"\n📊 Length-Weighted Majority Results:")
        logger.append(f"├─ Linear (1/length):")
        logger.append(f"│  ├─ Answer: {linear_weighted_answer}")
        logger.append(f"│  ├─ Confidence: {linear_confidence:.2f}")
        logger.append(f"│  └─ Correct? {'Yes' if is_linear_weighted_correct else 'No'}")
        logger.append(f"├─ Quadratic (1/length²):")
        logger.append(f"│  ├─ Answer: {quadratic_weighted_answer}")
        logger.append(f"│  ├─ Confidence: {quadratic_confidence:.2f}")
        logger.append(f"│  └─ Correct? {'Yes' if is_quadratic_weighted_correct else 'No'}")
        logger.append(f"├─ Square Root (1/sqrt(length)):")
        logger.append(f"│  ├─ Answer: {sqrt_weighted_answer}")
        logger.append(f"│  ├─ Confidence: {sqrt_confidence:.2f}")
        logger.append(f"│  └─ Correct? {'Yes' if is_sqrt_weighted_correct else 'No'}")
        logger.append(f"├─ Logarithmic (1/log(length+1)):")
        logger.append(f"│  ├─ Answer: {log_weighted_answer}")
        logger.append(f"│  ├─ Confidence: {log_confidence:.2f}")
        logger.append(f"│  └─ Correct? {'Yes' if is_log_weighted_correct else 'No'}")
        logger.append(f"\n📊 Thinking Length Statistics:")
        logger.append(f"├─ Avg thinking length: {avg_thinking_length:.1f} chars")
        logger.append(f"├─ Avg correct thinking length: {avg_correct_thinking:.1f} chars")
        logger.append(f"└─ Avg incorrect thinking length: {avg_incorrect_thinking:.1f} chars")
        
        # Add thinking length distributions
        if thinking_lengths:
            logger.append("\n📊 Thinking Length Distributions:")
            logger.append(correct_hist)
            logger.append(incorrect_hist)
            
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
                'correct_solution': example['solution'],
                'correct_answer': correct_answer,
                'model_solution': s['solution'],
                'model_answer': s['answer'],
                'is_correct': s['is_correct'],
                'thinking_length': s['thinking_length'],
                'attempt_number': i + 1,
                'total_attempts': len(solutions)
            })
        
        # Add statistics entry with length-weighted information
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [s['is_correct'] for s in solutions],
            'is_most_common_correct': is_most_common_correct,
            'is_linear_weighted_correct': is_linear_weighted_correct,
            'is_quadratic_weighted_correct': is_quadratic_weighted_correct,
            'is_sqrt_weighted_correct': is_sqrt_weighted_correct,
            'is_log_weighted_correct': is_log_weighted_correct,
            'linear_weighted_answer': linear_weighted_answer,
            'quadratic_weighted_answer': quadratic_weighted_answer,
            'sqrt_weighted_answer': sqrt_weighted_answer,
            'log_weighted_answer': log_weighted_answer,
            'linear_weighted_confidence': linear_confidence,
            'quadratic_weighted_confidence': quadratic_confidence,
            'sqrt_weighted_confidence': sqrt_confidence,
            'log_weighted_confidence': log_confidence,
            'success_rate': (correct_count/config.best_of)*100,
            'total_solutions': len(solutions),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(solutions) - correct_count,
            'avg_thinking_length': avg_thinking_length,
            'avg_correct_thinking': avg_correct_thinking,
            'avg_incorrect_thinking': avg_incorrect_thinking,
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
            'is_linear_weighted_correct': None,
            'is_quadratic_weighted_correct': None,
            'is_sqrt_weighted_correct': None,
            'is_log_weighted_correct': None,
            'linear_weighted_answer': None,
            'quadratic_weighted_answer': None,
            'sqrt_weighted_answer': None,
            'log_weighted_answer': None,
            'linear_weighted_confidence': 0.0,
            'quadratic_weighted_confidence': 0.0,
            'sqrt_weighted_confidence': 0.0,
            'log_weighted_confidence': 0.0,
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'avg_thinking_length': 0,
            'avg_correct_thinking': 0,
            'avg_incorrect_thinking': 0,
            'all_solutions_correct': None
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
    """Main function for benchmarking mathematical problem solving with multiple length-weighted majority functions."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems with advanced length-weighted majority')
    
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
