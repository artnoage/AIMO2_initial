import os
import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from dotenv import load_dotenv
import sys
import re
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

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification, self-reflection filtering, and majority voting"""
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
        solution_agent = ReflectiveSolutionAgent(main)
        solutions = []
        correct_count = 0
        best_solution = None
        
        # Track all solutions, including those where model thinks answer is incorrect
        all_solutions = []
        
        for attempt in range(config.best_of):
            try:
                prompt, current_solution = await solution_agent.generate(example["problem"], return_prompt=True)
                
                # Extract reflection content to check if model thinks answer is correct
                reflection_match = re.search(r'<reflection>(.*?)</reflection>', current_solution, re.DOTALL)
                reflection_content = reflection_match.group(1) if reflection_match else ""
                
                # Check if reflection indicates the answer is correct
                thinks_correct = "answer is correct" in reflection_content.lower()
                thinks_incorrect = "answer may not be correct" in reflection_content.lower() or "answer is not correct" in reflection_content.lower()
                
                # Create numeric verifier
                verifier = NumericVerifier(tolerance=config.tolerance)
                
                # Make sure correct_answer is a string before passing to verify
                correct_answer_str = str(correct_answer) if correct_answer is not None else ""
                
                is_correct, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer_str,
                    example["problem"]
                )
                
                # Add to all_solutions regardless of self-assessment
                all_solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'is_correct': is_correct,
                    'self_assessed_correct': thinks_correct,
                    'self_assessed_incorrect': thinks_incorrect
                })
                
                # Add to filtered solutions list only if model thinks it's correct
                if thinks_correct:
                    solutions.append({
                        'solution': current_solution,
                        'answer': current_answer,
                        'is_correct': is_correct,
                        'self_assessed_correct': thinks_correct
                    })
                else:
                    logger.append(f"Skipping solution attempt {attempt + 1} for filtered list because model did not self-assess as correct")
                
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
                        prompt, current_solution = await solution_agent.generate(example["problem"], return_prompt=True)
                        
                        # Extract reflection content to check if model thinks answer is correct
                        reflection_match = re.search(r'<reflection>(.*?)</reflection>', current_solution, re.DOTALL)
                        reflection_content = reflection_match.group(1) if reflection_match else ""
                        
                        # Check if reflection indicates the answer is correct
                        thinks_correct = "answer is correct" in reflection_content.lower()
                        thinks_incorrect = "answer may not be correct" in reflection_content.lower() or "answer is not correct" in reflection_content.lower()
                        
                        # Create numeric verifier
                        verifier = NumericVerifier(tolerance=config.tolerance)
                        
                        # Make sure correct_answer is a string before passing to verify
                        correct_answer_str = str(correct_answer) if correct_answer is not None else ""
                        
                        is_correct, current_answer = await verifier.verify(
                            current_solution,
                            correct_answer_str,
                            example["problem"]
                        )
                        
                        # Add to all_solutions regardless of self-assessment
                        all_solutions.append({
                            'solution': current_solution,
                            'answer': current_answer,
                            'is_correct': is_correct,
                            'self_assessed_correct': thinks_correct,
                            'self_assessed_incorrect': thinks_incorrect
                        })
                        
                        # Add to filtered solutions list only if model thinks it's correct
                        if thinks_correct:
                            solutions.append({
                                'solution': current_solution,
                                'answer': current_answer,
                                'is_correct': is_correct,
                                'self_assessed_correct': thinks_correct
                            })
                        else:
                            logger.append(f"Skipping solution retry {retry + 1} for filtered list because model did not self-assess as correct")
                        
                        if is_correct:
                            correct_count += 1
                            if best_solution is None:
                                best_solution = current_solution
                                
                        break  # Success, exit retry loop
                        
                    except Exception as retry_e:
                        logger.append(f"Retry {retry + 1} failed: {str(retry_e)}")
                        if retry == 2:  # Last retry failed
                            solution_info = {
                                'solution': f"Error occurred after 3 retries: {type(e).__name__} - {str(e)}",
                                'answer': None,
                                'is_correct': False,
                                'self_assessed_correct': False,
                                'self_assessed_incorrect': False
                            }
                            all_solutions.append(solution_info)
                continue  # Move to next attempt
        
        # Calculate reflection accuracy statistics
        total_reflections = len(all_solutions)
        correct_self_assessments = sum(1 for s in all_solutions if s['is_correct'] == s['self_assessed_correct'])
        incorrect_self_assessments = total_reflections - correct_self_assessments
        
        # True positives: correct answers that model thought were correct
        true_positives = sum(1 for s in all_solutions if s['is_correct'] and s['self_assessed_correct'])
        
        # False negatives: correct answers that model thought were incorrect
        false_negatives = sum(1 for s in all_solutions if s['is_correct'] and s.get('self_assessed_incorrect', False))
        
        # False positives: incorrect answers that model thought were correct
        false_positives = sum(1 for s in all_solutions if not s['is_correct'] and s['self_assessed_correct'])
        
        # True negatives: incorrect answers that model thought were incorrect
        true_negatives = sum(1 for s in all_solutions if not s['is_correct'] and s.get('self_assessed_incorrect', False))
        
        # Calculate initial majority vote (before filtering)
        initial_model_answers = [s['answer'] for s in all_solutions if s['answer'] is not None]
        initial_most_common_answer = None
        initial_is_most_common_correct = False
        initial_majority_count = 0
        initial_majority_percentage = 0
        
        if initial_model_answers:
            initial_answer_counts = Counter(str(ans) for ans in initial_model_answers)
            initial_most_common = initial_answer_counts.most_common(1)
            if initial_most_common:
                initial_most_common_answer = initial_most_common[0][0]
                initial_majority_count = initial_most_common[0][1]
                initial_majority_percentage = (initial_majority_count / len(initial_model_answers)) * 100
                
                # Check if the most common answer is correct by comparing with the expected answer
                numeric_verifier = NumericVerifier(tolerance=config.tolerance)
                initial_is_most_common_correct = False
                
                # Try to convert both to numeric values for comparison
                try:
                    most_common_numeric = extract_numeric_answer(initial_most_common_answer)[0]
                    correct_numeric = extract_numeric_answer(correct_answer)[0]
                    
                    if most_common_numeric is not None and correct_numeric is not None:
                        # Use tolerance-based comparison for numeric answers
                        initial_is_most_common_correct = abs(most_common_numeric - correct_numeric) <= config.tolerance
                    else:
                        # Fall back to string comparison for non-numeric answers
                        initial_is_most_common_correct = initial_most_common_answer.strip() == str(correct_answer).strip()
                except:
                    # If conversion fails, use direct string comparison
                    initial_is_most_common_correct = initial_most_common_answer.strip() == str(correct_answer).strip()
        
        # Calculate filtered majority vote (after filtering out self-assessed incorrect answers)
        filtered_model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        filtered_most_common_answer = None
        filtered_is_most_common_correct = False
        filtered_majority_count = 0
        filtered_majority_percentage = 0
        
        if filtered_model_answers:
            filtered_answer_counts = Counter(str(ans) for ans in filtered_model_answers)
            filtered_most_common = filtered_answer_counts.most_common(1)
            if filtered_most_common:
                filtered_most_common_answer = filtered_most_common[0][0]
                filtered_majority_count = filtered_most_common[0][1]
                filtered_majority_percentage = (filtered_majority_count / len(filtered_model_answers)) * 100
                
                # Check if the most common answer is correct by comparing with the expected answer
                numeric_verifier = NumericVerifier(tolerance=config.tolerance)
                filtered_is_most_common_correct = False
                
                # Try to convert both to numeric values for comparison
                try:
                    most_common_numeric = extract_numeric_answer(filtered_most_common_answer)[0]
                    correct_numeric = extract_numeric_answer(correct_answer)[0]
                    
                    if most_common_numeric is not None and correct_numeric is not None:
                        # Use tolerance-based comparison for numeric answers
                        filtered_is_most_common_correct = abs(most_common_numeric - correct_numeric) <= config.tolerance
                    else:
                        # Fall back to string comparison for non-numeric answers
                        filtered_is_most_common_correct = filtered_most_common_answer.strip() == str(correct_answer).strip()
                except:
                    # If conversion fails, use direct string comparison
                    filtered_is_most_common_correct = filtered_most_common_answer.strip() == str(correct_answer).strip()
        
        # Calculate think length statistics
        think_lengths = [get_think_length(s['solution']) for s in all_solutions]
        correct_think_lengths = [length for length, s in zip(think_lengths, all_solutions) if s['is_correct']]
        incorrect_think_lengths = [length for length, s in zip(think_lengths, all_solutions) if not s['is_correct']]
        
        avg_think_length = sum(think_lengths) / len(think_lengths) if think_lengths else 0
        avg_correct_think = sum(correct_think_lengths) / len(correct_think_lengths) if correct_think_lengths else 0
        avg_incorrect_think = sum(incorrect_think_lengths) / len(incorrect_think_lengths) if incorrect_think_lengths else 0
        
        # Create think length distribution visualization
        if think_lengths:
            # Create a simple ASCII histogram
            correct_hist = create_ascii_histogram(correct_think_lengths, "Correct solutions think length")
            incorrect_hist = create_ascii_histogram(incorrect_think_lengths, "Incorrect solutions think length")
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        logger.append(f"\n📊 Initial Statistics (All Solutions):")
        logger.append(f"├─ Total solutions: {len(all_solutions)}")
        logger.append(f"├─ Model answers: {[s['answer'] for s in all_solutions]}")
        logger.append(f"├─ Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in all_solutions]}")
        logger.append(f"├─ Self-assessed correct: {[1 if s['self_assessed_correct'] else 0 for s in all_solutions]}")
        logger.append(f"├─ Correct solutions: {sum(1 for s in all_solutions if s['is_correct'])}/{len(all_solutions)}")
        logger.append(f"├─ Success rate: {(sum(1 for s in all_solutions if s['is_correct'])/len(all_solutions))*100:.1f}%")
        logger.append(f"├─ Initial majority answer: {initial_most_common_answer} ({initial_majority_count} votes, {initial_majority_percentage:.1f}%)")
        logger.append(f"└─ Initial majority answer correct? {'Yes' if initial_is_most_common_correct else 'No'}")
        
        logger.append(f"\n📊 Filtered Statistics (Self-Assessed Correct Only):")
        logger.append(f"├─ Total filtered solutions: {len(solutions)}")
        logger.append(f"├─ Filtered model answers: {[s['answer'] for s in solutions]}")
        logger.append(f"├─ Filtered correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        logger.append(f"├─ Correct filtered solutions: {sum(1 for s in solutions if s['is_correct'])}/{len(solutions)}")
        logger.append(f"├─ Filtered success rate: {(sum(1 for s in solutions if s['is_correct'])/len(solutions))*100:.1f}% (of self-assessed correct solutions)")
        logger.append(f"├─ Filtered majority answer: {filtered_most_common_answer} ({filtered_majority_count} votes, {filtered_majority_percentage:.1f}%)")
        logger.append(f"└─ Filtered majority answer correct? {'Yes' if filtered_is_most_common_correct else 'No'}")
        
        # Add reflection statistics
        logger.append(f"\n📊 Reflection Statistics:")
        logger.append(f"├─ Total reflections: {total_reflections}")
        logger.append(f"├─ True positives (correct answers assessed as correct): {true_positives}")
        logger.append(f"├─ False positives (incorrect answers assessed as correct): {false_positives}")
        logger.append(f"├─ False negatives (correct answers assessed as incorrect): {false_negatives}")
        logger.append(f"├─ True negatives (incorrect answers assessed as incorrect): {true_negatives}")
        logger.append(f"└─ Self-assessment accuracy: {correct_self_assessments/total_reflections*100:.1f}%")
        
        # Add think length statistics
        logger.append(f"\n📊 Think Length Statistics:")
        logger.append(f"├─ Avg think length: {avg_think_length:.1f} chars")
        logger.append(f"├─ Avg correct think length: {avg_correct_think:.1f} chars")
        logger.append(f"└─ Avg incorrect think length: {avg_incorrect_think:.1f} chars")
        
        # Add think length distributions
        if think_lengths:
            logger.append("\n📊 Think Length Distributions:")
            logger.append(correct_hist)
            logger.append(incorrect_hist)
            
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create individual entries for each solution
        result_entries = []
        
        # Add individual solution entries for all solutions
        for i, s in enumerate(all_solutions):
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'problem': example['problem'],
                'correct_solution': example['solution'],
                'correct_answer': correct_answer,
                'model_solution': s['solution'],
                'model_answer': s['answer'],
                'is_correct': s['is_correct'],
                'self_assessed_correct': s['self_assessed_correct'],
                'self_assessed_incorrect': s.get('self_assessed_incorrect', False),
                'attempt_number': i + 1,
                'total_attempts': len(all_solutions)
            })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            
            # Initial statistics (all solutions)
            'initial_is_correct_list': [s['is_correct'] for s in all_solutions],
            'initial_self_assessed_correct_list': [s['self_assessed_correct'] for s in all_solutions],
            'initial_self_assessed_incorrect_list': [s.get('self_assessed_incorrect', False) for s in all_solutions],
            'initial_most_common_answer': initial_most_common_answer,
            'initial_is_most_common_correct': initial_is_most_common_correct,
            'initial_majority_count': initial_majority_count,
            'initial_majority_percentage': initial_majority_percentage,
            'initial_success_rate': (sum(1 for s in all_solutions if s['is_correct'])/len(all_solutions))*100 if all_solutions else 0,
            'initial_total_solutions': len(all_solutions),
            'initial_correct_solutions': sum(1 for s in all_solutions if s['is_correct']),
            'initial_incorrect_solutions': sum(1 for s in all_solutions if not s['is_correct']),
            
            # Filtered statistics (after removing self-assessed incorrect)
            'filtered_is_correct_list': [s['is_correct'] for s in solutions],
            'filtered_self_assessed_correct_list': [s['self_assessed_correct'] for s in solutions],
            'filtered_most_common_answer': filtered_most_common_answer,
            'filtered_is_most_common_correct': filtered_is_most_common_correct,
            'filtered_majority_count': filtered_majority_count,
            'filtered_majority_percentage': filtered_majority_percentage,
            'filtered_success_rate': (sum(1 for s in solutions if s['is_correct'])/len(solutions))*100 if solutions else 0,
            'filtered_total_solutions': len(solutions),
            'filtered_correct_solutions': sum(1 for s in solutions if s['is_correct']),
            'filtered_incorrect_solutions': sum(1 for s in solutions if not s['is_correct']),
            
            # Reflection statistics
            'true_positives': true_positives,
            'false_positives': false_positives, # FP
            'false_negatives': false_negatives, # FN
            'true_negatives': true_negatives, # TN
            'self_assessment_accuracy': (correct_self_assessments/total_reflections)*100 if total_reflections else 0,
            # Add renamed fields for ProgressTracker compatibility
            'correct_answers_assessed_correct': true_positives, # TP
            'correct_answers_assessed_incorrect': false_negatives, # FN
            'incorrect_answers_assessed_correct': false_positives, # FP
            'incorrect_answers_assessed_incorrect': true_negatives, # TN
            'total_reflections': total_reflections, # Total solutions generated
            
            # For compatibility with ProgressTracker - use separate fields for initial and filtered results
            'is_correct_list': [s['is_correct'] for s in all_solutions],  # Use initial list for compatibility
            'is_most_common_correct': initial_is_most_common_correct,  # Use initial result for compatibility
            'filtered_is_most_common_correct': filtered_is_most_common_correct,  # Add filtered result separately
            'success_rate': (sum(1 for s in solutions if s['is_correct'])/len(solutions))*100 if solutions else 0,
            'total_solutions': len(all_solutions),
            'correct_solutions': sum(1 for s in all_solutions if s['is_correct']),
            'incorrect_solutions': sum(1 for s in all_solutions if not s['is_correct']),
            'all_solutions_correct': all(s['is_correct'] for s in all_solutions) if all_solutions else False
        })
        
        return result_entries
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'initial_is_correct_list': [],
            'initial_self_assessed_correct_list': [],
            'initial_most_common_answer': None,
            'initial_is_most_common_correct': None,
            'initial_success_rate': 0,
            'filtered_is_correct_list': [],
            'filtered_self_assessed_correct_list': [],
            'filtered_most_common_answer': None,
            'filtered_is_most_common_correct': None,
            'filtered_success_rate': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'self_assessment_accuracy': 0,
            'is_correct_list': [],
            'is_most_common_correct': None,
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
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
    """Main function for benchmarking mathematical problem solving with reflection filtering and majority voting."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems with reflection filtering and majority voting')
    
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
