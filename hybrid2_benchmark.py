import os
import asyncio
import logging
from typing import Optional, Dict, List, Tuple, Any, Union
from collections import Counter
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

def calculate_answer_majority(answers, tolerance=1e-2, weights=None):
    """
    Calculate the most common answer by counting how many answers are within tolerance
    of each unique answer, with optional weighting.
    
    Args:
        answers: List of answers (can be numeric or string)
        tolerance: Numeric tolerance for grouping similar answers
        weights: Optional list of weights corresponding to each answer
        
    Returns:
        Tuple of (majority_answer, count_dict) where count_dict maps each answer to its weighted count
    """
    if not answers or all(ans is None for ans in answers):
        return None, {}
    
    # Filter out None values and their corresponding weights
    valid_answers = []
    valid_weights = []
    
    if weights:
        for ans, weight in zip(answers, weights):
            if ans is not None:
                valid_answers.append(ans)
                valid_weights.append(weight)
    else:
        valid_answers = [ans for ans in answers if ans is not None]
        valid_weights = [1.0] * len(valid_answers)  # Default weight of 1.0
    
    # Convert to numeric where possible
    numeric_answers = []
    for ans, weight in zip(valid_answers, valid_weights):
        try:
            if isinstance(ans, (int, float)):
                numeric_answers.append((ans, str(ans), weight))
            else:
                numeric_val, _ = extract_numeric_answer(str(ans))
                if numeric_val is not None:
                    numeric_answers.append((numeric_val, str(ans), weight))
                else:
                    # Keep non-numeric answers as is
                    numeric_answers.append((None, str(ans), weight))
        except:
            numeric_answers.append((None, str(ans), weight))
    
    # Count how many answers are within tolerance of each answer, with weighting
    count_dict = {}
    for i, (num_val, str_val, weight) in enumerate(numeric_answers):
        # Initialize count for this answer
        if str_val not in count_dict:
            count_dict[str_val] = 0
        
        # Count all answers within tolerance of this one, applying weights
        for other_num, other_str, other_weight in numeric_answers:
            if num_val is not None and other_num is not None:
                # Both are numeric, use tolerance
                if abs(num_val - other_num) <= tolerance:
                    count_dict[str_val] += other_weight
            else:
                # At least one is non-numeric, use exact string matching
                if str_val == other_str:
                    count_dict[str_val] += other_weight
    
    # Find the answer with the highest weighted count
    if count_dict:
        majority_answer = max(count_dict.items(), key=lambda x: x[1])[0]
        return majority_answer, count_dict
    else:
        return None, {}

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using both programming and standard solution agents,
    then take the weighted union of their answers"""
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

        # Get models for solution and programming agents
        main_model = get_model(config, role="main")
        
        # Initialize agents
        solution_agent = FullSolutionAgent(main_model)
        programming_agent = ProgrammingAgent(main_model)
        
        # Generate multiple programming solutions
        programming_solutions = []
        programming_correctness = []
        programming_answers = []
        
        for attempt in range(config.best_of):
            try:
                prompt, current_solution = await programming_agent.generate(example["problem"], return_prompt=True)
                
                # Extract code from solution - similar to programming_benchmark.py
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
                
                logger.append(f"Extracted code length: {len(code) if code else 0} characters")
                
                if not code:
                    logger.append(f"❌ No code found in programming solution {attempt+1}")
                    programming_solutions.append(current_solution)
                    programming_correctness.append(False)
                    programming_answers.append(None)
                    continue
                
                # Check code quality
                code_quality_passed, quality_message = check_code_quality(code)
                
                if not code_quality_passed:
                    logger.append(f"❌ Code quality check failed for attempt {attempt+1}: {quality_message}")
                    programming_solutions.append(current_solution)
                    programming_correctness.append(False)
                    programming_answers.append(None)
                    continue
                
                # Run code safely
                execution_success, result, error_message = run_code_safely(code, timeout=config.timeout)
                
                if not execution_success:
                    logger.append(f"❌ Code execution failed for attempt {attempt+1}: {error_message}")
                    programming_solutions.append(current_solution)
                    programming_correctness.append(False)
                    programming_answers.append(None)
                    continue
                
                # Compare with correct answer - match programming_benchmark.py logic
                is_correct = False
                try:
                    # Convert correct_answer to float if possible for comparison
                    numeric_correct_answer = None
                    if isinstance(correct_answer, (int, float)):
                        numeric_correct_answer = correct_answer
                    else:
                        try:
                            numeric_correct_answer, _ = extract_numeric_answer(correct_answer)
                        except:
                            pass
                    
                    if numeric_correct_answer is not None and isinstance(result, (int, float)):
                        # Use tolerance for numeric comparison
                        is_correct = abs(numeric_correct_answer - result) <= config.tolerance
                    else:
                        # Try string comparison as fallback
                        is_correct = str(correct_answer).strip() == str(result).strip()
                except Exception as e:
                    logger.append(f"Error comparing answers: {str(e)}")
                
                programming_solutions.append(current_solution)
                programming_correctness.append(is_correct)
                programming_answers.append(result)
                
            except Exception as e:
                logger.append(f"❌ Error in programming attempt {attempt+1}: {str(e)}")
                programming_solutions.append(f"Error: {str(e)}")
                programming_correctness.append(False)
                programming_answers.append(None)
        
        # Generate multiple standard solutions
        standard_solutions = []
        standard_correctness = []
        standard_answers = []
        
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
                
                standard_solutions.append(current_solution)
                standard_correctness.append(is_correct)
                standard_answers.append(current_answer)
                
            except Exception as e:
                logger.append(f"❌ Error in standard solution attempt {attempt+1}: {str(e)}")
                standard_solutions.append(f"Error: {str(e)}")
                standard_correctness.append(False)
                standard_answers.append(None)
        
        # Calculate statistics for programming solutions with tolerance-based grouping
        programming_success_rate = sum(programming_correctness) / len(programming_correctness) * 100 if programming_correctness else 0
        
        # Use the weighted approach to calculate programming majority answer
        programming_majority_answer, programming_answer_counts = calculate_answer_majority(
            programming_answers, tolerance=config.tolerance)
            
        # Create a backward-compatible programming_grouped_answers structure
        programming_grouped_answers = {}
        for ans_str, count in programming_answer_counts.items():
            programming_grouped_answers[ans_str] = []
            for i, ans in enumerate(programming_answers):
                if ans is None:
                    continue
                
                # Check if this answer is within tolerance of the group key
                ans_numeric = None
                key_numeric = None
                try:
                    if isinstance(ans, (int, float)):
                        ans_numeric = ans
                    else:
                        ans_numeric, _ = extract_numeric_answer(str(ans))
                        
                    if isinstance(ans_str, (int, float)):
                        key_numeric = ans_str
                    else:
                        key_numeric, _ = extract_numeric_answer(ans_str)
                except:
                    pass
                
                # Add to group if within tolerance
                if ans_numeric is not None and key_numeric is not None:
                    if abs(ans_numeric - key_numeric) <= config.tolerance:
                        programming_grouped_answers[ans_str].append((i, ans))
                elif str(ans) == ans_str:
                    programming_grouped_answers[ans_str].append((i, ans))
        
        # Check if majority answer is correct
        programming_majority_correct = False
        if programming_majority_answer is not None:
            for i, (ans, is_correct) in enumerate(zip(programming_answers, programming_correctness)):
                if ans is None:
                    continue
                
                # Check if this answer is the majority answer or within tolerance
                ans_numeric = None
                majority_numeric = None
                try:
                    if isinstance(ans, (int, float)):
                        ans_numeric = ans
                    else:
                        ans_numeric, _ = extract_numeric_answer(str(ans))
                        
                    if isinstance(programming_majority_answer, (int, float)):
                        majority_numeric = programming_majority_answer
                    else:
                        majority_numeric, _ = extract_numeric_answer(programming_majority_answer)
                except:
                    pass
                
                is_majority = False
                if ans_numeric is not None and majority_numeric is not None:
                    is_majority = abs(ans_numeric - majority_numeric) <= config.tolerance
                else:
                    is_majority = str(ans) == programming_majority_answer
                
                if is_majority and is_correct:
                    programming_majority_correct = True
                    break
        
        # Calculate statistics for standard solutions with tolerance-based grouping
        standard_success_rate = sum(standard_correctness) / len(standard_correctness) * 100 if standard_correctness else 0
        
        # Use the weighted approach to calculate standard majority answer
        standard_majority_answer, standard_answer_counts = calculate_answer_majority(
            standard_answers, tolerance=config.tolerance)
            
        # Create a backward-compatible standard_grouped_answers structure
        standard_grouped_answers = {}
        for ans_str, count in standard_answer_counts.items():
            standard_grouped_answers[ans_str] = []
            for i, ans in enumerate(standard_answers):
                if ans is None:
                    continue
                
                # Check if this answer is within tolerance of the group key
                ans_numeric = None
                key_numeric = None
                try:
                    if isinstance(ans, (int, float)):
                        ans_numeric = ans
                    else:
                        ans_numeric, _ = extract_numeric_answer(str(ans))
                        
                    if isinstance(ans_str, (int, float)):
                        key_numeric = ans_str
                    else:
                        key_numeric, _ = extract_numeric_answer(ans_str)
                except:
                    pass
                
                # Add to group if within tolerance
                if ans_numeric is not None and key_numeric is not None:
                    if abs(ans_numeric - key_numeric) <= config.tolerance:
                        standard_grouped_answers[ans_str].append((i, ans))
                elif str(ans) == ans_str:
                    standard_grouped_answers[ans_str].append((i, ans))
        
        # Check if majority answer is correct
        standard_majority_correct = False
        if standard_majority_answer is not None:
            for i, (ans, is_correct) in enumerate(zip(standard_answers, standard_correctness)):
                if ans is None:
                    continue
                
                # Check if this answer is the majority answer or within tolerance
                ans_numeric = None
                majority_numeric = None
                try:
                    if isinstance(ans, (int, float)):
                        ans_numeric = ans
                    else:
                        ans_numeric, _ = extract_numeric_answer(str(ans))
                        
                    if isinstance(standard_majority_answer, (int, float)):
                        majority_numeric = standard_majority_answer
                    else:
                        majority_numeric, _ = extract_numeric_answer(standard_majority_answer)
                except:
                    pass
                
                is_majority = False
                if ans_numeric is not None and majority_numeric is not None:
                    is_majority = abs(ans_numeric - majority_numeric) <= config.tolerance
                else:
                    is_majority = str(ans) == standard_majority_answer
                
                if is_majority and is_correct:
                    standard_majority_correct = True
                    break
        
        # Create weighted union of answers
        # Apply weights: 1.2 for programming, 0.8 for standard
        all_answers = programming_answers + standard_answers
        all_weights = [1.2] * len(programming_answers) + [0.8] * len(standard_answers)
        
        # Calculate weighted majority from the union
        final_answer, weighted_answer_counts = calculate_answer_majority(
            all_answers, tolerance=config.tolerance, weights=all_weights)
        
        # Check if final answer is correct
        final_answer_correct = False
        if final_answer is not None:
            # Check if any solution with this answer is marked as correct
            final_answer_correct = any(
                (str(ans) == final_answer and is_correct)
                for solutions_list, correctness_list in [
                    (programming_answers, programming_correctness),
                    (standard_answers, standard_correctness)
                ]
                for ans, is_correct in zip(solutions_list, correctness_list)
                if ans is not None
            )
            
            # If not found by exact match, try numeric tolerance
            if not final_answer_correct:
                final_numeric = None
                try:
                    final_numeric, _ = extract_numeric_answer(final_answer)
                except:
                    pass
                
                if final_numeric is not None:
                    for solutions_list, correctness_list in [
                        (programming_answers, programming_correctness),
                        (standard_answers, standard_correctness)
                    ]:
                        for ans, is_correct in zip(solutions_list, correctness_list):
                            if ans is None or not is_correct:
                                continue
                                
                            ans_numeric = None
                            try:
                                if isinstance(ans, (int, float)):
                                    ans_numeric = ans
                                else:
                                    ans_numeric, _ = extract_numeric_answer(str(ans))
                            except:
                                pass
                                
                            if ans_numeric is not None and abs(ans_numeric - final_numeric) <= config.tolerance:
                                final_answer_correct = True
                                break
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Programming solutions statistics
        logger.append(f"\n📊 Programming Solutions Statistics (Weight: 1.2):")
        for i, (sol_correct, sol_answer) in enumerate(zip(programming_correctness, programming_answers)):
            logger.append(f"├─ Solution {i+1}: {'✓' if sol_correct else '✗'} (Answer: {sol_answer})")
            # Add error messages for debugging
            if not sol_correct and i < len(programming_solutions):
                code = extract_code_from_response(programming_solutions[i])
                if not code:
                    logger.append(f"  └─ No code extracted")
                elif "Error:" in programming_solutions[i]:
                    logger.append(f"  └─ {programming_solutions[i]}")
        
        # Log programming groups
        logger.append(f"├─ Programming answer groups (tolerance: {config.tolerance}):")
        for group_key, group_items in programming_grouped_answers.items():
            group_indices = [idx+1 for idx, _ in group_items]
            logger.append(f"│  └─ Group '{group_key}': solutions {group_indices} ({len(group_items)} items)")
        
        logger.append(f"├─ Programming success rate: {programming_success_rate:.1f}%")
        logger.append(f"├─ Programming majority answer: {programming_majority_answer}")
        logger.append(f"├─ Programming majority correct? {'✓' if programming_majority_correct else '✗'}")
        
        # Standard solutions statistics
        logger.append(f"\n📊 Standard Solutions Statistics (Weight: 0.8):")
        for i, (sol_correct, sol_answer) in enumerate(zip(standard_correctness, standard_answers)):
            logger.append(f"├─ Solution {i+1}: {'✓' if sol_correct else '✗'} (Answer: {sol_answer})")
        
        # Log standard groups
        logger.append(f"├─ Standard answer groups (tolerance: {config.tolerance}):")
        for group_key, group_items in standard_grouped_answers.items():
            group_indices = [idx+1 for idx, _ in group_items]
            logger.append(f"│  └─ Group '{group_key}': solutions {group_indices} ({len(group_items)} items)")
            
        logger.append(f"├─ Standard success rate: {standard_success_rate:.1f}%")
        logger.append(f"├─ Standard majority answer: {standard_majority_answer}")
        logger.append(f"├─ Standard majority correct? {'✓' if standard_majority_correct else '✗'}")
        
        # Weighted union statistics
        logger.append(f"\n📊 Weighted Union Statistics:")
        logger.append(f"├─ Weights: Programming=1.2, Standard=0.8")
        
        # Log weighted groups
        logger.append(f"├─ Weighted answer groups (tolerance: {config.tolerance}):")
        for group_key, weighted_count in weighted_answer_counts.items():
            logger.append(f"│  └─ Group '{group_key}': weighted count {weighted_count:.1f}")
            
        logger.append(f"├─ Final answer (weighted union): {final_answer}")
        logger.append(f"├─ Final answer correct? {'✓' if final_answer_correct else '✗'}")
        logger.append(f"├─ Programming majority correct? {'✓' if programming_majority_correct else '✗'}")
        logger.append(f"├─ Standard majority correct? {'✓' if standard_majority_correct else '✗'}")
        logger.append(f"└─ Improvement from weighted union? {'✓' if final_answer_correct and not (programming_majority_correct and standard_majority_correct) else '✗'}")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add detailed entry
        result_entries.append({
            'id': example_id,
            'data_type': 'training',
            'problem': example['problem'],
            'correct_solution': example.get('solution', ''),
            'correct_answer': correct_answer,
            'programming_solutions': programming_solutions,
            'programming_correctness': programming_correctness,
            'programming_answers': programming_answers,
            'standard_solutions': standard_solutions,
            'standard_correctness': standard_correctness,
            'standard_answers': standard_answers,
            'weighted_answer_counts': weighted_answer_counts,
            'final_answer': final_answer,
            'final_answer_correct': final_answer_correct
        })
        
        # Add statistics entry - ensure compatibility with both programming_benchmark.py and tutor2_solution_benchmark.py
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            # Programming solutions statistics
            'programming_solutions_count': len(programming_solutions),
            'programming_correctness': programming_correctness,
            'programming_answers': programming_answers,
            'programming_success_rate': programming_success_rate,
            'programming_majority_answer': programming_majority_answer,
            'programming_majority_correct': programming_majority_correct,
            # Standard solution statistics
            'standard_solutions_count': len(standard_solutions),
            'standard_correctness': standard_correctness,
            'standard_answers': standard_answers,
            'standard_success_rate': standard_success_rate,
            'standard_majority_answer': standard_majority_answer,
            'standard_majority_correct': standard_majority_correct,
            # Weighted union statistics
            'weighted_answer_counts': weighted_answer_counts,
            'final_answer': final_answer,
            'final_answer_correct': final_answer_correct,
            'weighted_union_improved': final_answer_correct and not (programming_majority_correct and standard_majority_correct),
            'weighted_union_worsened': (programming_majority_correct or standard_majority_correct) and not final_answer_correct,
            
            # Compatibility fields for ProgressTracker statistics
            # Include both programming and standard solutions for "at least one correct" calculation
            'is_correct_list': programming_correctness + standard_correctness,  # Use both approaches for main stats
            'is_most_common_correct': final_answer_correct,  # Use final answer for main stats
            'total_solutions': len(programming_solutions) + len(standard_solutions),
            'correct_solutions': sum(programming_correctness) + sum(standard_correctness),
            'incorrect_solutions': (len(programming_correctness) - sum(programming_correctness)) + 
                                  (len(standard_correctness) - sum(standard_correctness)),
            'programming_grouped_answers': {k: [ans for _, ans in v] for k, v in programming_grouped_answers.items()},
            'standard_grouped_answers': {k: [ans for _, ans in v] for k, v in standard_grouped_answers.items()},
            
            # For compatibility with tutor2_solution_benchmark.py
            'initial_solutions_count': len(programming_solutions),
            'initial_correctness': programming_correctness,
            'initial_answers': programming_answers,
            'initial_success_rate': programming_success_rate,
            'initial_majority_answer': programming_majority_answer,
            'initial_majority_correct': programming_majority_correct,
            'tutor_responses': standard_solutions,
            'tutor_verdicts': ["Standard solution" for _ in standard_solutions],
            'final_solutions': standard_solutions,
            'final_correctness': standard_correctness,
            'final_answers': standard_answers,
            'final_success_rate': standard_success_rate,
            'final_majority_answer': standard_majority_answer,
            'final_majority_correct': standard_majority_correct,
            'has_clear_winner': True,  # Always true for weighted union
            'solution_sources': ["weighted_union"],
            'majority_vote_improved': final_answer_correct and not (programming_majority_correct and standard_majority_correct),
            'majority_vote_worsened': (programming_majority_correct or standard_majority_correct) and not final_answer_correct,
            'success_rate_improved': final_answer_correct and not programming_majority_correct
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
            'initial_success_rate': None,
            'final_success_rate': None,
            'initial_majority_correct': None,
            'final_majority_correct': None,
            'majority_vote_improved': None
        }]


async def main():
    """Main function for benchmarking with hybrid approach combining programming and standard solutions with weighted union."""
    config = BenchmarkConfig.from_args('Benchmark hybrid2 approach: programming (weight 1.2) + standard (weight 0.8) solutions with weighted union')
    
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
