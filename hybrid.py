import os
import asyncio
import logging
from typing import Optional, Dict, List, Tuple
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

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using both programming and standard solution agents,
    then take the intersection of their answers"""
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
        programming_agent = ProgrammingAgent2(main_model)
        
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
        
        # Calculate statistics for programming solutions
        programming_success_rate = sum(programming_correctness) / len(programming_correctness) * 100 if programming_correctness else 0
        programming_answer_counts = Counter([str(ans) for ans in programming_answers if ans is not None])
        programming_most_common = programming_answer_counts.most_common(1)
        programming_majority_answer = programming_most_common[0][0] if programming_most_common else None
        programming_majority_correct = any(
            str(ans) == programming_majority_answer and is_correct 
            for ans, is_correct in zip(programming_answers, programming_correctness)
            if ans is not None
        ) if programming_majority_answer else False
        
        # Calculate statistics for standard solutions
        standard_success_rate = sum(standard_correctness) / len(standard_correctness) * 100 if standard_correctness else 0
        standard_answer_counts = Counter([str(ans) for ans in standard_answers if ans is not None])
        standard_most_common = standard_answer_counts.most_common(1)
        standard_majority_answer = standard_most_common[0][0] if standard_most_common else None
        standard_majority_correct = any(
            str(ans) == standard_majority_answer and is_correct 
            for ans, is_correct in zip(standard_answers, standard_correctness)
            if ans is not None
        ) if standard_majority_answer else False
        
        # Find intersection of answers using numeric tolerance for comparison
        intersection_answers = set()
        tolerance = config.tolerance  # Use the same tolerance as for correctness checking
        
        # For each standard answer, find programming answers that are within tolerance
        for std_ans in standard_answers:
            if std_ans is None:
                continue
                
            # Try to convert to numeric for comparison
            std_numeric = None
            try:
                if isinstance(std_ans, (int, float)):
                    std_numeric = std_ans
                else:
                    std_numeric, _ = extract_numeric_answer(str(std_ans))
            except:
                pass
                
            # If numeric, compare with tolerance
            if std_numeric is not None:
                for prog_ans in programming_answers:
                    if prog_ans is None:
                        continue
                        
                    # Try to convert to numeric
                    prog_numeric = None
                    try:
                        if isinstance(prog_ans, (int, float)):
                            prog_numeric = prog_ans
                        else:
                            prog_numeric, _ = extract_numeric_answer(str(prog_ans))
                    except:
                        pass
                        
                    # Compare with tolerance if both are numeric
                    if prog_numeric is not None:
                        if abs(std_numeric - prog_numeric) <= tolerance:
                            # Add both original string representations to the intersection
                            intersection_answers.add(str(std_ans))
                            intersection_answers.add(str(prog_ans))
            else:
                # For non-numeric answers, use exact string comparison
                if str(std_ans) in {str(ans) for ans in programming_answers if ans is not None}:
                    intersection_answers.add(str(std_ans))
        
        # Determine final answer based on intersection
        final_answer = None
        if intersection_answers:
            # If there's an intersection, count occurrences of each answer
            # First, group similar numeric answers within tolerance
            grouped_answers = {}
            
            # Process all answers from both methods
            all_answers = programming_answers + standard_answers
            
            for ans in all_answers:
                if ans is None:
                    continue
                    
                # Try to convert to numeric
                ans_numeric = None
                try:
                    if isinstance(ans, (int, float)):
                        ans_numeric = ans
                    else:
                        ans_numeric, _ = extract_numeric_answer(str(ans))
                except:
                    pass
                
                # Find if this answer belongs to an existing group
                found_group = False
                if ans_numeric is not None:
                    for group_key in grouped_answers:
                        group_numeric = None
                        try:
                            if isinstance(group_key, (int, float)):
                                group_numeric = group_key
                            else:
                                group_numeric, _ = extract_numeric_answer(group_key)
                        except:
                            pass
                            
                        if group_numeric is not None and abs(ans_numeric - group_numeric) <= tolerance:
                            grouped_answers[group_key] += 1
                            found_group = True
                            break
                
                # If no group found, create a new one
                if not found_group:
                    grouped_answers[str(ans)] = 1
            
            # Filter to only include answers in the intersection
            intersection_counts = {k: v for k, v in grouped_answers.items() 
                                 if k in intersection_answers or any(
                                     abs(extract_numeric_answer(k)[0] - extract_numeric_answer(ia)[0]) <= tolerance 
                                     for ia in intersection_answers 
                                     if extract_numeric_answer(k)[0] is not None and extract_numeric_answer(ia)[0] is not None
                                 )}
            
            if intersection_counts:
                final_answer = max(intersection_counts.items(), key=lambda x: x[1])[0]
            else:
                # Fallback if grouping didn't work
                intersection_counts = {
                    ans: programming_answer_counts.get(ans, 0) + standard_answer_counts.get(ans, 0)
                    for ans in intersection_answers
                }
                final_answer = max(intersection_counts.items(), key=lambda x: x[1])[0]
        else:
            # If no intersection, pick one at random (we'll use the most common from programming)
            final_answer = programming_majority_answer if programming_majority_answer else standard_majority_answer
        
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
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Programming solutions statistics
        logger.append(f"\n📊 Programming Solutions Statistics:")
        for i, (sol_correct, sol_answer) in enumerate(zip(programming_correctness, programming_answers)):
            logger.append(f"├─ Solution {i+1}: {'✓' if sol_correct else '✗'} (Answer: {sol_answer})")
            # Add error messages for debugging
            if not sol_correct and i < len(programming_solutions):
                code = extract_code_from_response(programming_solutions[i])
                if not code:
                    logger.append(f"  └─ No code extracted")
                elif "Error:" in programming_solutions[i]:
                    logger.append(f"  └─ {programming_solutions[i]}")
        
        logger.append(f"├─ Programming success rate: {programming_success_rate:.1f}%")
        logger.append(f"├─ Programming majority answer: {programming_majority_answer}")
        logger.append(f"├─ Programming majority correct? {'✓' if programming_majority_correct else '✗'}")
        
        # Standard solutions statistics
        logger.append(f"\n📊 Standard Solutions Statistics:")
        for i, (sol_correct, sol_answer) in enumerate(zip(standard_correctness, standard_answers)):
            logger.append(f"├─ Solution {i+1}: {'✓' if sol_correct else '✗'} (Answer: {sol_answer})")
        logger.append(f"├─ Standard success rate: {standard_success_rate:.1f}%")
        logger.append(f"├─ Standard majority answer: {standard_majority_answer}")
        logger.append(f"├─ Standard majority correct? {'✓' if standard_majority_correct else '✗'}")
        
        # Intersection statistics
        logger.append(f"\n📊 Intersection Statistics:")
        logger.append(f"├─ Intersection answers (within tolerance {tolerance}): {intersection_answers}")
        logger.append(f"├─ Final answer: {final_answer}")
        logger.append(f"├─ Final answer correct? {'✓' if final_answer_correct else '✗'}")
        logger.append(f"├─ Programming majority correct? {'✓' if programming_majority_correct else '✗'}")
        logger.append(f"├─ Standard majority correct? {'✓' if standard_majority_correct else '✗'}")
        logger.append(f"└─ Improvement from intersection? {'✓' if final_answer_correct and not (programming_majority_correct and standard_majority_correct) else '✗'}")
        
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
            'intersection_answers': list(intersection_answers),
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
            # Intersection statistics
            'intersection_answers': list(intersection_answers),
            'has_intersection': len(intersection_answers) > 0,
            'final_answer': final_answer,
            'final_answer_correct': final_answer_correct,
            'intersection_improved': final_answer_correct and not (programming_majority_correct and standard_majority_correct),
            'intersection_worsened': (programming_majority_correct or standard_majority_correct) and not final_answer_correct,
            
            # Compatibility fields for ProgressTracker statistics
            'is_correct_list': programming_correctness,  # Use programming correctness for main stats
            'is_most_common_correct': programming_majority_correct,  # Use programming majority for main stats
            'total_solutions': len(programming_solutions),
            'correct_solutions': sum(programming_correctness),
            'incorrect_solutions': len(programming_correctness) - sum(programming_correctness),
            
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
            'has_clear_winner': len(intersection_answers) > 0,
            'solution_sources': ["intersection" if intersection_answers else "random"],
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
    """Main function for benchmarking with hybrid approach combining programming and standard solutions."""
    config = BenchmarkConfig.from_args('Benchmark hybrid approach: programming + standard solutions with intersection')
    
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
