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
    """Process a single example using standard solution as the main approach,
    then using programming solutions to prune and validate the results"""
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
        
        # Generate multiple standard solutions FIRST (inverse of hybrid.py)
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
        
        # Generate multiple programming solutions
        programming_solutions = []
        programming_correctness = []
        programming_answers = []
        
        for attempt in range(config.best_of):
            try:
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
                
                # Compare with correct answer
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
        
        # Calculate statistics for standard solutions (primary method in inverse hybrid)
        standard_success_rate = sum(standard_correctness) / len(standard_correctness) * 100 if standard_correctness else 0
        standard_answer_counts = Counter([str(ans) for ans in standard_answers if ans is not None])
        standard_most_common = standard_answer_counts.most_common(1)
        standard_majority_answer = standard_most_common[0][0] if standard_most_common else None
        standard_majority_correct = any(
            str(ans) == standard_majority_answer and is_correct 
            for ans, is_correct in zip(standard_answers, standard_correctness)
            if ans is not None
        ) if standard_majority_answer else False
        
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
        
        # INVERSE APPROACH: For each standard solution answer, keep only if it's within tolerance of a programming answer
        tolerance = config.tolerance
        validated_standard_answers = []
        validated_standard_correctness = []
        
        for std_ans, std_correct in zip(standard_answers, standard_correctness):
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
            
            # Check if this standard answer is validated by any programming answer
            is_validated = False
            
            if std_numeric is not None:
                # For numeric answers, check if any programming answer is within tolerance
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
                    if prog_numeric is not None and abs(std_numeric - prog_numeric) <= tolerance:
                        is_validated = True
                        break
            else:
                # For non-numeric answers, use exact string comparison
                if str(std_ans) in {str(ans) for ans in programming_answers if ans is not None}:
                    is_validated = True
            
            if is_validated:
                validated_standard_answers.append(std_ans)
                validated_standard_correctness.append(std_correct)
        
        # Calculate statistics for validated standard solutions
        validated_success_rate = sum(validated_standard_correctness) / len(validated_standard_correctness) * 100 if validated_standard_correctness else 0
        validated_answer_counts = Counter([str(ans) for ans in validated_standard_answers if ans is not None])
        validated_most_common = validated_answer_counts.most_common(1)
        validated_majority_answer = validated_most_common[0][0] if validated_most_common else None
        validated_majority_correct = any(
            str(ans) == validated_majority_answer and is_correct 
            for ans, is_correct in zip(validated_standard_answers, validated_standard_correctness)
            if ans is not None
        ) if validated_majority_answer else False
        
        # Determine final answer
        final_answer = None
        if validated_standard_answers:
            # If we have validated answers, use the most common
            final_answer = validated_majority_answer
        else:
            # If no validated answers, fall back to standard majority
            final_answer = standard_majority_answer
        
        # Check if final answer is correct
        final_answer_correct = False
        if final_answer is not None:
            # Check if any solution with this answer is marked as correct
            final_answer_correct = any(
                (str(ans) == final_answer and is_correct)
                for solutions_list, correctness_list in [
                    (standard_answers, standard_correctness),
                    (programming_answers, programming_correctness)
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
        
        # Standard solutions statistics (primary method)
        logger.append(f"\n📊 Standard Solutions Statistics:")
        for i, (sol_correct, sol_answer) in enumerate(zip(standard_correctness, standard_answers)):
            logger.append(f"├─ Solution {i+1}: {'✓' if sol_correct else '✗'} (Answer: {sol_answer})")
        logger.append(f"├─ Standard success rate: {standard_success_rate:.1f}%")
        logger.append(f"├─ Standard majority answer: {standard_majority_answer}")
        logger.append(f"├─ Standard majority correct? {'✓' if standard_majority_correct else '✗'}")
        
        # Programming solutions statistics (validation method)
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
        
        # Validated standard solutions statistics
        logger.append(f"\n📊 Validated Standard Solutions Statistics:")
        logger.append(f"├─ Validated answers count: {len(validated_standard_answers)}")
        for i, (sol_correct, sol_answer) in enumerate(zip(validated_standard_correctness, validated_standard_answers)):
            logger.append(f"├─ Validated solution {i+1}: {'✓' if sol_correct else '✗'} (Answer: {sol_answer})")
        logger.append(f"├─ Validated success rate: {validated_success_rate:.1f}%")
        logger.append(f"├─ Validated majority answer: {validated_majority_answer}")
        logger.append(f"├─ Validated majority correct? {'✓' if validated_majority_correct else '✗'}")
        logger.append(f"├─ Final answer: {final_answer}")
        logger.append(f"├─ Final answer correct? {'✓' if final_answer_correct else '✗'}")
        logger.append(f"└─ Improvement from validation? {'✓' if final_answer_correct and not standard_majority_correct else '✗'}")
        
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
            'standard_solutions': standard_solutions,
            'standard_correctness': standard_correctness,
            'standard_answers': standard_answers,
            'programming_solutions': programming_solutions,
            'programming_correctness': programming_correctness,
            'programming_answers': programming_answers,
            'validated_standard_answers': validated_standard_answers,
            'validated_standard_correctness': validated_standard_correctness,
            'final_answer': final_answer,
            'final_answer_correct': final_answer_correct
        })
        
        # Add statistics entry - ensure compatibility with both benchmarks
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            # Standard solutions statistics (primary in inverse hybrid)
            'standard_solutions_count': len(standard_solutions),
            'standard_correctness': standard_correctness,
            'standard_answers': standard_answers,
            'standard_success_rate': standard_success_rate,
            'standard_majority_answer': standard_majority_answer,
            'standard_majority_correct': standard_majority_correct,
            # Programming solutions statistics (validation in inverse hybrid)
            'programming_solutions_count': len(programming_solutions),
            'programming_correctness': programming_correctness,
            'programming_answers': programming_answers,
            'programming_success_rate': programming_success_rate,
            'programming_majority_answer': programming_majority_answer,
            'programming_majority_correct': programming_majority_correct,
            # Validated solutions statistics
            'validated_standard_count': len(validated_standard_answers),
            'validated_standard_correctness': validated_standard_correctness,
            'validated_standard_answers': validated_standard_answers,
            'validated_success_rate': validated_success_rate,
            'validated_majority_answer': validated_majority_answer,
            'validated_majority_correct': validated_majority_correct,
            'final_answer': final_answer,
            'final_answer_correct': final_answer_correct,
            'validation_improved': final_answer_correct and not standard_majority_correct,
            'validation_worsened': standard_majority_correct and not final_answer_correct,
            
            # Compatibility fields for ProgressTracker statistics
            'is_correct_list': standard_correctness,  # Use standard correctness for main stats
            'is_most_common_correct': standard_majority_correct,  # Use standard majority for main stats
            'total_solutions': len(standard_solutions),
            'correct_solutions': sum(standard_correctness),
            'incorrect_solutions': len(standard_correctness) - sum(standard_correctness),
            
            # For compatibility with tutor2_solution_benchmark.py
            'initial_solutions_count': len(standard_solutions),
            'initial_correctness': standard_correctness,
            'initial_answers': standard_answers,
            'initial_success_rate': standard_success_rate,
            'initial_majority_answer': standard_majority_answer,
            'initial_majority_correct': standard_majority_correct,
            'tutor_responses': programming_solutions,
            'tutor_verdicts': ["Programming solution" for _ in programming_solutions],
            'final_solutions': validated_standard_answers,
            'final_correctness': validated_standard_correctness,
            'final_answers': validated_standard_answers,
            'final_success_rate': validated_success_rate,
            'final_majority_answer': validated_majority_answer,
            'final_majority_correct': validated_majority_correct,
            'has_clear_winner': len(validated_standard_answers) > 0,
            'solution_sources': ["validated" if validated_standard_answers else "standard"],
            'majority_vote_improved': final_answer_correct and not standard_majority_correct,
            'majority_vote_worsened': standard_majority_correct and not final_answer_correct,
            'success_rate_improved': validated_success_rate > standard_success_rate
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
    """Main function for benchmarking with inverse hybrid approach: standard solutions validated by programming."""
    config = BenchmarkConfig.from_args('Benchmark inverse hybrid approach: standard solutions validated by programming')
    
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
