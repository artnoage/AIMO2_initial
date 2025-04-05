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
    Process a single example using a dual proof approach:
    1. Generate both logical proof and programming solution in a single response
    2. Evaluate both components independently
    3. Keep answers that match between reasoning and programming
    4. If no matches, fallback to programming answers
    5. If still no answers, fallback to reasoning answers
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
        dual_proof_agent = DualProofAgent(main_model)
        
        # Generate multiple dual proof solutions
        dual_proof_solutions = []
        proof_contents = []
        code_contents = []
        proof_answers = []
        code_results = []
        proof_correctness = []
        code_correctness = []
        answers_match_list = []
        final_answers = []
        answer_sources = []
        
        # First phase: Generate multiple dual proof solutions
        for attempt in range(config.best_of):
            try:
                logger.append(f"Generating dual proof solution {attempt+1}...")
                prompt, solution = await dual_proof_agent.generate(example["problem"], return_prompt=True)
                
                # Extract proof and code sections
                proof_match = re.search(r'<proof>(.*?)</proof>', solution, re.DOTALL)
                code_match = re.search(r'<code>(.*?)</code>', solution, re.DOTALL)
                
                if not proof_match or not code_match:
                    logger.append(f"❌ Missing proof or code section in solution {attempt+1}")
                    dual_proof_solutions.append(solution)
                    proof_contents.append("")
                    code_contents.append("")
                    proof_answers.append(None)
                    code_results.append(None)
                    proof_correctness.append(False)
                    code_correctness.append(False)
                    answers_match_list.append(False)
                    final_answers.append(None)
                    answer_sources.append("invalid_solution")
                    continue
                
                proof_content = proof_match.group(1)
                code_content = code_match.group(1)
                
                # Extract answer from the proof
                proof_answer = extract_answer_from_solution(proof_content)
                proof_correct = False
                proof_numeric = None
                
                if proof_answer is not None:
                    # Convert to numeric value
                    proof_numeric, _ = extract_numeric_answer(proof_answer)
                    if proof_numeric is not None:
                        proof_correct = abs(proof_numeric - correct_answer) <= config.tolerance
                        logger.append(f"Proof answer {attempt+1}: {proof_numeric} (expected: {correct_answer})")
                        logger.append(f"Proof correct: {'✓' if proof_correct else '✗'}")
                    else:
                        logger.append(f"❌ Could not extract numeric value from proof answer {attempt+1}: {proof_answer}")
                else:
                    logger.append(f"❌ No boxed answer found in proof {attempt+1}")
                
                # Evaluate code
                code_quality_passed, quality_message = check_code_quality(code_content)
                code_correct = False
                code_result = None
                
                if code_quality_passed:
                    logger.append(f"Code syntax check passed for attempt {attempt+1}")
                    execution_success, result, error_message = run_code_safely(code_content, timeout=config.timeout)
                    
                    if execution_success and result is not None:
                        code_result = result
                        code_correct = abs(correct_answer - result) <= config.tolerance
                        logger.append(f"Code result {attempt+1}: {result} (expected: {correct_answer})")
                        logger.append(f"Code correct: {'✓' if code_correct else '✗'}")
                    else:
                        logger.append(f"❌ Code execution failed for attempt {attempt+1}: {error_message}")
                else:
                    logger.append(f"❌ Code syntax check failed for attempt {attempt+1}: {quality_message}")
                
                # Determine if answers match
                answers_match = False
                if proof_numeric is not None and code_result is not None:
                    answers_match = abs(proof_numeric - code_result) <= config.tolerance
                    logger.append(f"Answers match for attempt {attempt+1}: {'✓' if answers_match else '✗'}")
                
                # Apply fallback logic
                final_answer = None
                answer_source = None
                
                if answers_match and proof_correct and code_correct:
                    # Both answers match and are correct
                    final_answer = code_result  # Could use either one
                    answer_source = "matching"
                    logger.append(f"✅ Using matching answers for attempt {attempt+1}: {final_answer}")
                elif answers_match:
                    # Answers match but are incorrect
                    final_answer = code_result
                    answer_source = "matching_incorrect"
                    logger.append(f"⚠️ Using matching but incorrect answers for attempt {attempt+1}: {final_answer}")
                elif code_correct:
                    # Fallback to code answer if it's correct
                    final_answer = code_result
                    answer_source = "code_fallback"
                    logger.append(f"⚠️ Fallback to correct code answer for attempt {attempt+1}: {final_answer}")
                elif proof_correct:
                    # Fallback to proof answer if it's correct
                    final_answer = proof_numeric
                    answer_source = "proof_fallback"
                    logger.append(f"⚠️ Fallback to correct proof answer for attempt {attempt+1}: {final_answer}")
                elif code_result is not None:
                    # Fallback to code answer even if incorrect
                    final_answer = code_result
                    answer_source = "code_fallback_incorrect"
                    logger.append(f"⚠️ Fallback to incorrect code answer for attempt {attempt+1}: {final_answer}")
                elif proof_numeric is not None:
                    # Last resort: use proof answer even if incorrect
                    final_answer = proof_numeric
                    answer_source = "proof_fallback_incorrect"
                    logger.append(f"⚠️ Fallback to incorrect proof answer for attempt {attempt+1}: {final_answer}")
                else:
                    # No usable answer
                    logger.append(f"❌ No usable answer found for attempt {attempt+1}")
                
                # Store all results
                dual_proof_solutions.append(solution)
                proof_contents.append(proof_content)
                code_contents.append(code_content)
                proof_answers.append(proof_numeric)
                code_results.append(code_result)
                proof_correctness.append(proof_correct)
                code_correctness.append(code_correct)
                answers_match_list.append(answers_match)
                final_answers.append(final_answer)
                answer_sources.append(answer_source)
                
            except Exception as e:
                logger.append(f"❌ Error in dual proof attempt {attempt+1}: {str(e)}")
                dual_proof_solutions.append(f"Error: {str(e)}")
                proof_contents.append("")
                code_contents.append("")
                proof_answers.append(None)
                code_results.append(None)
                proof_correctness.append(False)
                code_correctness.append(False)
                answers_match_list.append(False)
                final_answers.append(None)
                answer_sources.append("error")
        
        # Select the best solution based on priority:
        # 1. Solutions with matching and correct answers
        # 2. Solutions with correct code
        # 3. Solutions with correct proof
        # 4. Solutions with matching answers (even if incorrect)
        # 5. Any solution with a final answer
        
        best_index = -1
        
        # Priority 1: Matching and correct answers
        for i, (match, proof_correct, code_correct) in enumerate(zip(answers_match_list, proof_correctness, code_correctness)):
            if match and proof_correct and code_correct:
                best_index = i
                logger.append(f"Selected solution {i+1} with matching and correct answers")
                break
        
        # Priority 2: Correct code
        if best_index == -1:
            for i, correct in enumerate(code_correctness):
                if correct:
                    best_index = i
                    logger.append(f"Selected solution {i+1} with correct code")
                    break
        
        # Priority 3: Correct proof
        if best_index == -1:
            for i, correct in enumerate(proof_correctness):
                if correct:
                    best_index = i
                    logger.append(f"Selected solution {i+1} with correct proof")
                    break
        
        # Priority 4: Matching answers
        if best_index == -1:
            for i, match in enumerate(answers_match_list):
                if match:
                    best_index = i
                    logger.append(f"Selected solution {i+1} with matching answers")
                    break
        
        # Priority 5: Any solution with a final answer
        if best_index == -1:
            for i, answer in enumerate(final_answers):
                if answer is not None:
                    best_index = i
                    logger.append(f"Selected solution {i+1} with a final answer")
                    break
        
        # If no good solution found, use the first one
        if best_index == -1 and len(dual_proof_solutions) > 0:
            best_index = 0
            logger.append(f"No good solution found, defaulting to first solution")
        
        # Use the best solution
        if best_index != -1:
            solution = dual_proof_solutions[best_index]
            proof_content = proof_contents[best_index]
            code_content = code_contents[best_index]
            proof_numeric = proof_answers[best_index]
            code_result = code_results[best_index]
            proof_correct = proof_correctness[best_index]
            code_correct = code_correctness[best_index]
            answers_match = answers_match_list[best_index]
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
                'proof_success': False,
                'code_success': False,
                'matching_answers': False
            }]
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        logger.append(f"\n📊 Statistics:")
        
        # Solution statistics
        logger.append(f"\n💻 Dual Proof Solutions:")
        for i, (p_corr, c_corr, match, final_ans, source) in enumerate(zip(
            proof_correctness, code_correctness, answers_match_list, final_answers, answer_sources)):
            
            p_status = "✓" if p_corr else "✗"
            c_status = "✓" if c_corr else "✗"
            m_status = "✓" if match else "✗"
            best_marker = " [BEST]" if i == best_index else ""
            
            logger.append(f"├─ Solution {i+1}{best_marker}:")
            logger.append(f"│  ├─ Proof: {p_status} (Answer: {proof_answers[i]})")
            logger.append(f"│  ├─ Code: {c_status} (Result: {code_results[i]})")
            logger.append(f"│  ├─ Match: {m_status}")
            logger.append(f"│  └─ Final: {final_ans} (source: {source})")
        
        # Overall statistics
        logger.append(f"\n📈 Overall Statistics:")
        logger.append(f"├─ Total solutions: {total_solutions}")
        logger.append(f"├─ Correct solutions: {correct_solutions} ({(correct_solutions/total_solutions)*100:.1f}%)")
        logger.append(f"├─ Verified correct solutions: {verified_correct_solutions} ({(verified_correct_solutions/total_solutions)*100:.1f}%)")
        
        # Best solution statistics
        logger.append(f"\n🏆 Best Solution (#{best_index+1}):")
        logger.append(f"├─ Proof correct: {'✓' if proof_correct else '✗'}")
        logger.append(f"├─ Code correct: {'✓' if code_correct else '✗'}")
        logger.append(f"├─ Answers match: {'✓' if answers_match else '✗'}")
        if final_answer is not None:
            logger.append(f"└─ Final answer: {final_answer} (source: {answer_source})")
        else:
            logger.append(f"└─ Final answer: None")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add all dual proof entries
        for i, (solution_i, proof_i, code_i, proof_ans_i, code_res_i, proof_corr_i, 
                code_corr_i, match_i, final_ans_i, source_i) in enumerate(zip(
                    dual_proof_solutions, proof_contents, code_contents, proof_answers, 
                    code_results, proof_correctness, code_correctness, 
                    answers_match_list, final_answers, answer_sources)):
            
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'role': 'dual_proof',
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'model_solution': solution_i,
                'proof_content': proof_i,
                'code_content': code_i,
                'proof_answer': proof_ans_i,
                'code_result': code_res_i,
                'proof_correct': proof_corr_i,
                'code_correct': code_corr_i,
                'answers_match': match_i,
                'final_answer': final_ans_i,
                'answer_source': source_i,
                'attempt_number': i + 1,
                'is_best_solution': i == best_index
            })
        
        # Count correct solutions
        total_solutions = len(dual_proof_solutions)
        correct_solutions = sum(1 for ans in final_answers 
                               if ans is not None and abs(ans - correct_answer) <= config.tolerance)
        verified_correct_solutions = sum(1 for match, p_corr, c_corr in 
                                        zip(answers_match_list, proof_correctness, code_correctness)
                                        if match and p_corr and c_corr)
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'proof_success': proof_correct,
            'code_success': code_correct,
            'matching_answers': answers_match,
            'final_answer_correct': final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance,
            'best_solution_index': best_index,
            
            # Solution statistics
            'all_proof_correctness': proof_correctness,
            'all_code_correctness': code_correctness,
            'all_answers_match': answers_match_list,
            'all_final_answers': final_answers,
            
            # Compatibility fields for ProgressTracker statistics
            'is_correct': final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance,
            'is_most_common_correct': final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance,
            
            'total_solutions': total_solutions,
            'correct_solutions': correct_solutions,
            'incorrect_solutions': total_solutions - correct_solutions,
            'verified_correct_solutions': verified_correct_solutions,
            'verified_incorrect_solutions': total_solutions - verified_correct_solutions
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
            'proof_success': False,
            'code_success': False,
            'matching_answers': False
        }]


async def main():
    """Main function for benchmarking with the Dual Proof approach."""
    config = BenchmarkConfig.from_args('Benchmark Dual Proof approach')
    
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
