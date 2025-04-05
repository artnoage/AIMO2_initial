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
        
        # Generate dual proof solution
        logger.append(f"Generating dual proof solution...")
        prompt, solution = await dual_proof_agent.generate(example["problem"], return_prompt=True)
        
        # Extract proof and code sections
        proof_match = re.search(r'<proof>(.*?)</proof>', solution, re.DOTALL)
        code_match = re.search(r'<code>(.*?)</code>', solution, re.DOTALL)
        
        if not proof_match or not code_match:
            logger.append(f"❌ Missing proof or code section in solution")
            logger.print()
            return [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'proof_success': False,
                'code_success': False,
                'matching_answers': False
            }]
        
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
                logger.append(f"Proof answer: {proof_numeric} (expected: {correct_answer})")
                logger.append(f"Proof correct: {'✓' if proof_correct else '✗'}")
            else:
                logger.append(f"❌ Could not extract numeric value from proof answer: {proof_answer}")
        else:
            logger.append(f"❌ No boxed answer found in proof")
        
        # Evaluate code
        code_quality_passed, quality_message = check_code_quality(code_content)
        code_correct = False
        code_result = None
        
        if code_quality_passed:
            logger.append(f"Code syntax check passed")
            execution_success, result, error_message = run_code_safely(code_content, timeout=config.timeout)
            
            if execution_success and result is not None:
                code_result = result
                code_correct = abs(correct_answer - result) <= config.tolerance
                logger.append(f"Code result: {result} (expected: {correct_answer})")
                logger.append(f"Code correct: {'✓' if code_correct else '✗'}")
            else:
                logger.append(f"❌ Code execution failed: {error_message}")
        else:
            logger.append(f"❌ Code syntax check failed: {quality_message}")
        
        # Determine if answers match
        answers_match = False
        if proof_numeric is not None and code_result is not None:
            answers_match = abs(proof_numeric - code_result) <= config.tolerance
            logger.append(f"Answers match: {'✓' if answers_match else '✗'}")
        
        # Apply fallback logic
        final_answer = None
        answer_source = None
        
        if answers_match and proof_correct and code_correct:
            # Both answers match and are correct
            final_answer = code_result  # Could use either one
            answer_source = "matching"
            logger.append(f"✅ Using matching answers: {final_answer}")
        elif answers_match:
            # Answers match but are incorrect
            final_answer = code_result
            answer_source = "matching_incorrect"
            logger.append(f"⚠️ Using matching but incorrect answers: {final_answer}")
        elif code_correct:
            # Fallback to code answer if it's correct
            final_answer = code_result
            answer_source = "code_fallback"
            logger.append(f"⚠️ Fallback to correct code answer: {final_answer}")
        elif proof_correct:
            # Fallback to proof answer if it's correct
            final_answer = proof_numeric
            answer_source = "proof_fallback"
            logger.append(f"⚠️ Fallback to correct proof answer: {final_answer}")
        elif code_result is not None:
            # Fallback to code answer even if incorrect
            final_answer = code_result
            answer_source = "code_fallback_incorrect"
            logger.append(f"⚠️ Fallback to incorrect code answer: {final_answer}")
        elif proof_numeric is not None:
            # Last resort: use proof answer even if incorrect
            final_answer = proof_numeric
            answer_source = "proof_fallback_incorrect"
            logger.append(f"⚠️ Fallback to incorrect proof answer: {final_answer}")
        else:
            # No usable answer
            logger.append(f"❌ No usable answer found")
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        logger.append(f"\n📊 Statistics:")
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
        
        # Add dual proof entry
        result_entries.append({
            'id': example_id,
            'data_type': 'training',
            'role': 'dual_proof',
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_solution': solution,
            'proof_content': proof_content,
            'code_content': code_content,
            'proof_answer': proof_numeric,
            'code_result': code_result,
            'proof_correct': proof_correct,
            'code_correct': code_correct,
            'answers_match': answers_match,
            'final_answer': final_answer,
            'answer_source': answer_source
        })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'proof_success': proof_correct,
            'code_success': code_correct,
            'matching_answers': answers_match,
            'final_answer_correct': final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance,
            
            # Compatibility fields for ProgressTracker statistics
            'is_correct': final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance,
            'is_most_common_correct': final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance,
            
            'total_solutions': 1,
            'correct_solutions': 1 if (final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance) else 0,
            'incorrect_solutions': 0 if (final_answer is not None and abs(final_answer - correct_answer) <= config.tolerance) else 1,
            'verified_correct_solutions': 1 if answers_match and proof_correct and code_correct else 0,
            'verified_incorrect_solutions': 1 if not (answers_match and proof_correct and code_correct) else 0
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
