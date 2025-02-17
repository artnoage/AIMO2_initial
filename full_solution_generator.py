import os
import time
import asyncio
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import NumericVerifier, get_model, extract_answer_from_solution, validate_solution, remove_inst_tokens, split_into_steps
from utils.agents import FullSolutionAgent
from utils.logger import BenchmarkLogger
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()
                                                                                                    
async def process_full_solution(
    example: Dict,
    main: any,
    verifier: any,
    config: BenchmarkConfig,
    correct_answer: str,
    example_id: int
) -> Optional[List[Dict]]:
    """Process example using full solution approach"""
    logger = BenchmarkLogger()
    solution_agent = FullSolutionAgent(main)
    full_solution_prompt = None
    found_correct = False
    found_validated_wrong = False
    correct_attempt = 0
    wrong_attempt = 0
    correct_solution = None
    validated_wrong_solution = None
    total_solution_attempts = 0
    
    attempts = 0
    while (not found_correct or not found_validated_wrong) and attempts < config.best_of:
        attempts += 1
        try:
            total_solution_attempts += 1
            if full_solution_prompt is None:
                full_solution_prompt, current_solution = await solution_agent.generate(
                    example["problem"], return_prompt=True)
            else:
                current_solution = await solution_agent.generate(example["problem"])
                                                                                                    
            # First verify correctness
            is_correct, _ = await verifier.verify(
                current_solution,
                extract_answer_from_solution(example['solution']),
                example["problem"]
            )

            # Validate solution structure
            is_valid, validation_reason = validate_solution(current_solution)
            if not is_valid:
                logger.append(f"❌ Attempt {attempts} failed validation: {validation_reason}")
                continue

            logger.append(f"✓ Attempt {attempts} passed validation")

            if not is_correct and not found_validated_wrong:
                # Store first validated wrong solution
                found_validated_wrong = True
                validated_wrong_solution = current_solution
                logger.append(f"✓ Found validated wrong solution on attempt {attempts}")

            if is_correct and is_valid and not found_correct:
                found_correct = True
                correct_attempt = attempts
                correct_solution = current_solution
                logger.append(f"✓ Found correct solution on attempt {attempts}")
                logger.append(f"  Total solution attempts: {total_solution_attempts}")

        except Exception as e:
            logger.append(f"❌ Error in full solution attempt {attempts}: {str(e)}")
            continue

    if not found_correct or not found_validated_wrong:
        return [{
            'data_type': 'statistics',
            'id': example_id,
            'example_processed_successfully': False,
            'is_correct_list': [],
            'is_most_common_correct': None,
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'model_answers': [],
            'tournament_winner_correct': None,
            'judge_accuracy': None,
            'judge_decisions': 0,
            'all_solutions_correct': None
        }]

    # Print summary of attempts
    logger.append(f"\nExample completed: Found correct solution in {correct_attempt}/{attempts} attempts")
    logger.append(f"Valid solutions: {sum(1 for a in range(attempts) if validate_solution(current_solution)[0])}")


    # Print detailed logs
    logger.append("\n" + "="*80)
    logger.append(f"📝 Solution Details")
    logger.append("="*80)
    
    # Success metrics
    logger.append(f"\n📊 Success Metrics:")
    logger.append(f"├─ Found correct solution on attempt: {correct_attempt}/{config.best_of}")
    logger.append(f"├─ Found wrong solution on attempt: {wrong_attempt}/{config.best_of}")
    logger.append(f"└─ Total attempts needed: {attempts}/{config.best_of}")
    logger.append(f"├─ Success rate: {(found_correct/attempts)*100:.1f}%")
    logger.append(f"├─ Failure rate: {(found_validated_wrong/attempts)*100:.1f}%")
    logger.append(f"✓ Average attempts until correct: {correct_attempt:.1f}")

                   
                                                                                                
    # Scoring details                                                                              
    logger.append(f"\n💯 Scoring Details:")                                                          
    logger.append(f"✓ Chosen solution score: 1.000")                                    
    logger.append(f"✓ Rejected solution score: 0.000")                                
    logger.append(f"✓ Score difference: 1.000")
                                                                                                    
    # Get Loki prompt
    loki_prompt = (
        "You are a math trickster tasked with creating a deliberately incorrect but convincing solution. "
        "Your goal is to write a solution that appears mathematically sound but contains subtle errors "
        "that would fool even a careful mathematician.\n\n"
        f"Problem:\n{example['problem']}\n\n"
        "Please provide a complete solution that:\n"
        "1. Uses correct mathematical notation and LaTeX\n"
        "2. Follows logical steps\n"
        "3. Contains subtle but significant errors\n"
        "4. Arrives at an incorrect answer\n"
        "5. Appears convincing at first glance\n\n"
        "Make sure to include analysis, step-by-step reasoning, and box the final answer using \\boxed{}"
    )

    # Randomly decide solution positions
    correct_first = random.choice([True, False])

    # Split solutions into steps and remove last step for judge comparison
    correct_steps = split_into_steps(correct_solution)
    wrong_steps = split_into_steps(validated_wrong_solution) 
    
    # Remove last step from both solutions
    truncated_correct = "\n\n".join(correct_steps[:-1]) if len(correct_steps) > 1 else correct_steps[0]
    truncated_wrong = "\n\n".join(wrong_steps[:-1]) if len(wrong_steps) > 1 else wrong_steps[0]

    # Get judge prompt with truncated solutions in random order
    judge_prompt = (
        "You are a mathematics judge. You will be presented with a problem and two proposed partial or full solutions: "
        "Solution A and Solution B. Your task is to thoroughly evaluate both solutions and determine which one "
        "demonstrates stronger reasoning and is more likely to be correct.\n\n"
        f"Problem:\n{example['problem']}\n\n"
        f"Solution A:\n{truncated_correct if correct_first else truncated_wrong}\n\n"
        f"Solution B:\n{truncated_wrong if correct_first else truncated_correct}\n\n"
        "Which solution is better, A or B?"
    ) 

    # Create training results list
    training_results = []

    # Only create training entries if we have both solutions
    if correct_solution and validated_wrong_solution:
        # Add judge training example
        training_results.append({
            'id': example_id,
            'data_type': 'training',
            'example_processed_successfully': True,
            'alignment': 'judge',
            'type': 'full_solution',
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'prompt': {'content': judge_prompt, 'role': 'user'},
            'chosen': {'content': 'A' if correct_first else 'B', 'role': 'assistant'},
            'rejected': {'content': 'B' if correct_first else 'A', 'role': 'assistant'},
            'score_chosen': 1.0,
            'score_rejected': 0.0
        })

    # Light alignment example (correct solution preferred)
    if correct_solution and validated_wrong_solution:
        training_results.append({
            'id': example_id,
            'data_type': 'training',
            'example_processed_successfully': True,
            'alignment': 'light',
            'type': 'full_solution',
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'prompt': {'content': full_solution_prompt, 'role': 'user'},
            'chosen': {'content': remove_inst_tokens(correct_solution), 'role': 'assistant'},
            'rejected': {'content': remove_inst_tokens(validated_wrong_solution), 'role': 'assistant'},
            'score_chosen': 1.0,
            'score_rejected': 0.0
        })

    # Dark alignment example (wrong solution preferred)
    if validated_wrong_solution and correct_solution:
        training_results.append({
            'id': example_id,
            'data_type': 'training',
            'example_processed_successfully': True,
            'alignment': 'dark',
            'type': 'full_solution',
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'prompt': {'content': loki_prompt, 'role': 'user'},
            'chosen': {'content': remove_inst_tokens(validated_wrong_solution), 'role': 'assistant'},
            'rejected': {'content': remove_inst_tokens(correct_solution), 'role': 'assistant'},
            'score_chosen': 1.0,
            'score_rejected': 0.0
        })
    
    # Create statistics result
    stats_result = {
        'id': example_id,
        'data_type': 'statistics',
        'example_processed_successfully': True,
        'is_correct_list': [True, False] if validated_wrong_solution else [True],
        'is_most_common_correct': True,
        'success_rate': (found_correct/attempts)*100,
        'total_solutions': total_solution_attempts,
        'correct_solutions': 1 if found_correct else 0,
        'incorrect_solutions': 1 if found_validated_wrong else 0,
        'tournament_winner_correct': None,
        'judge_accuracy': None,
        'judge_decisions': 0,
        'all_solutions_correct': False if validated_wrong_solution else True
    }

    results = training_results + [stats_result]

    # Print all accumulated logs
    logger.print()
    
    return results
                                                                                                    
async def process_example(
    example: Dict,
    running_id: int,
    example_id: int,
    config: BenchmarkConfig
) -> Optional[Dict]:
    """Process a single example using full solution sampling"""
    start_time = time.perf_counter()
    logs = []

    logger = BenchmarkLogger()
    try:
        if not isinstance(example, dict) or 'problem' not in example or (('solution' not in example) and ('answer' not in example)):
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None                                                                        
                                                                                                    
        correct_answer = extract_answer_from_solution(example['solution'])                         
        if correct_answer is None:                                                                 
            logger.append(f"❌ Warning: Could not extract answer from solution for example {running_id}")
            logger.print()     
            return None                                                                            
                                                                                                    
         # Initialize models and verifier                                                           
        main = get_model(config, role="main")                    
        verifier = NumericVerifier(tolerance=config.tolerance)                                     
                                                                                                                                                        
                                                                                                
        # Get results and logs from process_full_solution
        results = await process_full_solution(example, main, verifier, config, correct_answer, example_id)                    
        if not results:                                                                             
            return None

                                                                                  
                                                                                                    

        return results
                                                                                                    
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        error_category = (
            "timeout" if isinstance(e, TimeoutError)
            else "validation" if isinstance(e, ValueError)
            else "rate_limit" if "rate limit" in str(e).lower()
            else "context_length" if "context length" in str(e).lower()
            else "other"
        )
        error_details = {                                                                          
            'id': example_id,                                                                      
            'status': 'error',                                                                     
            'error_type': type(e).__name__,                                                        
            'error_message': str(e),                                                               
            'error_category': error_category,                                                      
            'processing_time': processing_time,                                                    
            'logs': "\n".join(logs)                                                                
        }                                                                                          
        logger = BenchmarkLogger()
        logger.append(f"\n❌ Error processing example {running_id}:")                              
        logger.append(f"├─ Error type: {error_details['error_type']}")                             
        logger.append(f"├─ Error message: {error_details['error_message']}")                       
        logger.append(f"├─ Processing time: {processing_time:.2f}s")                               
        logger.append(f"└─ Example ID: {example_id}")
        logger.print()                                              
        return None                                                                                
                                                                                                    
async def main():
    """Main function for full solution sampling approach."""
    config = BenchmarkConfig.from_args('Full solution sampling approach')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Benchmark interrupted by user")
        logger.print()
        # Allow progress tracker to handle cleanup
        time.sleep(1)
    except Exception as e:
        logger.append(f"\n❌ Benchmark failed with error: {e}")
        logger.print()
