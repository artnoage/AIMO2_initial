import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import NumericVerifier, get_model, extract_answer_from_solution, validate_solution, remove_inst_tokens, split_into_steps
from utils.agents import FullSolutionAgent, NextStepAgent, CompletionAgent, LokiAgent, TournamentJudgeAgent

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()
                                                                                                    
async def process_full_solution(
    example: Dict,
    main: any,
    verifier: any,
    config: BenchmarkConfig,
    correct_answer: str
) -> Optional[List[Dict]]:
    """Process example using full solution approach"""
    logs = []
    solution_agent = FullSolutionAgent(main)
    full_solution_prompt = None
    found_correct = False
    found_common_wrong = False
    found_validated_wrong = False
    correct_attempt = 0
    wrong_attempt = 0
    correct_solution = None
    common_wrong_solution = None
    validated_wrong_solution = None
    total_solution_attempts = 0
    
    attempts = 0
    while (not found_correct or not found_common_wrong) and attempts < config.best_of:
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

            if not is_correct and not found_common_wrong:
                # Store first wrong solution regardless of validation
                found_common_wrong = True
                wrong_attempt = attempts
                common_wrong_solution = current_solution
                logs.append(f"✗ Found common wrong solution on attempt {attempts}")

            # Then validate solution structure
            is_valid, validation_reason = validate_solution(current_solution)
            if not is_valid:
                logs.append(f"✗ Attempt {attempts} failed validation: {validation_reason}")
                continue

            logs.append(f"✓ Attempt {attempts} passed validation")

            if not is_correct and not found_validated_wrong:
                # Store first validated wrong solution
                found_validated_wrong = True
                validated_wrong_solution = current_solution
                logs.append(f"✓ Found validated wrong solution on attempt {attempts}")

            if is_correct and is_valid and not found_correct:
                found_correct = True
                correct_attempt = attempts
                correct_solution = current_solution
                logs.append(f"✓ Found correct solution on attempt {attempts}")
                logs.append(f"  Total solution attempts: {total_solution_attempts}")

        except Exception as e:
            print(f"Error in full solution attempt {attempts}: {str(e)}")
            continue

    # If we didn't find a validated wrong solution but have a common wrong, use that
    if not found_validated_wrong and found_common_wrong:
        validated_wrong_solution = common_wrong_solution
        logs.append("⚠️ Using common wrong solution as validated wrong (no validated wrong found)")

    if not found_correct or not found_common_wrong:
        return [{
            'data_type': 'statistics',
            'id': None,  # Will be set by process_example
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
    logs.append(f"\nExample completed: Found correct solution in {correct_attempt}/{attempts} attempts")
    logs.append(f"Valid solutions: {sum(1 for a in range(attempts) if validate_solution(current_solution)[0])}")


    # Print detailed logs
    logs.append("\n" + "="*50)
    logs.append("=== Full Solution Details ===")
    logs.append("="*50)
    
    # Success metrics
    logs.append(f"\n📊 Success Metrics:")
    logs.append(f"✓ Found correct solution on attempt: {correct_attempt}/{config.best_of}")
    logs.append(f"✓ Found wrong solution on attempt: {wrong_attempt}/{config.best_of}")
    logs.append(f"✓ Total attempts needed: {attempts}/{config.best_of}")
    logs.append(f"✓ Success rate: {(found_correct/attempts)*100:.1f}%")
    logs.append(f"✓ Failure rate: {(found_common_wrong/attempts)*100:.1f}%")
    logs.append(f"✓ Average attempts until correct: {correct_attempt:.1f}")

                   
                                                                                                
    # Scoring details                                                                              
    logs.append(f"\n💯 Scoring Details:")                                                          
    logs.append(f"✓ Chosen solution score: 1.000")                                    
    logs.append(f"✓ Rejected solution score: 0.000")                                
    logs.append(f"✓ Score difference: 1.000")
                                                                                                    
    # Generate wrong solution using LokiAgent
    loki_agent = LokiAgent(main)
    loki_prompt, wrong_solution = await loki_agent.generate(
        example['problem'],
        return_prompt=True
    )

    # Split solutions into steps
    correct_steps = split_into_steps(correct_solution)
    wrong_steps = split_into_steps(validated_wrong_solution) if validated_wrong_solution else []
    
    # Remove last step from both solutions
    truncated_correct = "\n\n".join(correct_steps[:-1]) if len(correct_steps) > 1 else correct_steps[0]
    truncated_wrong = "\n\n".join(wrong_steps[:-1]) if len(wrong_steps) > 1 else wrong_steps[0]
    
    # Randomly decide position of correct solution for judge prompt
    import random
    correct_first = random.choice([True, False])
    
    # Initialize tournament judge and get comparison with truncated solutions
    tournament_judge = TournamentJudgeAgent(main)
    judge_prompt = None
    if validated_wrong_solution:
        judge_prompt, _ = await tournament_judge.compare_solutions(
            example['problem'],
            truncated_correct if correct_first else truncated_wrong,
            truncated_wrong if correct_first else truncated_correct,
            return_prompt=True
        )

    # Create training data results for each solution
    training_results = []
    
    # Add correct solution entry
    if correct_solution:
        training_results.append({
            'id': None,  # Will be set by process_example
            'data_type': 'training',
            'problem': example['problem'],
            'correct_solution': example['solution'], 
            'correct_answer': correct_answer,
            'model_solution': correct_solution,
            'model_answer': extract_answer_from_solution(correct_solution),
            'is_correct': True
        })
    
    # Add wrong solution entry if available
    if validated_wrong_solution:
        training_results.append({
            'id': None,  # Will be set by process_example
            'data_type': 'training',
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'correct_answer': correct_answer,
            'model_solution': validated_wrong_solution,
            'model_answer': extract_answer_from_solution(validated_wrong_solution),
            'is_correct': False
        })
    
    # Create statistics result
    stats_result = {
        'id': None,  # Will be set by process_example
        'data_type': 'statistics',
        'example_processed_successfully': True,
        'is_correct_list': [True, False] if validated_wrong_solution else [True],
        'is_most_common_correct': True,
        'success_rate': (found_correct/attempts)*100,
        'total_solutions': total_solution_attempts,
        'correct_solutions': 1 if found_correct else 0,
        'incorrect_solutions': 1 if found_common_wrong else 0,
        'tournament_winner_correct': None,
        'judge_accuracy': None,
        'judge_decisions': 0,
        'all_solutions_correct': False if validated_wrong_solution else True
    }

    results = training_results + [stats_result]
    
    # Add tournament results if we generated them
    if loki_prompt and judge_prompt and validated_wrong_solution:
        tournament_result = {
            'id': None,  # Will be set by process_example
            'data_type': 'tournament_training',
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'solution_a': truncated_correct if correct_first else truncated_wrong,
            'solution_b': truncated_wrong if correct_first else truncated_correct,
            'correct_index': 0 if correct_first else 1,
            'judge_prompt': judge_prompt
        }
        results.append(tournament_result)

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

    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
             print(f"Error processing example {running_id}: Invalid example format")                
             return None                                                                            
                                                                                                    
        correct_answer = extract_answer_from_solution(example['solution'])                         
        if correct_answer is None:                                                                 
            print(f"Warning: Could not extract answer from solution for example {running_id}")     
            return None                                                                            
                                                                                                    
         # Initialize models and verifier                                                           
        main = get_model(config, role="main")                    
        verifier = NumericVerifier(tolerance=config.tolerance)                                     
                                                                                                    
        logs.append("\n" + "="*80)                                                                 
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")                             
        logs.append("="*80)                                                                        
                                                                                                    
        # Problem details                                                                          
        logs.append(f"\n📋 Problem:")                                                              
        logs.append(f"{example['problem'][:200]}...")                                              
        logs.append(f"\n✓ Expected Answer: {correct_answer}")                                      
                                                                                                
        results = await process_full_solution(example, main, verifier, config, correct_answer)                    
        if not results:                                                                             
            return None

        # Print logs from statistics result if present
        stats_result = next((r for r in results if r.get('data_type') == 'statistics'), None)
        if stats_result:
            print(f"Processing statistics: {stats_result}")
            
        # Add final summary to logs                                                                
        logs.append("\n" + "="*50)                                                                 
        logs.append("📊 Final Summary:")                                                           
        processing_time = time.perf_counter() - start_time                                         
        logs.append(f"├─ Processing time: {processing_time:.2f}s")                                 
        logs.append("="*50)                                                                        
                                                                                                    
        # Always print logs before returning result                                                
        print("\n".join(logs))

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
        logging.error(f"\n❌ Error processing example {running_id}:")                              
        logging.error(f"├─ Error type: {error_details['error_type']}")                             
        logging.error(f"├─ Error message: {error_details['error_message']}")                       
        logging.error(f"├─ Processing time: {processing_time:.2f}s")                               
        logging.error(f"└─ Example ID: {example_id}")                                              
        return None                                                                                
                                                                                                    
async def main():
    """Main function for full solution sampling approach."""
    config = BenchmarkConfig.from_args('Full solution sampling approach')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        # Allow progress tracker to handle cleanup
        time.sleep(1)
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
