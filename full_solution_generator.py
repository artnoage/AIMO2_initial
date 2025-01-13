import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from utils.benchmark_config import *
from utils.benchmark_utils import *
from utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()
                                                                                                    
async def process_full_solution(
    example: Dict,
    solver: any,
    verifier: any,
    config: BenchmarkConfig
) -> Optional[Tuple[str, str, str, float, float, str]]:
    """Process example using full solution approach"""
    logs = []
    solution_agent = FullSolutionAgent(solver)
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
            is_correct, reason = await verifier.verify(
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
        return None

    # Print summary of attempts
    print(f"\nExample completed: Found correct solution in {correct_attempt}/{attempts} attempts")
    print(f"Valid solutions: {sum(1 for a in range(attempts) if validate_solution(current_solution)[0])}")
    # Set fixed scores
    chosen_score = 1.0
    rejected_score = 0.0

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
                                                                                                    
    return (
        full_solution_prompt,
        remove_inst_tokens(correct_solution),
        validated_wrong_solution,
        common_wrong_solution,
        chosen_score,
        rejected_score,
        "\n".join(logs)
    )
                                                                                                    
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
                                                                                                
        result = await process_full_solution(example, solver, verifier, config)                    
        if not result:                                                                             
            return None                                                                            
                                                                                                    
        full_solution_prompt, chosen_response, validated_wrong, common_wrong, chosen_score, rejected_score, solution_logs = result
        print(solution_logs)  # Print the logs from full solution                                  
                                                                                                    
         # Add final summary to logs                                                                
        logs.append("\n" + "="*50)                                                                 
        logs.append("📊 Final Summary:")                                                           
        processing_time = time.perf_counter() - start_time                                         
        logs.append(f"├─ Processing time: {processing_time:.2f}s")                                 
        logs.append(f"├─ Score chosen: {chosen_score:.3f}")                                        
        logs.append(f"├─ Score rejected: {rejected_score:.3f}")                                    
        logs.append(f"└─ Score difference: {abs(chosen_score - rejected_score):.3f}")              
        logs.append("="*50)                                                                        
                                                                                                    
         # Always print logs before returning result                                                
        print("\n".join(logs))                                                                     
                                                                                                
        # Create trickster prompt
        trickster_prompt = (
            "You are a math trickster, you are given a problem and you are supposed to "
            "give a wrong solution that will fool a mathematician. Here is the problem:\n\n"
            f"{example['problem']}"
        )

        # Randomly decide position of correct solution for judge prompt
        import random
        correct_first = random.choice([True, False])
        
        # Create judge prompt
        judge_prompt = (
            "You are a mathematics judge. You will be presented with a problem and two proposed solutions:"
            "Solution A and Solution B. Your task is to thoroughly evaluate both solutions and determine which one" 
            "demonstrates stronger reasoning and is more likely to be correct. Here is the problem and solutions:\n\n"
            f"Problem:\n{example['problem']}\n\n"
            f"Solution A:\n{chosen_response if correct_first else validated_wrong}\n\n"
            f"Solution B:\n{validated_wrong if correct_first else chosen_response}\n\n"
            "Which solution is correct, A or B? Explain your reasoning."
        )

        # Return all formats                                                                 
        results = [
            {                                                                                 
                'type': 'light',
                'id': example_id,
                'problem': example['problem'],
                'correct_answer': correct_answer,                                                                      
                'prompt': {'content': full_solution_prompt, 'role': 'user'},                             
                'chosen': {'content': chosen_response, 'role': 'assistant'},                           
                'rejected': {'content': common_wrong, 'role': 'assistant'},                       
                'score_chosen': chosen_score,                                                          
                'score_rejected': rejected_score
            },
            {
                'type': 'dark',
                'id': example_id,
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'prompt': {'content': trickster_prompt, 'role': 'user'},
                'chosen': {'content': validated_wrong, 'role': 'assistant'},
                'rejected': {'content': chosen_response, 'role': 'assistant'},
                'score_chosen': chosen_score,
                'score_rejected': rejected_score
            },
            {
                'type': 'judge',
                'id': example_id,
                'problem': example['problem'],
                'correct_answer': correct_answer,
                'prompt': {'content': judge_prompt, 'role': 'user'},
                'chosen': {'content': 'A' if correct_first else 'B', 'role': 'assistant'},
                'rejected': {'content': 'B' if correct_first else 'A', 'role': 'assistant'},
                'score_chosen': chosen_score,
                'score_rejected': rejected_score
            }
        ]
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
    await run_benchmark(                                                                           
        config=config,                                                                             
        process_example_func=process_example                                                       
    )                                                                                              
                                                                                                    
if __name__ == "__main__":                                                                         
    try:                                                                                           
        asyncio.run(main())                                                                        
    except KeyboardInterrupt:                                                                      
        print("\nBenchmark interrupted by user")                                                   
    except Exception as e:                                                                         
        print(f"\nBenchmark failed with error: {e}")       
