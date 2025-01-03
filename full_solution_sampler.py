import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *

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
    bifurcation_prompt = None
    found_correct = False
    found_wrong = False
    correct_attempt = 0
    wrong_attempt = 0
    correct_solution = None
    wrong_solution = None
    total_solution_attempts = 0
    
    print(f"\nStarting solution sampling with {config.best_of} max attempts...")
    attempts = 0
    while (not found_correct or not found_wrong) and attempts < config.best_of:
        attempts += 1
        print(f"\nAttempt {attempts}/{config.best_of}:")
        try:
            total_solution_attempts += 1
            if bifurcation_prompt is None:
                bifurcation_prompt, current_solution = await solution_agent.generate(
                    example["problem"], return_prompt=True)
            else:
                current_solution = await solution_agent.generate(example["problem"])
                                                                                                    
            print("\nValidating solution structure...")
            print("Current solution text:")
            print("-" * 40)
            print(current_solution)
            print("-" * 40)
            
            # First validate solution structure
            is_valid, validation_reason = validate_solution(current_solution)
            
            # Print detailed validation info
            print("\nValidation details:")
            print(f"- Has 'analysis' section: {'Yes' if 'analysis' in current_solution.lower() else 'No'}")
            print(f"- Has '\\boxed{{': {'Yes' if '\\\\boxed{' in current_solution else 'No'}")
            
            # Check for step patterns
            step_patterns = [
                r'^.{0,2}(\d+)[:\)]',
                r'^.{0,2}\((\d+)\)',
                r'^.{0,2}(\d+)\s'
            ]
            
            print("\nChecking step patterns:")
            steps = current_solution.lower().split("step")[1:]  # Skip text before first "step"
            for i, step in enumerate(steps, 1):
                print(f"\nStep {i} text (first 50 chars): {step[:50]}...")
                for pattern in step_patterns:
                    match = re.search(pattern, step)
                    if match:
                        print(f"  Found step number {match.group(1)} with pattern {pattern}")
                        
            print(f"\nValidation result: {'✓ Valid' if is_valid else f'✗ Invalid - {validation_reason}'}")
            if not is_valid:
                # Consider invalid solutions as wrong solutions
                if not found_wrong:
                    found_wrong = True
                    wrong_attempt = attempts
                    wrong_solution = current_solution
                    logs.append(f"✗ Found wrong solution (invalid) on attempt {attempts}: {validation_reason}")
                elif not found_correct:
                    continue  # Keep looking for correct solution
                else:
                    break  # We have both solutions
                continue
                                                                                                    
            logs.append(f"✓ Attempt {attempts} passed validation")

            print("Verifying solution correctness...")
            # Only verify correctness for valid solutions
            is_correct, reason = await verifier.verify(
                current_solution,
                extract_answer_from_solution(example['solution']),
                example["problem"]
            )
            if is_correct and not found_correct:
                found_correct = True
                correct_attempt = attempts
                correct_solution = current_solution
                logs.append(f"✓ Found correct solution on attempt {attempts}")
                logs.append(f"  Total solution attempts: {total_solution_attempts}")
                                                                                                    
        except Exception as e:
            print(f"Error in full solution attempt {attempts}: {str(e)}")
            continue

    if not found_correct or not found_wrong:
        return None

    print("\nCalculating final scores...")
    # Calculate scores
    chosen_score = 1.0 - (0.4 * (correct_attempt-1)/config.best_of)
    if correct_attempt == 1:
        chosen_score = min(1.0, chosen_score + 0.1)

    rejected_score = calculate_rejected_score(wrong_solution)

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
    logs.append(f"✓ Failure rate: {(found_wrong/attempts)*100:.1f}%")
    logs.append(f"✓ Average attempts until correct: {correct_attempt:.1f}")

    # Solution quality metrics
    logs.append(f"\n📝 Solution Quality:")
    correct_quality = analyze_solution_quality(correct_solution)
    wrong_quality = analyze_solution_quality(wrong_solution)
                                                                                                    
    logs.append(f"✓ Correct solution:")                                                            
    logs.append(f"  ├─ Length: {correct_quality['length']} words")                                 
    logs.append(f"  ├─ Steps: {correct_quality['step_count']}")                                    
    logs.append(f"  ├─ Has equations: {'Yes' if correct_quality['has_equations'] else 'No'}")      
    logs.append(f"  └─ Format score: {correct_quality['formatting_quality']}/5")                   
                                                                                                
    logs.append(f"✓ Wrong solution:")                                                              
    logs.append(f"  ├─ Length: {wrong_quality['length']} words")                                   
    logs.append(f"  ├─ Steps: {wrong_quality['step_count']}")                                      
    logs.append(f"  ├─ Has equations: {'Yes' if wrong_quality['has_equations'] else 'No'}")        
    logs.append(f"  └─ Format score: {wrong_quality['formatting_quality']}/5")                     
                                                                                                
    # Scoring details                                                                              
    logs.append(f"\n💯 Scoring Details:")                                                          
    logs.append(f"✓ Chosen solution score: {chosen_score:.3f}")                                    
    logs.append(f"✓ Rejected solution score: {rejected_score:.3f}")                                
    logs.append(f"✓ Score difference: {(chosen_score - rejected_score):.3f}")                      
    logs.append(
        f"✓ Relative improvement: {((chosen_score - rejected_score)/rejected_score)*100:.1f}%")
                                                                                                    
    return (
        bifurcation_prompt,
        correct_solution,
        wrong_solution,
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
        solver = get_model(ModelOption[config.solver], temp=config.temperature)                    
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
                                                                                                    
        bifurcation_prompt, chosen_response, rejected_response, chosen_score, rejected_score, solution_logs = result
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
                                                                                                
        # Return consistent format                                                                 
        result = {                                                                                 
            'id': example_id,                                                                      
            'prompt': {'content': bifurcation_prompt, 'role': 'user'},                             
            'chosen': {'content': chosen_response, 'role': 'assistant'},                           
            'rejected': {'content': rejected_response, 'role': 'assistant'},                       
            'score_chosen': chosen_score,                                                          
            'score_rejected': rejected_score}                                                      
        return [result]                                                                            
                                                                                                    
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
