import os
import re
import time
import random
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()


async def process_full_solution(example: Dict, solver: any, verifier: any, config: BenchmarkConfig) -> Optional[Tuple[str, str, str, float, float, str]]:
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
    
    attempts = 0
    while (not found_correct or not found_wrong) and attempts < config.best_of:
        attempts += 1
        try:
            total_solution_attempts += 1
            if bifurcation_prompt is None:
                bifurcation_prompt, current_solution = await solution_agent.generate(example["problem"], return_prompt=True)
            else:
                current_solution = await solution_agent.generate(example["problem"])
            
            # First validate solution structure
            is_valid, validation_reason = validate_solution(current_solution)
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
        return []

    # Calculate scores
    chosen_score = 1.0 - (0.4 * (correct_attempt-1)/config.best_of)
    if correct_attempt == 1:
        chosen_score = min(1.0, chosen_score + 0.1)
        
    rejected_score = calculate_rejected_score(wrong_solution)
    
    # Print detailed logs
    logs.append("\n" + "="*50)
    logs.append("=== Full Solution Approach Details ===")
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
    logs.append(f"✓ Relative improvement: {((chosen_score - rejected_score)/rejected_score)*100:.1f}%")
    
    return bifurcation_prompt, correct_solution, wrong_solution, chosen_score, rejected_score, "\n".join(logs)

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using hybrid approach"""
    start_time = time.perf_counter()
    try:
        # Initialize variables that might be referenced in any code path
        # Initialize variables
        answer_1 = None
        answer_2 = None
        score_path_1 = 0.0
        score_path_2 = 0.0
        response_1 = None
        response_2 = None
        first_path_valid = True
        second_path_valid = True
        current_solution = ""  # Initialize empty string for current solution

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

        # Random approach selection
        r = random.random()
        
        logs = []
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        
        # Problem details
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Approach info
        logs.append(f"\n🔄 Processing Details:")
        logs.append(f"├─ Strategy: {'Full solution' if r < 0.3 else 'Progressive building'}")
        
        if r < 0.8:  # Full solution approach
            result = await process_full_solution(example, solver, verifier, config)
            if not result or result is None:
                return None
            try:
                bifurcation_prompt, chosen_response, rejected_response, chosen_score, rejected_score, solution_logs = result
                print(solution_logs)  # Print the logs from full solution
            except ValueError:
                logging.error(f"Invalid result format from process_full_solution: {result}")
                return None
            
        else:  # Analysis/Steps approach
            logs.append("\n=== Analysis/Steps Details ===")
            logs.append("Approach: Progressive solution building")
            
            # Determine bifurcation point
            if r < 0.9:  # Analysis only (0.3-0.5 = 0.2 probability)
                n = 1
            else:
                # Exponentially decaying probability for steps 2+
                norm_const = sum(3**(-i) for i in range(1, 11))
                r_scaled = (r - 0.9) / 0.1  # Scale remaining probability space to [0,1]
                cumsum = 0
                n = 1
                while n <= 10:
                    cumsum += (3**(-n)) / norm_const
                    if r_scaled <= cumsum:
                        break
                    n += 1
            
            logs.append(f"└─ Bifurcation: After step {n}")
            logs.append(f"Completion attempts planned: {config.completions}")
            
            analysis_agent = AnalysisAgent(solver)
            step_agent = NextStepAgent(solver)
            completion_agent = CompletionAgent(solver)
            
            # Process using the analysis/steps approach from data_creator.py
            if n == 1:
                # Try up to 3 times for path_1
                for retry in range(10):
                    bifurcation_prompt, path_1 = await analysis_agent.generate(example["problem"], return_prompt=True)
                    is_valid, reason = validate_analysis(path_1)
                    if is_valid:
                        break
                    logs.append(f"Analysis validation failed for path_1 (retry {retry + 1}/3): {reason}")
                    if retry == 9:  # All retries failed
                        logs.append("Failed all retries for path_1 analysis")
                        print("\n".join(logs))
                        return None
                
                # Try up to 3 times for path_2
                for retry in range(10):
                    _, path_2 = await analysis_agent.generate(example["problem"], return_prompt=True)
                    is_valid, reason = validate_analysis(path_2)
                    if path_2 != path_1 and is_valid:
                        break
                    logs.append(f"Analysis validation failed for path_2 (retry {retry + 1}/3): {reason}")
                    if retry == 9:  # All retries failed
                        logs.append("Failed all retries for path_2 analysis")
                        print("\n".join(logs))
                        return None
                    
                response_1 = path_1
                response_2 = path_2
                
            else:
                # Generate and validate initial analysis
                for retry in range(10):
                    _, common_analysis = await analysis_agent.generate(example["problem"], return_prompt=True)
                    is_valid, reason = validate_analysis(common_analysis)
                    if is_valid and extract_answer_from_solution(common_analysis) is None:
                        current_solution = common_analysis
                        logs.append(f"✓ Valid analysis generated: {reason}")
                        break
                    logs.append(f"Analysis validation failed (retry {retry + 1}/3): {reason}")
                    if retry == 9:
                        print("\n".join(logs))
                        return None
                
                # Generate intermediate steps
                for step_num in range(n-2):
                    step_added = False
                    for retry in range(10):
                        _, next_step = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                        if validate_step(next_step):
                            test_solution = current_solution + next_step
                            premature_answer = extract_answer_from_solution(test_solution)
                            if premature_answer is None:
                                current_solution = test_solution
                                step_added = True
                                break
                            else:
                                logs.append(f"Step {step_num + 1} generated premature answer (retry {retry + 1}/3)")
                        else:
                            logs.append(f"Step {step_num + 1} validation failed (retry {retry + 1}/3)")
                    
                    if not step_added:
                        logs.append(f"Failed all retries for step {step_num + 1}")
                        print("\n".join(logs))
                        return None
                
                
                # Generate first bifurcation path with retries
                for retry in range(10):
                    bifurcation_prompt, response_1 = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                    path_1 = current_solution + response_1
                    first_path_valid = validate_step(response_1)
                    if first_path_valid:
                        logs.append(f"✓ Valid first bifurcation step generated (attempt {retry + 1}/3)")
                        break
                    logs.append(f"❌ Invalid first bifurcation step (attempt {retry + 1}/3)")
                
                if not first_path_valid:
                    logs.append("Failed all retries for first bifurcation step")
                    score_path_1 = 0.0
                    answer_1 = None
                    path_1 = current_solution  # Ensure path_1 is initialized
                else:
                    answer_1 = extract_answer_from_solution(path_1)
                    if answer_1 is not None:
                        # First path found answer - verify it
                        score, total_steps, _ = await verifier.verify(path_1, correct_answer, example["problem"])
                        if score == total_steps:
                            logs.append("✓ First path found correct answer at bifurcation")
                            logs.append("\n📊 Early Return Summary:")
                            logs.append("├─ Status: Success - correct answer found in first path")
                            logs.append("└─ Score: 1.0 vs 0.0")
                            print("\n".join(logs))
                            # Return immediately since we found a correct answer
                            return []

                # Generate second bifurcation path with retries
                for retry in range(10):
                    response_2 = await step_agent.generate(example["problem"], current_solution)
                    path_2 = current_solution + response_2
                    second_path_valid = response_2 != response_1 and validate_step(response_2)
                    if second_path_valid:
                        logs.append(f"✓ Valid second bifurcation step generated (attempt {retry + 1}/3)")
                        break
                    logs.append(f"❌ Invalid second bifurcation step (attempt {retry + 1}/3)")
                
                if not second_path_valid:
                    logs.append("Failed all retries for second bifurcation step")
                    score_path_2 = 0.0
                    answer_2 = None
                    path_2 = current_solution  # Ensure path_2 is initialized
                else:
                    answer_2 = extract_answer_from_solution(path_2)
                    if answer_2 is not None:
                        # Second path found answer - verify it
                        score, total_steps, _ = await verifier.verify(path_2, correct_answer, example["problem"])
                        if score == total_steps:
                            logs.append("✓ Second path found correct answer at bifurcation")
                            logs.append("\n📊 Early Return Summary:")
                            logs.append("├─ Status: Success - correct answer found in second path")
                            logs.append("└─ Score: 1.0 vs 0.0")
                            print("\n".join(logs))
                            # Return immediately since we found a correct answer
                            return [{
                                'id': example_id,
                                'prompt': {'content': bifurcation_prompt, 'role': 'user'},
                                'chosen': {'content': response_2, 'role': 'assistant'},
                                'rejected': {'content': response_1, 'role': 'assistant'},
                                'score_chosen': 1.0,
                                'score_rejected': 0.0
                            }]
            
            # Check if responses are equal or either is None
            if response_1 is None or response_2 is None or response_1 == response_2:
                logs.append("❌ Failed: Bifurcated responses are identical")
                print("\n".join(logs))
                return None

            # Update path validity based on step number
            first_path_valid = first_path_valid if n > 1 else True
            second_path_valid = second_path_valid if n > 1 else True

            # Initialize success counters
            successful_path_1 = 0
            successful_path_2 = 0

            # Check path validity before proceeding
            if not (first_path_valid and second_path_valid):
                logs.append("❌ Failed: One or both paths are invalid")
                print("\n".join(logs))
                return None
                
            # Verify no premature answers
            if answer_1 is not None or answer_2 is not None:
                logs.append("❌ Failed: Premature answer found")
                print("\n".join(logs))
                return None
            
            logs.append("\n✓ Both paths valid and need completions")
            
            # Do completions for both paths
            logs.append("\n🔍 Completion Attempts:")
            
            midpoint = config.completions // 3
            for attempt in range(config.completions):
                logs.append(f"\nAttempt {attempt + 1}/{config.completions}:")
                
                try:
                    complete_solution = path_1 + await completion_agent.generate(example["problem"], path_1)
                    is_correct, reason = await verifier.verify(complete_solution, correct_answer, example["problem"])
                    logs.append(f"Path 1:")
                    logs.append(f"├─ Verification Result: {reason}")
                    if is_correct:
                        successful_path_1 += 1
                        logs.append(f"└─ Success! ({successful_path_1} total successes)")
                    else:
                        logs.append(f"└─ Failed verification")
                except Exception as e:
                    logs.append(f"└─ Error: {str(e)}")
                
                try:
                    complete_solution = path_2 + await completion_agent.generate(example["problem"], path_2)
                    is_correct, reason = await verifier.verify(complete_solution, correct_answer, example["problem"])
                    logs.append(f"Path 2:")
                    logs.append(f"├─ Verification Result: {reason}")
                    if is_correct:
                        successful_path_2 += 1
                        logs.append(f"└─ Success! ({successful_path_2} total successes)")
                    else:
                        logs.append(f"└─ Failed verification")
                except Exception as e:
                    logs.append(f"└─ Error: {str(e)}")
                
                # Check at midpoint if paths are showing meaningful difference
                if attempt + 1 == midpoint:
                    current_score_diff = abs(successful_path_1 - successful_path_2)
                    if current_score_diff < 1:
                        logs.append("\n❌ Early termination at midpoint:")
                        logs.append(f"├─ Path 1 successes: {successful_path_1}")
                        logs.append(f"├─ Path 2 successes: {successful_path_2}")
                        logs.append("└─ Reason: Bifurcation steps don't make a difference")
                        print("\n".join(logs))
                        return None
            
            # Calculate success rates as ratios
            score_path_1 = successful_path_1 / config.completions
            score_path_2 = successful_path_2 / config.completions
            
            # Calculate relative scores if either has non-zero success
            # Only return None if both paths are invalid
            if not first_path_valid and not second_path_valid:
                logs.append("❌ Failed: Both paths are invalid")
                logs.append("\n📊 Early Return Summary:")
                logs.append("├─ Status: Failed - both paths invalid")
                logs.append("└─ Reason: No valid paths for sampling")
                print("\n".join(logs))
                return []

            # Calculate max score from valid paths only
            valid_scores = []
            if first_path_valid:
                valid_scores.append(score_path_1)
            if second_path_valid:
                valid_scores.append(score_path_2)
            
            max_score = max(valid_scores) if valid_scores else 0
            relative_path_1 = score_path_1 / max_score if max_score > 0 else 0
            relative_path_2 = score_path_2 / max_score if max_score > 0 else 0
            relative_diff = abs(relative_path_1 - relative_path_2)

            # Add performance metrics
            logs.append(f"\n📊 Performance Metrics:")
            logs.append(f"├─ Path 1 success: {score_path_1:.2%}")
            logs.append(f"├─ Path 2 success: {score_path_2:.2%}")
            logs.append(f"├─ Relative path 1 score: {relative_path_1:.2%}")
            logs.append(f"├─ Relative path 2 score: {relative_path_2:.2%}")
            logs.append(f"└─ Relative difference: {relative_diff:.2%}")
            
            # Add quality metrics for both solutions
            logs.append(f"\n🔍 Solution Quality:")
            logs.append("├─ Path 1:")
            path_1_quality = analyze_solution_quality(path_1)
            logs.append(f"│  ├─ Length: {path_1_quality['length']} words")
            logs.append(f"│  ├─ Steps: {path_1_quality['step_count']}")
            logs.append(f"│  └─ Format score: {path_1_quality['formatting_quality']}/5")
            
            logs.append("└─ Path 2:")
            path_2_quality = analyze_solution_quality(path_2)
            logs.append(f"   ├─ Length: {path_2_quality['length']} words")
            logs.append(f"   ├─ Steps: {path_2_quality['step_count']}")
            logs.append(f"   └─ Format score: {path_2_quality['formatting_quality']}/5")
            
            # Check if differences are too small (indicating statistical noise)
            score_diff = abs(successful_path_1 - successful_path_2)
            if relative_diff < 0.22 or score_diff < 2:
                logs.append(f"❌ Failed: Differences too small (relative: {relative_diff:.1%}, absolute: {score_diff})")
                logs.append("\n📊 Early Return Summary:")
                logs.append("├─ Status: Failed - insufficient score difference")
                logs.append(f"└─ Details: relative diff {relative_diff:.1%}, absolute diff {score_diff}")
                print("\n".join(logs))
                return None
                
            # Return higher scoring response as chosen
            if score_path_2 > score_path_1:
                chosen_response = response_2
                rejected_response = response_1
                chosen_score = score_path_2
                rejected_score = score_path_1
            else:
                chosen_response = response_1
                rejected_response = response_2
                chosen_score = score_path_1
                rejected_score = score_path_2

            logs.append(f"Score difference: {abs(chosen_score - rejected_score)/max(chosen_score, rejected_score):.1%}")
                    # Always print logs before returning result
            print("\n".join(logs))
            return [{
                'id': example_id,
                'prompt': {'content': bifurcation_prompt, 'role': 'user'},
                'chosen': {'content': chosen_response, 'role': 'assistant'},
                'rejected': {'content': rejected_response, 'role': 'assistant'},
                'score_chosen': chosen_score,
                'score_rejected': rejected_score}]

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
    """Main function for hybrid approach combining full solution and analysis/steps methods."""
    config = BenchmarkConfig.from_args('Hybrid approach combining full solution and analysis/steps methods')
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
