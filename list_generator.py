import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from hybrid_data_creator import validate_solution, validate_analysis, validate_step
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def evaluate_analysis(
    problem: str,
    analysis: str,
    solver: Any,
    verifier: Any,
    correct_answer: str,
    num_completions: int
) -> float:
    """Evaluate an analysis by attempting completions"""
    is_valid, reason = validate_analysis(analysis)
    if not is_valid:
        print(f"Analysis validation failed: {reason}")
        return 0.0

    # Check if analysis already contains answer
    if answer := extract_answer_from_solution(analysis):
        score, max_score, _ = await verifier.verify(analysis, correct_answer, problem)
        return float(score == max_score)

    # Try completions
    completion_agent = CompletionAgent(solver)
    successful = 0
    
    for _ in range(num_completions):
        try:
            completion = await completion_agent.generate(problem, analysis)
            score, max_score, _ = await verifier.verify(
                analysis + completion, 
                correct_answer, 
                problem
            )
            successful += (score == max_score)
        except Exception as e:
            print(f"Completion error: {e}")
            
    return successful / num_completions if num_completions > 0 else 0.0

async def evaluate_step(
    problem: str,
    current_solution: str,
    next_step: str,
    solver: Any, 
    verifier: Any,
    correct_answer: str,
    num_completions: int
) -> Tuple[float, bool]:
    """Evaluate a step by checking for answer or attempting completions"""
    solution_with_step = current_solution + next_step
    
    # Return 0 score if solution becomes invalid
    if not validate_solution(solution_with_step)[0]:
        return 0.0, False
        
    # Check if step contains answer
    if answer := extract_answer_from_solution(solution_with_step):
        score, max_score, _ = await verifier.verify(solution_with_step, correct_answer, problem)
        return float(score == max_score), True

    # Try completions
    completion_agent = CompletionAgent(solver)
    successful = 0
    
    for _ in range(num_completions):
        try:
            completion = await completion_agent.generate(problem, solution_with_step)
            score, max_score, _ = await verifier.verify(
                solution_with_step + completion,
                correct_answer,
                problem
            )
            successful += (score == max_score)
        except Exception as e:
            
            
    return (successful / num_completions if num_completions > 0 else 0.0), False

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[List[Dict]]:
    """Process a single example using list generation approach"""
    start_time = time.perf_counter()
    logs = []
    
    logs.append("\n" + "="*80)
    logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
    logs.append("="*80)
    
    # Problem details
    logs.append(f"\n📋 Problem:")
    logs.append(f"{example['problem'][:200]}...")
    
    # Validate input
    if not isinstance(example, dict) or not {'problem', 'solution'} <= example.keys():
        logs.append("❌ Invalid example format")
        print("\n".join(logs))
        return None
        
    if not (correct_answer := extract_answer_from_solution(example['solution'])):
        logs.append("❌ No answer found in solution")
        print("\n".join(logs))
        return None
        
    logs.append(f"✓ Expected Answer: {correct_answer}")

    try:
        # Setup models
        solver = get_model(ModelOption[config.solver], temp=config.temperature)
        verifier = create_verifier(
            config.verification_type,
            verifier_model=None if config.verification_type == 'numeric' else get_model(
                ModelOption[config.verifier], temp=config.verifier_temp),
            second_verifier_model=None if config.verification_type != 'solution' else get_model(
                ModelOption[config.second_verifier], temp=config.verifier_temp),
            tolerance=config.tolerance
        )

        analysis_agent = AnalysisAgent(solver)
        step_agent = NextStepAgent(solver)
        results = []
        
        logs.append("\n🔄 Processing Details:")
        logs.append("├─ Strategy: List generation")
        
        # First generate and evaluate analyses
        logs.append("\n📊 Generating analyses...")
        analyses = []
        prompts = []
        scores = []
        
        attempts = 0
        max_attempts = config.best_of * 2  # Allow more attempts to get diverse scores
        score_diff_threshold = 0.3  # Minimum score difference we want
        
        while attempts < max_attempts and len(analyses) < config.best_of:
            attempts += 1
            prompt, analysis = await analysis_agent.generate(example["problem"], return_prompt=True)
            
            if analysis not in analyses:
                score = await evaluate_analysis(
                    example["problem"],
                    analysis,
                    solver,
                    verifier,
                    correct_answer,
                    config.completions
                )
                
                # Only add if score is different enough from existing scores
                should_add = True
                for existing_score in scores:
                    if abs(score - existing_score) < score_diff_threshold:
                        should_add = False
                        break
                
                if should_add or not scores:  # Always add first analysis
                    analyses.append(analysis)
                    prompts.append(prompt)
                    scores.append(score)
                    logs.append(f"├─ Analysis score: {score:.2f}")
                    
                    # If we found a perfect score, add it to results and stop
                    if score == 1.0:
                        # Add all previous results if any
                        final_results = []
                        if len(analyses) > 1:  # If we have multiple analyses to compare
                            for i in range(len(analyses)-1):
                                if scores[i] != scores[i+1]:  # Only add pairs with different scores
                                    final_results.append({
                                        'id': example_id,
                                        'prompt': {'content': prompts[i], 'role': 'user'},
                                        'chosen': {'content': analyses[i], 'role': 'assistant'},
                                        'rejected': {'content': analyses[i+1], 'role': 'assistant'},
                                        'score_chosen': scores[i],
                                        'score_rejected': scores[i+1]
                                    })
                        
                        # Add the perfect score result
                        final_results.append({
                            'id': example_id,
                            'prompt': {'content': prompt, 'role': 'user'},
                            'chosen': {'content': analysis, 'role': 'assistant'},
                            'rejected': {'content': analyses[scores.index(min(scores))], 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': min(scores)
                        })
                        return final_results
        
        if not analyses:
            logs.append("❌ No valid analyses generated")
            print("\n".join(logs))
            return None
            
        # Sort analyses by score and pair them
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        
        # Add results comparing adjacent pairs with different scores
        for i in range(len(sorted_indices)-1):
            if scores[sorted_indices[i]] > scores[sorted_indices[i+1]]:
                results.append({
                    'id': example_id,
                    'prompt': {'content': prompts[sorted_indices[i]], 'role': 'user'},
                    'chosen': {'content': analyses[sorted_indices[i]], 'role': 'assistant'},
                    'rejected': {'content': analyses[sorted_indices[i+1]], 'role': 'assistant'},
                    'score_chosen': scores[sorted_indices[i]],
                    'score_rejected': scores[sorted_indices[i+1]]
                })
        
        # Continue with best analysis (first in sorted indices)
        current_solution = analyses[sorted_indices[0]]
        step = 1
        
        while True:
            logs.append(f"\n📝 Processing step {step}")
            
            # Generate multiple steps
            steps = []
            step_prompts = []
            step_scores = []
            found_answer = False
            
            max_step_score_found = False
            min_step_score_found = False
            
            attempts = 0
            max_attempts = config.best_of * 3  # Allow more attempts to get valid steps
            
            score_diff_threshold = 0.3  # Minimum score difference we want
            
            while len(steps) < config.best_of and attempts < max_attempts:
                attempts += 1
                prompt, step_text = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                score, has_answer = await evaluate_step(
                        example["problem"],
                        current_solution,
                        step_text,
                        solver,
                        verifier,
                        correct_answer,
                        config.completions
                    )
                
                # Only add if score is different enough from existing scores
                should_add = True
                for existing_score in step_scores:
                    if abs(score - existing_score) < score_diff_threshold:
                        should_add = False
                        break
                        
                if (should_add or not step_scores) and step_text not in steps:
                    steps.append(step_text)
                    step_prompts.append(prompt)
                    step_scores.append(score)
                    logs.append(f"├─ Step score: {score:.2f}")
                    
                    # If we found a perfect score with answer, we can stop
                    if has_answer and score == 1.0:
                        # Found correct answer in this step
                        return results + [{
                            'id': example_id,
                            'prompt': {'content': prompt, 'role': 'user'},
                            'chosen': {'content': current_solution + step_text, 'role': 'assistant'},
                            'rejected': {'content': "", 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0
                        }]
                    found_answer = found_answer or has_answer
                else:
                    logs.append(f"├─ Skipped duplicate step (attempt {attempts}/{max_attempts})")
            
            if not steps:
                logs.append("❌ No valid steps generated")
                print("\n".join(logs))
                break
                
            # Sort steps by score
            sorted_step_indices = sorted(range(len(step_scores)), key=lambda k: step_scores[k], reverse=True)
            best_step_idx = sorted_step_indices[0]
            
            # If best step has score 0 or we found wrong answer, stop
            if step_scores[best_step_idx] == 0 or (found_answer and max(step_scores) < 1.0):
                break
                
            # Add results comparing adjacent pairs with different scores
            for i in range(len(sorted_step_indices)-1):
                if step_scores[sorted_step_indices[i]] > step_scores[sorted_step_indices[i+1]]:
                    results.append({
                        'id': example_id,
                        'prompt': {'content': step_prompts[sorted_step_indices[i]], 'role': 'user'},
                        'chosen': {'content': steps[sorted_step_indices[i]], 'role': 'assistant'},
                        'rejected': {'content': steps[sorted_step_indices[i+1]], 'role': 'assistant'},
                        'score_chosen': step_scores[sorted_step_indices[i]],
                        'score_rejected': step_scores[sorted_step_indices[i+1]]
                    })
            
            # Update current solution with best step
            current_solution += steps[best_step_idx]
            step += 1
            
        return results if results else None
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        error_category = (
            "timeout" if isinstance(e, TimeoutError)
            else "validation" if isinstance(e, ValueError)
            else "rate_limit" if "rate limit" in str(e).lower()
            else "context_length" if "context length" in str(e).lower()
            else "other"
        )
        
        logs.append("\n❌ Error Details:")
        logs.append(f"├─ Error type: {type(e).__name__}")
        logs.append(f"├─ Error message: {str(e)}")
        logs.append(f"├─ Error category: {error_category}")
        logs.append(f"├─ Processing time: {processing_time:.2f}s")
        logs.append(f"└─ Example ID: {example_id}")
        
        print("\n".join(logs))
        return None

async def main():
    """Main function for list generation approach."""
    config = BenchmarkConfig.from_args('List generation approach for mathematical problems')
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
