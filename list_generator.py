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
            print(f"Completion error: {e}")
            
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
        
        max_score_found = False
        min_score_found = False
        
        for _ in range(config.best_of):
            prompt, analysis = await analysis_agent.generate(example["problem"], return_prompt=True)
            if analysis not in analyses:
                analyses.append(analysis)
                prompts.append(prompt)
                score = await evaluate_analysis(
                    example["problem"],
                    analysis,
                    solver,
                    verifier,
                    correct_answer,
                    config.completions
                )
                scores.append(score)
                logs.append(f"├─ Analysis score: {score:.2f}")
                
                # Check if we found max (1.0) and min (0.0) scores
                if score == 1.0:
                    max_score_found = True
                elif score == 0.0:
                    min_score_found = True
                    
                # Stop if we found both extremes or got a perfect score
                if max_score_found and min_score_found or score == 1.0:
                    return [{
                        'id': example_id,
                        'prompt': {'content': prompt, 'role': 'user'},
                        'chosen': {'content': analysis, 'role': 'assistant'},
                        'rejected': {'content': "", 'role': 'assistant'},
                        'score_chosen': 1.0,
                        'score_rejected': 0.0
                    }]
        
        if not analyses:
            logs.append("❌ No valid analyses generated")
            print("\n".join(logs))
            return None
            
        # Find best and worst analysis
        best_idx = scores.index(max(scores))
        worst_idx = scores.index(min(scores))
        
        # Add first result comparing analyses
        results.append({
            'id': example_id,
            'prompt': {'content': prompts[best_idx], 'role': 'user'},
            'chosen': {'content': analyses[best_idx], 'role': 'assistant'},
            'rejected': {'content': analyses[worst_idx], 'role': 'assistant'},
            'score_chosen': scores[best_idx],
            'score_rejected': scores[worst_idx]
        })
        
        # Continue with best analysis
        current_solution = analyses[best_idx]
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
                if step_text not in steps:  # Only check for duplicates
                    steps.append(step_text)
                    step_prompts.append(prompt)
                    step_scores.append(score)
                    logs.append(f"├─ Step score: {score:.2f}")
                    
                    # Check if we found max and min scores
                    if score == 1.0:
                        max_step_score_found = True
                    elif score == 0.0:
                        min_step_score_found = True
                    
                    # Stop if we found both extremes or a correct answer
                    if (max_step_score_found and min_step_score_found) or (has_answer and score == 1.0):
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
                
            # Find best and worst steps
            best_step_idx = step_scores.index(max(step_scores))
            worst_step_idx = step_scores.index(min(step_scores))
            
            # If best step has score 0 or we found wrong answer, stop
            if step_scores[best_step_idx] == 0 or (found_answer and max(step_scores) < 1.0):
                break
                
            # Add result for this step
            results.append({
                'id': example_id,
                'prompt': {'content': step_prompts[best_step_idx], 'role': 'user'},
                'chosen': {'content': steps[best_step_idx], 'role': 'assistant'},
                'rejected': {'content': steps[worst_step_idx], 'role': 'assistant'},
                'score_chosen': step_scores[best_step_idx],
                'score_rejected': step_scores[worst_step_idx]
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
