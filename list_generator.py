import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def generate_and_score_completions(
    problem: str,
    partial_solution: str,
    solver: Any,
    verifier: Any,
    correct_answer: str,
    num_completions: int
) -> Tuple[str, str, float, float]:
    """Generate completions and find highest/lowest scoring ones"""
    completion_agent = CompletionAgent(solver)
    highest_score = -1
    lowest_score = float('inf')
    highest_completion = ""
    lowest_completion = ""
    
    for _ in range(num_completions):
        try:
            complete_solution = partial_solution + await completion_agent.generate(problem, partial_solution)
            score, max_score, current_answer = await verifier.verify(
                complete_solution,
                correct_answer,
                problem
            )
                
            # Normalize score
            normalized_score = score / max_score if max_score > 0 else 0
            
            if normalized_score > highest_score:
                highest_score = normalized_score
                highest_completion = complete_solution
            if normalized_score < lowest_score:
                lowest_score = normalized_score
                lowest_completion = complete_solution
                
            # Early exit if we found correct answer
            if normalized_score == 1.0:
                return highest_completion, lowest_completion, 1.0, normalized_score
                
        except Exception as e:
            print(f"Error in completion: {str(e)}")
            continue
            
    return highest_completion, lowest_completion, highest_score, lowest_score

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[List[Dict]]:
    """Process a single example using list generation approach"""
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
        verifier_model = None if config.verification_type == 'numeric' else get_model(
            ModelOption[config.verifier], temp=config.verifier_temp)
        second_verifier_model = None if config.verification_type != 'solution' else get_model(
            ModelOption[config.second_verifier], temp=config.verifier_temp)
        verifier = create_verifier(
            config.verification_type,
            verifier_model=verifier_model,
            second_verifier_model=second_verifier_model,
            tolerance=config.tolerance
        )

        analysis_agent = AnalysisAgent(solver)
        step_agent = NextStepAgent(solver)
        results = []
        current_partial = ""
        step = 0
        
        while True:
            print(f"\nProcessing step {step} for example {running_id}")
            
            # Generate multiple continuations
            continuations = []
            prompts = []
            
            # Generate different continuations based on best_of parameter
            for _ in range(config.best_of):
                if step == 0:
                    # For first step, generate analysis
                    prompt, continuation = await analysis_agent.generate(example["problem"], return_prompt=True)
                    prompts.append(prompt)
                else:
                    # For subsequent steps, generate next step
                    prompt, continuation = await step_agent.generate(example["problem"], current_partial, return_prompt=True)
                    prompts.append(prompt)
                    
                if continuation not in continuations:
                    continuations.append(continuation)
                    
            # Score each continuation
            highest_overall_score = -1
            best_continuation = ""
            best_prompt = ""
            
            for idx, continuation in enumerate(continuations):
                partial_solution = current_partial + continuation
                
                # First check if this step already contains an answer
                answer = extract_answer_from_solution(partial_solution)
                if answer is not None:
                    # Verify if this answer is correct
                    score, max_score, _ = await verifier.verify(
                        partial_solution,
                        correct_answer,
                        example["problem"]
                    )
                    if score == max_score:  # Found correct solution in the step itself
                        return [{
                            'id': example_id,
                            'prompt': {'content': prompts[idx], 'role': 'user'},
                            'chosen': {'content': partial_solution, 'role': 'assistant'},
                            'rejected': {'content': "", 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0
                        }]
                
                # If no answer or wrong answer, proceed with completions
                highest_completion, lowest_completion, high_score, low_score = await generate_and_score_completions(
                    example["problem"],
                    partial_solution,
                    solver,
                    verifier,
                    correct_answer,
                    config.completions
                )
                
                # Check if this continuation already has the answer (before completion)
                answer = extract_answer_from_solution(partial_solution)
                if answer is not None:
                    score, max_score, _ = await verifier.verify(
                        partial_solution,
                        correct_answer,
                        problem
                    )
                    if score == max_score:  # Found correct solution in the step itself
                        results.append({
                            'id': example_id,
                            'prompt': {'content': prompts[idx], 'role': 'user'},
                            'chosen': {'content': partial_solution, 'role': 'assistant'},
                            'rejected': {'content': lowest_completion or "", 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0
                        })
                        return results
                    
                # Track best continuation
                if high_score > highest_overall_score:
                    highest_overall_score = high_score
                    best_continuation = continuation
                    best_prompt = prompts[idx]
                    best_high_completion = highest_completion
                    best_low_completion = lowest_completion
                    best_high_score = high_score
                    best_low_score = low_score
            
            # Print score range for this level
            print(f"\nLevel {step} scores:")
            print(f"Max score: {highest_overall_score:.3f}")
            print(f"Min score: {best_low_score:.3f}")
            print(f"Score range: {(highest_overall_score - best_low_score):.3f}")
            
            # If no continuation scored above 0, stop
            if highest_overall_score <= 0:
                break
                
            # Add result for this step
            results.append({
                'id': example_id,
                'prompt': {'content': best_prompt, 'role': 'user'},
                'chosen': {'content': best_high_completion, 'role': 'assistant'},
                'rejected': {'content': best_low_completion, 'role': 'assistant'},
                'score_chosen': best_high_score,
                'score_rejected': best_low_score
            })
            
            # Update current partial solution with best continuation
            current_partial += best_continuation
            step += 1
            
        return results if results else None
        
    except Exception as e:
        print(f"Error processing example {running_id}: {str(e)}")
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
