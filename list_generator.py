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
) -> Tuple[str, str, int, int]:
    """Generate completions and count correct/incorrect solutions"""
    completion_agent = CompletionAgent(solver)
    correct_count = 0
    total_count = 0
    best_completion = ""
    worst_completion = ""
    
    for _ in range(num_completions):
        try:
            complete_solution = partial_solution + await completion_agent.generate(problem, partial_solution)
            score, max_score, _ = await verifier.verify(
                complete_solution,
                correct_answer,
                problem
            )
            
            total_count += 1
            if score == max_score:  # Solution is correct
                correct_count += 1
                if not best_completion:  # Keep first correct solution
                    best_completion = complete_solution
            else:  # Solution is wrong
                if not worst_completion:  # Keep first wrong solution
                    worst_completion = complete_solution
                    
        except Exception as e:
            print(f"Error in completion: {str(e)}")
            continue
            
    return best_completion, worst_completion, correct_count, total_count

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
            best_success_rate = 0
            best_continuation = ""
            best_prompt = ""
            best_correct = ""
            best_wrong = ""
            
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
                correct_completion, wrong_completion, num_correct, total = await generate_and_score_completions(
                    example["problem"],
                    partial_solution,
                    solver,
                    verifier,
                    correct_answer,
                    config.completions
                )
                
                # Calculate success rate for this continuation
                success_rate = num_correct / total if total > 0 else 0
                
                # Track best performing continuation
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_continuation = continuation
                    best_prompt = prompts[idx]
                    best_correct = correct_completion
                    best_wrong = wrong_completion
            
            # Print statistics for this level
            print(f"\nLevel {step} completion stats:")
            print(f"Best success rate: {best_success_rate:.1%}")
            
            # Stop if no completions succeeded
            if best_success_rate == 0:
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
