import os
import asyncio
import random
import math
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, best_of: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using double analysis approach with multiple completions per analysis"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Initialize agents
        analysis_agent = AnalysisAgent(solver_model)
        step_agent = NextStepAgent(solver_model)
        completion_agent = CompletionAgent(solver_model)

        # Generate random n with probability 2^(-n)
        r = random.random()
        n = 1
        while r < 0.5 and n < 10:
            r = random.random()
            n += 1

        print(f"Selected n={n} for bifurcation")
        
        # Initialize prompts list for all cases
        prompts = []
        
        if n == 1:
            # Original behavior: two separate analyses with prompts
            analysis_prompt_1, analysis_1 = await analysis_agent.generate(example["problem"], return_prompt=True)
            analysis_prompt_2, analysis_2 = await analysis_agent.generate(example["problem"], return_prompt=True)
            prompts.append(("analysis_1", analysis_prompt_1))
            prompts.append(("analysis_2", analysis_prompt_2))
            path_1 = analysis_1
            path_2 = analysis_2
        else:
            # Common analysis and n-2 steps, then bifurcate
            # Get analysis with prompt
            analysis_prompt, common_analysis = await analysis_agent.generate(example["problem"], return_prompt=True)
            prompts.append(("analysis", analysis_prompt))
            current_solution = common_analysis
            
            # Add n-2 common steps
            for step_num in range(n-2):
                step_prompt, next_step = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                prompts.append((f"step_{step_num+1}", step_prompt))
                current_solution += next_step
                # Check if we already have an answer
                if extract_answer_from_solution(current_solution) is not None:
                    print(f"Found answer during common path generation for example {running_id}, skipping")
                    return None
            
            # Generate bifurcation prompt
            bifurcation_prompt, _ = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
            prompts.append(("bifurcation", bifurcation_prompt))
            
            # Get two different responses using step agent
            step_1 = await step_agent.generate(example["problem"], current_solution)
            step_2 = await step_agent.generate(example["problem"], current_solution)
            
            path_1 = current_solution + step_1
            path_2 = current_solution + step_2
            
            # Check if either path already has a solution
            answer_1 = extract_answer_from_solution(path_1)
            answer_2 = extract_answer_from_solution(path_2)

            # Initialize scores
            score_1 = 20 if answer_1 is not None else 0
            score_2 = 20 if answer_2 is not None else 0
        
        # Only process completions if we don't already have a solution
        if score_1 == 0:
            # Process completions for first analysis
            for _ in range(20):
                try:
                    complete_solution = path_1 + await completion_agent.generate(example["problem"], path_1)
                    verifier = create_verifier(
                        config.verification_type,
                        verifier_model=verifier_model,
                        second_verifier_model=second_verifier_model,
                        tolerance=config.tolerance
                    )
                    score, total_steps, _ = await verifier.verify(
                        complete_solution,
                        correct_answer,
                        example["problem"]
                    )
                    if score == total_steps:
                        score_1 += 1
                except Exception as e:
                    print(f"Error in completion for analysis 1: {str(e)}")

        if score_2 == 0:
            # Process completions for second analysis
            for _ in range(10):
                try:
                    complete_solution = path_2 + await completion_agent.generate(example["problem"], path_2)
                    verifier = create_verifier(
                        config.verification_type,
                        verifier_model=verifier_model,
                        second_verifier_model=second_verifier_model,
                        tolerance=config.tolerance
                    )
                    score, total_steps, _ = await verifier.verify(
                        complete_solution,
                        correct_answer,
                        example["problem"]
                    )
                    if score == total_steps:
                        score_2 += 1
                except Exception as e:
                    print(f"Error in completion for analysis 2: {str(e)}")

        # Print statistics
        print(f"\nExample {running_id + 1}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Analysis 1 score: {score_1}/10")
        print(f"Analysis 2 score: {score_2}/10")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'path_1': path_1,
            'path_2': path_2,
            'bifurcation_point': n,
            'score_1': score_1,
            'score_2': score_2,
            'prompts': prompts if 'prompts' in locals() else []  # Include prompts if they exist
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for benchmarking mathematical problem solving with double analysis."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems using double analysis')
    verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
    
    # Initialize models
    solver_model = get_model(ModelOption[config.solver], temp=config.temperature)
    
    global progress_tracker
    
    await run_benchmark(
        config=config,
        process_example_func=process_example,
        solver_model=solver_model,
        verifier_model=verifier_model,
        second_verifier_model=second_verifier_model
    )
    
    if progress_tracker:
        progress_tracker.print_final_stats()
        progress_tracker.save_results(config.solver, config.split)

if __name__ == "__main__":
    progress_tracker = None  # Will be initialized in run_benchmark
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if progress_tracker:
            progress_tracker.print_final_stats()
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        if progress_tracker:
            progress_tracker.print_final_stats()
