import os
import asyncio
from typing import Optional, Dict
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
        completion_agent = CompletionAgent(solver_model)
        
        # Generate two different analyses
        analysis_1 = await analysis_agent.generate(example["problem"])
        analysis_2 = await analysis_agent.generate(example["problem"])
        
        # Track scores for each analysis
        score_1 = 0
        score_2 = 0
        
        # Process completions for first analysis
        for _ in range(20):
            try:
                complete_solution = analysis_1 + await completion_agent.generate(example["problem"], analysis_1)
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
                
        # Process completions for second analysis
        for _ in range(10):
            try:
                complete_solution = analysis_2 + await completion_agent.generate(example["problem"], analysis_2)
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
            'analysis_1': analysis_1,
            'analysis_2': analysis_2,
            'score_1': score_1,
            'score_2': score_2
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
