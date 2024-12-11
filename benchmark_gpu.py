import asyncio
import torch
from bench_utils.benchmark_config import BenchmarkConfig, ModelOption
from bench_utils.benchmark_utils import run_benchmark, get_model_response, extract_answer_from_solution
from bench_utils.agents import AnalysisPlusStepAgent
import argparse

async def process_example(example, running_id, example_id, config):
    """Process a single example with GPU acceleration and multiple attempts"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")
        
    # Ensure we're using GPU device 1
    torch.cuda.set_device(1)
    
    agent = AnalysisPlusStepAgent(config)
    
    try:
        # Generate solution
        solution = await agent.generate(example['problem'])
        
        # Extract answer
        model_answer = extract_answer_from_solution(solution)
        
        # Calculate solution length and step count for metrics
        solution_length = len(solution.split())
        step_count = solution.count('Step') + solution.count('\n1.') + solution.count('\n2.')
        
        return {
            'id': example_id,
            'running_id': running_id,
            'problem': example['problem'],
            'solution': solution,
            'model_answer': model_answer,
            'correct_solution': example['solution'],
            'metrics': {
                'solution_length': solution_length,
                'step_count': step_count,
                'attempt': config.current_attempt
            }
        }
        
    except Exception as e:
        print(f"Error processing example {example_id}: {str(e)}")
        return None

async def main():
    parser = argparse.ArgumentParser(description='Run GPU-accelerated benchmark with multiple attempts per problem')
    parser.add_argument('--best_of', type=int, default=3, help='Number of attempts per problem')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model=ModelOption.LOCAL,
        dataset='filtered',
        split=args.split,
        max_concurrent=1,  # Force serial processing
        best_of=args.best_of,
        upload_results=True,
        save_all_attempts=True  # Save all attempts for statistical analysis
    )
    
    await run_benchmark(config, process_example)

if __name__ == "__main__":
    asyncio.run(main())
