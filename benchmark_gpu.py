import asyncio
import torch
from bench_utils.benchmark_config import BenchmarkConfig, ModelOption
from bench_utils.benchmark_utils import run_benchmark, get_model_response, extract_answer_from_solution
from bench_utils.agents import AnalysisPlusStepAgent
import argparse

async def process_example(example, running_id, example_id, config):
    """Process a single example with GPU acceleration"""
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
        
        return {
            'id': example_id,
            'running_id': running_id,
            'problem': example['problem'],
            'solution': solution,
            'model_answer': model_answer,
            'correct_solution': example['solution']
        }
        
    except Exception as e:
        print(f"Error processing example {example_id}: {str(e)}")
        return None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_concurrent', type=int, default=4)
    parser.add_argument('--best_of', type=int, default=1)
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model=ModelOption.LOCAL,
        dataset='filtered',
        split='train',
        max_concurrent=args.max_concurrent,
        best_of=args.best_of,
        upload_results=True
    )
    
    await run_benchmark(config, process_example)

if __name__ == "__main__":
    asyncio.run(main())
