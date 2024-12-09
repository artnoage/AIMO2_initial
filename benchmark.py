import os
import asyncio
import aiohttp
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from glob import glob
from pathlib import Path
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {str(running_id)}: Invalid example format")
            return None
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {str(running_id)}")
            return None

        solver = get_model(ModelOption[config.solver], temp=config.temperature)
        solution_agent = FullSolutionAgent(solver)
        solutions = []
        correct_count = 0
        best_solution = None
        
        for attempt in range(config.best_of):
            try:
                current_solution = await solution_agent.generate(example["problem"])
                
                # Create and use appropriate verifier
                verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
                second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
                verifier = create_verifier(
                    config.verification_type,
                    verifier_model=verifier_model,
                    second_verifier_model=second_verifier_model,
                    tolerance=config.tolerance
                )
                score, total_steps, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer,
                    example["problem"]
                )
                # For solution verification, consider it correct if it passes majority of steps
                threshold = total_steps 
                if score >= threshold:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
                        
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'verification_score': score,
                    'verification_steps': total_steps,
                    'is_correct': score == total_steps
                })
            except Exception as e:
                print(f"Error in attempt {str(attempt + 1)} for example {str(running_id)}: {str(e)}")
                # Handle error case
                solution_info = {
                    'solution': "Error occurred",
                    'answer': None,
                    'verification_score': 0,
                    'verification_steps': 1,
                    'is_correct': False
                }
                solutions.append(solution_info)
        
        
        # Print statistics
        print(f"\nExample {str(running_id + 1)}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{config.best_of}")
        print(f"Success rate: {(correct_count/config.best_of)*100:.1f}%")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'correct_answer': correct_answer,
            'model_solutions': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'verification_scores': [s['verification_score'] for s in solutions],
            'verification_steps': [s['verification_steps'] for s in solutions]
        }
        
    except Exception as e:
        print(f"Error processing example {str(running_id)}: {e}")
        return None

async def load_lora_adapter(lora_name: str, lora_path: str):
    """Send request to load LoRA adapter to local LLM server"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/v1/load_lora_adapter",
            json={
                "lora_name": lora_name,
                "lora_path": lora_path
            }
        ) as response:
            response_text = await response.text()
            print(f"Server response: {response_text}")
            if response.status != 200:
                if "already been loaded" in response_text:
                    print("LoRA adapter already loaded, continuing...")
                else:
                    raise Exception(f"Failed to load LoRA adapter: {response_text}")

def get_latest_lora_path():
    """Get the path of the most recent lora folder"""
    lora_folders = glob('loras/*/')
    if not lora_folders:
        return None
    return max(lora_folders, key=os.path.getctime)

async def main():
    """Main function for benchmarking mathematical problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems')
    
    # Handle LoRA loading based on config
    if config.upload_lora:
        lora_path = get_latest_lora_path()
        if lora_path:
            try:
                print(f"Using latest LoRA adapter from: {lora_path}")
                lora_name = Path(lora_path).name
                await load_lora_adapter(lora_name, str(Path(lora_path).absolute()))
            except Exception as e:
                print(f"Warning: Failed to load latest LoRA adapter: {e}")
                print("Continuing benchmark without LoRA adapter...")
    
    if config.lora_dir:
        lora_dir = Path(config.lora_dir)
        if not lora_dir.exists():
            print(f"Warning: LoRA directory {lora_dir} does not exist")
        else:
            for lora_path in lora_dir.glob('*'):
                if lora_path.is_dir():
                    try:
                        lora_name = lora_path.name
                        print(f"Loading LoRA adapter {lora_name} from: {lora_path}")
                        await load_lora_adapter(lora_name, str(lora_path.absolute()))
            except Exception as e:
                print(f"Warning: Failed to load LoRA adapter {lora_name}: {e}")
    
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
