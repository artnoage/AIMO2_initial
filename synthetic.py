import os
import re
import json
import asyncio
import argparse
from enum import Enum
from typing import Optional, List, Dict
from utils.augmented_data_handler import handle_augmented_data_file, save_augmented_data

from utils.utils import ModelOption
from datetime import datetime
from typing import List, Dict, Optional
from itertools import islice
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from huggingface_hub import HfApi
from tqdm import tqdm
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """You are a mathematical problem solver. When given a problem and partial solution as a hint.
Analyzer the problem and understand the techniques that are needed. 
Do a step by step new proof (you are allowed to copy parts)
Provide the final answer in the end \\boxed{}"""

def get_model(model: ModelOption, temp: float = 0.1):
    """
    Initialize the ChatOpenAI model based on the selected ModelOption.
    For LOCAL models, it connects to a local endpoint.
    For other models, it uses the OpenRouter API.
    """
    if model == ModelOption.LOCAL:
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key="EMPTY",
            base_url="http://localhost:8000/v1")
    elif model == ModelOption.SAMBA:
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=os.getenv("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1")
    else:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=openrouter_api_key)

def extract_answer_from_solution(solution: str) -> Optional[str]:
    """Extract boxed answer from solution"""
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start + 1
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    pattern = re.compile(r'\\boxed\{')
    for match in pattern.finditer(solution):
        start = match.end() - 1
        end = find_matching_brace(solution, start)
        if end != -1:
            return solution[start + 1:end].strip()
    return None


async def compare_math_answers(model_answer: Optional[str], correct_answer: Optional[str], problem: str, model) -> bool:
    """Use the model to compare two mathematical answers"""
    if model_answer is None or correct_answer is None:
        return False
        
    comparison_prompt = [
        SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
        HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
    ]
    
    try:
        response = await model.ainvoke(comparison_prompt)
        return response.content.strip().lower() == 'yes'
    except Exception:
        return False

def get_partial_solution(solution: str) -> str:
    """Get partial solution by removing last two lines"""
    lines = solution.strip().split('\n')
    if len(lines) <= 2:
        return solution
    return '\n'.join(lines[:-2])

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model) -> Optional[Dict]:
    """Process a single example"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Combine problem with partial solution
        partial_solution = get_partial_solution(example['solution'])
        combined_prompt = f"{example['problem']}\n\nPartial solution:\n{partial_solution}"
        
        prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=combined_prompt)
        ]
        
        response = await solver_model.ainvoke(prompt)
        solution = response.content
        model_answer = extract_answer_from_solution(solution)
        is_correct = await compare_math_answers(model_answer, correct_answer, example["problem"], verifier_model)
        
        # Print results for this example
        status = '✓' if is_correct else '✗'
        print(f"\nProblem {running_id + 1}: {status}")
        print(f"Expected Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'partial_solution': partial_solution,
            'correct_answer': correct_answer,
            'model_response': solution,
            'model_answer': model_answer,
            'is_correct': is_correct
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Synthetic Model Benchmark')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Model to use for solving problems')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Model to use for verifying answers')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--max-concurrent', type=int, default=4,
                       help='Maximum number of concurrent problems (default: 4)')
    args = parser.parse_args()

    if args.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina-Olympiads", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    dataset = dataset.shuffle(seed=42)
    num_examples = len(dataset)
    
    print("\nDataset Information:")
    print(f"Number of examples: {num_examples}")

    if num_examples == 0:
        print("Error: Dataset is empty!")
        return

    solver_model = get_model(ModelOption[args.solver], temp=0.1)
    verifier_model = get_model(ModelOption[args.verifier], temp=0.1)
    print(f"\nBenchmarking solver: {args.solver}, verifier: {args.verifier} on {args.split} split...")

    # Create example data with dataset IDs and build lookup map
    example_data = []
    example_map = {}
    for ex in dataset:
        example = {
            'id': ex['id'],
            'problem': ex['problem'],
            'solution': ex['solution']
        }
        example_data.append(example)
        example_map[ex['id']] = example
    
    def calculate_error_rate(results):
        if not results:
            return 0.0
        correct_count = sum(1 for r in results if r['is_correct'])
        return 1.0 - (correct_count / len(results))

    # Process examples with controlled concurrency
    results = []
    error_rate_points = []
    total_examples = len(example_data)
    print(f"\nStarting processing of {total_examples} examples...")

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, example['id'], solver_model, verifier_model)

    # Create tasks for all examples
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    # Initialize augmented dataset filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('augmented_datasets', exist_ok=True)
    augmented_filename = os.path.join('augmented_datasets', 
                                    f"synthetic_numina_augmented_{args.solver}_{args.verifier}_{timestamp}.json")
    
    # Check if user wants to proceed with augmented data handling
    if not handle_augmented_data_file(augmented_filename):
        print("Operation cancelled by user.")
        return
        
    # Process all examples with progress bar
    progress_bar = tqdm(total=total_examples, desc="Processing examples")
    current_batch = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            # Add to current batch using data we already have
            augmented_example = {
                'id': result['id'],
                'problem': result['problem'],
                'solution': example_map[result['id']]['solution'],
                'partial_solution': result['partial_solution'],
                'model_response': result['model_response'],
                'is_correct': result['is_correct']
            }
            current_batch.append(augmented_example)
            
            # Save intermediate results every time we process more examples
            current_error_rate = calculate_error_rate(results)
            error_rate_points.append({
                'examples_processed': len(results),
                'error_rate': current_error_rate,
                'timestamp': datetime.now().isoformat()
            })
            print(f"\nIntermediate Error Rate at {len(results)} examples: {current_error_rate:.4f}")
    
            # Save intermediate results
            intermediate_filename = os.path.join('synthetic_results', 
                f"synthetic_intermediate_{args.solver}_{args.verifier}_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
            output_data = {
                'results': results,
                'error_rate_points': error_rate_points,
                'current_error_rate': current_error_rate
            }
            os.makedirs('synthetic_results', exist_ok=True)
            with open(intermediate_filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Saved intermediate results to {intermediate_filename}")
            
            # Save current batch of augmented data
            if current_batch:
                save_augmented_data(current_batch, augmented_filename, len(results))
                current_batch = []
        progress_bar.update(1)
    progress_bar.close()

    if not results:
        print("\nNo examples were successfully processed.")
        return

    results.sort(key=lambda x: x['id'])

    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_count / len(results)) * 100 if results else 0

    print("\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    print(f"Final Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")

    # Save final results
    os.makedirs('synthetic_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join('synthetic_results', 
                                  f"synthetic_{timestamp}.json")
    
    # Save results with error rate points
    output_data = {
        'results': results,
        'error_rate_points': error_rate_points,
        'final_error_rate': calculate_error_rate(results),
        'accuracy': {
            'correct_count': correct_count,
            'total_count': len(results),
            'accuracy_percentage': (correct_count / len(results) * 100) if results else 0
        }
    }
    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {results_filename}")
    
    # Save final augmented data batch
    if current_batch:
        save_augmented_data(current_batch, augmented_filename, len(results))
    
    # Print error rate progression
    print("\nError Rate Progression:")
    for point in error_rate_points:
        print(f"After {point['examples_processed']} examples: {point['error_rate']:.4f}")

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
