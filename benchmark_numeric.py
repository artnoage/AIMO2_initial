import os
import json
import asyncio
import argparse
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from utils.utils import (
    ModelOption, 
    get_model, 
    extract_answer_from_solution, 
    async_retry,
    NUMERIC_SOLVER_SYSTEM_PROMPT
)
from utils.progress_tracker import ProgressTracker
from langchain_core.messages import HumanMessage, SystemMessage
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm


def extract_numeric_answer(solution: str) -> Optional[float]:
    """
    Extract numeric answer from a solution string.
    Looks for a number inside a LaTeX boxed environment.
    Returns float if found, None otherwise.
    """
    
    # First extract the raw boxed content
    raw_answer = extract_answer_from_solution(solution)
    if raw_answer is None:
        return None
        
    # Clean the answer and try to convert to float
    clean_answer = raw_answer.strip()
    try:
        return float(clean_answer)
    except ValueError:
        return None

def is_answer_correct(model_answer: Optional[float], correct_answer: Optional[float], tolerance: float = 0.01) -> bool:
    """Compare two numeric answers within tolerance"""
    if model_answer is None or correct_answer is None:
        return False
    return abs(model_answer - correct_answer) <= tolerance

@async_retry(max_retries=3, timeout=300)
async def get_model_response(solver_model, prompt, running_id: int, attempt: int) -> str:
    """Get response from model with retry logic"""
    response = await solver_model.ainvoke(prompt)
    return response.content

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, best_of: int = 1) -> Optional[Dict]:
    """Process a single example and return results"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        # Extract the correct answer
        try:
            correct_answer = extract_numeric_answer(example['solution'])
            if correct_answer is None:
                print(f"Warning: Could not extract numeric answer from solution for example {running_id}")
                return None
        except Exception as e:
            print(f"Error extracting answer from solution for example {running_id}: {str(e)}")
            return None

        prompt = [SystemMessage(content=NUMERIC_SOLVER_SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
        for attempt in range(best_of):
            try:
                current_solution = await get_model_response(solver_model, prompt, running_id, attempt)
                current_answer = extract_numeric_answer(current_solution)
                is_correct = is_answer_correct(current_answer, correct_answer)
                
                if is_correct:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
                        best_answer = current_answer
            except Exception as e:
                print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
                current_solution = "Error occurred"
                current_answer = None
                is_correct = False
            
            solutions.append({
                'solution': current_solution,
                'answer': current_answer,
                'is_correct': is_correct
            })
            
            if attempt >= best_of - 1:
                break
        
        solution = best_solution if best_solution is not None else solutions[0]['solution']
        model_answer = best_answer if best_answer is not None else solutions[0]['answer']
        
        success_ratio = f"{correct_count}/{best_of}"
        success_percentage = (correct_count / best_of) * 100
        print(f"\nProblem {running_id + 1}: {success_ratio} ({success_percentage:.1f}%)")
        print(f"Correct Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'model_answer_raw': model_answer,
            'correct_answer_raw': correct_answer,
            'attempts': {
                'total': len(solutions),
                'correct_count': correct_count
            }
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Benchmark model on numeric problems')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL', help='Model to use for solving problems')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--max-concurrent', type=int, default=256,
                       help='Maximum number of concurrent problems (default: 16)')
    parser.add_argument('--best-of', type=int, default=5,
                       help='Number of attempts per problem (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for model generation (default: 0.7)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                       help='Tolerance for numeric comparison (default: 0.001)')
    args = parser.parse_args()

    if args.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    dataset = dataset.shuffle(seed=42)

    print("\nDataset Information:")
    num_examples = len(dataset)
    print(f"Number of examples: {num_examples}")

    if num_examples == 0:
        print("Error: Dataset is empty! Check your source filter and split arguments.")
        return

    try:
        solver_model = get_model(ModelOption[args.solver], temp=args.temperature)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    print(f"\nBenchmarking solver: {args.solver} on {args.split} split...")

    example_data = []
    for example in dataset:
        processed = {
            'id': example['id'],
            'problem': example['problem'],
            'solution': example['solution']
        }
        example_data.append(processed)
    
    if not example_data:
        print("No valid examples to process after initial filtering.")
        return

    progress_tracker = ProgressTracker(total_examples=len(example_data), best_of=args.best_of)
    print(f"\nStarting processing of {progress_tracker.total_examples} examples...")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, example['id'], solver_model, args.best_of)

    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
    
    if not example_data:
        print("No examples to process!")
        return
        
    print(f"\nWill process {len(example_data)} examples")
    
    results = []
    progress_bar = tqdm(total=len(example_data), desc="Processing examples")
    current_batch = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            augmented_example = {
                'id': result['id'],
                'problem': result['problem'],
                'solution': str(result['correct_answer']),
                'model_responses': result['model_responses'],
                'is_correct_list': result['is_correct_list'],
                'solver': args.solver,
                'total_attempts': len(result['model_responses']),
                'correct_answer': result['correct_answer']
            }
            current_batch.append(augmented_example)
            
            progress_tracker.add_result(result)
            progress_tracker.print_progress()
            
        progress_bar.update(1)
    progress_bar.close()

    progress_bar.close()
    progress_tracker.print_final_stats()

if __name__ == "__main__":
    asyncio.run(main())
