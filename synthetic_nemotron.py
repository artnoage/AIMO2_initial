import os
import re
import json
import asyncio
import argparse
from enum import Enum
from typing import Optional, List, Dict
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

def get_model(temp: float = 0.1):
    """Initialize the NEMOTRON model"""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
    
    return ChatOpenAI(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
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


async def compare_math_answers(model_answer: Optional[str], correct_answer: Optional[str], model) -> bool:
    """Use the model to compare two mathematical answers"""
    if model_answer is None or correct_answer is None:
        print("\nSkipping comparison - one or both answers are None")
        return False
        
    print("\n=== Answer Comparison ===")
    print(f"Model Answer: {model_answer}")
    print(f"Correct Answer: {correct_answer}")
    
    comparison_prompt = [
        SystemMessage(content="You are a mathematical answer validator. Given two answers to a math problem, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
        HumanMessage(content=f"Are these two mathematical answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
    ]
    
    try:
        print("\nAsking model to verify...")
        response = await model.ainvoke(comparison_prompt)
        result = response.content.strip().lower() == 'yes'
        print(f"Model's response: {response.content}")
        print(f"Comparison result: {'Equivalent' if result else 'Different'}")
        print("=" * 30)
        return result
    except Exception as e:
        print(f"\nError during comparison: {e}")
        print("=" * 30)
        return False

def get_partial_solution(solution: str) -> str:
    """Get partial solution by removing last two lines"""
    lines = solution.strip().split('\n')
    if len(lines) <= 2:
        return solution
    return '\n'.join(lines[:-2])

async def process_example(example: Dict, idx: int, model) -> Optional[Dict]:
    """Process a single example"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {idx}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {idx}")
            return None

        # Combine problem with partial solution
        partial_solution = get_partial_solution(example['solution'])
        combined_prompt = f"{example['problem']}\n\nPartial solution:\n{partial_solution}"
        
        prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=combined_prompt)
        ]
        
        response = await model.ainvoke(prompt)
        solution = response.content
        model_answer = extract_answer_from_solution(solution)

        print(f"\n\nProcessing Problem {idx + 1}:")
        print(f"Problem text: {example['problem'][:200]}...")
        
        is_correct = await compare_math_answers(model_answer, correct_answer, model)
        
        status = '✓' if is_correct else '✗'
        print(f"\nFinal Result for Problem {idx + 1}: {status}")
        print("-" * 80)
        
        return {
            'id': idx,
            'problem': example['problem'],
            'partial_solution': partial_solution,
            'correct_answer': correct_answer,
            'model_solution': solution,
            'model_answer': model_answer,
            'is_correct': is_correct
        }
        
    except Exception as e:
        print(f"Error processing example {idx}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Synthetic NEMOTRON Benchmark')
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

    model = get_model(temp=0.1)
    print(f"\nBenchmarking NEMOTRON model on {args.split} split...")

    example_data = [{'id': idx, 'problem': ex['problem'], 'solution': ex['solution']} 
                   for idx, ex in enumerate(dataset)]
    
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

    async def process_with_semaphore(example):
        async with semaphore:
            return await process_example(example, example['id'], model)

    # Create tasks for all examples
    tasks = [process_with_semaphore(ex) for ex in example_data]
    
    # Process all examples with progress bar
    progress_bar = tqdm(total=total_examples, desc="Processing examples")
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            # Calculate error rate every 2000 points
            if len(results) % 2000 == 0:
                current_error_rate = calculate_error_rate(results)
                error_rate_points.append({
                    'examples_processed': len(results),
                    'error_rate': current_error_rate,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"\nIntermediate Error Rate at {len(results)} examples: {current_error_rate:.4f}")
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

    # Save results
    os.makedirs('synthetic_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join('synthetic_results', f"synthetic_nemotron_{timestamp}.json")
    
    # Save results with error rate points
    output_data = {
        'results': results,
        'error_rate_points': error_rate_points,
        'final_error_rate': calculate_error_rate(results)
    }
    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {results_filename}")
    
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
