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

SYSTEM_PROMPT = """You are a mathematical problem solver. When given a problem and partial solution, complete the solution.
Make sure to:
- Follow the reasoning pattern shown in the partial solution
- Complete the remaining steps clearly
- Provide your final answer inside \\boxed{}"""

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

def normalize_math_answer(answer: str) -> str:
    """Normalize mathematical answer for comparison"""
    if not answer:
        return ""
    
    answer = ''.join(answer.split())
    answer = answer.lower()
    
    latex_commands = [r'\text', r'\left', r'\right', r'\begin', r'\end', 
                     r'\frac', r'\sqrt', r'\cdot']
    for cmd in latex_commands:
        answer = answer.replace(cmd, '')
    
    replacements = {
        '{': '', '}': '', '\\': '', '÷': '/',
        '×': '*', '⋅': '*', '−': '-', '–': '-', '—': '-',
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)
    
    return answer

def compare_math_answers(model_answer: Optional[str], correct_answer: Optional[str]) -> bool:
    """Compare two mathematical answers after normalization"""
    if model_answer is None or correct_answer is None:
        return False
    return normalize_math_answer(model_answer) == normalize_math_answer(correct_answer)

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
        is_correct = compare_math_answers(model_answer, correct_answer)
        
        status = '✓' if is_correct else '✗'
        print(f"\nProblem {idx + 1}: {status}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        return {
            'id': idx,
            'problem': example['problem'],
            'partial_solution': partial_solution,
            'correct_answer': correct_answer,
            'model_solution': solution,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'normalized_model_answer': normalize_math_answer(model_answer) if model_answer else None,
            'normalized_correct_answer': normalize_math_answer(correct_answer) if correct_answer else None
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
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for concurrent processing (default: 1)')
    args = parser.parse_args()

    if args.batch_size < 1:
        print("Error: Batch size must be at least 1")
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
    
    results = []
    progress_bar = tqdm(total=len(example_data), desc="Processing examples")

    for i in range(0, len(example_data), args.batch_size):
        batch = example_data[i:i + args.batch_size]
        batch_results = await asyncio.gather(
            *[process_example(ex, ex['id'], model) for ex in batch]
        )
        valid_results = [r for r in batch_results if r]
        results.extend(valid_results)
        progress_bar.update(len(batch))
    
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
    
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_filename}")

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
