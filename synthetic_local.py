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
import random

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """You are a mathematical problem solver. When given a problem, first analyzie and hypothesize on 
the tools you have to use. After, solve it step by step, 
showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

In the end provide your final answer inside \\boxed{}"""

def get_model(temp: float = 0.1):
    """Initialize the local model with specified temperature"""
    return ChatOpenAI(
        model="mistralai/Mathstral-7B-v0.1",
        temperature=temp,
        api_key="EMPTY",
        base_url="http://localhost:8000/v1")

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

async def process_example(example: Dict, idx: int, model, num_responses: int = 3) -> Optional[Dict]:
    """Process a single example multiple times and collect results"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {idx}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {idx}")
            return None

        prompt = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=example["problem"])]
        
        # Generate multiple responses
        responses = []
        correct_responses = []
        wrong_responses = []
        
        for i in range(num_responses):
            response = await model.ainvoke(prompt)
            solution = response.content
            model_answer = extract_answer_from_solution(solution)
            is_correct = compare_math_answers(model_answer, correct_answer)
            
            responses.append({
                'solution': solution,
                'answer': model_answer,
                'is_correct': is_correct
            })
            
            if is_correct:
                correct_responses.append(solution)
            else:
                wrong_responses.append(solution)
        
        # Calculate score for this example
        correct_count = sum(1 for r in responses if r['is_correct'])
        score = (correct_count / num_responses) * 100
        
        # Determine overall correctness (at least one correct answer)
        is_correct = any(r['is_correct'] for r in responses)
        status = '✓' if is_correct else '✗'
        
        print(f"\nProblem {idx + 1}: {status} (Score: {score:.2f}%)")
        print(f"Correct Answer: {correct_answer}")
        print(f"Model Answers: {[r['answer'] for r in responses]}")
        print("-" * 80)
        
        return {
            'id': idx,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'responses': responses,
            'score': score,
            'is_correct': is_correct,
            'model_correct_response': random.choice(correct_responses) if correct_responses else None,
            'model_wrong_response': random.choice(wrong_responses) if wrong_responses else None
        }
        
    except Exception as e:
        print(f"Error processing example {idx}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Synthetic Local Benchmark')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--responses', type=int, default=3,
                       help='Number of responses to generate per problem')
    args = parser.parse_args()

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
    print(f"\nBenchmarking local model on {args.split} split...")

    example_data = [{'id': idx, 'problem': ex['problem'], 'solution': ex['solution']} 
                   for idx, ex in enumerate(dataset)]
    
    results = []
    progress_bar = tqdm(total=len(example_data), desc="Processing examples")
    
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for concurrent processing (default: 1)')
    args = parser.parse_args()

    # Validate batch size
    if args.batch_size < 1:
        print("Error: Batch size must be at least 1")
        return

    # Process all examples concurrently in batches
    for i in range(0, len(example_data), args.batch_size):
        batch = example_data[i:i + args.batch_size]
        # Process batch concurrently
        batch_results = await asyncio.gather(
            *[process_example(ex, ex['id'], model, args.responses) for ex in batch]
        )
        valid_results = [r for r in batch_results if r]
        results.extend(valid_results)
        progress_bar.update(len(batch))
    
    progress_bar.close()

    if not results:
        print("\nNo examples were successfully processed.")
        return

    results.sort(key=lambda x: x['id'])

    # Calculate final statistics
    binary_score = sum(1 for r in results if r['is_correct']) / len(results) * 100
    average_score = sum(r['score'] for r in results) / len(results)

    print("\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    print(f"Binary Score (✓/✗): {binary_score:.2f}%")
    print(f"Average Score: {average_score:.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs('synthetic_results', exist_ok=True)
    results_filename = os.path.join('synthetic_results', f"synthetic_results_{timestamp}.json")
    
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_filename}")

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
