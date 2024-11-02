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
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import HfApi
from tqdm import tqdm
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Load environment variables from .env file
load_dotenv()

class ModelOption(Enum):
    CLAUDE = "anthropic/claude-3.5-sonnet:beta"
    GEMINI_PRO_FREE = "google/gemini-pro-1.5-exp"
    GEMINI_FLASH_FREE="google/gemini-flash-1.5-exp"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GEMINI_FLASH="google/gemini-flash-1.5"
    GPT = "openai/gpt-4o"
    GPT_MINI="openai/gpt-4o-mini"
    MASTER = "openai/o1-preview-2024-09-12"
    LOCAL = "mistralai/Mathstral-7B-v0.1"
    GROQ = "llama-3.1-70b-versatile"
    NOUS ="nousresearch/hermes-3-llama-3.1-405b:free"

SYSTEM_PROMPT = """You are a mathematical problem solver. When given a problem, solve it step by step, 
showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

Never ask for confirmation. Just provide your final answer inside \\boxed{}"""

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
    else:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=openrouter_api_key)

import re
from typing import Optional

def extract_answer_from_solution(solution: str) -> Optional[str]:
    """
    Extract the first boxed answer from the solution text by searching for LaTeX boxed answers: \boxed{X}.
    Returns the raw answer string with LaTeX notation preserved, or None if no boxed answer is found.
    """
    def find_matching_brace(s: str, start: int) -> int:
        """
        Find the index of the matching closing brace for an opening brace at the given start position.
        
        Args:
            s (str): The string to search.
            start (int): The index of the opening brace '{'.
        
        Returns:
            int: The index of the matching closing brace '}', or -1 if not found.
        """
        count = 1  # Initialize brace count
        i = start + 1  # Start searching after the opening brace
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Pattern to find all occurrences of \boxed{ with proper escaping
    pattern = re.compile(r'\\boxed\{')
    for match in pattern.finditer(solution):
        start = match.end() - 1  # Position of the opening brace '{'
        end = find_matching_brace(solution, start)
        if end != -1:
            # Extract content between the braces
            content = solution[start + 1:end].strip()
            return content  # Return the first found boxed content

    return None  # Return None if no boxed content is found


def save_results(results: list, model_name: str):
    """
    Save the benchmarking results to a JSON file within the benchmark_results directory.
    The filename includes the model name and a timestamp for uniqueness.
    """
    os.makedirs('benchmark_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join('benchmark_results', f"benchmark_results_{model_name}_{timestamp}.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def normalize_math_answer(answer: str) -> str:
    """
    Normalize a mathematical answer for comparison by:
    - Removing all whitespace
    - Converting to lowercase
    - Removing LaTeX formatting commands
    - Normalizing common mathematical notations
    """
    if not answer:
        return ""
    
    # Remove all whitespace
    answer = ''.join(answer.split())
    
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove common LaTeX commands
    latex_commands = [r'\text', r'\left', r'\right', r'\begin', r'\end', 
                     r'\frac', r'\sqrt', r'\cdot']
    for cmd in latex_commands:
        answer = answer.replace(cmd, '')
    
    # Normalize mathematical notations
    replacements = {
        '{': '',
        '}': '',
        '\\': '',
        '÷': '/',
        '×': '*',
        '⋅': '*',
        '−': '-',
        '–': '-',
        '—': '-',
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)
    
    return answer

def compare_math_answers(model_answer: Optional[str], correct_answer: Optional[str]) -> bool:
    """
    Compare two mathematical answers after normalization.
    Returns True if they are equivalent, False otherwise.
    """
    if model_answer is None or correct_answer is None:
        return False
    
    normalized_model = normalize_math_answer(model_answer)
    normalized_correct = normalize_math_answer(correct_answer)
    
    return normalized_model == normalized_correct

async def process_example(example: Dict, idx: int, model) -> Optional[Dict]:
    """
    Process a single example and print its results immediately:
    - Count input tokens
    - Extract the correct answer from the solution
    - Generate the solution using the model
    - Extract the model's answer
    - Count output tokens
    - Determine correctness
    - Print results
    """
    try:
        # Validate input data
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {idx}: Invalid example format")
            return None
            
        # Extract the correct answer
        try:
            correct_answer = extract_answer_from_solution(example['solution'])
            if correct_answer is None:
                print(f"Warning: Could not extract answer from solution for example {idx}")
                print(f"Solution text: {example['solution']}...")
                return None
        except Exception as e:
            print(f"Error extracting answer from solution for example {idx}: {str(e)}")
            return None
        # Create the chat prompt
        prompt = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        # Generate the solution using the model
        response = await model.ainvoke(prompt)  # Await the async response
        solution = response.content
        
        # Extract the model's answer from the solution
        model_answer = extract_answer_from_solution(solution)
        # Compare answers using robust comparison
        is_correct = compare_math_answers(model_answer, correct_answer)
        
        # Print results immediately
        status = '✓' if is_correct else '✗'
        print(f"\nProblem {idx + 1}: {status}")
        print(f"Extracted Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)

        # Return the result
        return {
            'id': idx,
            'problem': example['problem'],
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
    # Start timing the entire process
    start_time = datetime.now()
    
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Benchmark model on NuminaMath-CoT dataset')
    parser.add_argument('--model', type=str, choices=[model.name for model in ModelOption],
                       default='NOUS', help='Model to benchmark')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--dataset', type=str, default='filtered',
                       choices=['original', 'filtered'],
                       help='Dataset to use: original (AI-MO/NuminaMath-CoT) or filtered (Numina-Olympiads)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for concurrent processing (default: 1)')
    args = parser.parse_args()

    # Validate batch size
    if args.batch_size < 1:
        print("Error: Batch size must be at least 1")
        return
    if args.batch_size > 20:
        print("Warning: Large batch sizes may cause memory issues")
    args = parser.parse_args()

    # Load the dataset based on selection
    try:
        if args.dataset == 'original':
            dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
        else:  # filtered
            username = HfApi().whoami()["name"]
            dataset = load_dataset(f"{username}/Numina-Olympiads", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter by source if specified
    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    # Shuffle the dataset for randomness
    dataset = dataset.shuffle(seed=42)

    # Print dataset information
    print("\nDataset Information:")
    num_examples = len(dataset)
    print(f"Number of examples: {num_examples}")

    if num_examples == 0:
        print("Error: Dataset is empty! Check your source filter and split arguments.")
        return

    # Initialize the model
    try:
        model = get_model(ModelOption[args.model])
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    print(f"\nBenchmarking {args.model} on {args.split} split...")


    # Prepare the list of examples to process
    example_data = []
    for idx, example in enumerate(dataset):
        processed = {
            'id': idx,
            'problem': example['problem'],
            'solution': example['solution']
        }
        example_data.append(processed)
    
    if not example_data:
        print("No valid examples to process after initial filtering.")
        return

    # Process all examples concurrently in batches of 8
    results = []
    total_examples = len(example_data)
    print(f"\nStarting processing of {total_examples} examples...")

    progress_bar = tqdm(total=total_examples, desc="Processing examples")
    for i in range(0, len(example_data), args.batch_size):
        batch = example_data[i:i + args.batch_size]
        # Process batch concurrently
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

    # Sort results by ID to maintain the original order
    results.sort(key=lambda x: x['id'])

    # Calculate final statistics
    correct_count = sum(1 for r in results if r['is_correct'])

    # Print final results
    print("\n\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    
    if len(results) > 0:
        accuracy = (correct_count / len(results)) * 100
        print(f"Final Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")
    else:
        print("No examples were successfully processed.")

    # Save the results to JSON files
    try:
        # Save benchmark results
        save_results(results, args.model)
        
        # Create augmented dataset
        augmented_data = []
        for result in results:
            augmented_example = {
                'id': result['id'],
                'problem': result['problem'],
                'solution': dataset[result['id']]['solution'],  # Original solution
                'model_response': result['model_solution'],  # Full model response
                'is_correct': result['is_correct']
            }
            augmented_data.append(augmented_example)
            
        # Save augmented dataset
        os.makedirs('augmented_datasets', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        augmented_filename = os.path.join('augmented_datasets', 
                                        f"augmented_dataset_{args.model}_{args.dataset}_{timestamp}.json")
        
        with open(augmented_filename, 'w') as f:
            json.dump(augmented_data, f, indent=2)
        print(f"\nAugmented dataset saved to {augmented_filename}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

    # Calculate and print timing information
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
