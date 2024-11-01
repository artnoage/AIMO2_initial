import os
import re
import json
import argparse
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import tiktoken

# Load environment variables from .env file
load_dotenv()

class ModelOption(Enum):
    CLAUDE = "anthropic/claude-3.5-sonnet:beta"
    GEMINI_PRO_FREE = "google/gemini-pro-1.5-exp"
    GEMINI_FLASH_FREE = "google/gemini-flash-1.5-exp"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GEMINI_FLASH = "google/gemini-flash-1.5"
    GPT = "gpt-4"  # Corrected model name
    GPT_MINI = "gpt-4-mini"  # Assuming this is the intended name
    MASTER = "o1-preview-2024-09-12"  # Ensure this is the correct model name
    LOCAL = "mistralai/Mathstral-7B-v0.1"
    GROQ = "llama-3.1-70b-versatile"
    NOUS = "nousresearch/hermes-3-llama-3.1-405b:free"

SYSTEM_PROMPT = """You are a mathematical problem solver. When given a problem, solve it step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

Never ask for confirmation. Just provide your final answer as a number at the end of your 
response prefixed with 'ANSWER: '."""

def get_model(model: ModelOption, temp: float = 0.1) -> ChatOpenAI:
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
            base_url="http://localhost:8000/v1",
            request_timeout=60  # Added timeout
        )
    else:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            request_timeout=60  # Added timeout
        )

def extract_answer_from_solution(solution: str) -> Optional[int]:
    """
    Extract the numerical answer from the solution text by searching for patterns like
    'ANSWER: 42' or 'answer is 42'. Falls back to the last number in the text if no pattern is found.
    """
    # Regex to find 'ANSWER: <number>' or 'answer is <number>'
    answer_pattern = re.compile(r'(?:ANSWER:\s*|answer\s+is\s*)(-?\d+)', re.IGNORECASE)
    matches = answer_pattern.findall(solution)
    if matches:
        return int(matches[-1])
    
    # Fallback: return the last number in the solution
    numbers = re.findall(r'-?\d+', solution)
    return int(numbers[-1]) if numbers else None

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

def process_example(example: Dict, idx: int, enc, model: ChatOpenAI) -> Optional[Dict]:
    """
    Process a single example:
    - Count input tokens
    - Extract the correct answer from the solution
    - Generate the solution using the model
    - Extract the model's answer
    - Count output tokens
    - Determine correctness
    """
    try:
        # Prepare the input text
        input_text = f"{SYSTEM_PROMPT}\n{example['problem']}"
        input_tokens = len(enc.encode(input_text))
        
        # Extract the correct answer
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {idx}")
            return None
        
        # Create the chat prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", example['problem'])
        ])
        
        # Generate the solution using the model
        response = model.invoke(prompt.format_messages())
        solution = response.content
        
        # Extract the model's answer from the solution
        model_answer = extract_answer_from_solution(solution)
        is_correct = model_answer == correct_answer
        
        # Count output tokens
        output_tokens = len(enc.encode(solution))
        
        # Compile the result
        return {
            'id': idx,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_solution': solution,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
        
    except Exception as e:
        print(f"Error processing example {idx}: {e}")
        return None

def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Benchmark model on NuminaMath-CoT dataset')
    parser.add_argument('--model', type=str, choices=[model.name for model in ModelOption],
                       default='NOUS', help='Model to benchmark')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--concurrency', type=int, default=4,
                       help='Number of concurrent tasks (default: 4)')
    args = parser.parse_args()

    # Load the dataset
    try:
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
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
        
    print("\nAvailable keys in each example:")
    example = dataset[0]
    for key in example.keys():
        print(f"- {key}: {type(example[key]).__name__}")
        if isinstance(example[key], str):
            print(f"  Sample value: {example[key][:100]}...")

    # Initialize the model
    try:
        model = get_model(ModelOption[args.model])
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    print(f"\nBenchmarking {args.model} on {args.split} split...")

    # Initialize the tokenizer for token counting
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return

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

    # Initialize ThreadPoolExecutor for multi-threaded processing
    results = []
    total_examples = len(example_data)
    processed = 0
    print(f"\nStarting processing of {total_examples} examples with concurrency={args.concurrency}...")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # Submit all tasks to the executor
        future_to_example = {
            executor.submit(process_example, ex, ex['id'], enc, model): ex for ex in example_data
        }
        
        for future in as_completed(future_to_example):
            ex = future_to_example[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    # Indicate success or failure
                    status = '✓' if result['is_correct'] else '✗'
                    print(f"Problem {result['id'] + 1}: {status}")
                else:
                    print(f"Problem {ex['id'] + 1}: Failed to process.")
            except Exception as e:
                print(f"Problem {ex['id'] + 1}: Exception occurred: {e}")
            finally:
                processed += 1
                # Optional: Display progress
                print(f"Progress: {processed}/{total_examples} examples processed", end='\r')

    if not results:
        print("\nNo examples were successfully processed.")
        return

    # Sort results by ID to maintain the original order
    results.sort(key=lambda x: x['id'])

    # Calculate final statistics
    correct_count = sum(1 for r in results if r['is_correct'])
    total_input_tokens = sum(r['input_tokens'] for r in results)
    total_output_tokens = sum(r['output_tokens'] for r in results)

    # Print final results
    print("\n\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    
    if len(results) > 0:
        accuracy = (correct_count / len(results)) * 100
        avg_input_tokens = total_input_tokens / len(results)
        avg_output_tokens = total_output_tokens / len(results)
        print(f"Final Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Average input tokens per problem: {avg_input_tokens:.1f}")
        print(f"Average output tokens per problem: {avg_output_tokens:.1f}")
    else:
        print("No examples were successfully processed.")

    # Save the results to a JSON file
    try:
        save_results(results, args.model)
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
