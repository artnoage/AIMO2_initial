import os
from enum import Enum
from typing import Optional, List, Dict
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import tiktoken
from tqdm import tqdm
import json
from datetime import datetime

# Load environment variables
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

SYSTEM_PROMPT = """You are a mathematical problem solver. When given a problem, solve it step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

Never ask for confirmation. Just provide your final answer as a number at the end of your 
response prefixed with 'ANSWER: '."""

def get_model(model: ModelOption, temp: float = 0.1):
    """Initialize the model with OpenRouter"""
    if model == ModelOption.LOCAL:
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
    else:
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

def extract_answer_from_solution(solution: str) -> Optional[int]:
    """Extract numerical answer from the solution text by looking for the final number"""
    import re
    # Look for numbers after "answer is" or "answer:" case insensitive
    answer_pattern = re.compile(r'(?:answer\s+is|answer:)\s*(-?\d+)', re.IGNORECASE)
    matches = answer_pattern.findall(solution)
    if matches:
        # Return the last match as it's likely the final answer
        return int(matches[-1])
    # Fallback: look for the last number in the text
    numbers = re.findall(r'-?\d+', solution)
    return int(numbers[-1]) if numbers else None

def save_results(results: list, model_name: str):
    """Save results to a JSON file in benchmark_results directory"""
    # Create benchmark_results directory if it doesn't exist
    os.makedirs('benchmark_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join('benchmark_results', f"benchmark_results_{model_name}_{timestamp}.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def process_example(example: Dict, idx: int, enc) -> Optional[Dict]:
    """Process a single example and prepare result dict"""
    try:
        # Count input tokens
        input_text = f"{SYSTEM_PROMPT}\n{example['problem']}"
        input_tokens = len(enc.encode(input_text))
        
        # Extract answers from solutions
        correct_answer = extract_answer_from_solution(example['solution'])
        
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {idx}")
            return None
            
        # Return dict with everything except model results
        return {
            'id': idx,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'input_tokens': input_tokens,
        }
        
    except Exception as e:
        print(f"Error processing example {idx}: {e}")
        return None

def process_model_result(result: Dict, example_data: Dict, enc) -> Optional[Dict]:
    """Process model output and combine with example data"""
    try:
        solution = result.content
        output_tokens = len(enc.encode(solution))
        model_answer = extract_answer_from_solution(solution)
        is_correct = model_answer == example_data['correct_answer']
        
        # Combine with example data
        return {
            **example_data,
            'model_solution': solution,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'output_tokens': output_tokens
        }
    except Exception as e:
        print(f"Error processing model result: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark model on NuminaMath-CoT dataset')
    parser.add_argument('--model', type=str, choices=[model.name for model in ModelOption],
                       default='NOUS', help='Model to benchmark')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--concurrency', type=int, default=8,
                       help='Number of concurrent tasks (default: 8)')
    args = parser.parse_args()

    # Load dataset and shuffle
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
    
    # Filter by source if specified
    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    dataset = dataset.shuffle()
    
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
    
    # Initialize model
    model = get_model(ModelOption[args.model])
    print(f"\nBenchmarking {args.model} on {args.split} split...")
    
    # Setup tokenizer for counting
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Create the chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{problem}")
    ])

    # Create the chain
    chain = prompt | model

    # Process examples and prepare inputs
    example_data = []
    with tqdm(total=len(dataset), desc="Preparing examples") as pbar:
        for idx, example in enumerate(dataset):
            result = process_example(example, idx, enc)
            if result:
                example_data.append(result)
            pbar.update(1)

    # Prepare batch inputs
    inputs = [{"problem": ex["problem"]} for ex in example_data]
    
    # Process batches
    results = []
    with tqdm(total=len(inputs), desc="Processing with model") as pbar:
        for batch_outputs in chain.batch(
            inputs, 
            {"max_concurrency": args.concurrency},
            batch_size=args.concurrency
        ):
            # Process each result in the batch
            for output, ex_data in zip(batch_outputs, example_data[len(results):len(results)+len(batch_outputs)]):
                result = process_model_result(output, ex_data, enc)
                if result:
                    results.append(result)
                    # Print progress
                    print(f"\nProblem {result['id'] + 1}:")
                    print(f"Model Answer: {result['model_answer']}")
                    print(f"Correct Answer: {result['correct_answer']}")
                    print(f"Correct: {result['is_correct']}")
            pbar.update(len(batch_outputs))
    
    # Sort results by ID to maintain order
    results.sort(key=lambda x: x['id'])
    
    # Calculate final statistics
    correct_count = sum(1 for r in results if r['is_correct'])
    total_input_tokens = sum(r['input_tokens'] for r in results)
    total_output_tokens = sum(r['output_tokens'] for r in results)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    
    if len(results) > 0:
        print(f"Final Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Average input tokens per problem: {total_input_tokens/len(results):.1f}")
        print(f"Average output tokens per problem: {total_output_tokens/len(results):.1f}")
    else:
        print("No examples were successfully processed")
    
    # Save results
    save_results(results, args.model)

if __name__ == "__main__":
    main()
