import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark model on NuminaMath-CoT dataset')
    parser.add_argument('--model', type=str, choices=[model.name for model in ModelOption],
                       default='NOUS', help='Model to benchmark')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--label', type=str, default='all',
                       help='Filter problems by source label (default: all)')
    args = parser.parse_args()

    # Load dataset and shuffle
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
    
    # Filter by label if specified
    if args.label.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.label)
    
    dataset = dataset.shuffle()
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Number of examples: {len(dataset)}")
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
    
    results = []
    correct_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for idx, example in enumerate(tqdm(dataset)):
        try:
            # Prepare input
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example['problem']}
            ]
            
            # Count input tokens
            input_text = "\n".join(msg["content"] for msg in messages)
            input_tokens = len(enc.encode(input_text))
            total_input_tokens += input_tokens
            
            # Get model response
            response = model.invoke(messages)
            solution = response.content
            
            # Count output tokens
            output_tokens = len(enc.encode(solution))
            total_output_tokens += output_tokens
            
            # Extract answers from solutions
            model_answer = extract_answer_from_solution(solution)
            correct_answer = extract_answer_from_solution(example['solution'])
            if correct_answer is None:
                print(f"Warning: Could not extract answer from solution for example {idx}")
                continue
            is_correct = model_answer == correct_answer
            
            if is_correct:
                correct_count += 1
            
            # Store result
            result = {
                'id': idx,
                'problem': example['problem'],
                'model_solution': solution,
                'model_answer': model_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
            results.append(result)
            
            # Print running statistics
            print(f"\nProblem {idx + 1}:")
            print(f"Model Answer: {model_answer}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Correct: {is_correct}")
            print(f"Running Accuracy: {correct_count}/{idx + 1} = {correct_count/(idx + 1):.2%}")
            
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
    
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
