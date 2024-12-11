import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
from tqdm import tqdm
import numpy as np
from bench_utils.benchmark_utils import extract_numeric_answer, extract_answer_from_solution, is_answer_correct


def setup_model(model_path: str):
    """Load model and tokenizer, setting up for GPU inference."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")
    
    # Ensure we're using GPU device 1
    torch.cuda.set_device(1)
    
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def generate_solution(model, tokenizer, problem: str, temperature: float = 0.0, max_length: int = 4096):
    """Generate a solution for a given problem."""
    prompt = f"""[INST]Here is a mathematical problem to solve:\n\n{problem}\n\n
                Please provide a complete solution following these guidelines:\n
                1. Start with '**Problem Analysis and Approach**:' section explaining:\n
                   - Problem type and key concepts involved\n
                   - Relevant theorems and techniques\n
                   - Overall solution strategy\n\n
                2. Then provide a detailed step-by-step solution:\n
                   - Number each step clearly (Step 1, Step 2, etc.)\n
                   - Show all work and intermediate calculations\n
                   - Use LaTeX notation for mathematical expressions\n
                   - Provide justification in [brackets] for key steps\n
                   - End with final answer in \\boxed{{}}\n\n [/INST]"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=temperature > 0.0,
            **({'temperature': temperature} if temperature > 0.0 else {}),
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_example(model, tokenizer, example, attempt: int, temperature: float) -> bool:
    """Process a single example and return if the answer was correct."""
    try:
        solution = generate_solution(model, tokenizer, example['problem'], temperature)
        
        # Extract and compare answers
        model_answer = extract_answer_from_solution(solution)
        correct_answer = extract_answer_from_solution(example['solution'])
        
        # Convert to numeric values
        model_numeric, model_debug = extract_numeric_answer(model_answer, debug=True)
        correct_numeric, correct_debug = extract_numeric_answer(correct_answer, debug=True)
        
        # Check correctness
        is_correct = is_answer_correct(model_numeric, correct_numeric, 0.001)
        
        if is_correct:
            print(f"✓ Example {example['id']} correct on attempt {attempt + 1}")
        elif attempt == 0:  # Only print failures on first attempt
            print(f"✗ Example {example['id']} incorrect")
            print(f"  Model boxed: {model_answer}")
            print(f"  Model numeric: {model_numeric} ({model_debug})")
            print(f"  Correct boxed: {correct_answer}") 
            print(f"  Correct numeric: {correct_numeric} ({correct_debug})")
            
        return is_correct
        
    except Exception as e:
        if attempt == 0:  # Only print errors on first attempt
            print(f"! Error on example {example['id']}: {str(e)}")
        return False

def print_final_stats(correct_count: int, total: int, args: argparse.Namespace):
    """Print final benchmark statistics."""
    print("\nFinal Statistics:")
    print(f"Total examples: {total}")
    print(f"Correct solutions: {correct_count}")
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nBenchmark settings:")
    print(f"Model: {args.model_path}")
    print(f"Attempts per problem: {args.best_of}")
    print(f"Dataset split: {args.split}")

def main():
    parser = argparse.ArgumentParser(description='Run GPU-accelerated benchmark with multiple attempts per problem')
    parser.add_argument('--model_path', type=str, required=True, help='Path to local model')
    parser.add_argument('--best_of', type=int, default=1, help='Number of attempts per problem')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation (default: 0.0)')
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = setup_model(args.model_path)
    
    # Load dataset
    dataset = load_dataset("artnoage/Numina", split=args.split)
    print(f"\nLoaded {len(dataset)} examples from dataset")
    
    correct_count = 0
    total = len(dataset)
    
    # Process each example
    for example in tqdm(dataset, desc="Processing examples"):
        for attempt in range(args.best_of):
            if process_example(model, tokenizer, example, attempt, args.temperature):
                correct_count += 1
                break
    
    # Print final statistics
    print_final_stats(correct_count, total, args)

if __name__ == "__main__":
    main()
