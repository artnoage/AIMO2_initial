import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import json
from datetime import datetime
from pathlib import Path
import re
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
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def generate_solution(model, tokenizer, problem: str, max_length: int = 4096):
    """Generate a solution for a given problem."""
    prompt = f"[INST]Solve this step by step and provide the final answer in a \\boxed{{}}:\n\n{problem}[/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_example(model, tokenizer, example, attempt: int) -> bool:
    """Process a single example and return if the answer was correct."""
    try:
        solution = generate_solution(model, tokenizer, example['problem'])
        
        # Extract and compare answers
        model_answer = extract_answer_from_solution(solution)
        correct_answer = extract_answer_from_solution(example['solution'])
        
        # Convert to numeric values
        model_numeric, _ = extract_numeric_answer(model_answer)
        correct_numeric, _ = extract_numeric_answer(correct_answer)
        
        # Check correctness
        is_correct = is_answer_correct(model_numeric, correct_numeric, 0.001)
        
        if is_correct:
            print(f"✓ Example {example['id']} correct on attempt {attempt + 1}")
        elif attempt == 0:  # Only print failures on first attempt
            print(f"✗ Example {example['id']} incorrect")
            
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
    parser.add_argument('--best_of', type=int, default=3, help='Number of attempts per problem')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = setup_model(args.model_path)
    
    # Load dataset
    dataset = load_dataset("AI-MO/Numina", split=args.split)
    print(f"\nLoaded {len(dataset)} examples from dataset")
    
    correct_count = 0
    total = len(dataset)
    
    # Process each example
    for example in tqdm(dataset, desc="Processing examples"):
        for attempt in range(args.best_of):
            if process_example(model, tokenizer, example, attempt):
                correct_count += 1
                break
    
    # Print final statistics
    print_final_stats(correct_count, total, args)

if __name__ == "__main__":
    main()
