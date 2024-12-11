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

def extract_answer_from_solution(solution: str) -> str:
    """Extract the first boxed answer from the solution."""
    pattern = re.compile(r'\\boxed\{([^}]+)\}')
    match = pattern.search(solution)
    return match.group(1) if match else None

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

def generate_solution(model, tokenizer, problem: str, max_length: int = 2048):
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

def process_example(model, tokenizer, example, attempt: int):
    """Process a single example."""
    try:
        # Generate solution
        solution = generate_solution(model, tokenizer, example['problem'])
        
        # Extract answer
        model_answer = extract_answer_from_solution(solution)
        
        # Calculate metrics
        solution_length = len(solution.split())
        step_count = solution.count('Step') + solution.count('\n1.') + solution.count('\n2.')
        
        return {
            'id': example['id'],
            'problem': example['problem'],
            'solution': solution,
            'model_answer': model_answer,
            'correct_solution': example['solution'],
            'metrics': {
                'solution_length': solution_length,
                'step_count': step_count,
                'attempt': attempt
            }
        }
        
    except Exception as e:
        print(f"Error processing example {example['id']}: {str(e)}")
        return None

def save_results(results: list, args: argparse.Namespace):
    """Save results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"gpu_benchmark_{timestamp}.json"
    
    # Calculate aggregate statistics
    solution_lengths = []
    step_counts = []
    
    for result in results:
        if result and 'metrics' in result:
            solution_lengths.append(result['metrics']['solution_length'])
            step_counts.append(result['metrics']['step_count'])
    
    stats = {
        'avg_solution_length': np.mean(solution_lengths),
        'std_solution_length': np.std(solution_lengths),
        'avg_step_count': np.mean(step_counts),
        'std_step_count': np.std(step_counts),
        'total_examples': len(results),
        'args': vars(args)
    }
    
    with output_file.open('w') as f:
        json.dump({
            'stats': stats,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

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
    
    all_results = []
    
    # Process each example
    for example in tqdm(dataset, desc="Processing examples"):
        example_results = []
        for attempt in range(args.best_of):
            result = process_example(model, tokenizer, example, attempt)
            if result:
                example_results.append(result)
        
        # Add best result to final results
        if example_results:
            # For now, just take the first result as "best"
            # Could implement more sophisticated selection later
            all_results.append(example_results[0])
    
    # Save results
    save_results(all_results, args)

if __name__ == "__main__":
    main()
