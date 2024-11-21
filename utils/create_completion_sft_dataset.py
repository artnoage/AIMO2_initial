import json
import random
import argparse
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

def load_augmented_data(filename: str) -> List[Dict]:
    """Load the augmented dataset file using UTF-8 encoding"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_solution_steps(solution: str) -> int:
    """Count the number of solution steps in a response"""
    # Look for "Step X" or "Step X." patterns
    steps = re.findall(r'Step\s+\d+\.?', solution, re.IGNORECASE)
    return len(steps)

def split_at_step(solution: str, step_num: int) -> Tuple[str, str]:
    """Split solution into prefix and completion at given step number"""
    pattern = f"Step\\s+{step_num+1}\\."
    match = re.search(pattern, solution, re.IGNORECASE)
    
    if not match:
        return None, None
        
    split_point = match.start()
    prefix = solution[:split_point].strip()
    completion = solution[split_point:].strip()
    
    return prefix, completion

def create_completion_example(entry: Dict, min_steps: int = 2) -> Optional[Dict]:
    """Create a completion training example from an augmented data entry"""
    
    # Get solutions that passed all verifications (level 4)
    valid_solutions = [
        resp for resp, ver in zip(entry['model_responses'], entry['verification_results'])
        if ver == 4
    ]
    
    if not valid_solutions:
        return None
        
    # Randomly select one valid solution
    solution = random.choice(valid_solutions)
    
    # Count total steps
    total_steps = count_solution_steps(solution)
    
    if total_steps < min_steps:
        return None
        
    # Choose random cutoff point between min_steps-1 and total_steps-1
    cutoff_step = random.randint(min_steps-1, total_steps-1)
    
    # Split solution at chosen step
    prefix, completion = split_at_step(solution, cutoff_step)
    
    if not prefix or not completion:
        return None
        
    # Create input prompt
    input_text = (
        f"{entry['problem']}\n\n"
        "Solve this step by step. Here is the start of a solution:\n\n"
        f"{prefix}\n\n"
        "Complete the remaining steps:"
    )
    
    return {
        "input": input_text,
        "output": completion
    }

def main():
    parser = argparse.ArgumentParser(description='Create completion SFT dataset from augmented data')
    parser.add_argument('--input', type=str, default='augmented_datasets/synthetic_augmented.json',
                       help='Input augmented dataset file')
    parser.add_argument('--output', type=str, default='datasets/completion_sft_dataset.json',
                       help='Output completion dataset file')
    parser.add_argument('--min-steps', type=int, default=2,
                       help='Minimum number of steps required in solution')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of times to process each problem')
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load augmented data
    print(f"Loading augmented data from {args.input}")
    data = load_augmented_data(args.input)
    
    # Create completion examples
    print("Creating completion examples...")
    completion_examples = []
    for _ in range(args.iterations):
        examples = [
            example for example in (
                create_completion_example(entry, args.min_steps)
                for entry in data
            ) if example is not None
        ]
        completion_examples.extend(examples)
    
    # Save completion dataset
    print(f"Saving {len(completion_examples)} examples to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(completion_examples, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
