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

def create_masked_completion_example(entry: Dict, min_steps: int = 3) -> Optional[Dict]:
    """Create a masked completion example where a single step needs to be completed"""
    
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
        
    # Choose random step to mask (not first or last step)
    step_to_mask = random.randint(2, total_steps-1)
    
    # Split solution at chosen step
    prefix, step_and_rest = split_at_step(solution, step_to_mask-1)
    if not prefix or not step_and_rest:
        return None
        
    # Split the rest to isolate the target step
    target_step, remainder = split_at_step(step_and_rest, step_to_mask)
    if not target_step or not remainder:
        return None
    
    # Create input prompt
    input_text = (
        "Here is a mathematical problem:\n\n"
        f"{entry['problem']}\n\n"
        "Below is a step-by-step solution with one step missing. "
        "Your task is to provide ONLY the missing step, maintaining the same style and rigor.\n\n"
        "Important guidelines:\n"
        "- Provide ONLY the missing step\n"
        "- Match the level of detail and explanation style\n"
        "- Use LaTeX notation consistently\n"
        "- Provide justification in [brackets]\n\n"
        "Here is the solution with a missing step:\n\n"
        f"{prefix}\n\n"
        "[MISSING STEP]\n\n"
        f"{remainder}\n\n"
        "Please provide the missing step:"
    )
    
    return {
        "conversations": [
            {
                "role": "human",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": target_step
            }
        ]
    }

def create_progressive_completion_example(entry: Dict, min_steps: int = 2) -> Optional[Dict]:
    """Create a completion training example in ShareGPT format from an augmented data entry"""
    
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
        "Here is a mathematical problem:\n\n"
        f"{entry['problem']}\n\n"
        "I will show you the beginning of a step-by-step mathematical solution. "
        "Your task is to complete the solution by continuing with the same style and rigor.\n\n"
        "Important guidelines:\n"
        "- Maintain the same level of detail and explanation as the previous steps\n"
        "- Continue the step numbering sequence\n"
        "- Use LaTeX notation consistently\n"
        "- Provide justification for each step in [brackets]\n"
        "- End with a clear boxed answer using \\boxed{}\n\n"
        "Here is the partial solution:\n\n"
        f"{prefix}\n\n"
        "Please complete the remaining steps following the same format:"
    )
    
    return {
        "conversations": [
            {
                "role": "human",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": completion
            }
        ]
    }

def main():
    parser = argparse.ArgumentParser(description='Create completion SFT dataset from augmented data')
    parser.add_argument('--input', type=str, default='augmented_datasets/synthetic_augmented.json',
                       help='Input augmented dataset file')
    parser.add_argument('--output-progressive', type=str, default='datasets/progressive_completion.json',
                       help='Output file for progressive completion examples')
    parser.add_argument('--output-masked', type=str, default='masked_completion.json',
                       help='Output file for masked completion examples')
    parser.add_argument('--min-steps', type=int, default=2,
                       help='Minimum number of steps required in solution')
    parser.add_argument('--iterations', type=int, default=2,
                       help='Number of times to process each problem')
    args = parser.parse_args()
    
    # Create output directories if needed
    Path(args.output_progressive).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_masked).parent.mkdir(parents=True, exist_ok=True)
    
    # Load augmented data
    print(f"Loading augmented data from {args.input}")
    data = load_augmented_data(args.input)
    
    # Create progressive completion examples
    print("Creating progressive completion examples...")
    progressive_examples = []
    for _ in range(args.iterations):
        examples = [
            example for example in (
                create_progressive_completion_example(entry, args.min_steps)
                for entry in data
            ) if example is not None
        ]
        progressive_examples.extend(examples)
    
    # Create masked completion examples
    print("Creating masked completion examples...")
    masked_examples = []
    for _ in range(args.iterations):
        examples = [
            example for example in (
                create_masked_completion_example(entry, args.min_steps)
                for entry in data
            ) if example is not None
        ]
        masked_examples.extend(examples)
    
    # Save datasets
    print(f"Saving {len(progressive_examples)} progressive examples to {args.output_progressive}")
    with open(args.output_progressive, 'w', encoding='utf-8') as f:
        json.dump(progressive_examples, f, indent=2)
        
    print(f"Saving {len(masked_examples)} masked examples to {args.output_masked}")
    with open(args.output_masked, 'w', encoding='utf-8') as f:
        json.dump(masked_examples, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
