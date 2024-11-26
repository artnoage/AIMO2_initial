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

def validate_solution_steps(solution: str) -> Tuple[bool, List[int]]:
    """
    Validate that solution steps are sequential and properly numbered.
    Returns (is_valid, step_numbers)
    """
    # Extract all step numbers from the solution
    step_matches = re.finditer(r'Step\s+(\d+)\.?', solution, re.IGNORECASE)
    step_numbers = [int(m.group(1)) for m in step_matches]
    
    if not step_numbers:
        return False, []
        
    # Ensure steps are sequential and start from 1
    is_valid = step_numbers[0] == 1 and all(b-a == 1 for a, b in zip(step_numbers, step_numbers[1:]))
    return is_valid, step_numbers

def count_solution_steps(solution: str) -> int:
    """Count the number of solution steps in a response"""
    _, step_numbers = validate_solution_steps(solution)
    return len(step_numbers)

def split_at_step(solution: str, step_num: int) -> Tuple[str, str]:
    """Split solution into prefix and completion at given step number"""
    # Find all step markers in the solution
    step_matches = list(re.finditer(r'Step\s+(\d+)\.?', solution, re.IGNORECASE))
    
    if not step_matches:
        return None, None
    
    # Find the indices for the requested step number
    current_step_idx = None
    next_step_idx = None
    
    for i, match in enumerate(step_matches):
        if int(match.group(1)) == step_num + 1:
            next_step_idx = match.start()
            # If this is Step 1, include everything before it
            if step_num == 0:
                current_step_idx = 0
            break
        elif int(match.group(1)) == step_num:
            current_step_idx = match.start()
    
    if current_step_idx is None or next_step_idx is None:
        return None, None
        
    # For step 0 (before Step 1), include all text from the beginning
    prefix = solution[:next_step_idx].strip()
    completion = solution[next_step_idx:].strip()
    
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
    
    # Validate solution steps
    is_valid, step_numbers = validate_solution_steps(solution)
    if not is_valid:
        return None
        
    total_steps = len(step_numbers)
    if total_steps < min_steps or total_steps == 1:
        return None
        
    # Choose random step to mask (not first or last step)
    if total_steps <= 3:
        # If only 3 steps, always mask step 2
        step_to_mask = 2
    else:
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
                "role": "user",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": target_step
            }
        ]
    }

def create_next_step_example(entry: Dict) -> Optional[Dict]:
    """Create an example where the model needs to provide just the next solution step"""
    
    # Get solutions that passed all verifications (level 4)
    valid_solutions = [
        resp for resp, ver in zip(entry['model_responses'], entry['verification_results'])
        if ver == 4
    ]
    
    if not valid_solutions:
        return None
        
    # Randomly select one valid solution
    solution = random.choice(valid_solutions)
    
    # Validate solution steps
    is_valid, step_numbers = validate_solution_steps(solution)
    if not is_valid:
        return None
        
    total_steps = len(step_numbers)
    if total_steps <= 1:
        return None
        
    # Randomly decide whether to start from scratch or from a partial solution
    include_steps = random.randint(0, total_steps - 1)
    
    if include_steps == 0:
        prefix = ""
        # Get the first step
        next_step, _ = split_at_step(solution, 0)
        if next_step is None:
            return None
    else:
        # Get the solution up to the chosen step
        prefix, remainder = split_at_step(solution, include_steps)
        if prefix is None or remainder is None:
            return None
            
        # Get just the next step from the remainder
        next_step, _ = split_at_step(remainder, include_steps + 1)
        if next_step is None:
            return None
        
    # Create input prompt
    input_text = (
        "Here is a mathematical problem:\n\n"
        f"{entry['problem']}\n\n"
        "Your task is to provide the next step in the solution. "
        "Make sure your step is detailed and mathematically rigorous.\n\n"
        "Guidelines:\n"
        "- Provide exactly ONE step\n"
        "- Include clear explanations\n"
        "- Use LaTeX notation where appropriate\n"
        "- Include justification in [brackets]\n"
        "- Number your step appropriately\n\n"
    )
    
    if prefix:
        input_text += f"Here are the steps so far:\n\n{prefix}\n\nProvide the next step:"
    else:
        input_text += "Start the solution with Step 1:"
    
    return {
        "conversations": [
            {
                "role": "user",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": next_step
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
    
    # Validate solution steps
    is_valid, step_numbers = validate_solution_steps(solution)
    if not is_valid:
        return None
        
    total_steps = len(step_numbers)
    if total_steps < min_steps or total_steps == 1:
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
                "role": "user",
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
    parser.add_argument('--output', type=str, default='datasets/sft_completion.json',
                       help='Output file for all completion examples')
    parser.add_argument('--min-steps', type=int, default=2,
                       help='Minimum number of steps required in solution')
    parser.add_argument('--iterations', type=int, default=2,
                       help='Number of times to process each problem')
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load augmented data
    print(f"Loading augmented data from {args.input}")
    data = load_augmented_data(args.input)
    
    print(f"Processing {len(data)} problems, {args.iterations} iterations each...")
    
    # Initialize counters
    non_sequential_counts = {
        'progressive': 0,
        'masked': 0,
        'next_step': 0
    }
    
    # Create progressive completion examples
    progressive_examples = []
    for _ in range(args.iterations):
        for entry in data:
            example = create_progressive_completion_example(entry, args.min_steps)
            if example is None:
                # Check if failure was due to non-sequential steps
                solution = random.choice([resp for resp, ver in zip(entry['model_responses'], entry['verification_results']) if ver == 4])
                if solution and not validate_solution_steps(solution)[0]:
                    non_sequential_counts['progressive'] += 1
            else:
                progressive_examples.append(example)
    
    # Create masked completion examples
    masked_examples = []
    for _ in range(args.iterations):
        for entry in data:
            example = create_masked_completion_example(entry, args.min_steps)
            if example is None:
                # Check if failure was due to non-sequential steps
                solution = random.choice([resp for resp, ver in zip(entry['model_responses'], entry['verification_results']) if ver == 4])
                if solution and not validate_solution_steps(solution)[0]:
                    non_sequential_counts['masked'] += 1
            else:
                masked_examples.append(example)
    
    # Create next step completion examples
    next_step_examples = []
    for _ in range(args.iterations):
        for entry in data:
            example = create_next_step_example(entry)
            if example is None:
                # Check if failure was due to non-sequential steps
                solution = random.choice([resp for resp, ver in zip(entry['model_responses'], entry['verification_results']) if ver == 4])
                if solution and not validate_solution_steps(solution)[0]:
                    non_sequential_counts['next_step'] += 1
            else:
                next_step_examples.append(example)
    
    # Combine and shuffle all examples
    all_examples = progressive_examples + masked_examples + next_step_examples
    random.shuffle(all_examples)
    
    # Save shuffled dataset and print summary
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2)
        
    print("\nResults summary:")
    print(f"Progressive completion examples: {len(progressive_examples)} (lost {non_sequential_counts['progressive']} to non-sequential steps)")
    print(f"Masked completion examples: {len(masked_examples)} (lost {non_sequential_counts['masked']} to non-sequential steps)")
    print(f"Next step completion examples: {len(next_step_examples)} (lost {non_sequential_counts['next_step']} to non-sequential steps)")
    print(f"Total examples saved to {args.output}: {len(all_examples)}")
    print(f"Total examples lost to non-sequential steps: {sum(non_sequential_counts.values())}")

if __name__ == "__main__":
    main()
