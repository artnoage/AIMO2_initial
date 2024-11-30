import json
import random
import argparse
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from transformers import AutoTokenizer
from utils import filter_by_token_ranges

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
        current_num = int(match.group(1))
        if current_num == step_num:
            current_step_idx = match.start()
        elif current_num == step_num + 1:
            next_step_idx = match.start()
            break
    
    # If we can't find both the current step and next step, return None
    if current_step_idx is None or next_step_idx is None:
        return None, None
        
    # Extract the text before the current step and the current step itself
    prefix = solution[:current_step_idx].strip()
    step_text = solution[current_step_idx:next_step_idx].strip()
    completion = solution[next_step_idx:].strip()
    
    # Combine prefix and step_text, ensuring no leading newlines
    combined = f"{prefix}\n\n{step_text}" if prefix else step_text
    return combined, completion

def extract_analysis_section(solution: str) -> Optional[str]:
    """Extract the analysis section that comes before the steps"""
    # Look for common analysis section markers
    analysis_markers = [
        "**Problem Analysis and Approach**:",
        "Problem Analysis and Approach:",
        "Analysis:",
        "Approach:"
    ]
    
    # Find the start of the analysis section
    start_idx = -1
    for marker in analysis_markers:
        if marker in solution:
            start_idx = solution.find(marker)
            break
            
    if start_idx == -1:
        return None
        
    # Find the end of the analysis section (start of steps)
    step_match = re.search(r'Step\s+1\.?', solution, re.IGNORECASE)
    if not step_match:
        return None
        
    end_idx = step_match.start()
    
    # Extract and clean the analysis section
    analysis = solution[start_idx:end_idx].strip()
    if len(analysis) < 50:  # Minimum length check
        return None
        
    return analysis

def create_masked_completion_example(entry: Dict, min_steps: int = 3) -> Optional[Dict]:
    """Create a masked completion example where a single step needs to be completed"""
    
    valid_solutions = get_valid_solutions(entry)
    if not valid_solutions:
        return None
        
    # Randomly select one valid solution
    solution = random.choice(valid_solutions)
    
    # Get step numbers for the chosen solution
    _, step_numbers = validate_solution_steps(solution)
    total_steps = len(step_numbers)
        
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
    """Creates an example where the model needs to provide just the next solution step"""
    
    valid_solutions = get_valid_solutions(entry)
    if not valid_solutions:
        return None
        
    # Randomly select one valid solution
    solution = random.choice(valid_solutions)
    
    # Get step numbers for the chosen solution
    _, step_numbers = validate_solution_steps(solution)
    total_steps = len(step_numbers)
        
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

def create_analysis_example(entry: Dict) -> Optional[Dict]:
    """Create an example focusing only on the problem analysis section"""
    valid_solutions = get_valid_solutions(entry)
    if not valid_solutions:
        return None
        
    # Find solutions with good analysis sections
    analyses = []
    for solution in valid_solutions:
        analysis = extract_analysis_section(solution)
        if analysis:
            analyses.append(analysis)
            
    if not analyses:
        return None
        
    # Choose random analysis
    analysis = random.choice(analyses)
    
    input_text = (
        "Here is a mathematical problem:\n\n"
        f"{entry['problem']}\n\n"
        "Before solving this problem step-by-step, provide a thorough analysis that:\n"
        "1. Categorizes the problem type\n"
        "2. Lists the specific theorems and techniques that will be useful\n"
        "3. Outlines the general approach to solving it\n\n"
        "Important guidelines:\n"
        "- Start with '**Problem Analysis and Approach**:'\n"
        "- Be specific about which theorems/techniques apply\n"
        "- Explain why these approaches are suitable\n"
        "- Do NOT provide the actual solution steps\n\n"
        "Please provide the analysis:"
    )
    
    return {
        "conversations": [
            {
                "role": "user",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": analysis
            }
        ]
    }

def create_full_solution_example(entry: Dict) -> Optional[Dict]:
    """Create an example requesting a complete solution with steps and analysis"""
    valid_solutions = get_valid_solutions(entry)
    if not valid_solutions:
        return None
        
    # Choose random solution
    solution = random.choice(valid_solutions)
    
    input_text = (
        "Here is a mathematical problem to solve:\n\n"
        f"{entry['problem']}\n\n"
        "Please provide a complete solution following these guidelines:\n"
        "1. Start with '**Problem Analysis and Approach**:' section explaining:\n"
        "   - Problem type and key concepts involved\n"
        "   - Relevant theorems and techniques\n"
        "   - Overall solution strategy\n\n"
        "2. Then provide a detailed step-by-step solution:\n"
        "   - Number each step clearly (Step 1, Step 2, etc.)\n"
        "   - Show all work and intermediate calculations\n"
        "   - Use LaTeX notation for mathematical expressions\n"
        "   - Provide justification in [brackets] for key steps\n"
        "   - End with final answer in \\boxed{}\n\n"
        "Please solve the problem completely:"
    )
    
    return {
        "conversations": [
            {
                "role": "user",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": solution
            }
        ]
    }

def get_valid_solutions(entry: Dict) -> List[str]:
    """Get solutions that are valid (verified and have 3+ sequential steps)"""
    # Get solutions that passed all verifications (level 4)
    valid_solutions = [
        resp for resp, ver in zip(entry['model_responses'], entry['verification_results'])
        if ver == 4
    ]
    
    # Further filter for solutions with 3+ sequential steps
    filtered_solutions = []
    for solution in valid_solutions:
        is_valid, step_numbers = validate_solution_steps(solution)
        if is_valid and len(step_numbers) > 2:
            filtered_solutions.append(solution)
            
    return filtered_solutions

def create_progressive_completion_example(entry: Dict, min_steps: int = 2) -> Optional[Dict]:
    """Create a completion training example in ShareGPT format from an augmented data entry"""
    
    valid_solutions = get_valid_solutions(entry)
    if not valid_solutions:
        return None
        
    # Randomly select one valid solution
    solution = random.choice(valid_solutions)
    
    # Get step numbers for the chosen solution
    _, step_numbers = validate_solution_steps(solution)
    total_steps = len(step_numbers)
        
    # Choose random cutoff point between min_steps and total_steps-1
    cutoff_step = random.randint(min_steps, total_steps-1)
    
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
    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return
    
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
    
    # Create analysis-only examples
    analysis_examples = []
    for _ in range(args.iterations):
        for entry in data:
            example = create_analysis_example(entry)
            if example is not None:
                analysis_examples.append(example)
                
    # Create progressive completion examples
    progressive_examples = []
    for _ in range(args.iterations):
        for entry in data:
            example = create_progressive_completion_example(entry, args.min_steps)
            if example is not None:
                progressive_examples.append(example)
    
    # Create masked completion examples
    masked_examples = []
    for _ in range(args.iterations):
        for entry in data:
            example = create_masked_completion_example(entry, args.min_steps)
            if example is not None:
                masked_examples.append(example)
    
    # Create next step completion examples
    next_step_examples = []
    for _ in range(args.iterations):
        for entry in data:
            example = create_next_step_example(entry)
            if example is not None:
                next_step_examples.append(example)
    
    # Create full solution examples
    full_solution_examples = []
    for _ in range(args.iterations):
        for entry in data:
            example = create_full_solution_example(entry)
            if example is not None:
                full_solution_examples.append(example)

    # Combine and shuffle all examples
    all_examples = (analysis_examples + progressive_examples + masked_examples + 
                   next_step_examples + full_solution_examples)
    random.shuffle(all_examples)
    
    # Save shuffled dataset and print summary
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2)
        
    # Filter examples by token count
    filtered_examples, token_ranges = filter_by_token_ranges(all_examples, tokenizer)

    print("\nResults summary:")
    print(f"Analysis-only examples: {len(analysis_examples)}")
    print(f"Progressive completion examples: {len(progressive_examples)}")
    print(f"Masked completion examples: {len(masked_examples)}")
    print(f"Next step completion examples: {len(next_step_examples)}")
    print(f"Full solution examples: {len(full_solution_examples)}")
    print(f"Total examples before filtering: {len(all_examples)}")
    print(f"Total examples after filtering (<= 4096 tokens): {len(filtered_examples)}")
    
    print("\nToken count distribution:")
    for range_name, count in token_ranges.items():
        percentage = (count / len(filtered_examples)) * 100 if filtered_examples else 0
        print(f"{range_name}: {count} examples ({percentage:.1f}%)")

    # Save filtered examples
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(filtered_examples, f, indent=2)

if __name__ == "__main__":
    main()
