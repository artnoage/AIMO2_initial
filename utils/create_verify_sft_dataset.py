import json
import random
import argparse
from typing import List, Dict
from pathlib import Path

def load_augmented_data(filename: str) -> List[Dict]:
    """Load the augmented dataset file using UTF-8 encoding"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_sft_example(entry: Dict, min_samples: int = 3, max_samples: int = 8) -> Dict:
    """Create a single SFT training example from an augmented data entry"""
    # Get all available responses
    responses = list(zip(entry['model_responses'], entry['verification_results']))
    
    # Determine number of samples based on available responses
    available = len(responses)
    if available < min_samples:
        return None  # Skip entries with too few responses
        
    # Choose random number of samples between min and max, but not more than available
    num_samples = random.randint(min_samples, min(max_samples, available))
    
    # Randomly sample responses
    selected = random.sample(responses, num_samples)
    
    # Create numbered list of solutions
    solutions_text = "\n\n".join(
        f"Solution {i+1}:\n{solution}" 
        for i, (solution, _) in enumerate(selected)
    )
    
    # Create input prompt
    input_text = (
        "Here is a mathematical problem to analyze:\n\n"
        f"{entry['problem']}\n\n"
        "I will now present several attempted solutions to this problem. "
        "Each solution represents a different approach to solving it. "
        "Your task is to carefully evaluate each solution based on two key criteria:\n"
        "1. The solution must be detailed, showing clear strategy and step-by-step reasoning\n"
        "2. The solution must be mathematically correct without any errors\n\n"
        "Here are the solutions to evaluate:\n\n"
        f"{solutions_text}\n\n"
        "After reviewing all solutions, provide a list of solution numbers that satisfy BOTH criteria.\n"
        "Format your response as a list [n1, n2, ...] containing only the numbers of correct and detailed solutions.\n"
        "If no solutions meet both criteria, return an empty list [].\n"
        "Remember: A solution must be both detailed AND correct to be included in the list."
    )
    
    # Create target output - list of correct solution numbers
    correct_solutions = [
        i+1 for i, (_, verification) in enumerate(selected)
        if verification == 4  # Level 4 means passed all checks
    ]
    
    output_text = f"{correct_solutions}"
    
    return {
        "input": input_text,
        "output": output_text
    }

def main():
    parser = argparse.ArgumentParser(description='Create SFT dataset from augmented data')
    parser.add_argument('--input', type=str, default='augmented_datasets/synthetic_augmented.json',
                       help='Input augmented dataset file')
    parser.add_argument('--output', type=str, default='datasets/sft_dataset.json',
                       help='Output SFT dataset file')
    parser.add_argument('--min-samples', type=int, default=3,
                       help='Minimum number of solution samples per problem')
    parser.add_argument('--max-samples', type=int, default=8,
                       help='Maximum number of solution samples per problem')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of times to process each problem')
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load augmented data
    print(f"Loading augmented data from {args.input}")
    data = load_augmented_data(args.input)
    
    # Create SFT examples
    print("Creating SFT examples...")
    sft_examples = []
    for _ in range(args.iterations):
        examples = [
            example for example in (
                create_sft_example(entry, args.min_samples, args.max_samples)
                for entry in data
            ) if example is not None
        ]
        sft_examples.extend(examples)
    
    # Save SFT dataset
    print(f"Saving {len(sft_examples)} examples to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(sft_examples, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
