import json
import random
import argparse
from typing import List, Dict
from pathlib import Path

def load_augmented_data(filename: str) -> List[Dict]:
    """Load the augmented dataset file"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_sft_example(entry: Dict, num_samples: int = 5) -> Dict:
    """Create a single SFT training example from an augmented data entry"""
    # Get all available responses
    responses = list(zip(entry['model_responses'], entry['verification_results']))
    
    # Ensure we have enough samples
    num_samples = min(num_samples, len(responses))
    
    # Randomly sample responses
    selected = random.sample(responses, num_samples)
    
    # Create numbered list of solutions
    solutions_text = "\n\n".join(
        f"Solution {i+1}:\n{solution}" 
        for i, (solution, _) in enumerate(selected)
    )
    
    # Create input prompt
    input_text = (
        f"Problem:\n{entry['problem']}\n\n"
        f"Please evaluate these solution attempts:\n\n{solutions_text}\n\n"
        "List the numbers of all solutions that are both detailed and mathematically correct."
    )
    
    # Create target output - list of correct solution numbers
    correct_solutions = [
        i+1 for i, (_, verification) in enumerate(selected)
        if verification == 4  # Level 4 means passed all checks
    ]
    
    output_text = (
        f"The following solutions are detailed and correct: {correct_solutions}\n"
        if correct_solutions else
        "None of the solutions are both detailed and correct.\n"
    )
    
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
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of solution samples per problem')
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load augmented data
    print(f"Loading augmented data from {args.input}")
    data = load_augmented_data(args.input)
    
    # Create SFT examples
    print("Creating SFT examples...")
    sft_examples = [
        create_sft_example(entry, args.samples)
        for entry in data
    ]
    
    # Save SFT dataset
    print(f"Saving {len(sft_examples)} examples to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(sft_examples, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
