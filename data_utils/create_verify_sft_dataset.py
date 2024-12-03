import json
import random
import argparse
from typing import List, Dict, Optional
from pathlib import Path
from transformers import AutoTokenizer
from benchmark_utils.utils import extract_answer_from_solution, filter_by_token_ranges

def load_augmented_data(filename: str) -> List[Dict]:
    """Load the augmented dataset file using UTF-8 encoding"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_answer_comparison_example(entry: Dict, tokenizer, max_tokens: int = 8192) -> List[Dict]:
    """Create examples for training answer comparison, returning both yes and no cases"""
    examples = []
    
    # Get the correct answer from the reference solution
    correct_ans = extract_answer_from_solution(entry['correct_solution'])
    if not correct_ans:
        return examples
        
    # Get solutions with verification results > 0 (passed format check)
    valid_responses = [(sol, ver) for sol, ver in zip(entry['model_responses'], entry['verification_results']) 
                      if ver > 0]
    
    if not valid_responses:
        return []
        
    # Split into correct (ver > 1) and incorrect (ver == 1) model answers
    correct_model_ans = [(sol, ver) for sol, ver in valid_responses if ver > 1]
    incorrect_model_ans = [(sol, ver) for sol, ver in valid_responses if ver == 1]
    
    # Create "yes" examples using correct model answers
    if correct_model_ans:
        # Take a random correct model answer
        sol, _ = random.choice(correct_model_ans)
        model_ans = extract_answer_from_solution(sol)
        if model_ans and model_ans != correct_ans:
            input_text = (
                "You are a mathematical answer validator. Given a problem and two answers, "
                "determine if they are mathematically equivalent.\n\n"
                f"Problem:\n{entry['problem']}\n\n"
                f"Answer 1: {correct_ans}\n"
                f"Answer 2: {model_ans}\n\n"
                "Are these answers mathematically equivalent? Respond with EXACTLY one word - ONLY 'yes' or 'no'."
            )
            
            # Check token count
            if len(tokenizer.encode(input_text)) + len(tokenizer.encode("yes")) <= max_tokens:
                examples.append({
                    "conversations": [
                        {"role": "user", "content": input_text},
                        {"role": "assistant", "content": "yes"}
                    ]
                })
    
    # Create "no" examples using incorrect model answers
    for sol, _ in incorrect_model_ans:
        incorrect_ans = extract_answer_from_solution(sol)
        if not incorrect_ans or incorrect_ans == correct_ans:
            continue
            
        input_text = (
            "You are a mathematical answer validator. Given a problem and two answers, "
            "determine if they are mathematically equivalent.\n\n"
            f"Problem:\n{entry['problem']}\n\n"
            f"Answer 1: {correct_ans}\n"
            f"Answer 2: {incorrect_ans}\n\n"
            "Are these answers mathematically equivalent? Respond with EXACTLY one word - ONLY 'yes' or 'no'."
        )
        
        # Check token count
        if len(tokenizer.encode(input_text)) + len(tokenizer.encode("no")) <= max_tokens:
            examples.append({
                "conversations": [
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": "no"}
                ]
            })
            break  # One "no" example is enough to balance the "yes" example
    
    return examples

def create_sft_example(entry: Dict, tokenizer, min_samples: int = 3, max_samples: int = 8, max_tokens: int = 8192) -> Optional[Dict]:
    """Create a single SFT training example in ShareGPT format from an augmented data entry"""
    # Get all available responses
    responses = list(zip(entry['model_responses'], entry['verification_results']))
    
    # Determine number of samples based on available responses
    available = len(responses)
    if available < min_samples:
        return None  # Skip entries with too few responses
        
    # Filter out responses with verification result of 0
    valid_responses = [(sol, ver) for sol, ver in responses if ver > 0]
    
    # Check if we have enough valid responses
    valid_count = len(valid_responses)
    if valid_count < min_samples:
        return None  # Skip entries with too few valid responses
    
    # Choose random number of samples between min and max, but not more than available valid responses
    num_samples = random.randint(min_samples, min(max_samples, valid_count))
    
    # Randomly sample from valid responses
    selected = random.sample(valid_responses, num_samples)
    
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
    
    # Count tokens in the conversation
    total_tokens = len(tokenizer.encode(input_text)) + len(tokenizer.encode(output_text))
    
    if total_tokens > max_tokens:
        return None
        
    return {
        "conversations": [
            {
                "role": "user",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": output_text
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
    
    parser = argparse.ArgumentParser(description='Create SFT dataset from augmented data')
    parser.add_argument('--input', type=str, default='augmented_datasets/synthetic_augmented.json',
                       help='Input augmented dataset file')
    parser.add_argument('--output', type=str, default='datasets/sft_verification.json',
                       help='Output SFT dataset file')
    parser.add_argument('--min-samples', type=int, default=2,
                       help='Minimum number of solution samples per problem')
    parser.add_argument('--max-samples', type=int, default=4,
                       help='Maximum number of solution samples per problem')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of times to process each problem')
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load augmented data
    print(f"Loading augmented data from {args.input}")
    data = load_augmented_data(args.input)
    
    # Create both types of SFT examples
    print("Creating SFT examples...")
    verification_examples = []
    comparison_examples = []
    
    for _ in range(args.iterations):
        # Create verification examples
        ver_examples = [
            example for example in (
                create_sft_example(entry, tokenizer, args.min_samples, args.max_samples, max_tokens=4096)
                for entry in data
            ) if example is not None
        ]
        verification_examples.extend(ver_examples)
        
        # Create answer comparison examples
        for entry in data:
            comp_examples = create_answer_comparison_example(entry, tokenizer)
            comparison_examples.extend(comp_examples)
    
    # Balance the number of comparison examples with verification examples
    num_verification = len(verification_examples)
    comparison_examples = comparison_examples[:num_verification]  # Truncate to match verification count
    
    # Combine all examples
    sft_examples = verification_examples + comparison_examples
    
    # Shuffle the combined examples
    random.shuffle(sft_examples)
    # Filter examples by token count
    filtered_examples, token_ranges = filter_by_token_ranges(sft_examples, tokenizer)

    print("\nResults summary:")
    print(f"Verification examples: {len(verification_examples)}")
    print(f"Answer comparison examples: {len(comparison_examples)}")
    print(f"Total examples before filtering: {len(sft_examples)}")
    print(f"Total examples after filtering (<= 4096 tokens): {len(filtered_examples)}")
    
    print("\nToken count distribution:")
    for range_name, count in token_ranges.items():
        percentage = (count / len(filtered_examples)) * 100 if filtered_examples else 0
        print(f"{range_name}: {count} examples ({percentage:.1f}%)")

    with open(args.output, 'w') as f:
        json.dump(filtered_examples, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
