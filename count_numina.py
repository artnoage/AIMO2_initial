import os
import argparse
from datasets import load_dataset
import tiktoken
from benchmark_numina import extract_answer_from_solution

def count_tokens_in_messages(messages):
    """Count tokens in all messages using tiktoken"""
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    
    for message in messages:
        if isinstance(message, dict) and 'content' in message:
            tokens = len(enc.encode(message['content']))
            total_tokens += tokens
            
    return total_tokens

def main():
    parser = argparse.ArgumentParser(description='Count tokens in NuminaMath-CoT dataset messages')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    args = parser.parse_args()

    # Load the dataset
    try:
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize counters
    total_examples = len(dataset)
    total_tokens = 0
    examples_with_messages = 0

    # Initialize counters
    tokens_0_1024 = 0
    tokens_1024_2048 = 0
    messages_with_box = 0
    examples_with_answers = 0
    
    # Process each example
    print(f"\nProcessing {total_examples} examples from {args.split} split...")
    print("\nDetailed Example Information:")
    print("-" * 80)
    
    for idx, example in enumerate(dataset):
        print(f"\nExample {idx + 1}:")
        
        # Process messages and count tokens
        tokens = 0
        has_box = False
        if 'messages' in example and example['messages']:
            examples_with_messages += 1
            tokens = count_tokens_in_messages(example['messages'])
            total_tokens += tokens
            
            # Check for \box in messages
            for message in example['messages']:
                if isinstance(message, dict) and 'content' in message:
                    if '\\box' in message['content']:
                        has_box = True
                        messages_with_box += 1
                        break
            
            # Count token ranges
            if 0 <= tokens <= 1024:
                tokens_0_1024 += 1
                token_bracket = "0-1024"
            elif 1024 < tokens <= 2048:
                tokens_1024_2048 += 1
                token_bracket = "1024-2048"
            else:
                token_bracket = ">2048"
        
        # Extract answer
        answer = None
        if 'solution' in example:
            answer = extract_answer_from_solution(example['solution'])
            if answer is not None:
                examples_with_answers += 1
        
        # Print example information
        print(f"Tokens: {tokens} (Bracket: {token_bracket})")
        print(f"Contains \\box: {'Yes' if has_box else 'No'}")
        print(f"Extracted Answer: {answer if answer else 'None'}")
        
        if (idx + 1) % 100 == 0:
            print(f"\nProgress: {idx + 1}/{total_examples}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Final Summary:")
    print(f"Total examples processed: {total_examples}")
    print(f"Examples with messages: {examples_with_messages}")
    print(f"Total tokens in messages: {total_tokens}")
    if examples_with_messages > 0:
        avg_tokens = total_tokens / examples_with_messages
        print(f"Average tokens per example with messages: {avg_tokens:.1f}")
    
    print("\nToken Range Distribution:")
    print(f"0-1024 tokens: {tokens_0_1024} examples")
    print(f"1024-2048 tokens: {tokens_1024_2048} examples")
    print(f"\nExamples with \\box notation: {messages_with_box}")
    print(f"Examples with extractable answers: {examples_with_answers}")

if __name__ == "__main__":
    main()
