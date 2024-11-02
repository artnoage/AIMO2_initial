import os
import argparse
from datasets import load_dataset
import tiktoken

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

    # Process each example
    print(f"\nProcessing {total_examples} examples from {args.split} split...")
    
    for idx, example in enumerate(dataset):
        if 'messages' in example and example['messages']:
            examples_with_messages += 1
            tokens = count_tokens_in_messages(example['messages'])
            total_tokens += tokens
            print(f"Example {idx + 1}: {tokens} tokens")
        
        if (idx + 1) % 100 == 0:
            print(f"Progress: {idx + 1}/{total_examples}")

    # Initialize counters
    tokens_0_1024 = 0
    tokens_1024_2048 = 0
    messages_with_box = 0
    
    # Print summary
    print("\nSummary:")
    print(f"Total examples processed: {total_examples}")
    print(f"Examples with messages: {examples_with_messages}")
    print(f"Total tokens in messages: {total_tokens}")
    if examples_with_messages > 0:
        avg_tokens = total_tokens / examples_with_messages
        print(f"Average tokens per example with messages: {avg_tokens:.1f}")
        
    # Count examples in token ranges
    for idx, example in enumerate(dataset):
        if 'messages' in example and example['messages']:
            tokens = count_tokens_in_messages(example['messages'])
            if 0 <= tokens <= 1024:
                tokens_0_1024 += 1
            elif 1024 < tokens <= 2048:
                tokens_1024_2048 += 1
    
    # Count messages with \box
    for idx, example in enumerate(dataset):
        if 'messages' in example and example['messages']:
            for message in example['messages']:
                if isinstance(message, dict) and 'content' in message:
                    if '\\box' in message['content']:
                        messages_with_box += 1
                        break  # Count only once per example
                        
    print("\nToken Range Distribution:")
    print(f"0-1024 tokens: {tokens_0_1024} examples")
    print(f"1024-2048 tokens: {tokens_1024_2048} examples")
    print(f"\nExamples with \\box notation: {messages_with_box}")

if __name__ == "__main__":
    main()
