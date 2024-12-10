import os
from datasets import load_dataset
from datetime import datetime
import argparse

def create_markdown(example, file):
    """Write a single example to the markdown file"""
    file.write(f"## Problem {example['problem']}\n\n")
    file.write("### Problem Statement\n")
    file.write(f"{example['problem']}\n\n")
    file.write("### Solution\n")
    file.write(f"{example['solution']}\n\n")
    file.write("### Answer\n")
    file.write(f"{example['answer']}\n\n")
    file.write("---\n\n")

def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace dataset to markdown')
    parser.add_argument('--dataset', required=True, help='HuggingFace dataset name (e.g. username/dataset-name)')
    parser.add_argument('--split', default='train', help='Dataset split to use (default: train)')
    parser.add_argument('--output', default=None, help='Output markdown file path')
    args = parser.parse_args()

    # Load the dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.split)
    
    # Create output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"dataset_{timestamp}.md"

    print(f"Writing markdown to {args.output}...")
    
    # Create the markdown file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"# Dataset: {args.dataset}\n\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Number of examples: {len(dataset)}\n\n")
        f.write("---\n\n")
        
        # Process each example
        for i, example in enumerate(dataset):
            create_markdown(example, f)
            if i % 100 == 0:
                print(f"Processed {i} examples...")

    print("Done!")

if __name__ == "__main__":
    main()
