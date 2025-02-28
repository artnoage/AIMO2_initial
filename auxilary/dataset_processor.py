import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfApi, login
import logging
import random
import sys

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Simple print function for status updates
def log_info(message):
    print(f"[INFO] {message}")

SYSTEM_PROMPT = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.\n\n
    <thinking>
    First, analyze the problem in depth and outline your approach.\n 
    This section should capture your reasoning, including any abstract thoughts or potential strategies.\n  
    Feel free to refine or correct your ideas as you work toward the solution.\n  
    </thinking>
    <response>\n
    <step>Step 1: Begin with the first calculation or operation\n
    Show your work clearly using LaTeX notation</step>\n\n
    <step>Step 2: Continue with the next logical step\n
    Each step should be numbered and self-contained</step>\n\n
    <step>Step N: In your final step, state your conclusion\n
    Put your final answer in \\boxed{}</step>\n
    </response>\n\n"""





def push_to_hub(dataset: Dataset, repo_name: str, token: Optional[str] = None) -> str:
    """
    Push a dataset to the Hugging Face Hub.
    
    Args:
        dataset: The dataset to push
        repo_name: The repository name on Hugging Face
        token: Hugging Face API token (optional if already logged in)
        
    Returns:
        URL of the pushed dataset
    """
    try:
        # Login if token is provided
        if token:
            login(token)
            print("Logged in to Hugging Face Hub")
        
        # Push the dataset
        print(f"Pushing dataset to {repo_name}")
        dataset.push_to_hub(repo_name)
        print(f"Successfully pushed dataset to {repo_name}")
        
        return f"https://huggingface.co/datasets/{repo_name}"
    except Exception as e:
        print(f"ERROR: Failed to push dataset to hub: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Process and combine datasets from Hugging Face")
    parser.add_argument("--datasets", nargs="+", default=["Metaskepsis/Olympiads_medium", "Metaskepsis/Olympiads_hard"], 
                        help="List of dataset names to load")
    parser.add_argument("--splits", nargs="+", default=["train"], 
                        help="Dataset splits to use")
    parser.add_argument("--format", type=str, default="default",
                        help="Format type to apply to datasets")
    parser.add_argument("--examples", type=int, default=None, 
                        help="Number of examples to select from each dataset (None for all)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for shuffling")
    parser.add_argument("--output-repo", type=str, required=True, 
                        help="Repository name for the output dataset")
    parser.add_argument("--token", type=str, default=None, 
                        help="Hugging Face API token")
    
    args = parser.parse_args()
    
    # Process each dataset and concatenate
    all_processed_datasets = []
    
    for dataset_name in args.datasets:
        try:
            print(f"Loading dataset: {dataset_name}")
            for split in args.splits:
                # Load dataset directly
                dataset = load_dataset(dataset_name, split=split)
                print(f"Loaded {len(dataset)} examples from {dataset_name} ({split})")
                # Format the dataset directly based on format type
                if args.format == "default":
                    # Default formatting for math problems
                    processed = dataset.map(lambda x: {
                        'prompt': '<|im_start|>system\\n' + SYSTEM_PROMPT + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                        'answer': x.get('answer', x.get('correct_answer', '')),
                        'problem': x['problem'],
                        'original_id': x.get('id', str(random.randint(0, 1000000)))
                    })
                elif args.format == "group":
                    # Group training format
                    processed = dataset.map(lambda x: {
                        'prompt': '<|im_start|>system\\n' + SYSTEM_PROMPT + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                        'answer': x.get('answer', x.get('correct_answer', '')),
                        'problem': x['problem']
                    })
                else:
                    # Custom format - just pass through the dataset
                    print(f"Using custom format: {args.format}")
                    processed = dataset
                
                # Shuffle and select examples if needed
                processed = processed.shuffle(seed=args.seed)
                if args.examples is not None and args.examples < len(processed):
                    print(f"Selecting {args.examples} examples from dataset of size {len(processed)}")
                    processed = processed.select(range(args.examples))
                
                all_processed_datasets.append(processed)
                print(f"Added {len(processed)} examples from {dataset_name} ({split})")
        except Exception as e:
            print(f"ERROR: Failed to load dataset {dataset_name}: {str(e)}")
    
    # Concatenate all datasets
    if all_processed_datasets:
        print(f"Concatenating {len(all_processed_datasets)} datasets")
        total_examples = sum(len(ds) for ds in all_processed_datasets)
        
        final_dataset = concatenate_datasets(all_processed_datasets)
        print(f"Created concatenated dataset with {len(final_dataset)} examples (from {total_examples} total)")
        
        # Push to hub
        hub_url = push_to_hub(final_dataset, args.output_repo, args.token)
        print(f"Dataset available at: {hub_url}")
    else:
        print("ERROR: No datasets were processed successfully")

if __name__ == "__main__":
    main()
