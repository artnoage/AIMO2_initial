import os
import json
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from utils import extract_answer_from_solution

def main():
    # Initialize Hugging Face API
    api = HfApi()
    parser = argparse.ArgumentParser(description='Filter NuminaMath-CoT dataset for olympiads with valid answers')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    args = parser.parse_args()

    # Load the dataset
    try:
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"\nOriginal dataset size: {len(dataset)}")

    # Filter for olympiads source
    olympiads_dataset = dataset.filter(lambda x: x['source'] == 'olympiads')
    print(f"After filtering for olympiads: {len(olympiads_dataset)}")

    # Filter for valid answers
    def has_valid_answer(example):
        if 'solution' not in example:
            return False
        answer = extract_answer_from_solution(example['solution'])
        return answer is not None and answer.strip() != ""

    filtered_dataset = olympiads_dataset.filter(has_valid_answer)
    print(f"After filtering for valid answers: {len(filtered_dataset)}")

    # Convert to Hugging Face dataset format with explicit schema
    from datasets import Features, Value
    
    features = Features({
        'id': Value('int64'),
        'problem': Value('string'),
        'solution': Value('string'), 
        'source': Value('string'),
        'answer': Value('string')
    })
    
    # Create dataset with IDs and explicit schema
    filtered_dataset_dict = DatasetDict({
        args.split: Dataset.from_dict(
            {
                'id': list(range(len(filtered_dataset))),
                'problem': filtered_dataset['problem'],
                'solution': filtered_dataset['solution'],
                'source': filtered_dataset['source'],
                'answer': [extract_answer_from_solution(sol) for sol in filtered_dataset['solution']]
            },
            features=features
        )
    })

    # Save locally first
    output_dir = "numina_olympiads"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset_info.json
    dataset_info = {
        "description": "Filtered NuminaMath-CoT dataset containing only olympiads problems with valid answers",
        "citation": "",
        "homepage": "",
        "license": "mit",
        "features": features,  # Use the same features object defined above
        "splits": {
            args.split: {
                "name": args.split,
                "num_bytes": None,
                "num_examples": len(filtered_dataset),
                "dataset_name": "Numina-Olympiads"
            }
        }
    }
    
    # Save dataset_info.json
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save the dataset
    filtered_dataset_dict.save_to_disk(output_dir)
    print(f"\nDataset saved locally to: {output_dir}")

    # Try to push to Hugging Face Hub
    try:
        # Get the username from huggingface-cli
        username = api.whoami()["name"]
        repo_id = f"{username}/Numina-Olympiads"
        
        # Create or get the repository
        try:
            api.create_repo(
                repo_id=repo_id,
                private=True,
                repo_type="dataset"
            )
        except Exception as repo_error:
            print(f"Note: Repository may already exist: {repo_error}")
        
        # Push the dataset
        filtered_dataset_dict.push_to_hub(repo_id)
        
        # Update the dataset card
        readme_content = f"""---
language: en
license: mit
pretty_name: Numina-Olympiads
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
task_ids:
  - math-word-problems
---

# Numina-Olympiads

Filtered NuminaMath-CoT dataset containing only olympiads problems with valid answers.

## Dataset Information
- Split: {args.split}
- Original size: {len(dataset)}
- Filtered size: {len(filtered_dataset)}
- Source: olympiads
- All examples contain valid boxed answers

## Dataset Description
This dataset is a filtered version of the NuminaMath-CoT dataset, containing only problems from olympiad sources that have valid boxed answers. Each example includes:
- A mathematical word problem
- A detailed solution with step-by-step reasoning
- A boxed final answer in LaTeX format

## Usage
The dataset is particularly useful for:
- Training and evaluating math problem-solving models
- Studying olympiad-style mathematical reasoning
- Testing model capabilities on complex word problems
"""
        with open("README.md", "w") as f:
            f.write(readme_content)
            
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("\nSuccessfully pushed dataset to Hugging Face Hub as 'Numina-Olympiads'")
    except Exception as e:
        print(f"\nFailed to push to Hugging Face Hub: {e}")
        print("You can still use the locally saved dataset")

    # Print some statistics
    print("\nSample problems from filtered dataset:")
    for idx, example in enumerate(filtered_dataset.select(range(min(3, len(filtered_dataset))))):
        print(f"\nProblem {idx + 1}:")
        print(f"Source: {example['source']}")
        print(f"Answer: {extract_answer_from_solution(example['solution'])}")
        print("-" * 80)

if __name__ == "__main__":
    main()
