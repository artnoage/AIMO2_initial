import os
import json
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from utils import extract_answer_from_solution

def count_boxed_answers(solution: str) -> int:
    """Count number of \boxed{...} occurrences in solution"""
    count = 0
    pos = 0
    while True:
        pos = solution.find('\\boxed{', pos)
        if pos == -1:
            break
        count += 1
        pos += 1
    return count

def contains_http(text: str) -> bool:
    """Check if text contains http links"""
    return 'http' in text.lower()

def contains_non_latin(text: str) -> bool:
    """Check if text contains Chinese or Russian characters"""
    for char in text:
        # Check for Chinese characters
        if '\u4e00' <= char <= '\u9fff':
            return True
        # Check for Russian characters
        if '\u0400' <= char <= '\u04FF':
            return True
    return False

def main():
    # Initialize Hugging Face API
    api = HfApi()
    parser = argparse.ArgumentParser(description='Filter NuminaMath-CoT dataset for olympiads with valid answers')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--repo-name', type=str, default='artnoage/Numina',
                       help='HuggingFace repository name')
    args = parser.parse_args()

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Metadata validation was skipped")
    warnings.filterwarnings("ignore", message="Found cached dataset")
    
    # Load the dataset
    try:
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"\nOriginal dataset size: {len(dataset)}")

    # Initialize detailed statistics
    stats = {
        'original': len(dataset),
        'removed_source': 0,
        'removed_no_boxed': 0,
        'removed_multiple_boxed': 0,
        'removed_http_problem': 0,
        'removed_http_solution': 0,
        'removed_non_latin_problem': 0,
        'removed_non_latin_solution': 0,
        'removed_invalid_answer': 0
    }

    # Filter by source if specified
    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
        stats['removed_source'] = stats['original'] - len(dataset)
        print(f"After filtering for {args.source}: {len(dataset)}")
    
    # Filter for solutions containing exactly one boxed answer
    def has_valid_answer(example):
        if 'solution' not in example:
            stats['removed_no_boxed'] += 1
            return False
            
        # Check for exactly one boxed answer
        boxed_count = count_boxed_answers(example['solution'])
        if boxed_count == 0:
            stats['removed_no_boxed'] += 1
            return False
        elif boxed_count > 1:
            stats['removed_multiple_boxed'] += 1
            return False
            
        # Check for HTTP links
        if contains_http(example['problem']):
            stats['removed_http_problem'] += 1
            return False
        if contains_http(example['solution']):
            stats['removed_http_solution'] += 1
            return False
            
        # Check for non-Latin characters
        if contains_non_latin(example['problem']):
            stats['removed_non_latin_problem'] += 1
            return False
        if contains_non_latin(example['solution']):
            stats['removed_non_latin_solution'] += 1
            return False
            
        # Verify answer extraction
        answer = extract_answer_from_solution(example['solution'])
        if answer is None or answer.strip() == "":
            stats['removed_invalid_answer'] += 1
            return False
            
        return True

    # Apply filters and update statistics
    filtered_dataset = dataset.filter(has_valid_answer)
    
    # Calculate detailed statistics
    stats['final'] = len(filtered_dataset)
    stats['removed_invalid'] = len(dataset) - len(filtered_dataset)
    
    # Print detailed statistics
    print("\nDetailed Filtering Statistics:")
    print(f"Original dataset size: {stats['original']}")
    if args.source.lower() != 'all':
        print(f"Removed due to source filter: {stats['removed_source']}")
    print("\nRemoved due to:")
    print(f"- Missing boxed answer: {stats['removed_no_boxed']}")
    print(f"- Multiple boxed answers: {stats['removed_multiple_boxed']}")
    print(f"- HTTP links in problem: {stats['removed_http_problem']}")
    print(f"- HTTP links in solution: {stats['removed_http_solution']}")
    print(f"- Non-Latin chars in problem: {stats['removed_non_latin_problem']}")
    print(f"- Non-Latin chars in solution: {stats['removed_non_latin_solution']}")
    print(f"- Invalid/empty answer: {stats['removed_invalid_answer']}")
    print(f"\nFinal dataset size: {stats['final']}")
    print(f"Total reduction: {((stats['original'] - stats['final'])/stats['original'])*100:.1f}%")

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
    output_dir = "../numina_olympiads"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset_info.json
    # Convert features to JSON-serializable format
    features_json = {
        'id': {'dtype': 'int64', 'id': None},
        'problem': {'dtype': 'string', 'id': None},
        'solution': {'dtype': 'string', 'id': None},
        'source': {'dtype': 'string', 'id': None},
        'answer': {'dtype': 'string', 'id': None}
    }
    
    dataset_info = {
        "description": "Filtered NuminaMath-CoT dataset containing only olympiads problems with valid answers in LaTeX format",
        "citation": "@misc{numina2024,\n  author={AI-MO},\n  title={NuminaMath-CoT Dataset},\n  year={2024},\n  howpublished={\\url{https://huggingface.co/datasets/AI-MO/NuminaMath-CoT}}\n}",
        "homepage": "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT",
        "license": "mit",
        "features": features_json,
        "splits": {
            args.split: {
                "name": args.split,
                "num_bytes": None,
                "num_examples": len(filtered_dataset),
                "dataset_name": "Numina-Olympiads"
            }
        },
        "tags": [
            "mathematics",
            "olympiads", 
            "problem-solving",
            "latex",
            "mathematical-reasoning",
            "math-word-problems",
            "olympiad-math"
        ],
        "task_categories": [
            "text-generation",
            "mathematical-reasoning"
        ],
        "task_ids": [
            "math-word-problems",
            "olympiad-math"  
        ],
        "metrics": [
            {
                "name": "filtered_ratio",
                "type": "ratio",
                "value": len(filtered_dataset) / len(dataset),
                "description": "Ratio of filtered dataset size to original dataset size"
            }
        ],
        "paper_authors": ["AI-MO"],
        "dataset_size": None,
        "config_name": args.split
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
        repo_id = args.repo_name
        
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
annotations_creators:
  - expert-generated
language:
  - en
language_creators:
  - expert-generated
license: mit
multilinguality:
  - monolingual
pretty_name: Numina-Olympiads
size_categories:
  - 1K<n<10K
source_datasets:
  - AI-MO/NuminaMath-CoT
task_categories:
  - text-generation
  - mathematical-reasoning
task_ids:
  - math-word-problems
  - olympiad-math
paperswithcode_id: numina-olympiads
tags:
  - mathematics
  - olympiads
  - problem-solving
  - latex
  - mathematical-reasoning
  - math-word-problems
  - olympiad-math
metrics:
  - name: filtered_ratio
    type: ratio 
    value: {len(filtered_dataset) / len(dataset):.3f}
    description: Ratio of filtered dataset size to original dataset size
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


if __name__ == "__main__":
    main()
