import argparse
from datasets import load_dataset
from huggingface_hub import HfApi
import json
from enum import Enum
from typing import Dict, List

class DatasetOption(str, Enum):
    NUMINA_OLYMPIADS = "numina-olympiads"
    NUMINA_MATH_COT = "numina-math-cot"
    AIME_VALIDATION = "aime-validation"

def get_dataset_info(dataset_type: DatasetOption) -> Dict[str, str]:
    """Get dataset loading information based on type"""
    dataset_map = {
        DatasetOption.NUMINA_OLYMPIADS: {
            "name": lambda username: f"{username}/Numina-Olympiads",
            "title": "Numina-Olympiads Dataset"
        },
        DatasetOption.NUMINA_MATH_COT: {
            "name": lambda _: "AI-MO/NuminaMath-CoT",
            "title": "NuminaMath-CoT Dataset"
        },
        DatasetOption.AIME_VALIDATION: {
            "name": lambda _: "AI-MO/aimo-validation-aime",
            "title": "AIME Validation Dataset"
        }
    }
    return dataset_map[dataset_type]

def load_examples(dataset_type: DatasetOption, split: str = 'train', num_examples: int = 10) -> List[Dict]:
    """Load examples from specified dataset"""
    username = HfApi().whoami()["name"]
    dataset_info = get_dataset_info(dataset_type)
    dataset_name = dataset_info["name"](username)
    
    dataset = load_dataset(dataset_name, split=split)
    examples = []
    
    for i, ex in enumerate(dataset):
        if i >= num_examples:
            break
        examples.append(dict(ex))
    
    return examples, dataset_info["title"]

def save_examples(examples: List[Dict], title: str, base_filename: str) -> None:
    """Save examples to markdown and JSON files"""
    # Create markdown content
    md_content = f"# {title} Sample\n\n"
    for i, ex in enumerate(examples, 1):
        md_content += f"## Entry {i}\n\n"
        for key, value in ex.items():
            md_content += f"### {key}\n\n"
            md_content += f"{value}\n\n"
        md_content += "---\n\n"
    
    # Save files
    md_filename = f"{base_filename}.md"
    json_filename = f"{base_filename}.json"
    
    with open(md_filename, 'w') as f:
        f.write(md_content)
    print(f"\nSaved {len(examples)} examples to {md_filename}")
    
    with open(json_filename, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"Saved raw data to {json_filename}")

def main():
    parser = argparse.ArgumentParser(description='Print dataset samples in Markdown format')
    parser.add_argument('--dataset', type=DatasetOption, choices=list(DatasetOption),
                       default=DatasetOption.NUMINA_OLYMPIADS,
                       help='Dataset to sample (default: numina-olympiads)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (default: train)')
    parser.add_argument('--num-examples', type=int, default=10,
                       help='Number of examples to include (default: 10)')
    args = parser.parse_args()

    try:
        examples, title = load_examples(args.dataset, args.split, args.num_examples)
        base_filename = f"dataset_sample_{args.dataset}"
        save_examples(examples, title, base_filename)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
