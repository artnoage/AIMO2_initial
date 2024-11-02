import os
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from benchmark_numina import extract_answer_from_solution

def main():
    # Initialize Hugging Face API
    api = HfApi()
    parser = argparse.ArgumentParser(description='Filter NuminaMath-CoT dataset for olympiads with valid answers')
    parser.add_argument('--split', type=str, default='test',
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
        return answer is not None

    filtered_dataset = olympiads_dataset.filter(has_valid_answer)
    print(f"After filtering for valid answers: {len(filtered_dataset)}")

    # Convert to Hugging Face dataset format
    filtered_dataset_dict = DatasetDict({
        args.split: Dataset.from_dict({
            'problem': filtered_dataset['problem'],
            'solution': filtered_dataset['solution'],
            'source': filtered_dataset['source'],
            'answer': [extract_answer_from_solution(sol) for sol in filtered_dataset['solution']]
        })
    })

    # Save locally first
    output_dir = "numina_olympiads"
    os.makedirs(output_dir, exist_ok=True)
    filtered_dataset_dict.save_to_disk(output_dir)
    print(f"\nDataset saved locally to: {output_dir}")

    # Try to push to Hugging Face Hub
    try:
        # Note: This requires being logged in with huggingface-cli login
        filtered_dataset_dict.push_to_hub(
            "Numina-Olympiads",
            private=True,
            description="Filtered NuminaMath-CoT dataset containing only olympiads problems with valid answers"
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
