import os
import argparse
from datasets import load_dataset
from benchmark_numina import extract_answer_from_solution

def main():
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

    # Save the filtered dataset
    output_dir = "filtered_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"olympiads_{args.split}.json")
    filtered_dataset.to_json(output_path)
    print(f"\nFiltered dataset saved to: {output_path}")

    # Print some statistics
    print("\nSample problems from filtered dataset:")
    for idx, example in enumerate(filtered_dataset.select(range(min(3, len(filtered_dataset))))):
        print(f"\nProblem {idx + 1}:")
        print(f"Source: {example['source']}")
        print(f"Answer: {extract_answer_from_solution(example['solution'])}")
        print("-" * 80)

if __name__ == "__main__":
    main()
