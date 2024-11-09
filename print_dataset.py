from datasets import load_dataset
from huggingface_hub import HfApi
import json

def main():
    try:
        # Get username and load dataset
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina-Olympiads", split='train')
        
        # Take first 10 examples
        examples = []
        for i, ex in enumerate(dataset):
            if i >= 10:  # Only get first 10
                break
            example = {
                'id': ex['id'],
                'problem': ex['problem'],
                'solution': ex['solution']
            }
            examples.append(example)
        
        # Print examples with nice formatting
        print("\nFirst 10 dataset entries:")
        print("=" * 80)
        for i, ex in enumerate(examples, 1):
            print(f"\nEntry {i}:")
            print(f"ID: {ex['id']}")
            print("\nProblem:")
            print(ex['problem'])
            print("\nSolution:")
            print(ex['solution'])
            print("-" * 80)
        
        # Also save to JSON file for easier viewing
        with open('dataset_sample.json', 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"\nSaved {len(examples)} examples to dataset_sample.json")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
