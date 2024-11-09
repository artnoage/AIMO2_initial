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
        
        # Create markdown content
        md_content = "# Numina-Olympiads Dataset Sample\n\n"
        for i, ex in enumerate(examples, 1):
            md_content += f"## Entry {i}\n\n"
            md_content += f"**ID**: {ex['id']}\n\n"
            md_content += "### Problem\n\n"
            md_content += f"{ex['problem']}\n\n"
            md_content += "### Solution\n\n"
            md_content += f"{ex['solution']}\n\n"
            md_content += "---\n\n"
        
        # Save to markdown file
        with open('dataset_sample.md', 'w') as f:
            f.write(md_content)
        print(f"\nSaved {len(examples)} examples to dataset_sample.md")
        
        # Also save raw data to JSON for reference
        with open('dataset_sample.json', 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"Saved raw data to dataset_sample.json")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
