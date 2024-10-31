from datasets import load_dataset
import random

def main():
    # Load dataset
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split='train')
    
    # Sample 10 random indices
    sample_indices = random.sample(range(len(dataset)), 10)
    
    # Create markdown output
    markdown = "# Random Samples from NuminaMath-CoT Training Set\n\n"
    
    for i, idx in enumerate(sample_indices, 1):
        example = dataset[idx]
        markdown += f"## Example {i} (Index: {idx})\n\n"
        markdown += f"### Problem\n{example['problem']}\n\n"
        markdown += f"### Solution\n{example['solution']}\n\n"
        markdown += f"### Source\n{example['source']}\n\n"
        markdown += f"### Level\n{example['level']}\n\n"
        markdown += f"### Topic\n{example['topic']}\n\n"
        markdown += "---\n\n"
    
    # Write to file
    with open('dataset_samples.md', 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print("Samples written to dataset_samples.md")

if __name__ == "__main__":
    main()
