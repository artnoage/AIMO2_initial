from typing import Dict, List
import json
import argparse
from pathlib import Path
from datasets import Dataset
import datetime

def process_dataset(input_path: str, output_path: str):
    """
    Process the entire dataset and convert to conversation format.
    Creates a single conversations list containing all examples.
    """
    # Read input JSON
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create list of conversations, each containing a complete exchange
    conversations = []
    for item in data:
        conversation = [
            {
                "role": "user",
                "content": item["problem"]
            },
            {
                "role": "assistant", 
                "content": item["solution"]
            }
        ]
        conversations.append(conversation)
    
    # Create the final format
    converted_data = {"conversations": conversations}
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict({"conversations": conversations})
    
    # Save as both JSON and HuggingFace dataset
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_output = output_path if output_path else f"conversation_dataset_{timestamp}.json"
    with open(json_output, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    # Save as HuggingFace dataset
    dataset_output = f"conversation_dataset_{timestamp}"
    dataset.save_to_disk(dataset_output)
    
    print(f"Saved JSON dataset to: {json_output}")
    print(f"Saved HuggingFace dataset to: {dataset_output}")

def main():
    parser = argparse.ArgumentParser(description='Convert JSON dataset to conversation format')
    parser.add_argument('input_path', type=str, help='Path to input JSON file')
    parser.add_argument('--output_path', type=str, help='Path for output JSON file (optional)')
    
    args = parser.parse_args()
    process_dataset(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
