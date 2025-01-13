import json
import sys
from pathlib import Path

def modify_judge_prompts(file_path: str):
    """
    Modifies judge prompts in the JSON file by replacing
    'Explain your reasoning' with 'Just answer with A or B'
    """
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process each item
    modified = False
    for item in data:
        if isinstance(item, dict) and 'prompt' in item and 'content' in item['prompt']:
            content = item['prompt']['content']
            if content.endswith('Explain your reasoning.'):
                item['prompt']['content'] = content.replace(
                    'Explain your reasoning.',
                    'Just answer with A or B.'
                )
                modified = True
    
    if modified:
        # Write back to file
        output_path = Path(file_path).with_stem(Path(file_path).stem + '_modified')
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Modified file saved as: {output_path}")
    else:
        print("No modifications were needed")

def main():
    if len(sys.argv) != 2:
        print("Usage: python modify_judge_prompt.py <json_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
        
    modify_judge_prompts(file_path)

if __name__ == "__main__":
    main()
