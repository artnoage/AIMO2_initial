import argparse
import json
from typing import List, Dict, Tuple
import sys

def load_json_file(filename: str) -> List[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        sys.exit(1)

def compare_entries(entry1: Dict, entry2: Dict) -> bool:
    """
    Compare two entries to check if they have matching problems and solutions.
    Returns True if they match, False otherwise.
    """
    return (entry1.get('problem', '') == entry2.get('problem', '') and 
            entry1.get('solution', '') == entry2.get('solution', ''))

def process_files(correct_file: str, incorrect_file: str, output_file: str) -> Tuple[int, int, int]:
    """
    Process two JSON files and create a new JSON file with matching entries.
    Also returns statistics about matching entries.
    Returns: (total_ids, matching_ids, matching_content)
    """
    correct_data = load_json_file(correct_file)
    incorrect_data = load_json_file(incorrect_file)
    
    # Create a dictionary for faster lookup of incorrect entries
    incorrect_dict = {entry.get('id'): entry for entry in incorrect_data}
    
    total_ids = len(correct_data)
    matching_ids = 0
    matching_content = 0
    matching_entries = []
    
    for correct_entry in correct_data:
        correct_id = correct_entry.get('id')
        if correct_id in incorrect_dict:
            matching_ids += 1
            incorrect_entry = incorrect_dict[correct_id]
            if compare_entries(correct_entry, incorrect_entry):
                matching_content += 1
                # Create new entry with the required DPO format
                new_entry = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": "You are a precise mathematical problem solver. You will be given a problem to solve.\n\nDO:\n List applicable theorems/techniques upfront\n If possible each step must contain a justification. \n Use LaTeX notation\n\nFORMAT:\n\n**Problem Analysis and Approach**:\n1. Start by categorizing the problem (e.g., \"This is an inequality problem involving algebraic identities\" or \"This is a combinatorial proof\").\n2. List specific tools or theorems that will guide your solution (e.g., \"AM-GM inequality,\" \"Basic algebraic manipulations\").\n\n**PROOF**:\nExample format for each step:\nGiven: \\( a, b, c > 0 \\) and \\( a + b + c = 3 \\). Prove that \\( abc \\leq 1 \\).\n\nStep 1. By the AM-GM inequality, \\( \\frac{a + b + c}{3} \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Apply AM-GM inequality to \\( a, b, c \\)]  \nStep 2. Substituting \\( a + b + c = 3 \\), we get \\( 1 \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Replace with given sum condition]  \nStep 3. Cube both sides to eliminate the root: \\( 1 \\geq abc \\) \\hspace{10pt} [Cube both sides to solve for \\( abc \\)]  \nStep 4. Thus, \\( abc \\leq 1 \\), as required.  \n\nFor each step, clearly state the action, use concise LaTeX notation, and provide a justification in brackets.\n\n**ANSWER**:\n\\(\\boxed{\\text{result}}\\)"
                        },
                        {
                            "from": "human",
                            "value": correct_entry['problem']
                        }
                    ],
                    "chosen": {
                        "from": "gpt",
                        "value": correct_entry.get('model_response', '')
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": incorrect_entry.get('model_response', '')
                    }
                }
                matching_entries.append(new_entry)
    
    # Save matching entries to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matching_entries, f, indent=2, ensure_ascii=False)
    
    return total_ids, matching_ids, matching_content

def main():
    parser = argparse.ArgumentParser(description='Create DPO dataset from correct and incorrect examples')
    parser.add_argument('-c', '--correct', required=True, help='JSON file with correct examples')
    parser.add_argument('-i', '--incorrect', required=True, help='JSON file with incorrect examples')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file for matching entries')
    
    args = parser.parse_args()
    
    total_ids, matching_ids, matching_content = process_files(args.correct, args.incorrect, args.output)
    
    print(f"Total entries in correct file: {total_ids}")
    print(f"Entries with matching IDs: {matching_ids}")
    print(f"Entries with matching problems and solutions: {matching_content}")
    print(f"Percentage of matching content: {(matching_content/total_ids)*100:.2f}%")

if __name__ == "__main__":
    main()
