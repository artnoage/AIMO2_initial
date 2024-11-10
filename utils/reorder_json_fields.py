import json
import argparse
from typing import List, Dict

def reorder_json_entries(data: List[Dict]) -> List[Dict]:
    """Reorder each dictionary in the list to put 'id' field first."""
    reordered = []
    for entry in data:
        if 'id' in entry:
            # Create new dict starting with id
            new_entry = {'id': entry['id']}
            # Add all other fields in their original order
            for key, value in entry.items():
                if key != 'id':
                    new_entry[key] = value
            reordered.append(new_entry)
        else:
            reordered.append(entry)  # Keep entries without id unchanged
    return reordered

def main():
    parser = argparse.ArgumentParser(description='Reorder JSON entries to put id field first')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    args = parser.parse_args()

    try:
        # Read input JSON
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reorder the entries
        reordered_data = reorder_json_entries(data)

        # Write output JSON
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(reordered_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully reordered JSON and saved to {args.output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
