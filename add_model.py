import json
import argparse
import os

def add_model_to_file(filename: str, model_name: str) -> None:
    """Add model name to all entries in a JSON file."""
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return

    try:
        # Read the file
        with open(filename, 'r') as f:
            data = json.load(f)

        # Check if it's a list of dictionaries
        if not isinstance(data, list):
            print(f"Error: File {filename} does not contain a list of examples")
            return

        # Add model name to each entry
        modified = False
        for entry in data:
            if isinstance(entry, dict) and 'model' not in entry:
                entry['model'] = model_name
                modified = True

        if not modified:
            print("No changes needed - all entries already have 'model' field")
            return

        # Write back to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Successfully added model name to {filename}")

    except json.JSONDecodeError:
        print(f"Error: File {filename} is not valid JSON")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Add model name to JSON files')
    parser.add_argument('--file', type=str, required=True,
                    help='JSON file to process')
    parser.add_argument('--model', type=str, required=True,
                    help='Model name to add')

    args = parser.parse_args()
    add_model_to_file(args.file, args.model)

if __name__ == "__main__":
    main()