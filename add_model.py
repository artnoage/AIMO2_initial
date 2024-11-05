import json
import argparse
import os

def add_models_to_file(filename: str, solver_name: str, verifier_name: str) -> None:
    """Add solver and verifier names to all entries in a JSON file."""
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return

    try:
        # Check file size
        file_size = os.path.getsize(filename)
        print(f"Processing file of size: {file_size / (1024*1024):.2f} MB")

        # Read the file in chunks if it's large
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON decode error at position {e.pos}: {e.msg}")
                print("Attempting to read file in chunks...")
                
                f.seek(0)
                content = ""
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    content += chunk
                
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON even with chunked reading: {e}")
                    return

        # Check if it's a list of dictionaries
        if not isinstance(data, list):
            print(f"Error: File {filename} does not contain a list of examples")
            return

        # Add model name to each entry
        modified = False
        for entry in data:
            if isinstance(entry, dict):
                if 'solver' not in entry or 'verifier' not in entry:
                    if 'solver' not in entry:
                        entry['solver'] = solver_name
                    if 'verifier' not in entry:
                        entry['verifier'] = verifier_name
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
    parser.add_argument('--solver', type=str, default='LOCAL_ORIGINAL',
                    help='Solver model name to add (default: LOCAL_ORIGINAL)')
    parser.add_argument('--verifier', type=str, default='GEMINI_FLASH',
                    help='Verifier model name to add (default: GEMINI_FLASH)')

    args = parser.parse_args()
    add_models_to_file(args.file, args.solver, args.verifier)

if __name__ == "__main__":
    main()
import json
import argparse
import os

def remove_bad_entries(filename: str) -> None:
    """
    Remove entries from a JSON file where model_response doesn't contain
    required keywords ('Problem Analysis' and 'STEP'/'Step').
    """
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

        # Count initial entries
        initial_count = len(data)

        # Filter entries
        filtered_data = []
        for entry in data:
            if isinstance(entry, dict) and 'model_response' in entry:
                response = entry['model_response']
                # Check if response contains required keywords
                has_analysis = 'Problem Analysis' in response
                has_step = 'STEP' in response.upper()
                if has_analysis and has_step:
                    filtered_data.append(entry)

        # Count removed entries
        removed_count = initial_count - len(filtered_data)

        if removed_count == 0:
            print("No entries needed to be removed")
            return

        # Write back to file
        with open(filename, 'w') as f:
            json.dump(filtered_data, f, indent=2)

        print(f"Removed {removed_count} entries from {filename}")
        print(f"Original count: {initial_count}")
        print(f"New count: {len(filtered_data)}")

    except json.JSONDecodeError:
        print(f"Error: File {filename} is not valid JSON")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Remove entries with invalid model responses from JSON files')
    parser.add_argument('--file', type=str, required=True,
                    help='JSON file to process')

    args = parser.parse_args()
    remove_bad_entries(args.file)

if __name__ == "__main__":
    main()
