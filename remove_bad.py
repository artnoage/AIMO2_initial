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

        # Count initial entries
        initial_count = len(data)

        # Filter entries
        filtered_data = []
        for entry in data:
            if isinstance(entry, dict) and 'model_response' in entry:
                response = entry['model_response']
                # Check if response contains required keywords (case insensitive)
                response_lower = response.lower()
                has_problem = 'problem' in response_lower
                has_analysis = 'analysis' in response_lower
                has_step = 'step' in response_lower
                if has_problem and has_analysis and has_step:
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
