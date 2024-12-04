import json
import argparse
from typing import List, Dict
from pathlib import Path

def recover_json_entries(file_path: str) -> List[Dict]:
    """
    Attempts to recover valid JSON entries from a corrupted file.
    Assumes the file contains one JSON object per line or an array of objects.
    """
    recovered = []
    
    # Read the entire file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First try to parse as a single JSON array
    try:
        data = json.loads(content)
        if isinstance(data, list):
            recovered.extend(data)
            return recovered
    except json.JSONDecodeError:
        pass

    # If that fails, try line-by-line parsing
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Try parsing as individual JSON object
                entry = json.loads(line)
                if isinstance(entry, dict):
                    recovered.append(entry)
            except json.JSONDecodeError:
                # Try to find and parse any JSON-like objects in the line
                try:
                    # Look for object boundaries
                    start_idx = line.find('{')
                    end_idx = line.rfind('}')
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        potential_json = line[start_idx:end_idx + 1]
                        entry = json.loads(potential_json)
                        if isinstance(entry, dict):
                            recovered.append(entry)
                except:
                    print(f"Could not recover entry from line {line_num}")
                    continue

    return recovered

def main():
    parser = argparse.ArgumentParser(description='Recover entries from corrupted JSON file')
    parser.add_argument('input_file', help='Path to corrupted JSON file')
    parser.add_argument('output_file', help='Path to save recovered entries')
    args = parser.parse_args()

    print(f"Attempting to recover entries from {args.input_file}")
    
    recovered_entries = recover_json_entries(args.input_file)
    
    print(f"Successfully recovered {len(recovered_entries)} entries")
    
    # Save recovered entries
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recovered_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Saved recovered entries to {args.output_file}")

if __name__ == "__main__":
    main()
