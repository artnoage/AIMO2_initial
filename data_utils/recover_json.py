import json
import argparse
from typing import List, Dict
from pathlib import Path

def process_json_entries(input_path: str, output_path: str, debug: bool = False):
    """
    Process JSON entries one by one, writing them to the output file as they're processed.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to write the recovered entries
        debug: If True, prints detailed error information
    """
    # Initialize output file with an opening bracket
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write('[\n')
    
    entry_count = 0
    first_entry = True
    
    def write_entry(entry: Dict, is_first: bool):
        with open(output_path, 'a', encoding='utf-8') as out_f:
            if not is_first:
                out_f.write(',\n')
            json.dump(entry, out_f, indent=2, ensure_ascii=False)
    
    # First try to parse as a single JSON array
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        data = json.loads(content)
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    write_entry(entry, first_entry)
                    first_entry = False
                    entry_count += 1
                    if entry_count % 100 == 0:
                        print(f"Processed {entry_count} entries...")
            
            # Close the JSON array
            with open(output_path, 'a', encoding='utf-8') as out_f:
                out_f.write('\n]')
            return entry_count
            
    except json.JSONDecodeError as e:
        if debug:
            print(f"Failed to parse as single JSON array: {str(e)}")
            lines = content.split('\n')
            if e.lineno <= len(lines):
                print(f"Error at line {e.lineno}:")
                print(lines[e.lineno - 1])
                print(" " * (e.colno - 1) + "^")
    
    # If that fails, try line-by-line parsing
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Try parsing as individual JSON object
                entry = json.loads(line)
                if isinstance(entry, dict):
                    write_entry(entry, first_entry)
                    first_entry = False
                    entry_count += 1
            except json.JSONDecodeError:
                # Try to find and parse any JSON-like objects in the line
                try:
                    start_idx = line.find('{')
                    end_idx = line.rfind('}')
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        potential_json = line[start_idx:end_idx + 1]
                        entry = json.loads(potential_json)
                        if isinstance(entry, dict):
                            write_entry(entry, first_entry)
                            first_entry = False
                            entry_count += 1
                except:
                    if debug:
                        print(f"Could not recover entry from line {line_num}")
                    continue
            
            if entry_count % 100 == 0:
                print(f"Processed {entry_count} entries...")
    
    # Close the JSON array
    with open(output_path, 'a', encoding='utf-8') as out_f:
        out_f.write('\n]')
    
    return entry_count

def main():
    parser = argparse.ArgumentParser(description='Recover entries from corrupted JSON file')
    parser.add_argument('input_file', help='Path to corrupted JSON file')
    parser.add_argument('output_file', help='Path to save recovered entries')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    print(f"Processing entries from {args.input_file}")
    
    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process entries and write them incrementally
    entry_count = process_json_entries(args.input_file, args.output_file, debug=args.debug)
    
    print(f"\nSuccessfully processed {entry_count} entries")
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
