import argparse
from json_utils import clean_json_file
import json
from typing import List, Dict

def combine_json_files(file1: str, file2: str, output_file: str = "combo.json") -> None:
    """
    Clean and combine two JSON files into a single output file.
    
    Args:
        file1: Path to first JSON file
        file2: Path to second JSON file
        output_file: Path to output combined JSON file
    """
    # Clean and load both files
    data1 = clean_json_file(file1)
    if data1 is None:
        print(f"Failed to clean {file1}")
        return
        
    data2 = clean_json_file(file2)
    if data2 is None:
        print(f"Failed to clean {file2}")
        return

    # Combine the data
    combined_data = data1 + data2
    
    # Write combined data to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2)
        print(f"Successfully combined files into {output_file}")
        print(f"Total entries: {len(combined_data)} ({len(data1)} from {file1}, {len(data2)} from {file2})")
    except Exception as e:
        print(f"Error writing combined file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Combine two JSON files after cleaning them')
    parser.add_argument('file1', help='First JSON file')
    parser.add_argument('file2', help='Second JSON file')
    parser.add_argument('--output', '-o', default='combo.json',
                      help='Output file name (default: combo.json)')
    
    args = parser.parse_args()
    combine_json_files(args.file1, args.file2, args.output)

if __name__ == "__main__":
    main()
