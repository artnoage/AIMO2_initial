import json
import glob
import os
import argparse
from typing import List, Dict

def merge_json_files(folder_path: str) -> List[Dict]:
    """
    Merge all JSON files in the specified folder that contain arrays of objects.
    Each file should have format [{},...,{}]
    """
    merged_data = []
    
    # Get all .json files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return merged_data
        
    # Read and merge each file
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"Merging {file_path}: {len(data)} items")
                    merged_data.extend(data)
                else:
                    print(f"Skipping {file_path}: Not a JSON array")
        except json.JSONDecodeError:
            print(f"Error: {file_path} is not valid JSON")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return merged_data

def main():
    parser = argparse.ArgumentParser(description='Merge JSON files containing arrays of objects')
    parser.add_argument('folder', help='Folder containing JSON files to merge')
    parser.add_argument('--output', '-o', default='merged.json',
                      help='Output file name (default: merged.json)')
    
    args = parser.parse_args()
    
    # Merge the files
    merged_data = merge_json_files(args.folder)
    
    # Write merged data to output file
    if merged_data:
        with open(args.output, 'w') as f:
            json.dump(merged_data, f, indent=2)
        print(f"\nMerged {len(merged_data)} total items into {args.output}")
    else:
        print("\nNo data was merged")

if __name__ == "__main__":
    main()
