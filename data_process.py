import json
import argparse
from typing import List, Dict

def remove_by_verification(data: List[Dict], remove_wrong: bool = False, remove_right: bool = False) -> List[Dict]:
    """
    Filter data based on verification results:
    - remove_wrong: Remove entries that don't have any level 4 verifications
    - remove_right: Remove entries that ONLY have level 4 verifications
    """
    if not (remove_wrong or remove_right):
        return data
        
    filtered_data = []
    for entry in data:
        if 'verification_results' not in entry:
            continue
            
        results = entry['verification_results']
        has_success = 4 in results
        has_failure = any(x != 4 for x in results)
        
        # Keep entry if:
        # - remove_wrong is True and it has at least one success
        # - remove_right is True and it has at least one failure
        # - neither flag is True
        if ((remove_wrong and has_success) or 
            (remove_right and has_failure) or 
            (not remove_wrong and not remove_right)):
            filtered_data.append(entry)
            
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description='Process augmented dataset JSON files')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--remove-wrong', action='store_true',
                      help='Remove entries without any successful verifications')
    parser.add_argument('--remove-right', action='store_true',
                      help='Remove entries with only successful verifications')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    filtered_data = remove_by_verification(data, args.remove_wrong, args.remove_right)
    
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        print(f"Processed {len(data)} entries -> {len(filtered_data)} entries")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
