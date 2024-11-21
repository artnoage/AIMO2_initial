import json
import argparse
from typing import List, Dict, Optional

def clean_json_file(filename: str) -> Optional[List[Dict]]:
    """Clean JSON file by removing corrupted entries and returning valid data."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # First try standard parsing
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Standard JSON parsing failed at position {e.pos}")
                print("Attempting line-by-line parsing...")
                
                # Reset file pointer
                f.seek(0)
                data = []
                line_num = 0
                
                # Read opening bracket
                first_line = f.readline().strip()
                if first_line != '[':
                    raise ValueError("File must start with '['")
                
                # Buffer for incomplete objects
                buffer = ""
                
                for line in f:
                    line_num += 1
                    if line_num % 10000 == 0:
                        print(f"Processing line {line_num}...")
                    
                    buffer += line.strip()
                    
                    if buffer.endswith('},'):  # Complete object
                        try:
                            obj = json.loads(buffer.rstrip(','))
                            data.append(obj)
                            buffer = ""
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON at line {line_num}")
                            buffer = ""
                    elif buffer.endswith('}'):  # Last object
                        try:
                            obj = json.loads(buffer)
                            data.append(obj)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON at line {line_num}")
                
                if not data:
                    raise ValueError("No valid JSON objects found")
                
                # Write cleaned data back to file
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Cleaned and saved file with {len(data)} valid entries")
                return data

    except Exception as e:
        print(f"Error cleaning file: {str(e)}")
        return None

def extract_successful_responses(data: List[Dict]) -> List[Dict]:
    """
    Extract only the successful model responses and format them with metadata.
    Returns a list of entries containing only the first successful response for each problem.
    """
    filtered_data = []
    for entry in data:
        if 'verification_results' not in entry:
            continue
            
        results = entry['verification_results']
        if 4 in results:
            # Find the first successful response
            success_index = results.index(4)
            new_entry = {
                'id': entry['id'],
                'model_response': entry['model_responses'][success_index],
                'is_correct': True,
                'metadata': {
                    'problem': entry['problem'],
                    'solution': entry.get('correct_solution', '')  # Include solution if available
                }
            }
            filtered_data.append(new_entry)
    return filtered_data

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
    parser.add_argument('--extract-right', action='store_true',
                      help='Extract only successful responses and mark as correct')
    args = parser.parse_args()

    # First clean the input file
    data = clean_json_file(args.input_file)
    if data is None:
        print("Failed to process the input file")
        return

    if args.extract_right:
        filtered_data = extract_successful_responses(data)
    else:
        filtered_data = remove_by_verification(data, args.remove_wrong, args.remove_right)
    
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        print(f"Processed {len(data)} entries -> {len(filtered_data)} entries")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
