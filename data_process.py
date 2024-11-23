import json
import random
import argparse
import tiktoken
from typing import List, Dict, Optional

def clean_json_file(filename: str) -> Optional[List[Dict]]:
    """Clean JSON file by removing corrupted entries and returning valid data."""
    try:
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
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
                with open(filename, 'w', encoding='utf-8', errors='replace') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"Cleaned and saved file with {len(data)} valid entries")
                return data

    except Exception as e:
        print(f"Error cleaning file: {str(e)}")
        return None

def validate_verification_data(data: List[Dict], operation: str) -> bool:
    """
    Validate that the data contains required fields for verification operations.
    Returns True if valid, False otherwise.
    """
    if not data:
        print(f"Error: Empty dataset cannot be processed for {operation}")
        return False
        
    required_fields = {'verification_results', 'model_responses'}
    
    # Check first entry for required fields
    first_entry = data[0]
    missing_fields = required_fields - set(first_entry.keys())
    
    if missing_fields:
        print(f"Error: Cannot perform {operation} - missing required fields: {', '.join(missing_fields)}")
        print("The JSON file must contain 'verification_results' and 'model_responses' fields")
        return False
        
    return True

def extract_correct_synthetic(data: List[Dict]) -> Optional[List[Dict]]:
    """
    Extract only the successful model responses from synthetic verification data.
    Returns a list of entries containing only the first successful response for each problem,
    or None if the data format is invalid.
    """
    if not validate_verification_data(data, "extract synthetic correct responses"):
        return None
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
                'problem': entry['problem'],
                'solution': entry.get('correct_solution', '')  # Include solution if available
            }
            filtered_data.append(new_entry)
    return filtered_data

def extract_correct_verifier(data: List[Dict]) -> Optional[List[Dict]]:
    """
    Extract entries where all verifiers agree the solution is correct.
    Returns a list of entries where there is unanimous agreement on correctness,
    or None if the data format is invalid.
    """
    filtered_data = []
    for entry in data:
        if 'verifications' not in entry:
            continue
            
        verifications = entry['verifications']
        correctness = verifications.get('correctness', [])
        
        # Only include if there are verifications and all are True
        if correctness and all(correctness):
            new_entry = {
                'id': entry['id'],
                'model_response': entry['model_response'],
                'is_correct': True,
                'problem': entry['problem'],
                'solution': entry.get('solution', '')
            }
            filtered_data.append(new_entry)
    return filtered_data

def remove_by_verification(data: List[Dict], remove_wrong: bool = False, remove_right: bool = False) -> Optional[List[Dict]]:
    """
    Filter data based on verification results:
    - remove_wrong: Remove entries that don't have any level 4 verifications
    - remove_right: Remove entries that ONLY have level 4 verifications
    Returns None if the data format is invalid.
    """
    if not (remove_wrong or remove_right):
        return data
        
    if not validate_verification_data(data, "remove by verification"):
        return None
        
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

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text using GPT tokenizer.
    """
    encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoder
    return len(encoder.encode(text, disallowed_special=()))  # Allow all special tokens

def filter_by_tokens(data: List[Dict], max_tokens: int) -> List[Dict]:
    """
    Remove entries where any text field exceeds the token limit.
    """
    if not data:
        return []
    
    filtered_data = []
    for entry in data:
        # Convert entry to string to check all text fields
        entry_text = json.dumps(entry, ensure_ascii=False)
        if count_tokens(entry_text) <= max_tokens:
            filtered_data.append(entry)
    
    return filtered_data

def remove_entries_with_links(data: List[Dict]) -> List[Dict]:
    """
    Remove entries that contain 'http' anywhere in their text fields.
    Returns a new list with those entries removed.
    """
    if not data:
        return []
    
    filtered_data = []
    for entry in data:
        # Convert all text fields to string and check for 'http'
        entry_text = str(entry).lower()
        if 'http' not in entry_text:
            filtered_data.append(entry)
    
    return filtered_data

def deduplicate_by_id(data: List[Dict]) -> List[Dict]:
    """
    Keep only the first occurrence of each ID.
    Returns a new list with duplicate IDs removed.
    """
    if not data:
        return []
    
    seen_ids = set()
    deduplicated_data = []
    
    for entry in data:
        current_id = entry.get('id')
        if current_id is not None and current_id not in seen_ids:
            seen_ids.add(current_id)
            deduplicated_data.append(entry)
    
    return deduplicated_data

def shuffle_data(data: List[Dict], seed: Optional[int] = None) -> List[Dict]:
    """
    Randomly shuffle the dataset.
    Args:
        data: List of dictionary entries to shuffle
        seed: Optional random seed for reproducibility
    Returns:
        Shuffled copy of the input data
    """
    if seed is not None:
        random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    return shuffled

def extract_ids(data: List[Dict]) -> List[Dict]:
    """
    Extract only the ID field from each entry.
    Args:
        data: List of dictionary entries
    Returns:
        List of dictionaries containing only the ID field
    """
    return [{'id': entry.get('id')} for entry in data]

def combine_json_files(file1: str, file2: str) -> Optional[List[Dict]]:
    """
    Clean and combine two JSON files.
    Returns the combined data or None if there was an error.
    """
    # Clean and load both files
    data1 = clean_json_file(file1)
    if data1 is None:
        print(f"Failed to clean {file1}")
        return None
        
    data2 = clean_json_file(file2)
    if data2 is None:
        print(f"Failed to clean {file2}")
        return None

    # Combine the data
    combined_data = data1 + data2
    print(f"Combined entries: {len(combined_data)} ({len(data1)} from {file1}, {len(data2)} from {file2})")
    return combined_data

def main():
    parser = argparse.ArgumentParser(description='Process augmented dataset JSON files')
    parser.add_argument('-i', '--input', help='Input JSON file path')
    parser.add_argument('-i1', '--input1', help='First input JSON file for combine operation')
    parser.add_argument('-i2', '--input2', help='Second input JSON file for combine operation')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file path')
    parser.add_argument('--remove-wrong', action='store_true',
                      help='Remove entries without any successful verifications')
    parser.add_argument('--remove-right', action='store_true',
                      help='Remove entries with only successful verifications')
    parser.add_argument('--extract-correct-synthetic', action='store_true',
                      help='Extract only successful responses from synthetic verification data')
    parser.add_argument('--extract-correct-verifier', action='store_true',
                      help='Extract entries where all verifiers agree the solution is correct')
    parser.add_argument('--deduplicate', action='store_true',
                      help='Remove duplicate IDs, keeping only the first occurrence')
    parser.add_argument('--remove-links', action='store_true',
                      help='Remove entries containing http links anywhere in their text')
    parser.add_argument('--clean-only', action='store_true',
                      help='Only clean and validate the JSON file structure')
    parser.add_argument('--combine', action='store_true',
                      help='Combine two JSON files (requires --input1 and --input2)')
    parser.add_argument('--shuffle', action='store_true',
                      help='Randomly shuffle the dataset')
    parser.add_argument('--seed', type=int,
                      help='Random seed for shuffling')
    parser.add_argument('--filter-tokens', type=int,
                      help='Remove entries with more than specified number of tokens')
    parser.add_argument('--extract-ids', action='store_true',
                      help='Extract only the ID field from each entry')
    args = parser.parse_args()

    # Validate arguments
    if args.combine:
        if not (args.input1 and args.input2):
            print("Error: --combine requires both --input1 and --input2")
            return
        data = combine_json_files(args.input1, args.input2)
    else:
        if not args.input:
            print("Error: --input is required when not using --combine")
            return
        # Clean the input file
        data = clean_json_file(args.input)
        
    if data is None:
        print("Failed to process the input file(s)")
        return

    if args.clean_only:
        filtered_data = data
    else:
        # Apply filters in sequence
        filtered_data = data
        
        if args.extract_correct_synthetic:
            result = extract_correct_synthetic(filtered_data)
            if result is None:
                return
            filtered_data = result
        elif args.extract_correct_verifier:
            result = extract_correct_verifier(filtered_data)
            if result is None:
                return
            filtered_data = result
        elif args.remove_wrong or args.remove_right:
            result = remove_by_verification(filtered_data, args.remove_wrong, args.remove_right)
            if result is None:
                return
            filtered_data = result
            
        if args.deduplicate:
            filtered_data = deduplicate_by_id(filtered_data)
            
        if args.remove_links:
            filtered_data = remove_entries_with_links(filtered_data)
            
        if args.filter_tokens:
            filtered_data = filter_by_tokens(filtered_data, args.filter_tokens)
            
        if args.extract_ids:
            filtered_data = extract_ids(filtered_data)
            
        if args.shuffle:
            filtered_data = shuffle_data(filtered_data, args.seed)
    
    try:
        with open(args.output, 'w', encoding='utf-8', errors='replace') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        print(f"Processed {len(data)} entries -> {len(filtered_data)} entries")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
