import json
import argparse
from typing import List, Dict

def filter_by_success_rate(data: List[Dict], threshold: float, above: bool) -> List[Dict]:
    """
    Filter entries based on their success_rate being above or below the threshold.
    
    Args:
        data: List of dictionaries containing success_rate
        threshold: Value between 0 and 1 to compare against
        above: If True, keep entries above threshold. If False, keep entries below threshold.
    """
    filtered_data = []
    
    for entry in data:
        if 'success_rate' not in entry:
            print(f"Warning: Entry missing success_rate field: {entry}")
            continue
            
        success_rate = entry['success_rate']
        if above and success_rate >= threshold:
            filtered_data.append(entry)
        elif not above and success_rate < threshold:
            filtered_data.append(entry)
            
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description='Filter JSON entries by success rate')
    parser.add_argument('input_file', help='Input JSON file to filter')
    parser.add_argument('--threshold', '-t', type=float, required=True,
                      help='Success rate threshold (between 0 and 1)')
    parser.add_argument('--above', '-a', action='store_true',
                      help='Keep entries above threshold (default: below)')
    parser.add_argument('--output', '-o', default='filtered.json',
                      help='Output file name (default: filtered.json)')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0 <= args.threshold <= 1:
        print("Error: Threshold must be between 0 and 1")
        return
        
    # Read input file
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print("Error: Input file must contain a JSON array")
                return
    except json.JSONDecodeError:
        print(f"Error: {args.input_file} is not valid JSON")
        return
    except Exception as e:
        print(f"Error reading {args.input_file}: {str(e)}")
        return
    
    # Filter the data
    filtered_data = filter_by_success_rate(data, args.threshold, args.above)
    
    # Write filtered data
    with open(args.output, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\nKept {len(filtered_data)} entries out of {len(data)} total")
    print(f"Filtered data written to {args.output}")

if __name__ == "__main__":
    main()
