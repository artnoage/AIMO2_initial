import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

def filter_by_alignment(data: List[Dict], alignments: List[str]) -> List[Dict]:
    """
    Filter a list of dictionaries to only include entries with specified alignments.
    
    Args:
        data: List of dictionaries containing entries
        alignments: List of alignments to keep
        
    Returns:
        Filtered list containing only entries of specified alignments
    """
    return [entry for entry in data if entry.get('alignment') in alignments]

def filter_by_types(data: List[Dict], types: List[str]) -> List[Dict]:
    """
    Filter a list of dictionaries to only include entries with specified types.
    
    Args:
        data: List of dictionaries containing entries
        types: List of types to keep (e.g. ['light', 'dark', 'judge'])
        
    Returns:
        Filtered list containing only entries of specified types
    """
    return [entry for entry in data if entry.get('type') in types]

def filter_by_success_rate_above(data: List[Dict], threshold: float) -> List[Dict]:
    """
    Filter entries to keep only those with success_rate above or equal to threshold.
    
    Args:
        data: List of dictionaries containing success_rate
        threshold: Value between 0 and 1 to compare against
    """
    filtered_data = []
    for entry in data:
        if 'success_rate' not in entry:
            print(f"Warning: Entry missing success_rate field: {entry}")
            continue
        success_rate = entry['success_rate']/100
        if success_rate >= threshold:
            filtered_data.append(entry)
    return filtered_data

def filter_by_success_rate_below(data: List[Dict], threshold: float) -> List[Dict]:
    """
    Filter entries to keep only those with success_rate below threshold.
    
    Args:
        data: List of dictionaries containing success_rate
        threshold: Value between 0 and 1 to compare against
    """
    filtered_data = []
    for entry in data:
        if 'success_rate' not in entry:
            print(f"Warning: Entry missing success_rate field: {entry}")
            continue
        success_rate = entry['success_rate']/100
        if success_rate < threshold:
            filtered_data.append(entry)
    return filtered_data

def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Filter JSON data by various criteria')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--types', nargs='+',
                      help='Types to keep (e.g. light dark judge)')
    parser.add_argument('--alignments', nargs='+',
                      help='Alignments to keep')
    parser.add_argument('--success-rate-above', type=float,
                      help='Keep entries with success rate above this threshold (0-1)')
    parser.add_argument('--success-rate-below', type=float,
                      help='Keep entries with success rate below this threshold (0-1)')
    
    args = parser.parse_args()
    
    # Validate thresholds if provided
    if args.success_rate_above and not 0 <= args.success_rate_above <= 1:
        print("Error: Success rate threshold must be between 0 and 1")
        return
    if args.success_rate_below and not 0 <= args.success_rate_below <= 1:
        print("Error: Success rate threshold must be between 0 and 1")
        return
    
    # Load data
    data = load_json(args.input_file)
    filtered_data = data
    
    # Apply filters in sequence
    if args.alignments:
        filtered_data = filter_by_alignment(filtered_data, args.alignments)
        print(f"After alignment filtering: {len(filtered_data)} entries")
        
    if args.types:
        filtered_data = filter_by_types(filtered_data, args.types)
        print(f"After type filtering: {len(filtered_data)} entries")
        
    if args.success_rate_above:
        filtered_data = filter_by_success_rate_above(filtered_data, args.success_rate_above)
        print(f"After success rate above filtering: {len(filtered_data)} entries")
        
    if args.success_rate_below:
        filtered_data = filter_by_success_rate_below(filtered_data, args.success_rate_below)
        print(f"After success rate below filtering: {len(filtered_data)} entries")
    
    # Save filtered data
    save_json(filtered_data, args.output_file)
    
    print(f"\nTotal: Filtered {len(data)} entries to {len(filtered_data)} entries")
    if args.types:
        print(f"Types kept: {', '.join(args.types)}")

if __name__ == "__main__":
    main()
