import json
import argparse
import os
from typing import List, Dict, Any, Set
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

def filter_most_common_correct(data: List[Dict]) -> List[Dict]:
    """
    Filter entries keeping only those where is_most_common_correct is true.
    
    Args:
        data: List of dictionaries containing entries
        
    Returns:
        Filtered list containing only entries where is_most_common_correct is true
    """
    return [entry for entry in data if entry.get('is_most_common_correct', False)]

def filter_by_agent(data: List[Dict], agent_type: str) -> List[Dict]:
    """
    Filter entries to keep only those from a specific agent type (main/auxiliary).
    
    Args:
        data: List of dictionaries containing entries
        agent_type: String specifying agent type ('main' or 'auxiliary')
        
    Returns:
        Filtered list containing only entries from specified agent
    """
    return [entry for entry in data if entry.get('agent_type') == agent_type]

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
    parser.add_argument('--exclude', type=str,
                      help='JSON file containing entries to exclude')
    parser.add_argument('--include', type=str,
                      help='JSON file containing entries to include (keep only these)')
    parser.add_argument('--types', nargs='+',
                      help='Types to keep (e.g. light dark judge)')
    parser.add_argument('--alignments', nargs='+',
                      help='Alignments to keep')
    parser.add_argument('--success-rate-above', type=float,
                      help='Keep entries with success rate above this threshold (0-1)')
    parser.add_argument('--success-rate-below', type=float,
                      help='Keep entries with success rate below this threshold (0-1)')
    parser.add_argument('--most-common-correct', action='store_true',
                      help='Keep only entries where is_most_common_correct is true')
    parser.add_argument('--agent', choices=['main', 'auxiliary'],
                      help='Keep only entries from specified agent type')
    parser.add_argument('--direction', choices=['in', 'out'], default='in',
                      help='Filter direction: "in" to keep matching entries, "out" to remove them')
    
    args = parser.parse_args()
    
    # Validate thresholds if provided
    if args.success_rate_above and not 0 <= args.success_rate_above <= 1:
        print("Error: Success rate threshold must be between 0 and 1")
        return
    if args.success_rate_below and not 0 <= args.success_rate_below <= 1:
        print("Error: Success rate threshold must be between 0 and 1")
        return
    
    # Load exclude/include lists if provided
    exclude_data = []
    include_data = []
    
    if args.exclude and os.path.exists(args.exclude):
        try:
            exclude_data = load_json(args.exclude)
            print(f"Loaded {len(exclude_data)} entries to exclude")
        except Exception as e:
            print(f"Error loading exclude file: {e}")
            return

    if args.include and os.path.exists(args.include):
        try:
            include_data = load_json(args.include)
            print(f"Loaded {len(include_data)} entries to include")
        except Exception as e:
            print(f"Error loading include file: {e}")
            return

    # Load data
    data = load_json(args.input_file)
    filtered_data = data

    # Apply include/exclude filters first
    if include_data:
        filtered_data = [entry for entry in filtered_data 
                        if any(entry.get('problem') == inc.get('problem')
                              for inc in include_data)]
        print(f"After including only specified entries: {len(filtered_data)}")
    
    if exclude_data:
        filtered_data = [entry for entry in filtered_data 
                        if not any(entry.get('problem') == exc.get('problem')
                                 for exc in exclude_data)]
        print(f"After excluding entries: {len(filtered_data)}")
    
    # Apply filters in sequence
    if args.alignments:
        result = filter_by_alignment(filtered_data, args.alignments)
        filtered_data = result if args.direction == 'in' else [x for x in filtered_data if x not in result]
        print(f"After alignment filtering: {len(filtered_data)} entries")
        
    if args.types:
        result = filter_by_types(filtered_data, args.types)
        filtered_data = result if args.direction == 'in' else [x for x in filtered_data if x not in result]
        print(f"After type filtering: {len(filtered_data)} entries")
        
    if args.success_rate_above:
        result = filter_by_success_rate_above(filtered_data, args.success_rate_above)
        filtered_data = result if args.direction == 'in' else [x for x in filtered_data if x not in result]
        print(f"After success rate above filtering: {len(filtered_data)} entries")
        
    if args.success_rate_below:
        result = filter_by_success_rate_below(filtered_data, args.success_rate_below)
        filtered_data = result if args.direction == 'in' else [x for x in filtered_data if x not in result]
        print(f"After success rate below filtering: {len(filtered_data)} entries")
        
    if args.most_common_correct:
        result = filter_most_common_correct(filtered_data)
        filtered_data = result if args.direction == 'in' else [x for x in filtered_data if x not in result]
        print(f"After most common correct filtering: {len(filtered_data)} entries")
        
    if args.agent:
        result = filter_by_agent(filtered_data, args.agent)
        filtered_data = result if args.direction == 'in' else [x for x in filtered_data if x not in result]
        print(f"After agent type filtering: {len(filtered_data)} entries")
    
    # Save filtered data
    save_json(filtered_data, args.output_file)
    
    print(f"\nTotal: Filtered {len(data)} entries to {len(filtered_data)} entries")
    if args.types:
        print(f"Types kept: {', '.join(args.types)}")

if __name__ == "__main__":
    main()
