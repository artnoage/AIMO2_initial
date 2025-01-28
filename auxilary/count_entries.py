import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def count_entries(data: List[Dict], group_by: Optional[str] = None) -> Dict:
    """
    Count entries in dataset, optionally grouping by a field
    
    Args:
        data: List of dictionaries containing entries
        group_by: Optional field name to group counts by
        
    Returns:
        Dictionary with count statistics
    """
    stats = {
        'total_entries': len(data)
    }
    
    if group_by:
        # Count occurrences of each value in the group_by field
        counter = Counter(entry.get(group_by) for entry in data)
        stats['groups'] = dict(counter.most_common())
        
    return stats

def main():
    parser = argparse.ArgumentParser(description='Count entries in JSON dataset')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('--group-by', type=str,
                      help='Field to group counts by (e.g. type, source)')
    
    args = parser.parse_args()
    
    try:
        # Load and count data
        data = load_json(args.input_file)
        stats = count_entries(data, args.group_by)
        
        # Print results
        print(f"\nTotal entries: {stats['total_entries']}")
        
        if args.group_by and 'groups' in stats:
            print(f"\nCounts by {args.group_by}:")
            for key, count in stats['groups'].items():
                if key is None:
                    key = 'None'
                print(f"- {key}: {count}")
                
    except Exception as e:
        print(f"Error processing file: {e}")
        return

if __name__ == "__main__":
    main()
