import json
import argparse
from typing import List, Dict
from pathlib import Path

def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def filter_completion_results(data: List[Dict]) -> List[Dict]:
    """
    Filter completion benchmark results to keep only entries where:
    - verdict_matches is True AND extension_possible is None, OR
    - verdict_matches is True AND extension_possible is True
    """
    filtered = []
    
    # Process entries in pairs (benchmark data + statistics)
    for i in range(0, len(data), 2):
        benchmark_entry = data[i]
        stats_entry = data[i + 1] if i + 1 < len(data) else None
        
        # First get verdict_matches
        verdict_matches = benchmark_entry.get('verdict_matches', [])
        # Convert to list if single value
        if not isinstance(verdict_matches, list):
            verdict_matches = [verdict_matches]
            
        # Must have at least one verdict match and ALL must be True
        if not verdict_matches or not all(verdict_matches):
            continue
            
        # Now check extension_possible criteria
        extension_possible = benchmark_entry.get('extension_possible')
        if extension_possible is None or extension_possible is True:
            filtered.extend([benchmark_entry, stats_entry])
    
    return filtered

def main():
    parser = argparse.ArgumentParser(description='Filter completion benchmark results')
    parser.add_argument('input_file', type=str, help='Input JSON file path')
    parser.add_argument('output_file', type=str, help='Output JSON file path')
    args = parser.parse_args()

    # Load data
    data = load_json(args.input_file)
    
    # Filter results
    filtered_data = filter_completion_results(data)
    
    # Save filtered data
    save_json(filtered_data, args.output_file)
    
    print(f"Filtered {len(data)} entries down to {len(filtered_data)}")

if __name__ == "__main__":
    main()
