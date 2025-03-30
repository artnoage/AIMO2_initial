#!/usr/bin/env python3
"""
Script to convert CSV data to JSON format.
Converts entries with id, problem, and correct_answer to a JSON list.
"""

import csv
import json
import argparse
from pathlib import Path


def convert_csv_to_json(csv_file_path, output_file_path=None):
    """
    Convert CSV file to JSON format.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to save the output JSON file (optional)
    
    Returns:
        Path to the output JSON file
    """
    # If no output path is provided, use the same name with .json extension
    if output_file_path is None:
        output_file_path = Path(csv_file_path).with_suffix('.json')
    
    # Read CSV file and convert to list of dictionaries
    entries = []
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Rename correct_answer to correct_solution as requested
            entry = {
                'id': row['id'],
                'problem': row['problem'],
                'correct_solution': row['correct_answer']
            }
            entries.append(entry)
    
    # Write to JSON file
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(entries, json_file, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(entries)} entries from {csv_file_path} to {output_file_path}")
    return output_file_path


def main():
    parser = argparse.ArgumentParser(description='Convert CSV data to JSON format')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Path to save the output JSON file (optional)')
    
    args = parser.parse_args()
    
    convert_csv_to_json(args.csv_file, args.output)


if __name__ == "__main__":
    main()
