import json
import argparse
from typing import Dict, List

def convert_ids_to_int(data: List[Dict]) -> List[Dict]:
    """Convert 'id' fields from string to integer in a list of dictionaries."""
    converted_data = []
    for item in data:
        new_item = item.copy()
        if 'id' in new_item:
            try:
                new_item['id'] = int(new_item['id'])
            except ValueError:
                print(f"Warning: Could not convert id '{new_item['id']}' to integer. Keeping as string.")
        converted_data.append(new_item)
    return converted_data

def process_file(input_file: str, output_file: str) -> None:
    """Read JSON file, convert IDs to integers, and save to new file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
            
        converted_data = convert_ids_to_int(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully converted IDs and saved to {output_file}")
        
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file")
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Convert ID fields from string to integer in JSON file')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    
    args = parser.parse_args()
    process_file(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
