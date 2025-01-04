import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

def clean_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single entry in the dataset by ensuring all required fields are strings."""
    
    # List of fields that should be strings
    string_fields = ["chosen", "rejected", "prompt"]
    
    cleaned = entry.copy()
    
    # Convert None to empty string and ensure string type
    for field in string_fields:
        if field in cleaned:
            if isinstance(cleaned[field], dict) and "content" in cleaned[field]:
                # Handle nested content structure
                if cleaned[field]["content"] is None:
                    cleaned[field]["content"] = ""
                else:
                    cleaned[field]["content"] = str(cleaned[field]["content"])
            elif cleaned[field] is None:
                cleaned[field] = ""
            else:
                cleaned[field] = str(cleaned[field])
    
    return cleaned

def clean_json_file(input_path: str, output_path: str) -> None:
    """
    Read JSON file, clean the data, and write back to a new file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to write cleaned JSON file
    """
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Clean each entry
    if isinstance(data, list):
        cleaned_data = [clean_entry(entry) for entry in data]
    else:
        cleaned_data = clean_entry(data)
    
    # Write cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned data written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Clean JSON data by fixing None values and ensuring string types')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('--output-file', help='Output JSON file path (default: input_file_cleaned.json)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if args.output_file:
        output_path = args.output_file
    else:
        # Create default output filename by adding _cleaned before the extension
        stem = input_path.stem
        output_path = str(input_path.with_name(f"{stem}_cleaned.json"))
    
    clean_json_file(str(input_path), output_path)

if __name__ == "__main__":
    main()
