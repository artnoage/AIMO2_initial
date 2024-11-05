import json
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
