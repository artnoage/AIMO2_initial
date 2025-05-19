import json
import sys

def filter_entries(input_file, output_file):
    """
    Filter JSON entries based on the following rules:
    1. For each ID, keep one entry if it has the is_correct flag set to true.
    2. If no entries for an ID have is_correct as true, keep no entries for that ID.
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Group entries by ID
        entries_by_id = {}
        for entry in data:
            id_val = entry.get('id')
            if id_val not in entries_by_id:
                entries_by_id[id_val] = []
            entries_by_id[id_val].append(entry)
        
        # Filter entries based on the rules
        filtered_data = []
        for id_val, entries in entries_by_id.items():
            # Check if any entry has is_correct=true
            correct_entries = [entry for entry in entries if entry.get('is_correct') == True]
            
            # If there's at least one correct entry, keep one of them
            if correct_entries:
                filtered_data.append(correct_entries[0])
        
        # Write the filtered data to the output file
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"Filtered data written to {output_file}")
        print(f"Original entries: {len(data)}")
        print(f"Filtered entries: {len(filtered_data)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_correct_entries.py <input_json_file> <output_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sys.exit(filter_entries(input_file, output_file))
