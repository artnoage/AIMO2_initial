import os
import json
from typing import List, Dict
from datetime import datetime

def handle_augmented_data_file(filename: str) -> bool:
    """
    Handle existing augmented data file.
    Returns True if should proceed, False if should cancel.
    """
    if not os.path.exists(filename):
        return True
        
    while True:
        response = input(f"\nFile {filename} already exists. Choose action:\n"
                        "1. Cancel operation\n"
                        "2. Replace file completely (will create backup)\n"
                        "3. Append new entries\n"
                        "Choice (1-3): ").strip()
                        
        if response == "1":
            return False
        elif response in ["2", "3"]:
            if response == "2":
                # Create backup before removing
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"{os.path.splitext(filename)[0]}_backup_{timestamp}.json"
                os.rename(filename, backup_name)
                print(f"Created backup: {backup_name}")
            return True
        print("Invalid choice. Please try again.")

def get_existing_ids(filename: str) -> set[int]:
    """
    Get set of existing IDs from augmented data file.
    Returns a set of integer IDs that have already been processed.
    Handles missing/corrupt files and invalid entries gracefully.
    """
    if not os.path.exists(filename):
        return set()
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            
        existing_ids = set()
        skipped = 0
        
        for item in existing_data:
            try:
                if isinstance(item, dict) and 'id' in item:
                    id_val = item['id']
                    if isinstance(id_val, (int, str)):
                        # Convert string IDs to int if needed
                        existing_ids.add(int(id_val))
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except (ValueError, TypeError):
                skipped += 1
                
        if skipped > 0:
            print(f"\nWarning: Skipped {skipped} entries with invalid/missing IDs")
            
        return existing_ids
        
    except json.JSONDecodeError as e:
        print(f"\nWarning: Could not parse existing augmented data file ({str(e)})")
        print("The file may be corrupted. Backing it up and starting fresh.")
        
        # Backup the problematic file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{os.path.splitext(filename)[0]}_backup_{timestamp}.json"
        os.rename(filename, backup_name)
        print(f"Backed up to: {backup_name}")
        
        return set()
    except Exception as e:
        print(f"\nWarning: Unexpected error reading augmented data file: {str(e)}")
        return set()

def save_augmented_data(data: List[Dict], filename: str, examples_processed: int) -> None:
    """Save augmented data to file"""
    os.makedirs('../augmented_datasets', exist_ok=True)
    
    # If file exists and we're appending, load existing data first
    existing_data = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            
    # Combine existing and new data
    combined_data = existing_data + data
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    print(f"\nSaved {len(data)} entries to augmented dataset ({examples_processed} total examples processed)")
