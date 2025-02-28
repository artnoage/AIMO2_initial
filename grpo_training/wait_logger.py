import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class WaitLogger:
    """
    Logger for tracking "wait a second" moments when the model corrects itself.
    Logs data to a JSON file for later analysis.
    """
    
    def __init__(self, log_file: str = "wait.json"):
        """
        Initialize the wait logger.
        
        Args:
            log_file: Path to the JSON file where data will be stored
        """
        self.log_file = log_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the log file exists with valid JSON structure"""
        if not os.path.exists(self.log_file):
            # Create the file with an empty array
            with open(self.log_file, 'w') as f:
                json.dump([], f)
        else:
            # Verify it's valid JSON
            try:
                with open(self.log_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                # If corrupted, reset to empty array
                with open(self.log_file, 'w') as f:
                    json.dump([], f)
    
    def log_wait_moment(self, 
                        problem: str, 
                        completion: str, 
                        correct_answer: str, 
                        prompt: Optional[str] = None) -> None:
        """
        Log a successful "wait a second" moment where the model corrected itself.
        
        Args:
            problem: The original problem text
            completion: The model's completion/solution
            correct_answer: The correct answer that the model found
            prompt: The prompt that was given to the model (optional)
        """
        # Load existing data
        with open(self.log_file, 'r') as f:
            data = json.load(f)
        
        # Create simple entry with just the essential fields
        entry = {
            "problem": problem,
            "prompt_completion": prompt + completion if prompt else completion,
            "correct_answer": correct_answer
        }
        
        # Append to data and save
        data.append(entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
