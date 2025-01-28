import logging
from typing import List, Optional

class BenchmarkLogger:
    """Handles logging for benchmark generators"""
    
    def __init__(self):
        self.logs: List[str] = []
        
    def append(self, message: str):
        """Add a message to logs"""
        self.logs.append(message)
        
    def extend(self, messages: List[str]):
        """Add multiple messages to logs"""
        self.logs.extend(messages)
        
    def print(self):
        """Print all accumulated logs"""
        if self.logs:
            print("\n".join(self.logs))
            
    def clear(self):
        """Clear all logs"""
        self.logs = []
