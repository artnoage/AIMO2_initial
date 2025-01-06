import os
from datetime import datetime
from typing import List, Optional

class MarkdownLogger:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def save_logs(self, logs: List[str], example_id: Optional[int] = None) -> str:
        """Save logs to markdown file and return the filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_str = f"_{example_id}" if example_id is not None else ""
        filename = f"log_{timestamp}{id_str}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("# Solution Generation Log\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if example_id is not None:
                f.write(f"Example ID: {example_id}\n")
            f.write("\n---\n\n")
            f.write("\n".join(str(log) for log in logs))
            
        return filepath
