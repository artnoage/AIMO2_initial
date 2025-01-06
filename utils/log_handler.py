import os
from datetime import datetime
from typing import List, Optional

class MarkdownLogger:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"log_{timestamp}.md"
        self.filepath = os.path.join(output_dir, self.filename)
        # Initialize the log file
        with open(self.filepath, 'w') as f:
            f.write("# Solution Generation Log\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def save_logs(self, logs: List[str], example_id: Optional[int] = None) -> str:
        """Append logs for an example to the main log file and return the filename"""
        with open(self.filepath, 'a') as f:
            f.write(f"\n\n## Example {example_id}\n\n")
            f.write("---\n\n")
            f.write("\n".join(str(log) for log in logs))
            f.write("\n\n" + "="*80 + "\n")
            
        return self.filepath
