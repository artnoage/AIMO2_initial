import os
import json
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ProgressTracker:
    """
    Tracks progress and statistics during benchmark runs.
    
    Attributes:
        total_examples: Total number of examples to process
        best_of: Number of attempts per example
        results: List of processed results
        start_time: Timestamp when tracking started
    """
    total_examples: int
    best_of: int
    results: List[Dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    def _save_progress_stats(self, stats: str) -> None:
        """Save progress statistics to a markdown file"""
        os.makedirs("results", exist_ok=True)
        stats_file = os.path.join("results", f"progress_stats_{self.start_time.strftime('%Y%m%d_%H%M%S')}.md")
        
        # Open in append mode, which will create the file if it doesn't exist
        with open(stats_file, 'a') as f:
            # If file is empty, write the header
            if f.tell() == 0:
                f.write(f"# Benchmark Progress Statistics\n\n")
                f.write(f"Started at: {self.start_time.isoformat()}\n\n")
            # Append new stats
            f.write(stats)

    def add_result(self, result: Dict) -> None:
        if result:
            self.results.append(result)
    
    def calculate_error_rate(self, results: List[Dict]) -> float:
        if not results:
            return 0.0
        correct_count = sum(1 for r in results if any(r.get('is_correct_list', [])))
        return 1.0 - (correct_count / len(results))

    def print_progress(self, model_name: str = None, split: str = None) -> None:
        if len(self.results) % 100 == 0 and self.results:
            last_hundred = self.results[-100:]
            batch_error_rate = self.calculate_error_rate(last_hundred)
            cumulative_error_rate = self.calculate_error_rate(self.results)
            
            stats = f"\nAt {len(self.results)} examples:\n"
            stats += f"Batch Success Rate (last 100): {1 - batch_error_rate:.4f}\n"
            stats += f"Cumulative Success Rate: {1 - cumulative_error_rate:.4f}\n"
            stats += "-" * 80 + "\n"
            
            print(stats)
            self._save_progress_stats(stats)
            
            # Automatically save results every 100 examples
            if model_name and split:
                self.save_results(model_name, split)

    def save_results(self, model_name: str = None, split: str = None) -> None:
        """Save results to a JSON file"""
        if not self.results:
            return
            
        try:
            # Create metadata
            metadata = {
                "timestamp": self.start_time.isoformat(),
                "total_examples": self.total_examples,
                "best_of": self.best_of,
                "model": model_name,
                "split": split,
                "results": self.results
            }
            
            # Use a fixed filename based on the start timestamp
            filename = f"benchmark_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("results", filename)
            
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            print(f"\nSaving {len(self.results)} results to: {filepath}")
            
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Results successfully saved to: {filepath}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def print_final_stats(self) -> None:
        if not self.results:
            msg = "\nNo examples were successfully processed."
            print(msg)
            self._save_progress_stats(msg + "\n")
            return

        total = len(self.results)
        
        # Calculate success metrics
        any_correct = sum(1 for r in self.results if any(r.get('is_correct_list', [])))
        majority_correct = sum(1 for r in self.results 
                             if sum(r.get('is_correct_list', [])) > self.best_of // 2)
        
        # Calculate per-attempt success rates
        total_attempts = sum(len(r.get('is_correct_list', [])) for r in self.results)
        successful_attempts = sum(sum(r.get('is_correct_list', [])) for r in self.results)
        
        # Calculate percentages
        any_accuracy = (any_correct / total) * 100 if total > 0 else 0
        majority_accuracy = (majority_correct / total) * 100 if total > 0 else 0
        attempt_accuracy = (successful_attempts / total_attempts) * 100 if total_attempts > 0 else 0

        end_time = datetime.now()
        total_duration = end_time - self.start_time

        stats = "\n\n## Final Results\n\n"
        stats += f"- Total examples processed: {total}\n"
        stats += f"- Any-Correct Accuracy: {any_correct}/{total} = {any_accuracy:.2f}%\n"
        stats += f"- Majority-Correct Accuracy: {majority_correct}/{total} = {majority_accuracy:.2f}%\n"
        stats += f"- Per-Attempt Accuracy: {successful_attempts}/{total_attempts} = {attempt_accuracy:.2f}%\n\n"

        stats += f"### Best-of-{self.best_of} Statistics\n\n"
        stats += f"- Problems with at least one correct solution: {any_correct}/{total} = {any_accuracy:.2f}%\n"
        stats += f"- Problems with majority correct solutions: {majority_correct}/{total} = {majority_accuracy:.2f}%\n"
        stats += f"- Total successful attempts: {successful_attempts}/{total_attempts} = {attempt_accuracy:.2f}%\n\n"

        stats += "### Timing Information\n\n"
        stats += f"- Total execution time: {total_duration}\n"
        stats += f"- Average time per example: {total_duration.total_seconds() / total:.2f} seconds\n"

        print(stats)
        self._save_progress_stats(stats)
