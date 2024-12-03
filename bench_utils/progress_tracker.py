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
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        stats_file = os.path.join("results", f"progress_stats_{timestamp}.md")
        
        # Create file if it doesn't exist
        if not os.path.exists(stats_file):
            with open(stats_file, 'w') as f:
                f.write(f"# Benchmark Progress Statistics\n\n")
                f.write(f"Started at: {self.start_time.isoformat()}\n\n")
        
        # Append new stats
        with open(stats_file, 'a') as f:
            f.write(stats)

    def add_result(self, result: Dict) -> None:
        if result:
            self.results.append(result)
    
    def calculate_error_rate(self, results: List[Dict]) -> float:
        if not results:
            return 0.0
        correct_count = sum(1 for r in results if any(r.get('is_correct_list', [])))
        return 1.0 - (correct_count / len(results))

    def print_progress(self) -> None:
        if len(self.results) % 100 == 0 and self.results:
            last_hundred = self.results[-100:]
            batch_error_rate = self.calculate_error_rate(last_hundred)
            cumulative_error_rate = self.calculate_error_rate(self.results)
            
            # Calculate success statistics
            level_counts = {i: 0 for i in range(5)}
            total_verifications = 0
            
            for r in last_hundred:
                for is_correct in r.get('is_correct_list', []):
                    if is_correct:
                        level_counts[4] += 1
                    else:
                        level_counts[0] += 1
                    total_verifications += 1
            
            level_ratios = {
                i: (level_counts[i] / total_verifications * 100) if total_verifications > 0 else 0 
                for i in range(5)
            }
            
            stats = f"\nAt {len(self.results)} examples:\n"
            stats += f"Batch Error Rate (last 100): {batch_error_rate:.4f}\n"
            stats += f"Cumulative Error Rate: {cumulative_error_rate:.4f}\n"
            stats += "\nVerification Level Distribution (last 100):\n"
            stats += f"Format Check Failed: {level_ratios[0]:.2f}%\n"
            stats += f"Answer Check Failed: {level_ratios[1]:.2f}%\n"
            stats += f"First Verifier Failed: {level_ratios[2]:.2f}%\n"
            stats += f"Second Verifier Failed: {level_ratios[3]:.2f}%\n"
            stats += f"All Checks Passed: {level_ratios[4]:.2f}%\n"
            stats += "-" * 80 + "\n"
            
            print(stats)
            self._save_progress_stats(stats)

    def save_results(self, model_name: str, split: str) -> None:
        """Save results to a JSON file with timestamp"""
        # Always try to save, even if results list is empty
        try:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate aggregate statistics
        success_rates = []
        # Initialize empty stats collectors
        step_stats = {}
        
        for result in self.results:
            # Track success rate
            correct_count = sum(1 for is_correct in result.get('is_correct_list', []) if is_correct)
            total_attempts = len(result.get('is_correct_list', []))
            if total_attempts > 0:
                success_rates.append(correct_count / total_attempts)
            
            # Collect all available step-related metrics
            for key, value in result.items():
                if any(metric in key.lower() for metric in ['step', 'solution_type']):
                    if key not in step_stats:
                        step_stats[key] = []
                    if isinstance(value, list):
                        step_stats[key].extend(value)
                    else:
                        step_stats[key].append(value)
        
        # Calculate statistics
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Calculate step statistics for each type
        step_statistics = {}
        for step_type, steps in step_stats.items():
            if steps:
                step_statistics[f'avg_{step_type}'] = sum(steps) / len(steps)
                step_statistics[f'max_{step_type}'] = max(steps)
                step_statistics[f'min_{step_type}'] = min(steps)
        
        # Create unified output structure
        output = {
            'metadata': {
                'model': model_name,
                'split': split,
                'timestamp': timestamp,
                'total_examples': len(self.results),
                'statistics': {
                    'average_success_rate': avg_success_rate,
                    **step_statistics
                }
            },
            'results': self.results
        }
        
        filename = f"benchmark_{model_name}_{timestamp}.json"
        filepath = os.path.join("results", filename)
        print(f"\nAttempting to save results to: {filepath}")
        print(f"Number of results to save: {len(self.results)}")
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results successfully saved to: {filepath}")
        
        # Print statistics
        print("\nBenchmark Statistics:")
        print(f"Average Success Rate: {avg_success_rate:.2f}")
        
        # Print step statistics for each type
        for step_type, steps in step_stats.items():
            if steps:
                avg = sum(steps) / len(steps)
                max_val = max(steps)
                print(f"Average {step_type}: {avg:.2f}")
                print(f"Maximum {step_type}: {max_val}")

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
