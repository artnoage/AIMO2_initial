import os
import json
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ProgressTracker:
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
        correct_count = sum(1 for r in results if any(r['is_correct_list']))
        return 1.0 - (correct_count / len(results))

    def print_progress(self) -> None:
        if len(self.results) % 100 == 0 and self.results:
            last_hundred = self.results[-100:]
            batch_error_rate = self.calculate_error_rate(last_hundred)
            cumulative_error_rate = self.calculate_error_rate(self.results)
            
            majority_correct_count = sum(1 for r in last_hundred 
                                       if r['attempts']['correct_count'] > self.best_of // 2)
            majority_correct_rate = majority_correct_count / len(last_hundred)
            
            stats = f"\nAt {len(self.results)} examples:\n"
            stats += f"Batch Error Rate (last 100): {batch_error_rate:.4f}\n"
            stats += f"Cumulative Error Rate: {cumulative_error_rate:.4f}\n"
            stats += f"Batch Majority Correct Rate (last 100): {majority_correct_rate:.4f}\n"
            stats += "-" * 80 + "\n"
            
            print(stats)
            self._save_progress_stats(stats)

    def save_results(self, model_name: str, split: str) -> None:
        """Save results to a JSON file with timestamp"""
        if not self.results:
            print("No results to save")
            return
            
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Extract metadata about solution steps
        step_stats = {
            'total_steps': [],
            'steps_before_completion': [],
            'steps_per_attempt': []
        }
        
        for result in self.results:
            # Get total solution steps if available (test2)
            if 'total_solution_steps' in result:
                step_stats['total_steps'].extend(result['total_solution_steps'])
                
            # Get steps before completion if available (test2)
            if 'steps_before_completion' in result:
                step_stats['steps_before_completion'].extend(result['steps_before_completion'])
                
            # Get steps per attempt if available (test1)
            if 'steps_taken' in result:
                step_stats['steps_per_attempt'].extend(result['steps_taken'])
        
        # Calculate step statistics
        stats = {
            'step_statistics': {
                'average_total_steps': sum(step_stats['total_steps']) / len(step_stats['total_steps']) if step_stats['total_steps'] else None,
                'average_steps_before_completion': sum(step_stats['steps_before_completion']) / len(step_stats['steps_before_completion']) if step_stats['steps_before_completion'] else None,
                'average_steps_per_attempt': sum(step_stats['steps_per_attempt']) / len(step_stats['steps_per_attempt']) if step_stats['steps_per_attempt'] else None,
                'max_steps': max(step_stats['total_steps'] + step_stats['steps_before_completion'] + step_stats['steps_per_attempt'], default=None)
            }
        }
        
        # Add metadata to results
        output = {
            'metadata': {
                'model': model_name,
                'split': split,
                'timestamp': timestamp,
                'total_examples': len(self.results),
                'step_statistics': stats['step_statistics']
            },
            'results': self.results
        }
        
        # Determine benchmark type from results structure
        if any('total_solution_steps' in r for r in self.results):
            benchmark_type = 'test2'
        elif any('steps_taken' in r for r in self.results):
            benchmark_type = 'test1'
        else:
            benchmark_type = 'benchmark'
            
        filename = f"{benchmark_type}_{model_name}_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        with open(os.path.join("results", filename), 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {filename}")
        
        # Print step statistics
        print("\nStep Statistics:")
        for key, value in stats['step_statistics'].items():
            if value is not None:
                print(f"{key}: {value:.2f}")

    def print_final_stats(self) -> None:
        if not self.results:
            msg = "\nNo examples were successfully processed."
            print(msg)
            self._save_progress_stats(msg + "\n")
            return

        total = len(self.results)
        any_correct_count = sum(1 for r in self.results if any(r['is_correct_list']))
        majority_correct_count = sum(1 for r in self.results 
                                   if r['attempts']['correct_count'] > self.best_of // 2)

        any_accuracy = (any_correct_count / total) * 100
        majority_accuracy = (majority_correct_count / total) * 100

        at_least_one_correct = sum(1 for r in self.results if r['attempts']['correct_count'] > 0)
        majority_correct = sum(1 for r in self.results 
                             if r['attempts']['correct_count'] > self.best_of // 2)

        end_time = datetime.now()
        total_duration = end_time - self.start_time

        stats = "\n\n## Final Results\n\n"
        stats += f"- Total examples processed: {total}\n"
        stats += f"- Any-Correct Accuracy: {any_correct_count}/{total} = {any_accuracy:.2f}%\n"
        stats += f"- Majority-Correct Accuracy: {majority_correct_count}/{total} = {majority_accuracy:.2f}%\n\n"

        stats += f"### Best-of-{self.best_of} Statistics\n\n"
        stats += f"- Problems with at least one correct solution: {at_least_one_correct}/{total} = "
        stats += f"{(at_least_one_correct/total)*100:.2f}%\n"
        stats += f"- Problems with majority correct solutions: {majority_correct}/{total} = "
        stats += f"{(majority_correct/total)*100:.2f}%\n\n"

        stats += "### Timing Information\n\n"
        stats += f"- Total execution time: {total_duration}\n"
        stats += f"- Average time per example: {total_duration.total_seconds() / total:.2f} seconds\n"

        print(stats)
        self._save_progress_stats(stats)
