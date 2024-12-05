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
        config: BenchmarkConfig instance for accessing settings
    """
    total_examples: int
    best_of: int
    results: List[Dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        from bench_utils.benchmark_config import BenchmarkConfig
        self.config = BenchmarkConfig.from_args('Get defaults')
    
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
    
    def calculate_score_stats(self, results: List[Dict]) -> Dict:
        if not results:
            return {'avg_chosen': 0.0, 'avg_rejected': 0.0, 'avg_diff': 0.0}
        total = len(results)
        avg_chosen = sum(r.get('score_chosen', 0) for r in results) / total
        avg_rejected = sum(r.get('score_rejected', 0) for r in results) / total
        avg_diff = sum(r.get('score_chosen', 0) - r.get('score_rejected', 0) for r in results) / total
        return {
            'avg_chosen': avg_chosen,
            'avg_rejected': avg_rejected,
            'avg_diff': avg_diff
        }

    def print_progress(self, model_name: str = None, split: str = None) -> None:
        if len(self.results) % 100 == 0 and self.results:
            last_hundred = self.results[-100:]
            
            # Calculate statistics
            total_examples = len(last_hundred)
            stats = self.calculate_score_stats(last_hundred)
            avg_bifurcation = sum(r.get('bifurcation_point', 0) for r in last_hundred) / total_examples
            
            # Count bifurcation points
            bifurcation_counts = {}
            for r in last_hundred:
                point = r.get('bifurcation_point', 0)
                bifurcation_counts[point] = bifurcation_counts.get(point, 0) + 1
            
            stats_str = f"\nAt {len(self.results)} examples:\n"
            stats_str += f"Last 100 examples statistics:\n"
            stats_str += f"- Average chosen score: {stats['avg_chosen']:.2f}/20\n"
            stats_str += f"- Average rejected score: {stats['avg_rejected']:.2f}/20\n"
            stats_str += f"- Average score difference: {stats['avg_diff']:.2f}\n"
            stats_str += f"- Average bifurcation point: {avg_bifurcation:.2f}\n"
            stats_str += f"- Completions per path: {self.config.completions}\n"
            stats_str += "\nBifurcation distribution:\n"
            for point, count in sorted(bifurcation_counts.items()):
                percentage = (count / total_examples) * 100
                stats_str += f"- Point {point}: {count} examples ({percentage:.1f}%)\n"
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
            # Get model and split from config if not provided
            from bench_utils.benchmark_config import BenchmarkConfig
            config = BenchmarkConfig.from_args('Get defaults')
            
            # Use a fixed filename based on the start timestamp
            filename = f"benchmark_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("results", filename)
            
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            print(f"\nSaving {len(self.results)} results to: {filepath}")
            
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
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
        
        # Calculate statistics
        stats = self.calculate_score_stats(self.results)
        avg_bifurcation = sum(r.get('bifurcation_point', 0) for r in self.results) / total
        
        # Count bifurcation points
        bifurcation_counts = {}
        for r in self.results:
            point = r.get('bifurcation_point', 0)
            bifurcation_counts[point] = bifurcation_counts.get(point, 0) + 1

        end_time = datetime.now()
        total_duration = end_time - self.start_time

        stats_str = "\n\n## Final Results\n\n"
        stats_str += f"### Dataset Statistics\n\n"
        stats_str += f"- Total examples processed: {total}\n"
        stats_str += f"- Average chosen score: {stats['avg_chosen']:.2f}/20\n"
        stats_str += f"- Average rejected score: {stats['avg_rejected']:.2f}/20\n"
        stats_str += f"- Average score difference: {stats['avg_diff']:.2f}\n"
        stats_str += f"- Average bifurcation point: {avg_bifurcation:.2f}\n"
        stats_str += f"- Completions per path: {self.config.completions}\n\n"

        stats += "### Bifurcation Point Distribution\n\n"
        for point, count in sorted(bifurcation_counts.items()):
            percentage = (count / total) * 100
            stats += f"- Point {point}: {count} examples ({percentage:.1f}%)\n"
        
        stats += "\n### Timing Information\n\n"
        stats += f"- Total execution time: {total_duration}\n"
        stats += f"- Average time per example: {total_duration.total_seconds() / total:.2f} seconds\n"

        print(stats_str)
        self._save_progress_stats(stats_str)
