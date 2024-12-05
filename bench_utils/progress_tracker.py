import os
import json
from typing import List, Dict, Any
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
    config: Any
    results: List[Dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    def _save_progress_stats(self, stats: str) -> None:
        """Save progress statistics to a log file"""
        if not self.config.produce_statistics:
            return
            
        os.makedirs("results", exist_ok=True)
        stats_file = os.path.join("results", f"progress_stats_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
        with open(stats_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {stats}\n")

    def add_result(self, result: Dict) -> None:
        if result:
            self.results.append(result)
    
    def _has_field(self, results: List[Dict], field: str) -> bool:
        """Check if field exists in any result"""
        return any(field in r for r in results)

    def calculate_score_stats(self, results: List[Dict]) -> Dict:
        if not results:
            return {}
        
        stats = {}
        total = len(results)
        
        # Only calculate stats if fields exist
        if self._has_field(results, 'score_chosen'):
            stats['avg_chosen'] = sum(r.get('score_chosen', 0) for r in results) / total
        if self._has_field(results, 'score_rejected'):
            stats['avg_rejected'] = sum(r.get('score_rejected', 0) for r in results) / total
        if 'avg_chosen' in stats and 'avg_rejected' in stats:
            stats['avg_diff'] = stats['avg_chosen'] - stats['avg_rejected']
            
        return stats

    def print_progress(self, model_name: str = None, split: str = None) -> None:
        if len(self.results) % self.config.stats_update_freq == 0 and self.results:
            last_batch = self.results[-self.config.stats_update_freq:]
            
            # Calculate statistics
            total_examples = len(last_batch)
            stats = self.calculate_score_stats(last_batch)
            avg_bifurcation = sum(r.get('bifurcation_point', 0) for r in last_batch) / total_examples
            
            # Count bifurcation points
            bifurcation_counts = {}
            for r in last_batch:
                point = r.get('bifurcation_point', 0)
                bifurcation_counts[point] = bifurcation_counts.get(point, 0) + 1
            
            # Build statistics string
            stats_str = f"N={len(self.results)} "
            
            # For benchmark.py style results
            if self._has_field(last_batch, 'is_correct_list'):
                # Count problems with at least one correct solution
                at_least_one = sum(1 for r in last_batch if any(r.get('is_correct_list', [])))
                
                # Calculate average correct solutions per problem
                avg_correct = sum(sum(r.get('is_correct_list', [])) for r in last_batch) / total_examples
                
                # Count problems with above average correct solutions
                above_avg = sum(1 for r in last_batch if sum(r.get('is_correct_list', [])) > avg_correct)
                
                stats_str += (
                    f"\nInterim Benchmark Statistics:\n"
                    f"- Problems with at least one correct solution: {at_least_one}/{total_examples} ({at_least_one/total_examples*100:.1f}%)\n"
                    f"- Average correct solutions per problem: {avg_correct:.2f}\n"
                    f"- Problems with above average correct solutions: {above_avg}/{total_examples} ({above_avg/total_examples*100:.1f}%)\n"
                    f"- Runtime so far: {(datetime.now() - self.start_time).total_seconds():.1f}s"
                )
            
            # For data_creator.py style results
            if any(key in stats for key in ['avg_chosen', 'avg_rejected', 'avg_diff']):
                stats_str += (
                    f"\nInterim Data Creation Statistics:\n"
                    f"- Average score for chosen solutions: {stats.get('avg_chosen', 0):.2f}\n"
                    f"- Average score for rejected solutions: {stats.get('avg_rejected', 0):.2f}\n"
                    f"- Average score difference: {stats.get('avg_diff', 0):.2f}\n"
                )
                if self._has_field(last_batch, 'bifurcation_point'):
                    avg_bifurcation = sum(r.get('bifurcation_point', 0) for r in last_batch) / total_examples
                    bifurcation_counts = {}
                    for r in last_batch:
                        point = r.get('bifurcation_point', 0)
                        bifurcation_counts[point] = bifurcation_counts.get(point, 0) + 1
                    
                    stats_str += (
                        f"- Average bifurcation point: {avg_bifurcation:.2f}\n"
                        f"- Bifurcation point distribution: {dict(sorted(bifurcation_counts.items()))}\n"
                    )
                stats_str += f"- Runtime so far: {(datetime.now() - self.start_time).total_seconds():.1f}s"
            
            print(stats_str)
            self._save_progress_stats(stats_str)
            
            # Automatically save results every 100 examples
            if model_name and split:
                self.save_results(model_name, split)

    def save_results(self, model_name: str = None, split: str = None) -> None:
        """Save results to a JSON file"""
        if not self.results or not self.config.produce_statistics:
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

        stats_str = f"FINAL: N={total} "
        
        # For benchmark.py style results
        if self._has_field(self.results, 'is_correct_list'):
            # Count problems with at least one correct solution
            at_least_one = sum(1 for r in self.results if any(r.get('is_correct_list', [])))
            
            # Calculate average correct solutions per problem
            avg_correct = sum(sum(r.get('is_correct_list', [])) for r in self.results) / total
            
            # Count problems with above average correct solutions
            above_avg = sum(1 for r in self.results if sum(r.get('is_correct_list', [])) > avg_correct)
            
            stats_str += (
                f"\nBenchmark Statistics:\n"
                f"- Problems with at least one correct solution: {at_least_one}/{total} ({at_least_one/total*100:.1f}%)\n"
                f"- Average correct solutions per problem: {avg_correct:.2f}\n"
                f"- Problems with above average correct solutions: {above_avg}/{total} ({above_avg/total*100:.1f}%)\n"
                f"- Total runtime: {total_duration.total_seconds():.1f}s"
            )
        
        # For data_creator.py style results
        if any(key in stats for key in ['avg_chosen', 'avg_rejected', 'avg_diff']):
            stats_str += (
                f"\nFinal Data Creation Statistics:\n"
                f"- Average score for chosen solutions: {stats.get('avg_chosen', 0):.2f}\n"
                f"- Average score for rejected solutions: {stats.get('avg_rejected', 0):.2f}\n"
                f"- Average score difference: {stats.get('avg_diff', 0):.2f}\n"
            )
            if self._has_field(self.results, 'bifurcation_point'):
                avg_bifurcation = sum(r.get('bifurcation_point', 0) for r in self.results) / total
                bifurcation_counts = {}
                for r in self.results:
                    point = r.get('bifurcation_point', 0)
                    bifurcation_counts[point] = bifurcation_counts.get(point, 0) + 1
                
                stats_str += (
                    f"- Average bifurcation point: {avg_bifurcation:.2f}\n"
                    f"- Bifurcation point distribution: {dict(sorted(bifurcation_counts.items()))}\n"
                )
            stats_str += f"- Total runtime: {total_duration.total_seconds():.1f}s"

        print(stats_str)
        self._save_progress_stats(stats_str)
