import os
import json
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from datasets import Dataset

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
    config: Any
    results: List[Dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    accumulated_stats: Dict = field(default_factory=dict)
    success_rate_history: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def _save_progress_stats(self, stats: str) -> None:
        """Save progress statistics to a log file"""
        if not self.config.produce_statistics:
            return
            
        os.makedirs("results", exist_ok=True)
        stats_file = os.path.join("results", f"progress_stats_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
        with open(stats_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {stats}\n")

    def add_result(self, results: List[Dict]) -> None:
        """Add a list of results to the tracker"""
        if results:
            self.results.extend(results)
    
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

    def print_progress(self) -> None:
        if len(self.results) % self.config.stats_update_freq == 0 and self.results:
            last_batch = self.results[-self.config.stats_update_freq:]
            
            # Calculate batch statistics
            total_examples = len(last_batch)
            batch_stats = self.calculate_score_stats(last_batch)
            
            # Calculate accumulated statistics
            accumulated_stats = self.calculate_score_stats(self.results)
            
            # Build statistics string
            stats_str = f"N={len(self.results)} "
            stats_str += "\nBatch Statistics (last {total_examples}):\n"
            
            # For benchmark.py style results
            if self._has_field(last_batch, 'is_correct_list'):
                # Count problems with at least one correct solution
                at_least_one = sum(1 for r in last_batch if any(r.get('is_correct_list', [])))
                
                # Calculate average correct solutions per problem
                total_correct = sum(sum(r.get('is_correct_list', [])) for r in last_batch)
                avg_correct = total_correct / total_examples if total_examples > 0 else 0
                
                # Debug prints
                print(f"\nDebug - Interim Statistics:")
                print(f"Number of problems in batch: {total_examples}")
                print(f"Total correct solutions in batch: {total_correct}")
                print(f"Average correct solutions: {avg_correct:.3f}")
                # Debug prints for interim statistics
                print("\nDebug - Checking is_correct_list contents:")
                for idx, r in enumerate(last_batch):
                    print(f"Result {idx}: is_correct_list = {r.get('is_correct_list')}")
                
                print("Success rates per problem:", [
                    sum(r.get('is_correct_list', [])) / len(r.get('is_correct_list', []))
                    for r in last_batch
                ])
                
                # Count problems with success rate above 50%
                above_avg = sum(1 for r in last_batch if sum(r.get('is_correct_list', [])) / len(r.get('is_correct_list', [])) > 0.5)
                
                # Count problems where most common answer is correct
                most_common_correct = 0
                for r in last_batch:
                    if not r.get('model_answers'):
                        continue
                    # Get most common answer
                    answers = [str(ans) for ans in r.get('model_answers', []) if ans is not None]
                    if not answers:
                        continue
                    from collections import Counter
                    most_common = Counter(answers).most_common(1)[0][0]
                    # Check if most common answer is in list of correct answers
                    if any(r.get('is_correct_list', [])[i] for i, ans in enumerate(r.get('model_answers', [])) 
                          if ans is not None and str(ans) == most_common):
                        most_common_correct += 1
                
                # Batch statistics
                stats_str += (
                    f"\nBatch Statistics (last {total_examples}):\n"
                    f"- Problems with at least one correct solution: {at_least_one}/{total_examples} ({at_least_one/total_examples*100:.1f}%)\n"
                    f"- Average correct solutions per problem: {avg_correct:.2f}\n"
                    f"- Problems with above average correct solutions: {above_avg}/{total_examples} ({above_avg/total_examples*100:.1f}%)\n"
                    f"- Problems where most common answer is correct: {most_common_correct}/{total_examples} ({most_common_correct/total_examples*100:.1f}%)\n"
                )

                # Accumulated statistics
                total_acc = len(self.results)
                at_least_one_acc = sum(1 for r in self.results if any(r.get('is_correct_list', [])))
                avg_correct_acc = sum(sum(r.get('is_correct_list', [])) for r in self.results) / total_acc
                above_avg_acc = sum(1 for r in self.results if sum(r.get('is_correct_list', [])) / len(r.get('is_correct_list', [])) > 0.5)
                
                stats_str += (
                    f"\nAccumulated Statistics (N={total_acc}):\n"
                    f"- Problems with at least one correct solution: {at_least_one_acc}/{total_acc} ({at_least_one_acc/total_acc*100:.1f}%)\n"
                    f"- Average correct solutions per problem: {avg_correct_acc:.2f}\n"
                    f"- Problems with above average correct solutions: {above_avg_acc}/{total_acc} ({above_avg_acc/total_acc*100:.1f}%)\n"
                    f"- Runtime so far: {(datetime.now() - self.start_time).total_seconds():.1f}s"
                )
            
            # For data_creator.py style results
            if any(key in batch_stats for key in ['avg_chosen', 'avg_rejected', 'avg_diff']):
                # Batch statistics
                stats_str += (
                    f"\nBatch Statistics (last {total_examples}):\n"
                    f"- Average score for chosen solutions: {batch_stats.get('avg_chosen', 0):.2f}\n"
                    f"- Average score for rejected solutions: {batch_stats.get('avg_rejected', 0):.2f}\n"
                    f"- Average score difference: {batch_stats.get('avg_diff', 0):.2f}\n"
                )
                
                # Accumulated statistics
                stats_str += (
                    f"\nAccumulated Statistics (N={len(self.results)}):\n"
                    f"- Average score for chosen solutions: {accumulated_stats.get('avg_chosen', 0):.2f}\n"
                    f"- Average score for rejected solutions: {accumulated_stats.get('avg_rejected', 0):.2f}\n"
                    f"- Average score difference: {accumulated_stats.get('avg_diff', 0):.2f}\n"
                )
                stats_str += f"- Runtime so far: {(datetime.now() - self.start_time).total_seconds():.1f}s"
            
            print(stats_str)
            self._save_progress_stats(stats_str)
            
            # Automatically save results every 100 examples
            self.save_results()

    def save_results(self) -> None:
        """Save results to a JSON file"""
        if not self.results or not self.config.produce_statistics:
            return
            
        try:
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

    def create_hf_dataset(self) -> None:
        """Create a HuggingFace dataset from the results"""
        if not self.results or not self.config.create_dataset:
            return
            
        # Create timestamp-based directory
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        dataset_dir = os.path.join("local_datasets", timestamp)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Convert results to HuggingFace dataset
        dataset = Dataset.from_list(self.results)
        
        # Save locally in Arrow format
        dataset.save_to_disk(dataset_dir)
        print(f"\nDataset saved to: {dataset_dir}")

    def print_final_stats(self) -> None:
        if not self.results:
            msg = "\nNo examples were successfully processed."
            print(msg)
            self._save_progress_stats(msg + "\n")
            return
            
        # Save final results first
        self.save_results()
            
        # Create dataset if requested
        self.create_hf_dataset()

        total = len(self.results)
        
        # Calculate statistics
        stats = self.calculate_score_stats(self.results)
        
        # Initialize statistics tracking
        stats = self.calculate_score_stats(self.results)
        
        # Initialize bifurcation tracking
        bifurcation_stats = {
            'counts': {},
            'total': 0,
            'valid_points': 0,
            'average': 0
        }
        
        # Count valid bifurcation points
        for r in self.results:
            if r and isinstance(r, dict) and 'bifurcation_point' in r:
                point = r['bifurcation_point']
                if isinstance(point, (int, float)):
                    bifurcation_stats['counts'][point] = bifurcation_stats['counts'].get(point, 0) + 1
                    bifurcation_stats['total'] += point
                    bifurcation_stats['valid_points'] += 1
        
        if bifurcation_stats['valid_points'] > 0:
            bifurcation_stats['average'] = bifurcation_stats['total'] / bifurcation_stats['valid_points']

        end_time = datetime.now()
        total_duration = end_time - self.start_time

        stats_str = f"FINAL: N={total} "
        
        # For benchmark.py style results
        if self._has_field(self.results, 'is_correct_list'):
            # Count problems with at least one correct solution
            at_least_one = sum(1 for r in self.results if any(r.get('is_correct_list', [])))
            
            # Calculate average correct solutions per problem
            total_correct = sum(sum(r.get('is_correct_list', [])) for r in self.results)
            avg_correct = total_correct / total if total > 0 else 0
            
            # Debug prints
            print(f"\nDebug - Final Statistics:")
            print(f"Total number of problems: {total}")
            print(f"Total correct solutions: {total_correct}")
            print(f"Average correct solutions per problem: {avg_correct:.3f}")
            # Debug prints for final statistics
            print("\nDebug - Checking final is_correct_list contents:")
            for idx, r in enumerate(self.results):
                print(f"Result {idx}: is_correct_list = {r.get('is_correct_list')}")
            
            print("Success rates per problem:", [
                sum(r.get('is_correct_list', [])) / len(r.get('is_correct_list', []))
                for r in self.results
            ])
            
            # Count problems with success rate above 50%
            above_avg = sum(1 for r in self.results 
                if r.get('is_correct_list') and 
                (sum(r.get('is_correct_list', [])) / len(r.get('is_correct_list', [])) > 0.5))
            
            # Count problems where most common answer is correct
            most_common_correct = 0
            tournament_winners_correct = 0
            total_with_tournament = 0
            total_judge_decisions = 0
            total_judge_successes = 0
            total_judge_failsafes = 0
            
            for r in self.results:
                if not r.get('model_answers'):
                    continue
                # Get most common answer
                answers = [str(ans) for ans in r.get('model_answers', []) if ans is not None]
                if not answers:
                    continue
                from collections import Counter
                most_common = Counter(answers).most_common(1)[0][0]
                # Check if most common answer is in list of correct answers
                if any(r.get('is_correct_list', [])[i] for i, ans in enumerate(r.get('model_answers', []))
                      if ans is not None and str(ans) == most_common):
                    most_common_correct += 1
                
                # Tournament statistics
                if 'tournament_winner_correct' in r:
                    total_with_tournament += 1
                    if r['tournament_winner_correct']:
                        tournament_winners_correct += 1
                if 'judge_success_rate' in r:
                    total_judge_decisions += 1
                    total_judge_successes += r['judge_success_rate']
                if 'judge_failsafe_rate' in r:
                    total_judge_failsafes += r['judge_failsafe_rate']
            
            # Calculate averages for judge statistics
            avg_judge_success = total_judge_successes / total_judge_decisions if total_judge_decisions > 0 else 0
            avg_judge_failsafe = total_judge_failsafes / total_judge_decisions if total_judge_decisions > 0 else 0
            
            # Calculate failsafe statistics
            problems_with_failsafe = sum(1 for r in self.results if r.get('judge_failsafe_rate', 0) > 0)
            
            stats_str += (
                f"\nBenchmark Statistics:\n"
                f"- Problems with at least one correct solution: {at_least_one}/{total} ({at_least_one/total*100:.1f}%)\n"
                f"- Average correct solutions per problem: {avg_correct:.2f}\n"
                f"- Problems with above average correct solutions: {above_avg}/{total} ({above_avg/total*100:.1f}%)\n"
                f"- Problems where most common answer is correct: {most_common_correct}/{total} ({most_common_correct/total*100:.1f}%)\n"
                f"- Tournament winners correct: {tournament_winners_correct}/{total_with_tournament} ({tournament_winners_correct/total_with_tournament*100:.1f}%)\n"
                f"- Average judge success rate: {avg_judge_success:.1f}%\n"
                f"- Average judge failsafe rate: {avg_judge_failsafe:.1f}%\n"
                f"- Problems requiring failsafe: {problems_with_failsafe}/{total} ({problems_with_failsafe/total*100:.1f}%)\n"
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
            stats_str += f"- Total runtime: {total_duration.total_seconds():.1f}s"

        # Add judge accuracy statistics if available
        if self._has_field(self.results, 'judge_was_correct'):
            judge_predictions = [r['judge_was_correct'] for r in self.results if 'judge_was_correct' in r]
            correct_predictions = sum(1 for x in judge_predictions if x)
            total_predictions = len(judge_predictions)
            
            if total_predictions > 0:
                stats_str += (
                    f"\nJudge Performance:\n"
                    f"- Correct predictions: {correct_predictions}/{total_predictions} "
                    f"({correct_predictions/total_predictions*100:.1f}%)\n"
                )

        print(stats_str)
        self._save_progress_stats(stats_str)
