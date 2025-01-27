import os
import json
import time
import shutil
import asyncio
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

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
        """Add a list of results to the tracker and update progress"""
        if results:
            self.results.extend(results)
            # Force immediate save at checkpoints
            if len(self.results) % self.config.stats_update_freq == 0:
                self.print_progress()
                self._save_progress_stats(f"Checkpoint at {len(self.results)} examples")
                self.save_results()
    
    def _has_field(self, results: List[Dict], field: str) -> bool:
        """Check if field exists in any result"""
        return any(field in r for r in results)


    def print_progress(self) -> None:
        # Only proceed if we have results
        if not self.results:
            return
            
        last_batch = self.results[-self.config.stats_update_freq:]
        
        # Calculate batch statistics with null safety
        # Only count examples that have benchmark statistics
        benchmark_examples = [r for r in last_batch if 'is_correct_list' in r]
        total_examples = len(benchmark_examples)
        if total_examples == 0:
            # If no benchmark examples in this batch, just show total count
            stats_str = f"N={len(self.results)} (processing tournament results...)\n"
            print(stats_str)
            return
                
        # Build statistics string with safe access
        stats_str = f"N={len(self.results)} "
            
        # Track judge statistics if present
        judge_decisions = sum(1 for r in last_batch if 'judge_decisions' in r and r['judge_decisions'] > 0)
        judge_accuracy = 0
        if judge_decisions > 0:
            judge_accuracy = sum(r.get('judge_accuracy', 0) for r in last_batch if 'judge_decisions' in r and r.get('judge_accuracy') is not None) / judge_decisions if judge_decisions > 0 else 0
            
        # Basic statistics for all benchmark types
        if self._has_field(last_batch, 'is_correct_list'):
            # Only process benchmark examples (not tournament results)
            # Count problems with at least one correct solution - safely handle None and empty lists
            at_least_one = sum(1 for r in benchmark_examples if any(r.get('is_correct_list') or []))
            
            # Calculate average correct solutions per problem - safely handle None and empty lists
            total_correct = sum(sum(r.get('is_correct_list') or []) for r in benchmark_examples)
            avg_correct = total_correct / total_examples if total_examples > 0 else 0
            
            # Count problems with success rate above 50% - safely handle None and empty lists
            above_avg = sum(1 for r in benchmark_examples 
                          if r.get('is_correct_list') 
                          and len(r.get('is_correct_list', [])) > 0 
                          and (sum(r.get('is_correct_list', [])) / len(r.get('is_correct_list', [])) > 0.5))
            
            # Count problems where most common answer is correct - with additional null safety
            most_common_correct = 0
            for r in benchmark_examples:
                try:
                    if not r.get('model_answers'):
                        continue
                    # Get most common answer - safely handle None values
                    answers = [str(ans) for ans in r.get('model_answers', []) if ans is not None]
                    if not answers:
                        continue
                    from collections import Counter
                    most_common = Counter(answers).most_common(1)[0][0]
                    # Check if most common answer is in list of correct answers - safely handle None and index errors
                    is_correct_list = r.get('is_correct_list') or []
                    model_answers = r.get('model_answers') or []
                    if any(i < len(is_correct_list) and is_correct_list[i] 
                          for i, ans in enumerate(model_answers) 
                          if ans is not None and str(ans) == most_common):
                        most_common_correct += 1
                except (IndexError, KeyError, TypeError):
                    continue
            
            # Count tournament winners if present - with null safety
            tournament_winners = 0
            total_with_tournament = 0
            if self._has_field(last_batch, 'tournament_winner_correct'):
                total_with_tournament = sum(1 for r in last_batch if r.get('tournament_winner_correct') is not None)
                tournament_winners = sum(1 for r in last_batch if r.get('tournament_winner_correct', False))

            # Batch statistics
            stats_str += (
                f"\nBatch Statistics (last {total_examples}):\n"
                f"- Problems with at least one correct solution: {at_least_one}/{total_examples} "
                f"({at_least_one/total_examples*100:.1f}%)\n"
                f"- Average correct solutions per problem: {avg_correct:.2f}\n"
                f"- Problems with above average correct solutions: {above_avg}/{total_examples} "
                f"({above_avg/total_examples*100:.1f}%)\n"
                f"- Problems where most common answer is correct: {most_common_correct}/{total_examples} "
                f"({most_common_correct/total_examples*100:.1f}%)\n"
            )
            
            # Add tournament and judge statistics if present
            if total_with_tournament > 0:
                stats_str += (
                    f"- Tournament winners correct: {tournament_winners}/{total_with_tournament} "
                    f"({tournament_winners/total_with_tournament*100:.1f}%)\n"
                )
            if judge_decisions > 0:
                stats_str += (
                    f"- Judge decisions made: {judge_decisions}\n"
                    f"- Judge accuracy: {judge_accuracy:.1f}%\n"
                )

            # Accumulated statistics
            total_acc = len(self.results)
            at_least_one_acc = sum(1 for r in self.results if any(r.get('is_correct_list', [])))
            avg_correct_acc = sum(sum(r.get('is_correct_list', [])) for r in self.results) / total_acc
            above_avg_acc = sum(1 for r in self.results if sum(r.get('is_correct_list', [])) / len(r.get('is_correct_list', [])) > 0.5)
            most_common_correct_acc = sum(1 for r in self.results if r.get('is_most_common_correct', False))

            stats_str += (
                f"\nAccumulated Statistics (N={total_acc}):\n"
                f"- Problems with at least one correct solution: {at_least_one_acc}/{total_acc} "
                f"({at_least_one_acc/total_acc*100:.1f}%)\n"
                f"- Average correct solutions per problem: {avg_correct_acc:.2f}\n"
                f"- Problems with above average correct solutions: {above_avg_acc}/{total_acc} "
                f"({above_avg_acc/total_acc*100:.1f}%)\n"
                f"- Problems where most common answer is correct: {most_common_correct_acc}/{total_acc} "
                f"({most_common_correct_acc/total_acc*100:.1f}%)\n"
            )

            if total_with_tournament > 0:
                acc_tournament_winners = sum(1 for r in self.results if r.get('tournament_winner_correct', False))
                acc_total_tournaments = sum(1 for r in self.results if 'tournament_winner_correct' in r)
                stats_str += (
                    f"- Tournament winners correct: {acc_tournament_winners}/{acc_total_tournaments} "
                    f"({acc_tournament_winners/acc_total_tournaments*100:.1f}%)\n"
                )

        
        
        # Always print and save results, regardless of which style they are
        print(stats_str)
        self._save_progress_stats(stats_str)
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
            
    async def run_benchmark(
        self,
        process_example_func: Callable
    ) -> None:
        """Generic benchmark runner that handles dataset loading and example processing"""
        if self.config.max_concurrent < 1:
            print("Error: Maximum concurrent problems must be at least 1")
            return

        # Load exclude list if provided
        excluded_problems = set()
        if self.config.exclude and os.path.exists(self.config.exclude):
            try:
                with open(self.config.exclude, 'r') as f:
                    exclude_data = json.load(f)
                    excluded_problems = {item['problem'] for item in exclude_data if 'problem' in item}
                print(f"Loaded {len(excluded_problems)} problems to exclude")
            except Exception as e:
                print(f"Error loading exclude file: {e}")
                return

        try:
            # Create a unique cache directory using timestamp
            timestamp = int(time.time())
            cache_dir = os.path.join("cache", f"huggingface_{timestamp}")
            os.makedirs(cache_dir, exist_ok=True)

            def load_dataset_with_retry(max_retries=3, cleanup_on_fail=True):
                for attempt in range(max_retries):
                    try:
                        if os.path.exists(self.config.dataset):  # Local path
                            dataset = load_from_disk(self.config.dataset)
                            if self.config.split and hasattr(dataset, self.config.split):
                                dataset = dataset[self.config.split]
                        else:  # HuggingFace dataset
                            if self.config.dataset == 'Metaskepsis/Numina':
                                dataset = load_dataset(
                                    "Metaskepsis/Numina", 
                                    split=self.config.split,
                                    cache_dir=cache_dir,
                                    download_mode="force_redownload" if attempt > 0 else "reuse_cache_if_exists"
                                )
                            else:
                                dataset = load_dataset(
                                    self.config.dataset,
                                    split=self.config.split,
                                    cache_dir=cache_dir,
                                    download_mode="force_redownload" if attempt > 0 else "reuse_cache_if_exists"
                                )
                        return dataset
                    except Exception as e:
                        print(f"Dataset loading attempt {attempt + 1} failed: {str(e)}")
                        if cleanup_on_fail and attempt < max_retries - 1:
                            print("Cleaning up cache and retrying...")
                            try:
                                shutil.rmtree(cache_dir)
                                os.makedirs(cache_dir, exist_ok=True)
                            except Exception as cleanup_error:
                                print(f"Cache cleanup failed: {cleanup_error}")
                        time.sleep(2)  # Wait before retry
                        
                raise Exception("Failed to load dataset after all retries")

            try:
                dataset = load_dataset_with_retry()
            except Exception as e:
                print(f"Fatal error loading dataset: {e}")
                return
                
            # First sort by ID to ensure consistent ordering
            dataset = dataset.sort('id')
                
            # Filter out excluded problems
            if excluded_problems:
                dataset = dataset.filter(lambda x: x['problem'] not in excluded_problems)
                print(f"Filtered dataset to exclude {len(excluded_problems)} problems")
                
            # Shuffle dataset with seed if specified
            if self.config.seed is not None:
                dataset = dataset.shuffle(seed=self.config.seed)
                
            if self.config.split_slice:
                dataset = dataset.select(range(*self.config.split_slice.indices(len(dataset))))
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        if self.config.split_slice:
            dataset_length = min(self.config.split_slice.stop, len(dataset))
        else:
            dataset_length = len(dataset)

        self.total_examples = dataset_length

        example_data = []
        for example in dataset:
            processed = {
                'id': example['id'],
                'problem': example['problem'],
                'solution': example['solution']
            }
            example_data.append(processed)

        if not example_data:
            print("No valid examples to process after initial filtering.")
            return

        print(f"\nStarting processing of {self.total_examples} examples...")
        try:
            semaphore = asyncio.Semaphore(self.config.max_concurrent)

            async def process_with_semaphore(example: Dict, running_id: int) -> Optional[Dict]:
                async with semaphore:
                    result = await process_example_func(
                        example=example,
                        running_id=running_id,
                        example_id=example['id'],
                        config=self.config
                    )
                    if result:
                        self.add_result(result)
                    return result

            tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(example_data)]
            
            print(f"\nWill process {len(example_data)} examples")
            
            progress_bar = tqdm(total=len(example_data), desc="Processing examples")
            all_logs = []
            
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result and 'logs' in result:
                        all_logs.append(result['logs'])
                    if result and 'total_solution_attempts' in result:
                        all_logs.append(f"\nTotal solution attempts for example {len(self.results)}: {result['total_solution_attempts']}")
                    progress_bar.update(1)
                except Exception as e:
                    all_logs.append(f"Error processing example: {str(e)}")
            progress_bar.close()
        
        finally:
            # Print all collected logs
            print("\n" + "="*80)
            print("COMPLETE LOG OUTPUT")
            print("="*80)
            for log in all_logs:
                print("\n" + log)
            print("\n" + "="*80)
            
            self.print_final_stats()
            self.save_results()
            
            # Cleanup cache directory at the end
            try:
                shutil.rmtree(cache_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup cache directory: {e}")
        # Save final results first
        self.save_results()
            
        # Create dataset if requested
        self.create_hf_dataset()

        total = len(self.results)
        
    
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
            
            # Calculate accumulated tournament and judge statistics
            acc_tournament_winners = sum(1 for r in self.results if r.get('tournament_winner_correct', False))
            acc_total_tournaments = sum(1 for r in self.results if 'tournament_winner_correct' in r)
            
            stats_str += (
                f"\nBenchmark Statistics:\n"
                f"- Problems with at least one correct solution: {at_least_one}/{total} ({at_least_one/total*100:.1f}%)\n"
                f"- Average correct solutions per problem: {avg_correct:.2f}\n"
                f"- Problems with above average correct solutions: {above_avg}/{total} ({above_avg/total*100:.1f}%)\n"
                f"- Problems where most common answer is correct: {most_common_correct}/{total} ({most_common_correct/total*100:.1f}%)\n"
                + (f"- Tournament winners correct (batch): {tournament_winners_correct}/{total_with_tournament} ({tournament_winners_correct/total_with_tournament*100:.1f}%)\n"
                   f"- Tournament winners correct (accumulated): {acc_tournament_winners}/{acc_total_tournaments} ({acc_tournament_winners/acc_total_tournaments*100:.1f}%)\n"
                   if total_with_tournament > 0 else "") +
                (f"- Judge decisions made: {total_judge_decisions}\n"
                 f"- Overall judge accuracy: {total_judge_successes/total_judge_decisions:.1f}%\n"
                 if total_judge_decisions > 0 else "") +
                f"- Total runtime: {total_duration.total_seconds():.1f}s"
            )
        

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
