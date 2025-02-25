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
        config: BenchmarkConfig instance for accessing settings
        results: List of processed results
        start_time: Timestamp when tracking started
    """
    total_examples: int
    config: Any
    results: List[Dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    def _has_field(self, data_list: List[Dict], field_name: str) -> bool:
        """Check if any dictionary in the list contains the specified field"""
        return any(field_name in item for item in data_list)
    
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
            # Count only statistics entries for checkpoints
            stats_count = len([r for r in self.results if r.get('data_type') == 'statistics'])
            if stats_count > 0 and stats_count % self.config.stats_update_freq == 0:
                self.print_progress()
                self._save_progress_stats(f"Checkpoint at {stats_count} examples")
                # Save intermediate results
                self.save_results()
    
    def _calculate_statistics(self, entries: List[Dict]) -> Dict:
        """Calculate statistics from a list of statistics entries"""
        if not entries:
            return {}
            
        total = len(entries)
        stats = {}
        
        # Basic statistics
        stats['total'] = total
        successfully_processed = sum(1 for r in entries if r.get('example_processed_successfully', False))
        stats['successfully_processed'] = successfully_processed
        stats['processing_success_rate'] = (successfully_processed / total * 100) if total > 0 else 0
        
        # Calculate correct verdicts
        at_least_one = 0
        total_correct = 0
        above_avg = 0
        most_common_correct = 0
        
        for r in entries:
            matches = r.get('is_correct_list', [])
            if matches:
                # Check if any verdict matches
                matches_count = sum(1 for match in matches if match)
                if matches_count > 0:
                    at_least_one += 1
                total_correct += matches_count
                if matches_count / len(matches) > 0.5:
                    above_avg += 1
                # Check most common verdict
                if r.get('is_most_common_correct', False):
                    most_common_correct += 1
                
        stats['at_least_one'] = at_least_one
        stats['avg_correct'] = total_correct / total if total > 0 else 0
        stats['above_avg'] = above_avg
        stats['most_common_correct'] = most_common_correct
        
        # Judge statistics
        judge_entries = [r for r in entries if r.get('judge_accuracy') is not None]
        if judge_entries:
            stats['judge_decisions'] = len(judge_entries)
            stats['avg_judge_accuracy'] = sum(r['judge_accuracy'] for r in judge_entries) / len(judge_entries)
        
        # Joined benchmark statistics
        if any('main_model_correct_count' in r for r in entries):
            main_correct = sum(r.get('main_model_correct_count', 0) for r in entries)
            aux_correct = sum(r.get('aux_model_correct_count', 0) for r in entries)
            total_attempts = sum(r.get('total_attempts_per_model', 0) for r in entries)
            
            if total_attempts > 0:
                stats['main_model_success_rate'] = (main_correct / total_attempts) * 100
                stats['aux_model_success_rate'] = (aux_correct / total_attempts) * 100
                stats['main_vs_aux_diff'] = stats['main_model_success_rate'] - stats['aux_model_success_rate']
            
            # Use direct statistics from entries
            stats['both_correct_count'] = sum(r.get('both_correct_count', 0) for r in entries)
            stats['both_wrong_count'] = sum(r.get('both_wrong_count', 0) for r in entries)
            stats['disagreement_count'] = sum(r.get('disagreement_count', 0) for r in entries)
            stats['main_better_when_disagree'] = sum(r.get('main_better_when_disagree', 0) for r in entries)
            stats['aux_better_when_disagree'] = sum(r.get('aux_better_when_disagree', 0) for r in entries)
            
            # Calculate rates - use total_attempts_per_model to get the correct denominator
            total_attempts = sum(r.get('total_attempts_per_model', 0) for r in entries)
            stats['total_attempts'] = total_attempts  # Store for display
            if total_attempts > 0:
                stats['both_correct_rate'] = (stats['both_correct_count'] / total_attempts) * 100
                stats['both_wrong_rate'] = (stats['both_wrong_count'] / total_attempts) * 100
                stats['agreement_rate'] = ((stats['both_correct_count'] + stats['both_wrong_count']) / total_attempts) * 100
            else:
                stats['both_correct_rate'] = 0
                stats['both_wrong_rate'] = 0
                stats['agreement_rate'] = 0
            stats['disagreement_rate'] = (stats['disagreement_count'] / total) * 100 if total > 0 else 0
            
            if stats['disagreement_count'] > 0:
                stats['main_win_rate_when_disagree'] = (stats['main_better_when_disagree'] / stats['disagreement_count']) * 100
                stats['aux_win_rate_when_disagree'] = (stats['aux_better_when_disagree'] / stats['disagreement_count']) * 100
            
            # Track most common answer statistics
            stats['main_most_common_correct_count'] = sum(1 for r in entries if r.get('main_most_common_correct', False))
            stats['aux_most_common_correct_count'] = sum(1 for r in entries if r.get('aux_most_common_correct', False))
            stats['combined_most_common_correct_count'] = sum(1 for r in entries if r.get('combined_most_common_correct', False))
            
            stats['main_most_common_correct_rate'] = (stats['main_most_common_correct_count'] / total) * 100 if total > 0 else 0
            stats['aux_most_common_correct_rate'] = (stats['aux_most_common_correct_count'] / total) * 100 if total > 0 else 0
            stats['combined_most_common_correct_rate'] = (stats['combined_most_common_correct_count'] / total) * 100 if total > 0 else 0
            
        return stats


    def print_progress(self) -> None:
        """Print progress statistics for the last batch"""
        if not self.results:
            return
            
        # Get all statistics entries since last checkpoint
        total_stats = len([r for r in self.results if r.get('data_type') == 'statistics'])
        last_checkpoint = max(0, total_stats - self.config.stats_update_freq)
        stats_entries = [r for r in self.results if r.get('data_type') == 'statistics'][last_checkpoint:total_stats]
        if not stats_entries:
            return
            
        # Calculate statistics
        batch_stats = self._calculate_statistics(stats_entries)
        if not batch_stats:
            return
            
        # Build statistics string
        total_stats = len([r for r in self.results if r.get('data_type') == 'statistics'])
        stats_str = f"N={total_stats}\n\nBatch Statistics (last {self.config.stats_update_freq} examples):\n"
        
        # Basic statistics
        stats_str += (
            f"- Processing success rate: {batch_stats['processing_success_rate']:.1f}%\n"
            f"- Successfully processed examples: {batch_stats['successfully_processed']}/{batch_stats['total']} "
            f"({(batch_stats['successfully_processed']/batch_stats['total']*100):.1f}%)\n"
            f"- Problems with at least one correct solution: {batch_stats['at_least_one']}/{batch_stats['total']} "
            f"({(batch_stats['at_least_one']/batch_stats['total']*100):.1f}%)\n"
            f"- Average correct solutions per problem: {batch_stats['avg_correct']:.2f}\n"
            f"- Problems with above average correct solutions: {batch_stats['above_avg']}/{batch_stats['total']} "
            f"({(batch_stats['above_avg']/batch_stats['total']*100):.1f}%)\n"
            f"- Problems where most common answer is correct: {batch_stats['most_common_correct']}/{batch_stats['total']} "
            f"({(batch_stats['most_common_correct']/batch_stats['total']*100):.1f}%)\n"
        )
        
        # Joined benchmark statistics if present
        if 'main_model_success_rate' in batch_stats:
            stats_str += (
                f"\nModel Comparison:\n"
                f"- Main model success rate: {batch_stats['main_model_success_rate']:.1f}%\n"
                f"- Auxiliary model success rate: {batch_stats['aux_model_success_rate']:.1f}%\n"
                f"- Performance difference (main - aux): {batch_stats['main_vs_aux_diff']:.1f}%\n"
                f"\nModel Agreement:\n"
                f"- Both models correct: {batch_stats['both_correct_count']}/{batch_stats['total_attempts']} "
                f"({batch_stats['both_correct_rate']:.1f}%)\n"
                f"- Both models wrong: {batch_stats['both_wrong_count']}/{batch_stats['total_attempts']} "
                f"({batch_stats['both_wrong_rate']:.1f}%)\n"
                f"- Overall agreement rate: {batch_stats['agreement_rate']:.1f}%\n"
            )
            
            if 'disagreement_count' in batch_stats and batch_stats['disagreement_count'] > 0:
                stats_str += (
                    f"\nDisagreement Analysis:\n"
                    f"- Disagreement count: {batch_stats['disagreement_count']}/{batch_stats['total']} "
                    f"({batch_stats['disagreement_rate']:.1f}%)\n"
                    f"- Main model wins when disagreeing: {batch_stats['main_better_when_disagree']}/{batch_stats['disagreement_count']} "
                    f"({batch_stats['main_win_rate_when_disagree']:.1f}%)\n"
                    f"- Auxiliary model wins when disagreeing: {batch_stats['aux_better_when_disagree']}/{batch_stats['disagreement_count']} "
                    f"({batch_stats['aux_win_rate_when_disagree']:.1f}%)\n"
                )
                
            # Add most common answer statistics
            if 'main_most_common_correct_count' in batch_stats:
                stats_str += (
                    f"\nMost Common Answer Analysis:\n"
                    f"- Main model most common answer correct: {batch_stats['main_most_common_correct_count']}/{batch_stats['total']} "
                    f"({batch_stats['main_most_common_correct_rate']:.1f}%)\n"
                    f"- Auxiliary model most common answer correct: {batch_stats['aux_most_common_correct_count']}/{batch_stats['total']} "
                    f"({batch_stats['aux_most_common_correct_rate']:.1f}%)\n"
                    f"- Combined models most common answer correct: {batch_stats['combined_most_common_correct_count']}/{batch_stats['total']} "
                    f"({batch_stats['combined_most_common_correct_rate']:.1f}%)\n"
                )
        
        # Judge statistics if present
        if 'judge_decisions' in batch_stats:
            stats_str += (
                f"\nJudge Statistics:\n"
                f"- Judge decisions made: {batch_stats['judge_decisions']}\n"
                f"- Judge accuracy: {batch_stats['avg_judge_accuracy']:.1f}%\n"
            )
            
        # Calculate accumulated statistics
        acc_stats = self._calculate_statistics([r for r in self.results if r.get('data_type') == 'statistics'])
        if acc_stats:
            stats_str += f"\nAccumulated Statistics (N={acc_stats['total']}):\n"
            stats_str += (
                f"- Processing success rate: {acc_stats['processing_success_rate']:.1f}%\n"
                f"- Successfully processed examples: {acc_stats['successfully_processed']}/{acc_stats['total']} "
                f"({acc_stats['processing_success_rate']:.1f}%)\n"
                f"- Problems with at least one correct solution: {acc_stats['at_least_one']}/{acc_stats['total']} "
                f"({(acc_stats['at_least_one']/acc_stats['total']*100):.1f}%)\n"
                f"- Average correct solutions per problem: {acc_stats['avg_correct']:.2f}\n"
                f"- Problems with above average correct solutions: {acc_stats['above_avg']}/{acc_stats['total']} "
                f"({(acc_stats['above_avg']/acc_stats['total']*100):.1f}%)\n"
                f"- Problems where most common answer is correct: {acc_stats['most_common_correct']}/{acc_stats['total']} "
                f"({(acc_stats['most_common_correct']/acc_stats['total']*100):.1f}%)\n"
            )
            
            if 'tournament_winners' in acc_stats:
                stats_str += (
                    f"- Tournament winners correct: {acc_stats['tournament_winners']}/{acc_stats['total_tournaments']} "
                    f"({(acc_stats['tournament_winners']/acc_stats['total_tournaments']*100):.1f}%)\n"
                )
                    
            if 'judge_decisions' in acc_stats:
                stats_str += (
                    f"- Judge decisions made: {acc_stats['judge_decisions']}\n"
                    f"- Overall judge accuracy: {acc_stats['avg_judge_accuracy']:.1f}%\n"
                )
        
        print(stats_str)
        self._save_progress_stats(stats_str)
        
        # Create dataset if requested
        self.create_hf_dataset()

    def save_results(self) -> None:
        """Save results to JSON files by data type"""
        if not self.results:
            print("No results to save")
            return
        if not self.config.produce_statistics:
            print("Statistics production disabled")
            return

            
        try:
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            print(f"Total results to process: {len(self.results)}")
            
            # Group results by data type
            results_by_type = defaultdict(list)
            for r in self.results:
                data_type = r.get('data_type')
                if data_type:
                    results_by_type[data_type].append(r)
            
            print(f"Found data types: {list(results_by_type.keys())}")
            
            # Save timestamp for consistent filenames
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
            
            # Only save training data
            if 'training' in results_by_type and results_by_type['training']:
                training_results = results_by_type['training']
                filename = f"training_{timestamp}.json"
                filepath = os.path.join("results", filename)
                print(f"Attempting to save {len(training_results)} training results to: {filepath}")
                with open(filepath, 'w') as f:
                    json.dump(training_results, f, indent=2)
                print(f"Successfully saved {len(training_results)} training results to: {filepath}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            import traceback
            traceback.print_exc()

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

        # Get only statistics entries
        stats_entries = [r for r in self.results if r.get('data_type') == 'statistics']
        if not stats_entries:
            msg = "\nNo statistics entries were found in results."
            print(msg)
            self._save_progress_stats(msg + "\n")
            return

        # Use the common calculation method
        final_stats = self._calculate_statistics(stats_entries)
        total = final_stats['total']
        end_time = datetime.now()
        total_duration = end_time - self.start_time

        stats_str = (
            f"\nFinal Statistics (N={total}):\n"
            f"- Processing success rate: {final_stats['processing_success_rate']:.1f}%\n"
            f"- Successfully processed examples: {final_stats['successfully_processed']}/{total} "
            f"({final_stats['processing_success_rate']:.1f}%)\n"
            f"- Problems with at least one correct solution: {final_stats['at_least_one']}/{total} "
            f"({(final_stats['at_least_one']/total*100) if total > 0 else 0:.1f}%)\n"
            f"- Average correct solutions per problem: {final_stats['avg_correct']:.2f}\n"
            f"- Problems with above average correct solutions: {final_stats['above_avg']}/{total} "
            f"({(final_stats['above_avg']/total*100) if total > 0 else 0:.1f}%)\n"
            f"- Problems where most common answer is correct: {final_stats['most_common_correct']}/{total} "
            f"({(final_stats['most_common_correct']/total*100) if total > 0 else 0:.1f}%)\n"
        )

        # Joined benchmark statistics if present
        if 'main_model_success_rate' in final_stats:
            stats_str += (
                f"\nModel Comparison:\n"
                f"- Main model success rate: {final_stats['main_model_success_rate']:.1f}%\n"
                f"- Auxiliary model success rate: {final_stats['aux_model_success_rate']:.1f}%\n"
                f"- Performance difference (main - aux): {final_stats['main_vs_aux_diff']:.1f}%\n"
                f"\nModel Agreement:\n"
                f"- Both models correct: {final_stats['both_correct_count']}/{final_stats['total_attempts']} "
                f"({final_stats['both_correct_rate']:.1f}%)\n"
                f"- Both models wrong: {final_stats['both_wrong_count']}/{final_stats['total_attempts']} "
                f"({final_stats['both_wrong_rate']:.1f}%)\n"
                f"- Overall agreement rate: {final_stats['agreement_rate']:.1f}%\n"
            )
            
            if 'disagreement_count' in final_stats and final_stats['disagreement_count'] > 0:
                stats_str += (
                    f"\nDisagreement Analysis:\n"
                    f"- Disagreement count: {final_stats['disagreement_count']}/{total} "
                    f"({final_stats['disagreement_rate']:.1f}%)\n"
                    f"- Main model wins when disagreeing: {final_stats['main_better_when_disagree']}/{final_stats['disagreement_count']} "
                    f"({final_stats['main_win_rate_when_disagree']:.1f}%)\n"
                    f"- Auxiliary model wins when disagreeing: {final_stats['aux_better_when_disagree']}/{final_stats['disagreement_count']} "
                    f"({final_stats['aux_win_rate_when_disagree']:.1f}%)\n"
                )
                
            # Add most common answer statistics to final stats
            if 'main_most_common_correct_count' in final_stats:
                stats_str += (
                    f"\nMost Common Answer Analysis:\n"
                    f"- Main model most common answer correct: {final_stats['main_most_common_correct_count']}/{total} "
                    f"({final_stats['main_most_common_correct_rate']:.1f}%)\n"
                    f"- Auxiliary model most common answer correct: {final_stats['aux_most_common_correct_count']}/{total} "
                    f"({final_stats['aux_most_common_correct_rate']:.1f}%)\n"
                    f"- Combined models most common answer correct: {final_stats['combined_most_common_correct_count']}/{total} "
                    f"({final_stats['combined_most_common_correct_rate']:.1f}%)\n"
                )

        # Judge statistics if present
        if 'judge_decisions' in final_stats:
            stats_str += (
                f"\nJudge Statistics:\n"
                f"- Judge decisions made: {final_stats['judge_decisions']}\n"
                f"- Overall judge accuracy: {final_stats['avg_judge_accuracy']:.1f}%\n"
            )

        stats_str += f"\n- Total runtime: {total_duration.total_seconds():.1f}s"

        print(stats_str)
        self._save_progress_stats(stats_str)
    async def run_benchmark(
        self,
        process_example_func: Callable
    ) -> None:
        # Set up signal handlers
        import signal
        
        def signal_handler(signum, frame):
            print("\nReceived interrupt signal. Saving current results...")
            # Force save by temporarily setting produce_statistics to True
            original_setting = self.config.produce_statistics
            self.config.produce_statistics = True
            self.save_results()
            self.config.produce_statistics = original_setting
            self.print_final_stats()
            print("\nResults saved. Exiting...")
            exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
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
                            full_dataset = load_from_disk(self.config.dataset)
                            
                            # Handle slicing for local dataset
                            dataset = full_dataset
                            if self.config.split:
                                if '[' in self.config.split:
                                    # Extract slice indices
                                    base_split, slice_part = self.config.split.split('[')
                                    slice_part = slice_part.rstrip(']')
                                    if ':' in slice_part:
                                        start, end = map(lambda x: int(x) if x else None, slice_part.split(':'))
                                        # Apply slice
                                        dataset = dataset.select(range(start if start else 0, end if end else len(dataset)))
                            
                        else:  # HuggingFace dataset
                            # Handle split and slice
                            split_name = self.config.split or 'train'
                            if '[' in split_name:
                                # Extract slice indices
                                base_split, slice_part = split_name.split('[')
                                slice_part = slice_part.rstrip(']')
                                if ':' in slice_part:
                                    start, end = map(lambda x: int(x) if x else None, slice_part.split(':'))
                                    # Load full split then slice
                                    if self.config.dataset == 'Metaskepsis/Numina':
                                        dataset = load_dataset(
                                            "Metaskepsis/Numina",
                                            split=base_split,
                                            cache_dir=cache_dir,
                                            download_mode="force_redownload" if attempt > 0 else "reuse_cache_if_exists"
                                        )
                                    else:
                                        dataset = load_dataset(
                                            self.config.dataset,
                                            split=base_split,
                                            cache_dir=cache_dir,
                                            download_mode="force_redownload" if attempt > 0 else "reuse_cache_if_exists"
                                        )
                                    # Apply slice
                                    dataset = dataset.select(range(start if start else 0, end if end else len(dataset)))
                            else:
                                # No slice, load normally
                                if self.config.dataset == 'Metaskepsis/Numina':
                                    dataset = load_dataset(
                                        "Metaskepsis/Numina",
                                        split=split_name,
                                        cache_dir=cache_dir,
                                        download_mode="force_redownload" if attempt > 0 else "reuse_cache_if_exists"
                                    )
                                else:
                                    dataset = load_dataset(
                                        self.config.dataset,
                                        split=split_name,
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
                
                # Handle DatasetDict
                if isinstance(dataset, dict) and 'train' in dataset:
                    dataset = dataset['train']
                elif hasattr(dataset, 'train'):  # DatasetDict object
                    dataset = dataset['train']
                
                # Now that we have a Dataset object, process features
                if hasattr(dataset, 'features'):
                    # Add auto-incrementing ID if it doesn't exist
                    if 'id' not in dataset.features:
                        dataset = dataset.map(lambda x, idx: {'id': idx}, with_indices=True)
                    
                    # Convert 'question' to 'problem' if needed
                    if 'question' in dataset.features and 'problem' not in dataset.features:
                        dataset = dataset.map(lambda x: {'problem': x['question'], **{k:v for k,v in x.items() if k != 'question'}})
                    
                    # Create solution from answer if needed
                    if 'answer' in dataset.features and 'solution' not in dataset.features:
                        dataset = dataset.map(lambda x: {'solution': f"\\boxed{{{x['answer']}}}", **x})
                else:
                    print("Warning: Dataset does not have features attribute")
                    
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
            # Preserve all fields from the original dataset
            processed = {key: example[key] for key in example.keys()}
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
            
            # Force final save of results
            original_setting = self.config.produce_statistics
            self.config.produce_statistics = True
            self.save_results()
            self.config.produce_statistics = original_setting
            
            self.print_final_stats()
            
            # Cleanup cache directory at the end
            try:
                shutil.rmtree(cache_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup cache directory: {e}")
