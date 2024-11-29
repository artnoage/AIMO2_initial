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
            
            print(f"\nAt {len(self.results)} examples:")
            print(f"Batch Error Rate (last 100): {batch_error_rate:.4f}")
            print(f"Cumulative Error Rate: {cumulative_error_rate:.4f}")
            print(f"Batch Majority Correct Rate (last 100): {majority_correct_rate:.4f}")

    def save_results(self, model_name: str, split: str) -> None:
        """Save results to a JSON file with timestamp"""
        if not self.results:
            print("No results to save")
            return
            
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"bm_numeric_{model_name}_{timestamp}.json"
        
        output = {
            "metadata": {
                "model": model_name,
                "split": split,
                "total_examples": self.total_examples,
                "best_of": self.best_of,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            },
            "results": self.results
        }
        
        os.makedirs("results", exist_ok=True)
        with open(os.path.join("results", filename), 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {filename}")

    def print_final_stats(self) -> None:
        if not self.results:
            print("\nNo examples were successfully processed.")
            return

        total = len(self.results)
        any_correct_count = sum(1 for r in self.results if any(r['is_correct_list']))
        majority_correct_count = sum(1 for r in self.results 
                                   if r['attempts']['correct_count'] > self.best_of // 2)

        any_accuracy = (any_correct_count / total) * 100
        majority_accuracy = (majority_correct_count / total) * 100

        print("\n\nFinal Results:")
        print(f"Total examples processed: {total}")
        print(f"Any-Correct Accuracy: {any_correct_count}/{total} = {any_accuracy:.2f}%")
        print(f"Majority-Correct Accuracy: {majority_correct_count}/{total} = {majority_accuracy:.2f}%")

        at_least_one_correct = sum(1 for r in self.results if r['attempts']['correct_count'] > 0)
        majority_correct = sum(1 for r in self.results 
                             if r['attempts']['correct_count'] > self.best_of // 2)

        print(f"\nBest-of-{self.best_of} Statistics:")
        print(f"Problems with at least one correct solution: {at_least_one_correct}/{total} = "
              f"{(at_least_one_correct/total)*100:.2f}%")
        print(f"Problems with majority correct solutions: {majority_correct}/{total} = "
              f"{(majority_correct/total)*100:.2f}%")

        end_time = datetime.now()
        total_duration = end_time - self.start_time
        print(f"\nTotal execution time: {total_duration}")
        print(f"Average time per example: {total_duration.total_seconds() / total:.2f} seconds")
