import os
import asyncio
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
from utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification"""
    logs = []
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            logs.append(f"Error processing example {str(running_id)}: Invalid example format")
            print("\n".join(logs))
            return None
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            logs.append(f"Warning: Could not extract answer from solution for example {str(running_id)}")
            print("\n".join(logs))
            return None

        main = get_model(config, role="main")
        solution_agent = FullSolutionAgent(main)
        solutions = []
        correct_count = 0
        best_solution = None
        
        for attempt in range(config.best_of):
            try:
                current_solution = await solution_agent.generate(example["problem"])
                
                # Create numeric verifier
                verifier = NumericVerifier(tolerance=config.tolerance)
                is_correct, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer,
                    example["problem"]
                )
                # Always append the solution, regardless of correctness
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'is_correct': is_correct
                })
                
                # Update statistics if correct
                if is_correct:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
            except Exception as e:
                logs.append(f"Error in attempt {str(attempt + 1)} for example {str(running_id)}:")
                logs.append(f"Exception type: {type(e).__name__}")
                logs.append(f"Exception message: {str(e)}")
                import traceback
                logs.append(f"Traceback:\n{traceback.format_exc()}")
                
                # Retry this attempt up to 3 times
                for retry in range(3):
                    try:
                        logs.append(f"Retrying attempt {attempt + 1} (retry {retry + 1}/3)...")
                        current_solution = await solution_agent.generate(example["problem"])
                        
                        # Create numeric verifier
                        verifier = NumericVerifier(tolerance=config.tolerance)
                        is_correct, current_answer = await verifier.verify(
                            current_solution,
                            correct_answer,
                            example["problem"]
                        )
                        
                        if is_correct:
                            correct_count += 1
                            if best_solution is None:
                                best_solution = current_solution
                                
                        solutions.append({
                            'solution': current_solution,
                            'answer': current_answer,
                            'is_correct': is_correct
                        })
                        break  # Success, exit retry loop
                        
                    except Exception as retry_e:
                        logs.append(f"Retry {retry + 1} failed: {str(retry_e)}")
                        if retry == 2:  # Last retry failed
                            solution_info = {
                                'solution': f"Error occurred after 3 retries: {type(e).__name__} - {str(e)}",
                                'answer': None,
                                'is_correct': False
                            }
                            solutions.append(solution_info)
                continue  # Move to next attempt
        
        
        # Calculate most common answer statistics
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        most_common_answer = None
        is_most_common_correct = False
        if model_answers:
            from collections import Counter
            most_common_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_most_common_correct = any(str(s['answer']) == most_common_answer and s['is_correct'] for s in solutions)

        # Add statistics to logs
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n✓ Expected Answer: {correct_answer}")
        logs.append(f"\n📊 Statistics:")
        logs.append(f"├─ Model answers: {[s['answer'] for s in solutions]}")
        logs.append(f"├─ Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        logs.append(f"├─ Correct solutions: {correct_count}/{config.best_of}")
        logs.append(f"├─ Success rate: {(correct_count/config.best_of)*100:.1f}%")
        logs.append(f"├─ Most common answer: {most_common_answer}")
        logs.append(f"└─ Most common answer correct? {'Yes' if is_most_common_correct else 'No'}")
        logs.append("="*80)
        
        # Print all logs
        print("\n".join(logs))
        
        return [
            {
                'id': example_id,
                'data_type': 'training',
                'problem': example['problem'],
                'correct_solution': example['solution'],
                'correct_answer': correct_answer,
                'model_solutions': [s['solution'] for s in solutions],
                'model_answers': [s['answer'] for s in solutions],
            },
            {
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': True,
                'is_correct_list': [s['is_correct'] for s in solutions],
                'is_most_common_correct': is_most_common_correct,
                'success_rate': (correct_count/config.best_of)*100
            }
        ]
        
    except Exception as e:
        print(f"Error processing example {str(running_id)}: {e}")
        return None


async def main():
    """Main function for benchmarking mathematical problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
