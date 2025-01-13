import os
import asyncio
from typing import Optional, Dict, Tuple, List
from dotenv import load_dotenv
from utils.benchmark_config import *
from utils.benchmark_utils import *
from utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification and tournament judging"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {str(running_id)}: Invalid example format")
            return None
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {str(running_id)}")
            return None

        main = get_model(config, role="main")
        auxiliary = get_model(config, role="auxiliary")
        solution_agent = FullSolutionAgent(main)
        judge_agent = TournamentJudgeAgent(auxiliary)
        
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
                
                if is_correct:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
                        
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'is_correct': is_correct
                })
            except Exception as e:
                print(f"Error in attempt {str(attempt + 1)} for example {str(running_id)}: {str(e)}")
                solution_info = {
                    'solution': "Error occurred",
                    'answer': None,
                    'verification_score': 0,
                    'verification_steps': 1,
                    'is_correct': False
                }
                solutions.append(solution_info)

        # Tournament judging
        winning_solution = None
        winning_solution_correct = False
        
        if len(solutions) > 1:
            # Create tournament brackets
            tournament_solutions = [s for s in solutions if s['solution'] != "Error occurred"]
            if len(tournament_solutions) > 1:
                while len(tournament_solutions) > 1:
                    next_round = []
                    # Handle cases where number of solutions is odd
                    if len(tournament_solutions) % 2 != 0:
                        # Last solution automatically advances to next round
                        next_round.append(tournament_solutions.pop())
                    
                    # Process pairs
                    while len(tournament_solutions) >= 2:
                        solution_a = tournament_solutions.pop(0)
                        solution_b = tournament_solutions.pop(0)
                        
                        judge_result = await judge_agent.compare_solutions(
                            example["problem"],
                            solution_a['solution'],
                            solution_b['solution']
                        )
                        
                        # Determine winner based on judge's response
                        if "A" in judge_result.upper():
                            next_round.append(solution_a)
                        else:
                            next_round.append(solution_b)
                    
                    tournament_solutions = next_round
                
                winning_solution = tournament_solutions[0]
                winning_solution_correct = winning_solution['is_correct']

        # Calculate most common answer statistics
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        most_common_answer = None
        is_most_common_correct = False
        if model_answers:
            from collections import Counter
            most_common_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_most_common_correct = any(str(s['answer']) == most_common_answer and s['is_correct'] for s in solutions)

        # Print statistics
        print(f"\nExample {str(running_id + 1)}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{config.best_of}")
        print(f"Success rate: {(correct_count/config.best_of)*100:.1f}%")
        print(f"Most common answer: {most_common_answer}")
        print(f"Is most common answer correct? {'Yes' if is_most_common_correct else 'No'}")
        print(f"Tournament winner correct? {'Yes' if winning_solution_correct else 'No'}")
        print("-" * 80)
        
        return [{
            'id': example_id,
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'correct_answer': correct_answer,
            'model_solutions': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'is_most_common_correct': is_most_common_correct,
            'success_rate': (correct_count/config.best_of)*100,
            'tournament_winner_correct': winning_solution_correct
        }]
        
    except Exception as e:
        print(f"Error processing example {str(running_id)}: {e}")
        return None


async def main():
    """Main function for benchmarking mathematical problem solving with tournament judging."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems with tournament judging')
    
    await run_benchmark(
        config=config,
        process_example_func=process_example
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
