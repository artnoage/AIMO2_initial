import os
import asyncio
import logging
from typing import Optional, Dict
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
from utils.agents import *
from utils.tournament_utils import Tournament
from utils.logger import BenchmarkLogger
from collections import Counter
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification and tournament judging"""
    logger = BenchmarkLogger()
    try:
        if not isinstance(example, dict) or 'problem' not in example or (('solution' not in example) and ('answer' not in example)):
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            logger.append(f"❌ Warning: Could not extract answer from solution for example {str(running_id)}")
            logger.print()
            return None

        main = get_model(config, role="main")
        
        # Initialize solution and judge agents
        solution_agent = FullSolutionAgent(main)
        
        # Use auxiliary2 for judging if specified, otherwise use auxiliary with temp=0
        judge_model = get_model(config, role="auxiliary2")
            
        judge_agent = TournamentJudgeAgent(judge_model)
        
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
                logger.append(f"❌ Error in attempt {str(attempt + 1)} for example {str(running_id)}: {str(e)}")
                solution_info = {
                    'solution': "Error occurred",
                    'answer': None,
                    'is_correct': False
                }
                solutions.append(solution_info)

        # Run tournament with logger
        tournament = Tournament(judge_agent, logger=logger)
        tournament_solutions = [(s['solution'], s['is_correct'], '') for s in solutions if s['solution'] != "Error occurred"]
        
        # Get solution content for tournament
        def get_solution_content(solution_tuple):
            return solution_tuple[0]
            
        # Run tournament if we have enough solutions
        tournament_stats = {}
        tournament_results = []
        if len(tournament_solutions) > 1:
            _, tournament_results, tournament_stats = await tournament.run_tournament(
                tournament_solutions,
                example["problem"],
                correct_answer,
                get_content=get_solution_content
            )
            
        winning_solution_correct = tournament_stats.get('solution_ranking', [False])[0] if tournament_stats else False
        judge_accuracy = tournament_stats.get('judge_accuracy', 0) * 100 if tournament_stats and tournament_stats.get('judge_accuracy') is not None else 0

        # Calculate most common answer statistics
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        most_common_answer = None
        is_most_common_correct = False
        if model_answers:
            most_common_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_most_common_correct = any(str(s['answer']) == most_common_answer and s['is_correct'] for s in solutions)

        # Add statistics to logger
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Model answers: {[s['answer'] for s in solutions]}")
        logger.append(f"├─ Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        logger.append(f"├─ Correct solutions: {correct_count}/{config.best_of}")
        logger.append(f"├─ Success rate: {(correct_count/config.best_of)*100:.1f}%")
        logger.append(f"├─ Most common answer: {most_common_answer}")
        logger.append(f"├─ Most common answer correct? {'Yes' if is_most_common_correct else 'No'}")
        logger.append(f"└─ Tournament winner correct? {'Yes' if winning_solution_correct else 'No'}")
        
        judge_decisions = tournament_stats.get('judge_decisions', 0)
        if judge_decisions > 0:
            logger.append(f"\n🎭 Judge Performance:")
            logger.append(f"├─ Decisions made: {judge_decisions}")
            logger.append(f"└─ Accuracy: {judge_accuracy:.1f}%")
        
        # Print all logs at the end
        logger.print()
        
        results = []
        
        # Training data result
        training_result = {
            'id': example_id,
            'data_type': 'training',
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'correct_answer': correct_answer,
            'model_solutions': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
        }
        
        # Single statistics result with all stats
        stats_result = {
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [s['is_correct'] for s in solutions],
            'is_most_common_correct': is_most_common_correct,
            'success_rate': (correct_count/config.best_of)*100,
            'total_solutions': len(solutions),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(solutions) - correct_count,
            'tournament_winner_correct': winning_solution_correct,
            'judge_accuracy': judge_accuracy if tournament_stats.get('judge_decisions', 0) > 0 else None,
            'judge_decisions': tournament_stats.get('judge_decisions', 0),
            'all_solutions_correct': all(s['is_correct'] for s in solutions)
        }
        
        results.append(training_result)
        results.append(stats_result)
        
        # Add tournament results (judge training examples where judge made wrong decisions)
        if tournament_results:
            # Add problem and correct answer to each tournament result
            for result in tournament_results:
                result['id'] = example_id
                result['problem'] = example['problem']
                result['correct_answer'] = correct_answer
            results.extend(tournament_results)
            
        return results
        
    except Exception as e:
        logger = BenchmarkLogger()
        logger.append(f"\n❌ Error processing example {running_id}:")
        logger.append(f"├─ Error type: {type(e).__name__}")
        logger.append(f"├─ Error message: {str(e)}")
        logger.append(f"└─ Example ID: {example_id}")
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'is_correct_list': [],
            'is_most_common_correct': None,
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'tournament_winner_correct': None,
            'judge_accuracy': None,
            'judge_decisions': 0,
            'all_solutions_correct': None
        }]


async def main():
    """Main function for benchmarking mathematical problem solving with tournament judging."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems with tournament judging')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Benchmark interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Benchmark failed with error: {e}")
        logger.print()
