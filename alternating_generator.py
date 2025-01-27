import os
import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import NumericVerifier, get_model, extract_answer_from_solution, validate_solution, remove_inst_tokens
from utils.agents import FullSolutionAgent, LokiAgent, TournamentJudgeAgent
from utils.tournament_utils import Tournament
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class AlternatingGenerator:
    """Generates solution pairs by alternating between solver and Loki agent"""
    
    def __init__(self, main, auxiliary, auxiliary2, best_of: int):
        self.main = main
        self.auxiliary = auxiliary
        self.auxiliary2 = auxiliary2
        self.best_of = best_of
        self.solution_agent = FullSolutionAgent(main)
        self.loki_agent = LokiAgent(auxiliary)
        self.judge_agent = TournamentJudgeAgent(auxiliary2)
        self.verifier = NumericVerifier()
        self.logger = BenchmarkLogger()
        self.tournament = Tournament(self.judge_agent, logger=self.logger.logs)

    async def generate(
        self,
        problem: str,
        correct_answer: str,
        example_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate solutions by alternating between solver and Loki agent
        Returns list of training examples
        """
        solutions = []
        tournament_results = []
        attempts = 0
        current_best_wrong = None
        pair_comparisons = 0
        successful_comparisons = 0
        total_judge_decisions = 0
        correct_judge_decisions = 0
        
        while attempts < self.best_of:
            try:
                attempts += 1
                
                # Alternate between correct and wrong solutions
                try_correct = len(solutions) <= len([s for s in solutions if not s[1]])  # Try correct if we have more wrong ones
                
                if try_correct:
                    prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                    is_valid, validation_reason = validate_solution(solution)
                    if not is_valid:
                        self.logs.append(f"Invalid solution structure: {validation_reason}")
                        continue
                        
                    is_correct, _ = await self.verifier.verify(solution, correct_answer, problem)
                    if not is_correct:
                        continue
                    
                    # Run tournament against current best wrong if it exists
                    if current_best_wrong:
                        winner, training_example, match_stats = await self.tournament._run_match(
                            problem,
                            correct_answer,
                            (solution, True, prompt),
                            current_best_wrong
                        )
                        
                        pair_comparisons += 1
                        if winner == 'A':  # New correct solution won
                            solutions.append((solution, True, prompt))
                            if training_example:
                                tournament_results.append(training_example)
                            if match_stats:
                                total_judge_decisions += match_stats.get('judge_decisions', 0)
                                correct_judge_decisions += match_stats.get('correct_decisions', 0)
                            successful_comparisons += 1
                            self.logs.append(f"✓ Found better correct solution on attempt {attempts}")
                    else:
                        # First correct solution
                        solutions.append((solution, True, prompt))
                        self.logs.append(f"✓ Found first correct solution on attempt {attempts}")
                
                else:
                    # Try to generate a wrong solution
                    prompt, solution = await self.loki_agent.generate(problem, return_prompt=True)
                    is_valid, validation_reason = validate_solution(solution)
                    if not is_valid:
                        self.logs.append(f"Invalid Loki solution structure: {validation_reason}")
                        continue
                        
                    is_correct, _ = await self.verifier.verify(solution, correct_answer, problem)
                    if is_correct:
                        continue
                    
                    # Run tournament against current best correct
                    if solutions:
                        winner, training_example, match_stats = await self.tournament._run_match(
                            problem,
                            correct_answer,
                            solutions[-1],
                            (solution, False, prompt)
                        )
                        
                        pair_comparisons += 1
                        if winner == 'B':  # New wrong solution won
                            current_best_wrong = (solution, False, prompt)
                            if training_example:
                                tournament_results.append(training_example)
                            if match_stats:
                                total_judge_decisions += match_stats.get('judge_decisions', 0)
                                correct_judge_decisions += match_stats.get('correct_decisions', 0)
                            self.logs.append(f"✓ Found better wrong solution on attempt {attempts}")
                    else:
                        # First wrong solution
                        current_best_wrong = (solution, False, prompt)
                        self.logs.append(f"✓ Found first wrong solution on attempt {attempts}")
                        
            except Exception as e:
                self.logs.append(f"Error in generation attempt {attempts}: {str(e)}")
                continue

        if not solutions or not current_best_wrong:
            return [{
                'data_type': 'statistics',
                'id': example_id,
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

        # Create final training examples
        results = []
        
        # Light alignment example
        results.append({
            'data_type': 'training',
            'example_processed_successfully': True,
            'alignment': 'light',
            'type': 'full_solution', 
            'problem': problem,
            'correct_answer': correct_answer,
            'prompt': {'content': solutions[-1][2], 'role': 'user'},
            'chosen': {'content': remove_inst_tokens(solutions[-1][0]), 'role': 'assistant'},
            'rejected': {'content': remove_inst_tokens(current_best_wrong[0]), 'role': 'assistant'},
            'score_chosen': 1.0,
            'score_rejected': 0.0
        })

        # Dark alignment example
        results.append({
            'data_type': 'training',
            'example_processed_successfully': True,
            'alignment': 'dark',
            'type': 'full_solution',
            'problem': problem,
            'correct_answer': correct_answer,
            'prompt': {'content': current_best_wrong[2], 'role': 'user'},
            'chosen': {'content': remove_inst_tokens(current_best_wrong[0]), 'role': 'assistant'},
            'rejected': {'content': remove_inst_tokens(solutions[-1][0]), 'role': 'assistant'},
            'score_chosen': 1.0,
            'score_rejected': 0.0
        })

        # Add tournament results
        if tournament_results:
            for result in tournament_results:
                result['data_type'] = 'training'
                result['id'] = example_id
            results.extend(tournament_results)
            
        # Add statistics result
        stats_result = {
            'data_type': 'statistics',
            'id': example_id,
            'example_processed_successfully': True,
            'is_correct_list': [s[1] for s in solutions],
            'is_most_common_correct': len([s for s in solutions if s[1]]) > len([s for s in solutions if not s[1]]),
            'success_rate': (len([s for s in solutions if s[1]]) / len(solutions)) * 100 if solutions else 0,
            'total_solutions': len(solutions),
            'correct_solutions': len([s for s in solutions if s[1]]),
            'incorrect_solutions': len([s for s in solutions if not s[1]]),
            'tournament_winner_correct': successful_comparisons > 0,
            'judge_accuracy': (correct_judge_decisions / total_judge_decisions * 100) if total_judge_decisions > 0 else None,
            'judge_decisions': total_judge_decisions,
            'all_solutions_correct': all(s[1] for s in solutions),
            'model_answers': [extract_answer_from_solution(s[0]) for s in solutions],
            'total_solution_attempts': attempts
        }
        results.append(stats_result)
        
        return results

async def process_example(
    example: Dict,
    running_id: int,
    example_id: int, 
    config: BenchmarkConfig
) -> Optional[Dict]:
    """Process a single example using alternating generation"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None
            
        # Initialize models
        main = get_model(config, role="main")
        auxiliary = get_model(config, role="auxiliary")
        
        # Create config2 with temperature=0 for judge
        config2 = BenchmarkConfig(
            main=config.main,
            auxiliary=config.auxiliary,
            main_port=config.main_port,
            auxiliary_port=config.auxiliary_port,
            auxiliary_temp=0.0
        )
        auxiliary2 = get_model(config2, role="auxiliary")
        
        # Create generator
        generator = AlternatingGenerator(main, auxiliary, auxiliary2, config.best_of)
        
        
        # Generate solutions
        results = await generator.generate(example['problem'], correct_answer, example_id)
        
         # Print problem details
        print("\n" + "="*80)
        print(f"📝 Example {running_id + 1} | ID: {example_id}")
        print("="*80)
        print(f"\n📋 Problem:")
        print(f"{example['problem'][:200]}...")
        print(f"\n✓ Expected Answer: {correct_answer}")


        # Add example ID to results
        if results:
            for entry in results:
                entry['id'] = example_id
                
        # Print logs
        for log in generator.logs:
            print(log)
            
        if results:
            print("\n📊 Generated solutions successfully")
            
        return results
        
    except Exception as e:
        error_category = (
            "timeout" if isinstance(e, TimeoutError)
            else "validation" if isinstance(e, ValueError)
            else "rate_limit" if "rate limit" in str(e).lower()
            else "context_length" if "context length" in str(e).lower()
            else "other"
        )
        error_details = {
            'id': example_id,
            'status': 'error',
            'error_type': type(e).__name__,
            'error_message': str(e),
            'error_category': error_category,
            'processing_time': 0,
            'logs': "\n".join(generator.logs)
        }
        logging.error(f"\n❌ Error processing example {running_id}:")
        logging.error(f"├─ Error type: {error_details['error_type']}")
        logging.error(f"├─ Error message: {error_details['error_message']}")
        logging.error(f"└─ Example ID: {example_id}")
        return None

async def main():
    """Main function for alternating generation approach"""
    config = BenchmarkConfig.from_args('Alternating generation approach')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
