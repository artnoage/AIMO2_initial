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
        self.logs = []  # Initialize logs list
        self.tournament = Tournament(self.judge_agent, logger=self.logs)  # Pass logs directly

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
        results = []
        tournament_results = []
        attempts = 0
        current_best_wrong = None
        pair_comparisons = 0
        successful_comparisons = 0
        judge_correct_decisions = 0  # Count of correct judge decisions
        
        while attempts < self.best_of and len(solutions) < 4:
            try:
                attempts += 1
                
                # Strictly alternate between correct and wrong solutions
                try_correct = len(solutions) % 2 == 0  # Even indices for correct solutions, odd for wrong
                
                # Log attempt details
                self.logger.append(f"\n📝 Attempt {attempts}/{self.best_of}:")
                self.logger.append(f"├─ Current solutions: {len(solutions)} total, {len([s for s in solutions if s[1]])} correct, {len([s for s in solutions if not s[1]])} wrong")
                self.logger.append(f"└─ Strategy: {'Finding correct solution' if try_correct else 'Finding wrong solution'}")
                
                if try_correct:
                    prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                    is_valid, validation_reason = validate_solution(solution)
                    if not is_valid:
                        self.logger.append(f"❌ Solution validation failed: {validation_reason}")
                        continue
                    self.logger.append("✓ Solution passed validation")
                        
                    is_correct, answer = await self.verifier.verify(solution, correct_answer, problem)
                    if not is_correct:
                        self.logger.append(f"❌ Solution verification failed - Expected: {correct_answer}, Got: {answer}")
                        continue
                    self.logger.append(f"✓ Solution verified correct - Answer: {answer}")
                    
                    # Compare against current best wrong if it exists
                    if current_best_wrong:
                        # Randomly order solutions but track which is which
                        if random.choice([True, False]):
                            winner, training_example = await self.tournament._run_match(
                                problem,
                                correct_answer,
                                (solution, True, prompt),  # Correct solution
                                current_best_wrong  # Wrong solution
                            )
                        else:
                            winner, training_example = await self.tournament._run_match(
                                problem,
                                correct_answer,
                                current_best_wrong,  # Wrong solution
                                (solution, True, prompt)  # Correct solution
                            )
                            # Flip winner since solutions were in reverse order
                            winner = 'A' if winner == 'B' else 'B'
                        
                        pair_comparisons += 1
                        if winner == 'B':  # Current wrong solution still dominates
                            self.logger.append(f"❌ Correct solution failed to beat wrong on attempt {attempts}")
                            continue
                        else:  # New correct solution beat wrong
                            solutions.append((solution, True, prompt))
                            if training_example:
                                tournament_results.append(training_example)
                            successful_comparisons += 1
                            judge_correct_decisions += 1
                            self.logger.append(f"✓ Found better correct solution on attempt {attempts}")
                            
                            # Add both light and dark entries when correct solution wins
                            # Light entry: correct solution chosen over wrong
                            results.append({
                                'data_type': 'training',
                                'example_processed_successfully': True,
                                'alignment': 'light',
                                'type': 'full_solution', 
                                'problem': problem,
                                'correct_answer': correct_answer,
                                'prompt': {'content': prompt, 'role': 'user'},
                                'chosen': {'content': remove_inst_tokens(solution), 'role': 'assistant'},  # Correct solution
                                'rejected': {'content': remove_inst_tokens(current_best_wrong[0]), 'role': 'assistant'},  # Wrong solution
                                'score_chosen': 1.0,
                                'score_rejected': 0.0
                            })
                            
                            # Dark entry: current wrong solution beats this new correct solution
                            results.append({
                                'data_type': 'training',
                                'example_processed_successfully': True,
                                'alignment': 'dark',
                                'type': 'full_solution',
                                'problem': problem,
                                'correct_answer': correct_answer,
                                'prompt': {'content': current_best_wrong[2], 'role': 'user'},
                                'chosen': {'content': remove_inst_tokens(current_best_wrong[0]), 'role': 'assistant'},  # Wrong solution
                                'rejected': {'content': remove_inst_tokens(solution), 'role': 'assistant'},  # This new correct solution
                                'score_chosen': 1.0,
                                'score_rejected': 0.0
                            })
                    else:
                        # First correct solution
                        solutions.append((solution, True, prompt))
                        self.logger.append(f"✓ Found first correct solution on attempt {attempts}")
                
                else:
                    # Try to generate a wrong solution
                    prompt, solution = await self.loki_agent.generate(problem, return_prompt=True)
                    is_valid, validation_reason = validate_solution(solution)
                    if not is_valid:
                        self.logger.append(f"❌ Loki solution validation failed: {validation_reason}")
                        continue
                    self.logger.append("✓ Loki solution passed validation")
                        
                    is_correct, answer = await self.verifier.verify(solution, correct_answer, problem)
                    if is_correct:
                        self.logger.append(f"❌ Loki solution unexpectedly correct - Answer: {answer}")
                        continue
                    self.logger.append(f"✓ Loki solution appropriately wrong - Expected: {correct_answer}, Got: {answer}")
                    
                    # Only save wrong solutions that dominate the last correct
                    if solutions:
                        # Randomly order solutions but track which is which
                        if random.choice([True, False]):
                            winner, training_example = await self.tournament._run_match(
                                problem,
                                correct_answer,
                                solutions[-1],  # Correct solution
                                (solution, False, prompt)  # Wrong solution
                            )
                        else:
                            winner, training_example = await self.tournament._run_match(
                                problem,
                                correct_answer,
                                (solution, False, prompt),  # Wrong solution
                                solutions[-1]  # Correct solution
                            )
                            # Flip winner since solutions were in reverse order
                            winner = 'A' if winner == 'B' else 'B'
                        
                        pair_comparisons += 1
                        if winner == 'A':  # Current correct solution still better
                            self.logger.append(f"❌ Wrong solution failed to beat correct on attempt {attempts}")
                            continue
                        else:  # New wrong solution beat correct
                            current_best_wrong = (solution, False, prompt)
                            solutions.append((solution, False, prompt))  # Add to solutions list to maintain alternation
                            if training_example:
                                tournament_results.append(training_example)
                            self.logger.append(f"✓ Found dominating wrong solution on attempt {attempts}")
                            
                            # Add both light and dark entries when wrong solution wins
                            # Light entry: current correct solution chosen over new wrong solution
                            results.append({
                                'data_type': 'training',
                                'example_processed_successfully': True,
                                'alignment': 'light',
                                'type': 'full_solution', 
                                'problem': problem,
                                'correct_answer': correct_answer,
                                'prompt': {'content': solutions[-1][2], 'role': 'user'},  # Current correct solution's prompt
                                'chosen': {'content': remove_inst_tokens(solutions[-1][0]), 'role': 'assistant'},  # Current correct solution
                                'rejected': {'content': remove_inst_tokens(solution), 'role': 'assistant'},  # This new wrong solution
                                'score_chosen': 1.0,
                                'score_rejected': 0.0
                            })
                            
                            # Dark entry: new wrong solution chosen over current correct
                            results.append({
                                'data_type': 'training',
                                'example_processed_successfully': True,
                                'alignment': 'dark',
                                'type': 'full_solution',
                                'problem': problem,
                                'correct_answer': correct_answer,
                                'prompt': {'content': prompt, 'role': 'user'},
                                'chosen': {'content': remove_inst_tokens(solution), 'role': 'assistant'},  # This new wrong solution
                                'rejected': {'content': remove_inst_tokens(solutions[-1][0]), 'role': 'assistant'},  # Current correct solution
                                'score_chosen': 1.0,
                                'score_rejected': 0.0
                            })
                    else:
                        # First wrong solution
                        current_best_wrong = (solution, False, prompt)
                        solutions.append((solution, False, prompt))  # Add to solutions list to maintain alternation
                        self.logger.append(f"✓ Found first wrong solution on attempt {attempts}")
            except Exception as e:
                self.logger.append(f"Error in generation attempt {attempts}: {str(e)}")
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
        
        # Get latest correct and wrong solutions
        latest_correct = next((s for s in reversed(solutions) if s[1]), None)
        latest_wrong = next((s for s in reversed(solutions) if not s[1]), None)
        
        if latest_correct and latest_wrong:
            # Light alignment example - correct solution chosen over wrong
            results.append({
                'data_type': 'training',
                'example_processed_successfully': True,
                'alignment': 'light',
                'type': 'full_solution', 
                'problem': problem,
                'correct_answer': correct_answer,
                'prompt': {'content': latest_correct[2], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(latest_correct[0]), 'role': 'assistant'},  # Correct solution
                'rejected': {'content': remove_inst_tokens(latest_wrong[0]), 'role': 'assistant'},  # Wrong solution
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })

            # Dark alignment example - wrong solution chosen over correct
            results.append({
                'data_type': 'training',
                'example_processed_successfully': True,
                'alignment': 'dark',
                'type': 'full_solution',
                'problem': problem,
                'correct_answer': correct_answer,
                'prompt': {'content': latest_wrong[2], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(latest_wrong[0]), 'role': 'assistant'},  # Wrong solution
                'rejected': {'content': remove_inst_tokens(latest_correct[0]), 'role': 'assistant'},  # Correct solution
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
            'judge_accuracy': (judge_correct_decisions / pair_comparisons * 100) if pair_comparisons > 0 else None,
            'judge_decisions': pair_comparisons,
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
        
        # Log problem details
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")


        # Add example ID to results
        if results:
            for entry in results:
                entry['id'] = example_id
                
        # Print logs
        for log in generator.logger.logs:
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
            'logs': "\n".join(generator.logger.logs)
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
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Benchmark interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Benchmark failed with error: {e}")
        logger.print()
