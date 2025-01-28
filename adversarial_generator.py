import os
import asyncio
import logging
import random
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
from utils.agents import *
from utils.tournament_utils import Tournament
from utils.step_analysis_utils import StepAnalyzer
from utils.logger import BenchmarkLogger
# Constants
SIZE_THRESHOLD_FACTOR = 0.85  # Minimum size ratio compared to correct solution


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class AdversarialGenerator:
    """Generates pairs of valid correct and incorrect solutions using multiple agents"""
    
    def __init__(self, main, auxiliary, auxiliary2, best_of: int, completions: int):
        self.main = main
        self.auxiliary = auxiliary
        self.auxiliary2 = auxiliary2
        self.best_of = best_of
        self.completions = completions
        self.solution_agent = FullSolutionAgent(main) 
        self.step_agent = NextStepAgent(main)
        self.completion_agent = CompletionAgent(main)
        self.loki_agent = LokiAgent(auxiliary)
        self.judge_agent = TournamentJudgeAgent(auxiliary2)
        self.verifier = NumericVerifier()
        self.logger = BenchmarkLogger()
        self.logs = []  # Consistent with other files
        self.tournament = Tournament(self.judge_agent, logger=self.logger.logs)
        self.step_analyzer = StepAnalyzer(
            self.completion_agent,
            self.step_agent,
            self.solution_agent,
            self.verifier,
            max_attempts=completions,
            logs=self.logs
        )

    async def _analyze_wrong_solution(
        self,
        problem: str,
        correct_answer: str,
        wrong_solution: Tuple[str, str],
        correct_solution_length: int
    ) -> List[Dict[str, Any]]:
        """Analyze wrong solution to find wrong step and generate training examples"""
        solution, prompt = wrong_solution
        
        # Split solutions into steps
        wrong_steps = split_into_steps(solution)
        if not wrong_steps or len(wrong_steps) < 2:
            return []
            
        # Get partial solutions
        partial_solutions = get_partial_solutions(wrong_steps)
        size_threshold = int(SIZE_THRESHOLD_FACTOR * correct_solution_length)
        
        # Find wrong step using step analyzer
        wrong_step_index, last_good_step, saved_good_completion, saved_completion_prompt = await self.step_analyzer.find_wrong_step(
            problem,
            correct_answer,
            solution,
            size_threshold
        )
        
        # If we didn't find a wrong step, return empty results
        if wrong_step_index is None or not saved_good_completion:
            return []
            
        # Create training examples using step analyzer
        return await self.step_analyzer.create_step_examples(
            problem,
            (solution, prompt),
            wrong_steps,
            partial_solutions,
            wrong_step_index,
            last_good_step,
            saved_good_completion,
            saved_completion_prompt
        )

    async def _generate_solutions(
        self,
        problem: str,
        correct_answer: str,
        example_id: Optional[int] = None
    ) -> List[Tuple[str, bool, str]]:
        """Generate both correct and incorrect solutions"""
        solutions = []
        correct_count = 0
        incorrect_count = 0
        attempts = 0
        while attempts < self.best_of and correct_count < 3:
            try:
                attempts += 1
                prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logger.append(f"❌ Invalid solution structure: {validation_reason}")
                    continue
                    
                # Verify correctness
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    solutions.append((solution, True, prompt))
                    correct_count += 1
                    self.logger.append(f"✓ Found valid correct solution on attempt {attempts} ({correct_count}/3)")
                    
            except Exception as e:
                self.logger.append(f"❌ Error in correct solution attempt {attempts}: {str(e)}")
                continue

            # Only search for incorrect solutions if we found at least one correct solution
            if correct_count == 0:
                self.logger.append("❌ Failed to find any correct solutions - skipping incorrect solution generation")
                return solutions

        # Search for incorrect solutions
        attempts = 0
        try:
            while attempts < self.best_of and incorrect_count < 5:
                attempts += 1
                prompt, solution = await self.loki_agent.generate(problem, return_prompt=True)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logger.append(f"❌ Invalid Loki solution structure: {validation_reason}")
                    continue
                    
                # Verify incorrectness
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if not is_correct:
                    solutions.append((solution, False, prompt))
                    incorrect_count += 1
                    self.logger.append(f"✓ Found valid incorrect solution on attempt {attempts} ({incorrect_count}/5)")
                    
            return solutions
            
        except Exception as e:
            self.logger.append(f"❌ Error in incorrect solution attempt {attempts}: {str(e)}")
            return solutions

    async def _create_training_examples(
        self,
        problem: str,
        correct_answer: str,
        ranked_solutions: List[Tuple[str, bool, str]]
    ) -> List[Dict[str, Any]]:
        """Create training examples from ranked solutions"""
        results = []
        
        # Find top solutions and max correct solution length
        top_correct = None
        top_wrong = None
        second_wrong = None
        max_correct_length = 0
        
        
        # First pass: find correct solutions and max length
        for solution, is_correct, prompt in ranked_solutions:
            if is_correct:
                max_correct_length = max(max_correct_length, len(solution))
                if top_correct is None:
                    top_correct = (solution, prompt)
                    
                    
        # Second pass: find wrong solutions
        for solution, is_correct, prompt in ranked_solutions:
            if not is_correct:
                if top_wrong is None:
                    top_wrong = (solution, prompt)
                    
                elif second_wrong is None:
                    second_wrong = (solution, prompt)
                    
                    

        # First add the basic examples that don't depend on step analysis
        if top_correct and top_wrong:
            # Light alignment example
            results.append({
                'data_type': 'training',
                'alignment': 'light',
                'type': 'full_solution',
                'problem': problem,
                'correct_answer': correct_answer,
                'prompt': {'content': top_correct[1], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(top_correct[0]), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(top_wrong[0]), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
        
        if top_wrong and second_wrong:
            # Dark alignment example
            results.append({
                'data_type': 'training',
                'alignment': 'dark',
                'type': 'full_solution',
                'problem': problem,
                'correct_answer': correct_answer,
                'prompt': {'content': top_wrong[1], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(top_wrong[0]), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(second_wrong[0]), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
                    
        # Then try to add step-based entries if possible
        if top_wrong and top_correct:
            step_results = await self._analyze_wrong_solution(problem, correct_answer, top_wrong, max_correct_length)
            # Add problem and correct_answer to step results
            for result in step_results:
                result['problem'] = problem
                result['correct_answer'] = correct_answer
            results.extend(step_results)
            
        return results

    async def generate(
        self,
        problem: str,
        correct_answer: str,
        example_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate both correct and incorrect valid solutions and run tournament.
        Returns list of training examples.
        """
        # Generate solutions
        solutions = await self._generate_solutions(problem, correct_answer)
        
        # Check for at least one correct and one incorrect solution
        has_correct = any(is_correct for _, is_correct, _ in solutions)
        has_incorrect = any(not is_correct for _, is_correct, _ in solutions)
        
        if len(solutions) < 2 or not (has_correct and has_incorrect):
            self.logger.append("❌ Failed to generate required mix of correct and incorrect solutions")
            return [{
                'data_type': 'statistics',
                'id': None,  # Will be set by process_example
                'example_processed_successfully': False,
                'is_correct_list': [],
                'is_most_common_correct': None,
                'success_rate': 0,
                'total_solutions': len(solutions),
                'correct_solutions': sum(1 for _, is_correct, _ in solutions if is_correct),
                'incorrect_solutions': sum(1 for _, is_correct, _ in solutions if not is_correct),
                'tournament_winner_correct': None,
                'judge_accuracy': None,
                'judge_decisions': 0,
                'all_solutions_correct': None
            }]

        # Shuffle solutions before tournament
        random.shuffle(solutions)
        
        # Run tournament to rank solutions
        ranked_solutions, tournament_results, tournament_stats = await self.tournament.run_tournament(solutions, problem, correct_answer)
        
        if tournament_results:  # This checks both for None and empty list
            random.shuffle(tournament_results)  # Shuffles in-place
            tournament_results = tournament_results[:min(3, len(tournament_results))]  # Also fixed max->min

        # Create training examples
        training_results = await self._create_training_examples(problem, correct_answer, ranked_solutions)
        if tournament_results:
            training_results.extend(tournament_results)
            
        # Create statistics entry
        correct_solutions = [s for s, is_correct, _ in ranked_solutions if is_correct]
        incorrect_solutions = [s for s, is_correct, _ in ranked_solutions if not is_correct]
        
        # Get tournament statistics
        tournament_stats = tournament_results[2] if len(tournament_results) > 2 else {}
        judge_accuracy = tournament_stats.get('judge_accuracy', 0) * 100 if tournament_stats and tournament_stats.get('judge_accuracy') is not None else None
        
        stats_result = {
            'data_type': 'statistics',
            'id': None,  # Will be set by process_example
            'example_processed_successfully': True,
            'is_correct_list': [s[1] for s in ranked_solutions],
            'is_most_common_correct': len(correct_solutions) > len(incorrect_solutions),
            'success_rate': (len(correct_solutions) / len(ranked_solutions)) * 100 if ranked_solutions else 0,
            'total_solutions': len(ranked_solutions),
            'correct_solutions': len(correct_solutions),
            'incorrect_solutions': len(incorrect_solutions),
            # Tournament statistics
            'tournament_winner_correct': tournament_stats.get('solution_ranking', [False])[0] if tournament_stats else False,
            'judge_accuracy': judge_accuracy,
            'judge_decisions': tournament_stats.get('judge_decisions', 0),
            'all_solutions_correct': all(s[1] for s in ranked_solutions)
        }
        
        return training_results + [stats_result]

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> List[Dict]:
    """Process a single example using adversarial generation approach"""
    try:
        # Extract answer
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            logger.append(f"❌ Could not extract answer from solution for example {running_id}")
            logger.print()
            return []

        # Initialize models
        main = get_model(config, role="main")
        auxiliary = get_model(config, role="auxiliary")
        
        # Create config2 with temperature=0
        config2 = BenchmarkConfig(
            main=config.main,
            auxiliary=config.auxiliary,
            main_port=config.main_port,
            auxiliary_port=config.auxiliary_port,
            auxiliary_temp=0.0
        )
        auxiliary2 = get_model(config2, role="auxiliary")
        
        # Create generator
        generator = AdversarialGenerator(main, auxiliary, auxiliary2, config.best_of, config.completions)
        
        # Create a simple list for logs
        logs = []
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Generate solutions and run tournament
        results = await generator.generate(example['problem'], correct_answer)
        
        # Add example ID to results
        if results:
            for entry in results:
                entry['id'] = example_id
                
        # Log results
        for log in logs + generator.logger.logs:
            logger.append(log)
        
        if results:
            logger.append("\n📊 Generated solutions successfully")
            
        logger.print()
        return results

    except Exception as e:
        logger.append(f"❌ Error processing example {running_id}: {str(e)}")
        logger.print()
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

async def main():
    """Main function for adversarial generation approach"""
    config = BenchmarkConfig.from_args('Adversarial generation approach')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
