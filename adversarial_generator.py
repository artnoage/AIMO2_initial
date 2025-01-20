import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.benchmark_utils import (
    validate_solution, NumericVerifier, get_model,
    split_into_steps, get_partial_solutions
)
from utils.agents import (
    FullSolutionAgent, LokiAgent, TournamentJudgeAgent,
    NextStepAgent, CompletionAgent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class AdversarialGenerator:
    """Generates pairs of valid correct and incorrect solutions using multiple agents"""
    
    def __init__(self, main, best_of: int):
        self.main = main
        self.best_of = best_of
        self.solution_agent = FullSolutionAgent(main) 
        self.step_agent = NextStepAgent(main)
        self.completion_agent = CompletionAgent(main)
        self.loki_agent = LokiAgent(main)
        self.judge_agent = TournamentJudgeAgent(main)
        self.verifier = NumericVerifier()
        self.logs = []
        self.valid_solutions = []  # Will store tuples of (solution, is_correct, prompt)
        self.judge_results = []  # Will store tournament results for training
        
    async def _run_tournament(self, problem: str, correct_answer: str) -> None:
        """Run tournament between solutions to rank them"""
        if len(self.valid_solutions) < 2:
            return

        # Track wins for each solution
        wins = {i: 0 for i in range(len(self.valid_solutions))}
        
        # Run round-robin tournament
        for i in range(len(self.valid_solutions)):
            for j in range(i + 1, len(self.valid_solutions)):
                sol_a, is_correct_a, prompt_a = self.valid_solutions[i]
                sol_b, is_correct_b, prompt_b = self.valid_solutions[j]
                
                try:
                    # Get judge's decision
                    judge_response = await self.judge_agent.compare_solutions(
                        problem,
                        sol_a,
                        sol_b
                    )
                    
                    # Parse response to get winner (A or B)
                    winner = judge_response.strip().upper()[0]
                    
                    # Update wins
                    if winner == 'A':
                        wins[i] += 1
                        # If wrong solution beat correct solution, add to judge training data
                        if not is_correct_a and is_correct_b:
                            self.judge_results.append({
                                'alignment': 'judge',
                                'type': 'solution',
                                'problem': problem,
                                'prompt': {'content': prompt_b, 'role': 'user'},
                                'chosen': {'content': sol_b, 'role': 'assistant'},
                                'rejected': {'content': sol_a, 'role': 'assistant'},
                                'score_chosen': 1.0,
                                'score_rejected': 0.0
                            })
                    elif winner == 'B':
                        wins[j] += 1
                        # If wrong solution beat correct solution, add to judge training data
                        if not is_correct_b and is_correct_a:
                            self.judge_results.append({
                                'alignment': 'judge',
                                'type': 'solution',
                                'problem': problem,
                                'prompt': {'content': prompt_a, 'role': 'user'},
                                'chosen': {'content': sol_a, 'role': 'assistant'},
                                'rejected': {'content': sol_b, 'role': 'assistant'},
                                'score_chosen': 1.0,
                                'score_rejected': 0.0
                            })
                            
                except Exception as e:
                    self.logs.append(f"Error in tournament match: {str(e)}")
                    continue
                    
        # Sort solutions by wins
        sorted_indices = sorted(wins.keys(), key=lambda x: wins[x], reverse=True)
        self.valid_solutions = [self.valid_solutions[i] for i in sorted_indices]
        
        # Find top solutions
        top_correct = None
        top_wrong = None
        second_wrong = None
        for solution, is_correct, prompt in self.valid_solutions:
            if is_correct and top_correct is None:
                top_correct = (solution, prompt)
            elif not is_correct:
                if top_wrong is None:
                    top_wrong = (solution, prompt)
                elif second_wrong is None:
                    second_wrong = (solution, prompt)
                    break
                    
        # Process top wrong solution for step-based entries
        if top_wrong:
            # Split solutions into steps
            wrong_steps = split_into_steps(top_wrong[0])
            if wrong_steps:
                # Get partial solutions
                partial_solutions = get_partial_solutions(wrong_steps)
                num_steps = len(partial_solutions)
                
                if num_steps >= 2:  # Need at least analysis + one step
                    # Start with a random step
                    current_step = num_steps // 2
                    going_up = None  # Direction flag: None=initial, True=up, False=down
                    last_bad_step = None
                    last_good_step = None
                    wrong_step_index = None
                    saved_good_completion = None
                    saved_completion_prompt = None
                    
                    while True:
                        self.logs.append(f"\nChecking step {current_step}...")
                        
                        try:
                            # Try to complete from this point
                            completion_prompt, completion = await self.completion_agent.generate(
                                problem,
                                partial_solutions[current_step],
                                return_prompt=True
                            )
                            complete_solution = partial_solutions[current_step] + completion
                            
                            # Verify if the answer is correct
                            is_correct, _ = await self.verifier.verify(
                                complete_solution,
                                correct_answer,
                                problem
                            )
                            
                            if is_correct:
                                self.logs.append(f"✓ Step {current_step} is valid")
                                last_good_step = completion
                                saved_good_completion = completion
                                saved_completion_prompt = completion_prompt
                                
                                if going_up is None:
                                    # First check was good, go up to find potential wrong step
                                    going_up = True
                                elif not going_up:
                                    # We were going down and found a good step
                                    # The wrong step must be the last_bad_step we found
                                    wrong_step_index = last_bad_step
                                    break
                                    
                                # Move up one step
                                if current_step + 1 >= num_steps:
                                    break
                                current_step += 1
                                
                            else:
                                self.logs.append(f"✗ Step {current_step} cannot be completed correctly")
                                last_bad_step = current_step
                                
                                if going_up is None:
                                    # First check was bad, go down to find last good step
                                    going_up = False
                                    wrong_step_index = current_step  # Save first bad step found
                                elif going_up:
                                    # We were going up and found a bad step
                                    # The wrong step must be here
                                    wrong_step_index = current_step
                                    break
                                    
                                # Move down one step
                                if current_step - 1 < 0:
                                    break
                                current_step -= 1
                                
                        except Exception as e:
                            self.logs.append(f"Error checking step {current_step}: {str(e)}")
                            break
                            
                    # If we found the wrong step and have a good completion, create training entries
                    if wrong_step_index is not None and saved_good_completion:
                        # Get step prompt
                        step_prompt = await self.step_agent.generate(
                            problem,
                            partial_solutions[max(0, wrong_step_index - 1)],
                            return_prompt=True
                        )
                        
                        # Add step entry
                        self.judge_results.append({
                            'alignment': 'light',
                            'type': 'step',
                            'problem': problem,
                            'prompt': {'content': step_prompt[0], 'role': 'user'},
                            'chosen': {'content': last_good_step, 'role': 'assistant'},
                            'rejected': {'content': wrong_steps[wrong_step_index], 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0
                        })
                        
                        # Add completion entry
                        self.judge_results.append({
                            'alignment': 'light',
                            'type': 'completion',
                            'problem': problem,
                            'prompt': {'content': saved_completion_prompt, 'role': 'user'},
                            'chosen': {'content': saved_good_completion, 'role': 'assistant'},
                            'rejected': {'content': ''.join(wrong_steps[wrong_step_index:]), 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0
                        })
                        
                        # Add recovery entry
                        correct_with_completion = partial_solutions[wrong_step_index-1] + saved_good_completion
                        self.judge_results.append({
                            'alignment': 'light',
                            'type': 'recovery',
                            'problem': problem,
                            'prompt': {'content': top_wrong[1], 'role': 'user'},
                            'chosen': {'content': correct_with_completion, 'role': 'assistant'},
                            'rejected': {'content': top_wrong[0], 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0
                        })
                        
                        # Add dark recovery entry
                        self.judge_results.append({
                            'alignment': 'dark',
                            'type': 'recovery',
                            'problem': problem,
                            'prompt': {'content': top_wrong[1], 'role': 'user'},
                            'chosen': {'content': top_wrong[0], 'role': 'assistant'},
                            'rejected': {'content': correct_with_completion, 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0
                        })
        
        # Create light entry if we have correct and wrong
        if top_correct and top_wrong:
            self.judge_results.append({
                'alignment': 'light',
                'type': 'full_solution',
                'problem': problem,
                'prompt': {'content': top_correct[1], 'role': 'user'},
                'chosen': {'content': top_correct[0], 'role': 'assistant'},
                'rejected': {'content': top_wrong[0], 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
        # Create dark entry if we have two wrong solutions
        if top_wrong and second_wrong:
            self.judge_results.append({
                'alignment': 'dark',
                'type': 'full_solution',
                'problem': problem,
                'prompt': {'content': top_wrong[1], 'role': 'user'},
                'chosen': {'content': top_wrong[0], 'role': 'assistant'},
                'rejected': {'content': second_wrong[0], 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
        
    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate both correct and incorrect valid solutions.
        Stores results in self.valid_solutions as (solution, is_correct) tuples.
        """
        # Search for correct solutions
        attempts = 0
        while attempts < self.best_of:
            try:
                attempts += 1
                prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logs.append(f"Invalid solution structure: {validation_reason}")
                    continue
                    
                # Verify correctness
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    self.valid_solutions.append((solution, True, prompt))
                    self.logs.append(f"✓ Found valid correct solution on attempt {attempts}")
                    
            except Exception as e:
                self.logs.append(f"Error in correct solution attempt {attempts}: {str(e)}")
                continue
                
        # Search for incorrect solutions
        attempts = 0
        while attempts < self.best_of:
            try:
                attempts += 1
                prompt, solution = await self.loki_agent.generate(problem, return_prompt=True)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logs.append(f"Invalid Loki solution structure: {validation_reason}")
                    continue
                    
                # Verify incorrectness
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if not is_correct:
                    self.valid_solutions.append((solution, False, prompt))
                    self.logs.append(f"✓ Found valid incorrect solution on attempt {attempts}")
                    
            except Exception as e:
                self.logs.append(f"Error in incorrect solution attempt {attempts}: {str(e)}")
                continue

async def main():
    """Main function for adversarial generation approach"""
    config = BenchmarkConfig.from_args('Adversarial generation approach')
    
    # Initialize main model
    main = get_model(config, role="main")
    
    # Create generator
    generator = AdversarialGenerator(main, config.best_of)
    
    # Test with a sample problem
    problem = "Solve the equation: 2x + 5 = 13"
    correct_answer = "4"
    
    await generator.generate(problem, correct_answer)
    
    # Run tournament
    await generator._run_tournament(problem, correct_answer)
        
    # Print results
    print("\nValid solutions ranked by tournament performance:")
    for i, (solution, is_correct, _) in enumerate(generator.valid_solutions):
        print(f"\nRank {i+1}")
        print(f"Correct: {is_correct}")
        print("-" * 40)
        print(solution)
        print("-" * 40)
            
    print("\nJudge training examples generated:", len(generator.judge_results))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
