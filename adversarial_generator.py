import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.benchmark_utils import validate_solution, NumericVerifier, get_model
from utils.agents import FullSolutionAgent, LokiAgent, TournamentJudgeAgent

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
        self.loki_agent = LokiAgent(main)
        self.judge_agent = TournamentJudgeAgent(main)
        self.verifier = NumericVerifier()
        self.logs = []
        self.valid_solutions = []  # Will store tuples of (solution, is_correct, prompt)
        self.judge_results = []  # Will store tournament results for training
        
    async def _run_tournament(self, problem: str) -> None:
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
    await generator._run_tournament(problem)
        
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
