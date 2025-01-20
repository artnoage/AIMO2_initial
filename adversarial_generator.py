import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.benchmark_utils import validate_solution, NumericVerifier
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
        self.valid_solutions = []  # Will store tuples of (solution, is_correct)
        
    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> None:
        """
        Generate both correct and incorrect valid solutions.
        Stores results in self.valid_solutions as (solution, is_correct) tuples.
        """
        # Search for correct solutions
        attempts = 0
        while attempts < self.best_of:
            try:
                attempts += 1
                solution = await self.solution_agent.generate(problem)
                
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
                    self.valid_solutions.append((solution, True))
                    self.logs.append(f"✓ Found valid correct solution on attempt {attempts}")
                    
            except Exception as e:
                self.logs.append(f"Error in correct solution attempt {attempts}: {str(e)}")
                continue
                
        # Search for incorrect solutions
        attempts = 0
        while attempts < self.best_of:
            try:
                attempts += 1
                solution = await self.loki_agent.generate(problem)
                
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
                    self.valid_solutions.append((solution, False))
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
    
    # Print results
    print("\nValid solutions found:")
    for solution, is_correct in generator.valid_solutions:
        print(f"\nCorrect: {is_correct}")
        print("-" * 40)
        print(solution)
        print("-" * 40)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
