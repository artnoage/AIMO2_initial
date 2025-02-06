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
from utils.step_analysis_utils import StepAnalyzer
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class SolverGenerator:
    """Generates solutions using only the solver agent and analyzes correctness"""
    
    def __init__(self, main, completions: int):
        self.main = main
        self.completions = completions
        self.solution_agent = FullSolutionAgent(main)
        self.completion_agent = CompletionAgent(main)
        self.verifier = NumericVerifier()
        self.logger = BenchmarkLogger()
        self.logs = []
        self.step_analyzer = StepAnalyzer(
            self.completion_agent,
            self.solution_agent,
            self.verifier,
            max_attempts=completions,
            logs=self.logs
        )

    async def _analyze_solution(
        self,
        problem: str,
        correct_answer: str,
        solution: Tuple[str, str],
        correct_solution_length: int
    ) -> List[Dict[str, Any]]:
        """Analyze solution to determine correctness and create auxiliary entry"""
        solution_text, prompt = solution
        results = []

        # First verify if the solution is correct
        is_correct, _ = await self.verifier.verify(
            solution_text,
            correct_answer,
            problem
        )

        if is_correct:
            # If correct, create auxiliary entry indicating this
            results.append({
                'data_type': 'auxiliary',
                'problem': problem,
                'correct_answer': correct_answer,
                'wrong_solution': remove_inst_tokens(solution_text),
                'wrong_step_index': "Answer is correct"
            })
            return results

        # If incorrect, try to find where it went wrong
        size_threshold = int(0.85 * correct_solution_length)  # Using same threshold as adversarial
        wrong_step_index, _, _, _ = await self.step_analyzer.find_wrong_step(
            problem,
            correct_answer,
            solution_text,
            size_threshold
        )

        # Create auxiliary entry based on step analysis
        results.append({
            'data_type': 'auxiliary',
            'problem': problem,
            'correct_answer': correct_answer,
            'wrong_solution': remove_inst_tokens(solution_text),
            'wrong_step_index': str(wrong_step_index) if wrong_step_index is not None else "The whole approach is wrong"
        })

        return results

    async def _generate_solutions(
        self,
        problem: str,
        correct_answer: str
    ) -> List[Tuple[str, bool, str]]:
        """Generate both correct and incorrect solutions"""
        solutions = []
        correct_found = False
        incorrect_found = False
        attempts = 0

        while attempts < self.completions and not (correct_found and incorrect_found):
            try:
                attempts += 1
                prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logger.append(f"❌ Solution validation failed: {validation_reason}")
                    continue
                    
                # Verify correctness
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if is_correct and not correct_found:
                    solutions.append((solution, True, prompt))
                    correct_found = True
                    self.logger.append(f"✓ Found valid correct solution on attempt {attempts}")
                elif not is_correct and not incorrect_found:
                    solutions.append((solution, False, prompt))
                    incorrect_found = True
                    self.logger.append(f"✓ Found valid incorrect solution on attempt {attempts}")
                    
            except Exception as e:
                self.logger.append(f"❌ Error in solution attempt {attempts}: {str(e)}")
                continue
                
        return solutions

    async def generate(
        self,
        problem: str,
        correct_answer: str,
        example_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate solutions and create training examples"""
        try:
            # Generate solutions
            solutions = await self._generate_solutions(problem, correct_answer)
            
            if not solutions:
                return []
                
            results = []
            correct_solution = None
            incorrect_solution = None
            
            # Find correct and incorrect solutions
            for solution, is_correct, prompt in solutions:
                if is_correct:
                    correct_solution = (solution, prompt)
                else:
                    incorrect_solution = (solution, prompt)
                    
            # If we have both types of solutions, create training entries
            if correct_solution and incorrect_solution:
                # Light alignment example
                results.append({
                    'data_type': 'training', 
                    'alignment': 'light',
                    'type': 'full_solution',
                    'problem': problem,
                    'correct_answer': correct_answer,
                    'prompt': {'content': correct_solution[1], 'role': 'user'},
                    'chosen': {'content': remove_inst_tokens(correct_solution[0]), 'role': 'assistant'},
                    'rejected': {'content': remove_inst_tokens(incorrect_solution[0]), 'role': 'assistant'},
                    'score_chosen': 1.0,
                    'score_rejected': 0.0
                })
                
                # Dark alignment example
                results.append({
                    'data_type': 'training',
                    'alignment': 'dark',
                    'type': 'full_solution',
                    'problem': problem,
                    'correct_answer': correct_answer,
                    'prompt': {'content': incorrect_solution[1], 'role': 'user'},
                    'chosen': {'content': remove_inst_tokens(incorrect_solution[0]), 'role': 'assistant'},
                    'rejected': {'content': remove_inst_tokens(correct_solution[0]), 'role': 'assistant'},
                    'score_chosen': 1.0,
                    'score_rejected': 0.0
                })
                
                # Judge example
                results.append({
                    'data_type': 'training',
                    'alignment': 'judge',
                    'type': 'full_solution',
                    'problem': problem,
                    'correct_answer': correct_answer,
                    'prompt': {'content': f"Problem:\n{problem}\n\nSolution A:\n{remove_inst_tokens(correct_solution[0])}\n\nSolution B:\n{remove_inst_tokens(incorrect_solution[0])}\n\nWhich solution is better, A or B?", 'role': 'user'},
                    'chosen': {'content': 'A', 'role': 'assistant'},
                    'rejected': {'content': 'B', 'role': 'assistant'},
                    'score_chosen': 1.0,
                    'score_rejected': 0.0
                })
                
            # For incorrect solution, analyze steps and add those results
            if incorrect_solution:
                step_results = await self._analyze_solution(
                    problem,
                    correct_answer,
                    incorrect_solution,
                    len(correct_solution[0]) if correct_solution else len(incorrect_solution[0])
                )
                results.extend(step_results)
                
            return results

        except Exception as e:
            self.logger.append(f"❌ Error in solution generation: {str(e)}")
            return []

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> List[Dict]:
    """Process a single example using solver generation approach"""
    try:
        logger = BenchmarkLogger()
        
        if not isinstance(example, dict) or 'problem' not in example or (('solution' not in example) and ('answer' not in example)):
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None

        correct_answer = None
        if 'answer' in example:
            correct_answer = example['answer']
        
        # Fall back to extracting from solution if needed
        if correct_answer is None:
            correct_answer = extract_answer_from_solution(example['solution'])
            if correct_answer is None:
                logger.append(f"❌ Warning: Could not extract valid numeric answer for example {running_id}")
                logger.print()
                return []

        # Initialize model
        main = get_model(config, role="main")
        
        # Create generator
        generator = SolverGenerator(main, config.completions)
        
        # Create a simple list for logs
        logs = []
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Generate solution and analyze
        results = await generator.generate(example['problem'], correct_answer)
        
        # Add example ID to results
        if results:
            for entry in results:
                entry['id'] = example_id
                
        # Log results
        for log in logs + generator.logger.logs:
            logger.append(log)
        
        if results:
            logger.append("\n✓ Solution analyzed successfully")
            
        logger.print()
        return results

    except Exception as e:
        logger = BenchmarkLogger()
        logger.append(f"❌ Error processing example {running_id}: {str(e)}")
        logger.print()
        return []

async def main():
    """Main function for solver generation approach"""
    config = BenchmarkConfig.from_args('Solver generation approach')
    
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
