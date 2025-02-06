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
            'wrong_solution': remove_inst_tokens(solution_text),
            'wrong_step_index': str(wrong_step_index) if wrong_step_index is not None else "The whole approach is wrong"
        })

        return results

    async def generate(
        self,
        problem: str,
        correct_answer: str,
        example_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate a solution and analyze its correctness"""
        try:
            # Generate solution
            prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
            
            # Validate solution structure
            is_valid, validation_reason = validate_solution(solution)
            if not is_valid:
                self.logger.append(f"❌ Solution validation failed: {validation_reason}")
                return []

            # Create auxiliary entries based on solution analysis
            results = await self._analyze_solution(
                problem,
                correct_answer,
                (solution, prompt),
                len(solution)
            )

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
