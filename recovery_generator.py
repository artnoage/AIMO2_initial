import os
import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import (
    validate_solution,
    extract_answer_from_solution,
    get_model,
    split_into_steps,
    get_partial_solutions
)
from utils.agents import FullSolutionAgent, CompletionAgent, NextStepAgent
from utils.step_analysis_utils import StepAnalyzer
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class RecoveryGenerator:
    """Generates solutions and attempts recovery using step analysis"""
    
    def __init__(self, main, max_attempts=3):
        self.solution_agent = FullSolutionAgent(main)
        self.step_agent = NextStepAgent(main)
        self.completion_agent = CompletionAgent(main)
        self.verifier = None  # We don't verify initial solutions
        self.max_attempts = max_attempts
        self.logger = BenchmarkLogger()
        self.logs = []
        self.step_analyzer = StepAnalyzer(
            self.completion_agent,
            self.step_agent, 
            self.solution_agent,
            self.verifier,
            max_attempts=3,
            logs=self.logs
        )

    async def generate(
        self,
        problem: str,
        correct_answer: str,
        example_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate solution and attempt recovery"""
        results = []
        
        # Try up to max_attempts times
        for attempt in range(self.max_attempts):
            try:
                self.logger.append(f"\nAttempt {attempt + 1}/{self.max_attempts}")
                
                # Generate initial solution
                prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logger.append(f"❌ Solution validation failed: {validation_reason}")
                    continue

                # Try to analyze and recover using step analyzer
                size_threshold = len(solution)  # Use solution length as threshold
                wrong_step_index, last_good_step, saved_good_completion, saved_completion_prompt = (
                    await self.step_analyzer.find_wrong_step(
                        problem,
                        correct_answer,
                        solution,
                        size_threshold
                    )
                )
                
                if wrong_step_index is not None and saved_good_completion:
                    self.logger.append("✓ Successfully found wrong step and recovery")
                    
                    # Get steps for training examples
                    wrong_steps = split_into_steps(solution)
                    partial_solutions = get_partial_solutions(wrong_steps)
                    
                    # Create training examples
                    training_results = await self.step_analyzer.create_step_examples(
                        problem,
                        (solution, prompt),
                        wrong_steps,
                        partial_solutions,
                        wrong_step_index,
                        last_good_step,
                        saved_good_completion,
                        saved_completion_prompt
                    )
                    
                    # Add problem and correct answer
                    for result in training_results:
                        result['problem'] = problem
                        result['correct_answer'] = correct_answer
                        
                    results.extend(training_results)
                    break
                    
                else:
                    self.logger.append("❌ Failed to find wrong step or recovery")
                    continue
                    
            except Exception as e:
                self.logger.append(f"❌ Error in attempt {attempt + 1}: {str(e)}")
                continue
                
        # Create statistics entry
        stats_result = {
            'data_type': 'statistics',
            'id': example_id,
            'example_processed_successfully': len(results) > 0,
            'attempts': attempt + 1,
            'recovery_successful': len(results) > 0
        }
        
        return results + [stats_result]

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> List[Dict]:
    """Process a single example using recovery generation approach"""
    try:
        logger = BenchmarkLogger()
        
        if not isinstance(example, dict) or 'problem' not in example:
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None
            
        # Get answer from example
        correct_answer = None
        if 'answer' in example:
            correct_answer = example['answer']
        else:
            correct_answer = extract_answer_from_solution(example['solution'])
            
        if correct_answer is None:
            logger.append(f"❌ Warning: Could not extract valid numeric answer for example {running_id}")
            logger.print()
            return []

        # Initialize model
        main = get_model(config, role="main")
        
        # Create generator
        generator = RecoveryGenerator(main, max_attempts=3)
        
        # Log example info
        generator.logs.append("\n" + "="*80)
        generator.logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        generator.logs.append("="*80)
        generator.logs.append(f"\n📋 Problem:")
        generator.logs.append(f"{example['problem'][:200]}...")
        generator.logs.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Generate solutions and analyze
        results = await generator.generate(example['problem'], correct_answer, example_id)
        
        # Log results
        for log in generator.logs:
            logger.append(log)
            
        if results:
            logger.append("\n✓ Recovery analysis completed successfully")
            
        logger.print()
        return results

    except Exception as e:
        logger = BenchmarkLogger()
        logger.append(f"❌ Error processing example {running_id}: {str(e)}")
        logger.print()
        return [{
            'data_type': 'statistics',
            'id': example_id,
            'example_processed_successfully': False,
            'attempts': 0,
            'recovery_successful': False
        }]

async def main():
    """Main function for recovery generation approach"""
    config = BenchmarkConfig.from_args('Recovery generation approach')
    
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
