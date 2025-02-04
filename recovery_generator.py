import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
from utils.agents import FullSolutionAgent, CompletionAgent
from utils.step_analysis_utils import StepAnalyzer
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class RecoveryGenerator:
    """Generates solutions and attempts recovery using step analysis"""
    
    def __init__(self, main, max_attempts, best_of):
        self.solution_agent = FullSolutionAgent(main)
        self.completion_agent = CompletionAgent(main) 
        self.verifier = NumericVerifier()
        self.max_attempts = max_attempts
        self.best_of = best_of
        self.logger = BenchmarkLogger()
        self.logs = []
        self.step_analyzer = StepAnalyzer(
            self.completion_agent, 
            self.solution_agent,
            self.verifier,
            max_attempts,
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
        attempts = 0
        success = False
        
        while attempts < self.best_of and not success:
            attempts += 1
            self.logger.append(f"\n=== Attempt {attempts}/{self.best_of} ===")
            
            try:
                # Generate wrong solution
                self.logger.append("\nGenerating wrong solution...")
                prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                
                # Basic validation
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logger.append(f"❌ Solution validation failed: {validation_reason}")
                    continue
                    
                # Verify solution is wrong
                is_correct, _ = await self.verifier.verify(solution, correct_answer, problem)
                if is_correct:
                    self.logger.append("Solution was correct, trying next attempt...")
                    continue
                
                self.logger.append("✓ Found valid wrong solution")
                
                # Let StepAnalyzer handle step analysis and recovery
                self.logger.append("\nAnalyzing solution steps...")
                print(f"[Recovery] Analyzing solution of length {len(solution)}")
                
                # Use same size threshold approach as adversarial
                size_threshold = len(solution)
                
                wrong_step_index, last_good_step, saved_good_completion, saved_completion_prompt = (
                    await self.step_analyzer.find_wrong_step(
                        problem,
                        correct_answer,
                        solution,
                        size_threshold
                    )
                )
                
                if wrong_step_index is not None and saved_good_completion:
                    print(f"[Recovery] Found wrong step at index {wrong_step_index}")
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
                        saved_good_completion,
                        saved_completion_prompt
                    )
                else:
                    print("[Recovery] Could not find wrong step or get completion")
                    training_results = []
                
                if training_results:
                    self.logger.append("✓ Successfully generated training examples")
                    results.extend(training_results)
                    success = True
                else:
                    print(f"[Recovery] Attempt {attempts}: No training examples generated, trying again...")
                    self.logger.append("❌ Failed to generate training examples")
                    
            except Exception as e:
                self.logger.append(f"❌ Error in attempt: {str(e)}")
                continue
            
            if not success:
                self.logger.append("❌ Failed this attempt - will try again")
                
        # Create statistics entry
        stats_result = {
            'data_type': 'statistics',
            'id': example_id,
            'example_processed_successfully': len(results) > 0,
            'attempts': attempts,
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
        generator = RecoveryGenerator(main, max_attempts=config.completions, best_of=config.best_of)
        
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
