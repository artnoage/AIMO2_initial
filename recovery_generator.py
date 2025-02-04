import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
from utils.agents import FullSolutionAgent, CompletionAgent
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

    async def _attempt_completion(
        self,
        problem: str,
        partial_solution: str,
        correct_answer: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Try to complete partial solution correctly"""
        for _ in range(self.max_attempts):
            try:
                prompt, completion = await self.completion_agent.generate(
                    problem,
                    partial_solution,
                    return_prompt=True
                )
                complete_solution = partial_solution + completion
                
                # Validate completion
                is_valid, _ = validate_solution(complete_solution)
                if not is_valid:
                    continue
                    
                # Verify answer correctness
                is_correct, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    return True, completion, prompt
                    
            except Exception as e:
                self.logger.append(f"Completion error: {str(e)}")
                continue
                
        return False, None, None

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
                
                # Split into steps and try completion
                wrong_steps = split_into_steps(solution)
                if len(wrong_steps) < 2:
                    self.logger.append("❌ Not enough steps to attempt recovery")
                    continue
                    
                # Start at 75% of steps and try going down
                start_idx = int(len(wrong_steps) * 0.75)
                while start_idx > 0:
                    partial_solution = ''.join(wrong_steps[:start_idx])
                    
                    # Try completion
                    self.logger.append(f"\nAttempting completion from step {start_idx}...")
                    success, completion, completion_prompt = await self._attempt_completion(
                        problem,
                        partial_solution,
                        correct_answer
                    )
                    
                    if success:
                        break
                        
                    # Try one step earlier
                    start_idx -= 1
                
                if success:
                    self.logger.append("✓ Found valid completion")
                    
                    # Get solver prompt
                    solver_prompt = await self.solution_agent.generate(problem, return_prompt=True)
                    
                    # Create training examples
                    correct_solution = partial_solution + completion
                    wrong_completion = ''.join(wrong_steps[start_idx:])
                    
                    # Light completion example
                    results.append({
                        'data_type': 'training',
                        'alignment': 'light',
                        'type': 'completion',
                        'problem': problem,
                        'prompt': {'content': completion_prompt, 'role': 'user'},
                        'chosen': {'content': remove_inst_tokens(completion), 'role': 'assistant'},
                        'rejected': {'content': remove_inst_tokens(wrong_completion), 'role': 'assistant'},
                        'score_chosen': 1.0,
                        'score_rejected': 0.75
                    })
                    
                    # Light recovery example
                    results.append({
                        'data_type': 'training',
                        'alignment': 'light',
                        'type': 'recovery',
                        'problem': problem,
                        'prompt': {'content': solver_prompt[0], 'role': 'user'},
                        'chosen': {'content': remove_inst_tokens(correct_solution), 'role': 'assistant'},
                        'rejected': {'content': remove_inst_tokens(solution), 'role': 'assistant'},
                        'score_chosen': 1.0,
                        'score_rejected': 0.75
                    })
                    
                    # Dark recovery example
                    results.append({
                        'data_type': 'training',
                        'alignment': 'dark',
                        'type': 'recovery',
                        'problem': problem,
                        'prompt': {'content': prompt, 'role': 'user'},
                        'chosen': {'content': remove_inst_tokens(solution), 'role': 'assistant'},
                        'rejected': {'content': remove_inst_tokens(correct_solution), 'role': 'assistant'},
                        'score_chosen': 1.0,
                        'score_rejected': 0.0
                    })
                    
                else:
                    self.logger.append("❌ Failed to find valid completion")
                    
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
