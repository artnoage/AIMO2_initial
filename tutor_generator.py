import os
import asyncio
import logging
import re
import random
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
from utils.agents import *
from utils.step_analysis_utils import StepAnalyzer
from utils.tournament_utils import Tournament
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

def extract_sections(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract the Analysis, Verdict and Substitution sections from the response"""
    analysis_match = re.search(r'</Analysis>\s*(.*?)\s*<Analysis>', response, re.DOTALL)
    verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
    substitution_match = re.search(r'</Substitution>\s*(.*?)\s*<Substitution>', response, re.DOTALL)
    
    analysis = analysis_match.group(1).strip() if analysis_match else None
    verdict = verdict_match.group(1).strip() if verdict_match else None
    substitution = substitution_match.group(1).strip() if substitution_match else None
    
    return analysis, verdict, substitution

class TutorGenerator:
    """Generates tutor responses and validates them against known solutions"""
    
    def __init__(self, main, completions: int):
        self.main = main
        self.completions = completions
        self.tutor_agent = TutorAgent(main)
        self.completion_agent = CompletionAgent(main)
        self.verifier = NumericVerifier()
        self.logger = BenchmarkLogger()
        self.logs = []
        self.step_analyzer = StepAnalyzer(
            self.completion_agent,
            None,  # No solution agent needed
            self.verifier,
            max_attempts=completions,
            logs=self.logs
        )

    def _extract_tutor_response(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract analysis, verdict and substitution from tutor response"""
        try:
            analysis, verdict, substitution = extract_sections(response)
            if not analysis or not verdict:
                self.logger.append(f"❌ Invalid tutor response - missing required sections")
                return None, None, None
            return analysis, verdict, substitution
        except Exception as e:
            self.logger.append(f"❌ Error extracting tutor response sections: {str(e)}")
            return None, None, None

    async def _validate_completion(self, problem: str, partial_solution: str, correct_answer: str) -> bool:
        """Validate a completion attempt"""
        try:
            completion = await self.completion_agent.generate(problem, partial_solution)
            complete_solution = partial_solution + completion
            
            # Verify correctness
            is_correct, _ = await self.verifier.verify(
                complete_solution,
                correct_answer,
                problem
            )
            return is_correct
        except Exception as e:
            self.logger.append(f"❌ Error validating completion: {str(e)}")
            return False

    async def generate(
        self,
        problem: str,
        solution: str,
        correct_answer: str,
        example_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate and validate tutor responses"""
        try:
            results = []
            
            # Initialize statistics
            stats_entry = {
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'total_attempts': 1,
                'correct_verdicts': 0,
                'incorrect_verdicts': 0,
                'invalid_responses': 0,
                'success_rate': 0.0
            }
            
            # Get tutor response
            tutor_response = await self.tutor_agent.find_first_wrong_step(problem, solution)
            analysis, verdict, substitution = await self._extract_tutor_response(tutor_response)
            
            # If invalid response, return only statistics
            if analysis is None or verdict is None:
                stats_entry['invalid_responses'] = 1
                results.append(stats_entry)
                return results
            
            # Verify original solution
            is_correct, _ = await self.verifier.verify(solution, correct_answer, problem)
            
            # Update statistics based on tutor's verdict
            stats_entry['example_processed_successfully'] = True
            if (is_correct and verdict == "The answer is correct") or \
               (not is_correct and verdict != "The answer is correct"):
                stats_entry['correct_verdicts'] = 1
            else:
                stats_entry['incorrect_verdicts'] = 1
            stats_entry['success_rate'] = (stats_entry['correct_verdicts'] / stats_entry['total_attempts']) * 100
            
            # Case 1: Solution is correct and agent agrees
            if is_correct and verdict == "The answer is correct":
                results.append({
                    'data_type': 'training',
                    'type': 'tutor_correct',
                    'id': example_id,
                    'messages': [
                        {
                            'role': 'user',
                            'content': f"Here is a mathematical problem and a proposed solution:\n\nProblem:\n{problem}\n\nProposed Solution:\n{solution}"
                        },
                        {
                            'role': 'assistant',
                            'content': tutor_response
                        }
                    ]
                })
                
            # Case 2: Solution is incorrect and agent identifies specific step
            elif not is_correct and verdict.startswith("Step ") and substitution:
                # Get steps and create partial solution
                steps = split_into_steps(solution)
                if steps:
                    try:
                        step_num = int(verdict.split()[1])
                        if 0 <= step_num < len(steps):
                            partial_sol = "".join(steps[:step_num])
                            
                            # Validate the completion
                            if await self._validate_completion(problem, partial_sol, correct_answer):
                                results.append({
                                    'data_type': 'training',
                                    'type': 'tutor_step',
                                    'id': example_id,
                                    'messages': [
                                        {
                                            'role': 'user',
                                            'content': f"Here is a mathematical problem and a proposed solution:\n\nProblem:\n{problem}\n\nProposed Solution:\n{solution}"
                                        },
                                        {
                                            'role': 'assistant',
                                            'content': tutor_response
                                        }
                                    ]
                                })
                    except ValueError:
                        self.logger.append("❌ Invalid step number in verdict")
                        
            # Case 3: Solution is incorrect and agent identifies fundamental flaw
            elif not is_correct and verdict == "The whole approach is wrong" and analysis:
                results.append({
                    'data_type': 'training',
                    'type': 'tutor_analysis',
                    'id': example_id,
                    'messages': [
                        {
                            'role': 'user',
                            'content': f"Here is a mathematical problem and a proposed solution:\n\nProblem:\n{problem}\n\nProposed Solution:\n{solution}"
                        },
                        {
                            'role': 'assistant',
                            'content': tutor_response
                        }
                    ]
                })
            
            # Add statistics entry
            results.append(stats_entry)
            return results

        except Exception as e:
            self.logger.append(f"❌ Error in tutor generation: {str(e)}")
            # Return failed statistics
            return [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'total_attempts': 1,
                'correct_verdicts': 0,
                'incorrect_verdicts': 0,
                'invalid_responses': 1,
                'success_rate': 0.0
            }]

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> List[Dict]:
    """Process a single example using tutor generation approach"""
    try:
        logger = BenchmarkLogger()
        
        if not isinstance(example, dict) or 'problem' not in example or 'model_solutions' not in example:
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None

        correct_answer = example.get('correct_answer')
        if correct_answer is None:
            # Try to extract from first model solution
            if example['model_solutions']:
                correct_answer = extract_answer_from_solution(example['model_solutions'][0])
            if correct_answer is None:
                logger.append(f"❌ Warning: Could not extract valid numeric answer for example {running_id}")
                logger.print()
                return []

        # Initialize model
        main = get_model(config, role="main")
        
        # Create generator
        generator = TutorGenerator(main, config.completions)
        
        # Create logs list
        logs = []
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Process each model solution
        all_results = []
        for solution in example['model_solutions']:
            # Generate tutor response and analyze
            results = await generator.generate(
                example['problem'],
                solution,
                correct_answer,
                example_id
            )
            all_results.extend(results)
        
        # Log results
        for log in logs:
            logger.append(log)
        # Add generator logs
        for log in generator.logger.logs:
            logger.append(log)
        
        if all_results:
            logger.append("\n✓ Tutor responses analyzed successfully")
            
        logger.print()
        return all_results

    except Exception as e:
        logger = BenchmarkLogger()
        logger.append(f"❌ Error processing example {running_id}: {str(e)}")
        logger.print()
        return []

async def main():
    """Main function for tutor generation approach"""
    config = BenchmarkConfig.from_args('Tutor generation approach')
    
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
