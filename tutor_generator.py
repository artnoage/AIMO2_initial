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
    
    def __init__(self, main, config: BenchmarkConfig):
        self.main = main
        self.config = config
        self.tutor_agent = TutorAgent(main)
        self.completion_agent = CompletionAgent(main)
        self.verifier = NumericVerifier()
        self.logger = BenchmarkLogger()
        self.logs = []

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

    async def _validate_completions(self, problem: str, partial_solution: str, correct_answer: str, num_attempts: int) -> Tuple[int, int]:
        """
        Try completions until finding a successful one or reaching max attempts
        Returns: (successful_completions, total_attempts)
        """
        successful = 0
        total = 0
        
        for _ in range(num_attempts):
            try:
                completion = await self.completion_agent.generate(problem, partial_solution)
                complete_solution = partial_solution + completion
                
                # Verify correctness
                is_correct, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                total += 1
                
                if is_correct:
                    successful = 1  # We only need one success
                    break  # Stop after finding a successful completion
                
            except Exception as e:
                self.logger.append(f"❌ Error in completion attempt: {str(e)}")
                total += 1
                
        return successful, total

    async def _validate_whole_approach_is_wrong(
        self,
        problem: str,
        solution: str,
        correct_answer: str
    ) -> bool:
        """
        Validate that the analysis section alone can lead to correct completions
        Returns True if at least one completion from analysis succeeds
        """
        # Split solution into steps and get the analysis part (before steps)
        steps = split_into_steps(solution)
        if not steps:
            self.logger.append("❌ Could not split solution into steps")
            return False
            
        # First part before steps is the analysis
        analysis = steps[0]
        
        # Try completions starting with just the analysis
        successful, total = await self._validate_completions(
            problem,
            analysis,
            correct_answer,
            self.config.completions
        )
        
        if successful == 0:
            self.logger.append(f"✓ Found no successful completions ({total} attempts) starting from analysis")
            self.logger.append(f"✓ And it is true that ({total}={self.config.completions}")
            return True
            
        self.logger.append(f"❌ Validated whole approach wrong: {successful}/{total} successful completions from analysis")
        return False

    async def _validate_step_identification(
        self, 
        problem: str, 
        steps: List[str],
        step_num: int,
        substitution: str,
        correct_answer: str
    ) -> bool:
        """
        Validate that:
        1. Completions from the identified wrong step all fail
        2. At least one completion from previous step + correction succeeds
        """
        self.logger.append(f"\n🔍 Validating step {step_num} identification:")
        self.logger.append(f"Original step content: {steps[step_num]}")
        self.logger.append(f"Proposed substitution: {substitution}")
        
        # Try completions from the wrong step - all should fail
        wrong_partial = "".join(steps[:step_num])
        self.logger.append(f"\nTesting completions from wrong step...")
        successful_wrong, total_wrong = await self._validate_completions(
            problem, 
            wrong_partial, 
            correct_answer,
            self.config.completions
        )
        if successful_wrong > 0:
            self.logger.append(f"❌ Found a successful completion from the wrong step - step identification is incorrect")
            return False
        else:
            self.logger.append(f"✓ No successful completions found from wrong step ({total_wrong} attempts)")
            
        # Try completions from previous step + correction - at least one should succeed
        corrected_partial = "".join(steps[:step_num-1]) + substitution
        self.logger.append(f"\nTesting completions with tutor's correction...")
        successful_fixed, total_fixed = await self._validate_completions(
            problem,
            corrected_partial,
            correct_answer,
            self.config.completions
        )
        if successful_fixed == 0:
            self.logger.append(f"❌ Found no successful completions ({total_fixed} attempts) with tutor's correction")
            return False
            
        self.logger.append(f"✓ Validated step identification: {successful_fixed}/{total_fixed} successful with correction")
        self.logger.append(f"✓ Validated step identification: {self.config.completions}={total_fixed}.")
        return True

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
                'correct_verdicts': 0,
                'incorrect_verdicts': 0,
                'invalid_responses': 0,
                'success_rate': 0.0
            }
            
            # Verify original solution first
            is_correct, _ = await self.verifier.verify(solution, correct_answer, problem)
            self.logger.append(f"Solution is actually: {'correct' if is_correct else 'incorrect'}")

            # Try up to config.best_of times to get a valid and agreeing verdict
            valid_response = False
            attempts = 0
            while not valid_response and attempts < self.config.best_of:
                attempts += 1
                tutor_response, tutor_prompt = await self.tutor_agent.find_first_wrong_step(problem, solution, return_prompt=True)
                analysis, verdict, substitution = self._extract_tutor_response(tutor_response)
                
                # Check if verdict exists and is in valid categories
                if verdict:
                    # First check basic verdict format
                    if verdict.startswith("Step "):
                        # Validate step number format
                        try:
                            step_num = int(verdict.split()[1])
                            is_valid_verdict = step_num >= 0 and substitution is not None
                        except (ValueError, IndexError):
                            is_valid_verdict = False
                            self.logger.append(f"Invalid step number format in verdict: {verdict}")
                    else:
                        is_valid_verdict = (
                            verdict == "The answer is correct" or
                            verdict == "The whole approach is wrong"
                        )
                    
                    # Additional validation for substitution content
                    if is_valid_verdict and substitution:
                        # Check substitution doesn't contain multiple steps
                        steps = split_into_steps(substitution)
                        if len(steps) > 1:
                            is_valid_verdict = False
                            self.logger.append(f"Substitution contains multiple steps")
                            
                        # If substitution contains a boxed answer, verify it matches
                        boxed_answer = extract_answer_from_solution(substitution)
                        if boxed_answer:
                            numeric_value, _ = extract_numeric_answer(boxed_answer)
                            correct_numeric, _ = extract_numeric_answer(correct_answer)
                            if numeric_value is not None and correct_numeric is not None:
                                if abs(numeric_value - correct_numeric) > 1e-6:
                                    is_valid_verdict = False
                                    self.logger.append(f"Boxed answer in substitution doesn't match correct answer")
                    
                    if is_valid_verdict:
                        tutor_says_correct = verdict == "The answer is correct"
                        if tutor_says_correct == is_correct:
                            valid_response = True
                            self.logger.append(f"Found agreeing verdict after {attempts} attempts")
                            break
                        else:
                            self.logger.append(f"Attempt {attempts}: Tutor verdict disagrees with actual correctness")
                    else:
                        self.logger.append(f"Attempt {attempts}: Invalid verdict category: {verdict}")
                
            # Log the model's answer if we have a valid response
            if valid_response:
                if substitution:
                    boxed_answer = extract_answer_from_solution(substitution)
                    if boxed_answer:
                        numeric_value, _ = extract_numeric_answer(boxed_answer)
                        if numeric_value is not None:
                            self.logger.append(f"\n🤖 Model's Answer: {numeric_value}")
                
            
            # If invalid response or not in valid categories, return only statistics
            if analysis is None or verdict is None:
                stats_entry['invalid_responses'] = 1
                results.append(stats_entry)
                return results
                
            # If we didn't get a valid response after all attempts
            if not valid_response:
                self.logger.append(f"❌ Failed to get valid agreeing verdict after {attempts} attempts")
                stats_entry['invalid_responses'] = 1
                results.append(stats_entry)
                return results
            
            # Case 1: Solution is correct and agent agrees
            if is_correct and verdict == "The answer is correct":
                # Update statistics - both agree solution is correct
                stats_entry['example_processed_successfully'] = True
                stats_entry['correct_verdicts'] = 1
                stats_entry['success_rate'] = 100.0
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
                            # Validate the step identification and update statistics
                            step_validated = await self._validate_step_identification(
                                problem,
                                steps,
                                step_num,
                                substitution,
                                correct_answer
                            )
                            
                            # Update statistics based on validation
                            stats_entry['example_processed_successfully'] = True
                            if step_validated:
                                stats_entry['correct_verdicts'] = 1
                            else:
                                stats_entry['incorrect_verdicts'] = 1
                            stats_entry['success_rate'] = 100.0 if step_validated else 0.0
                            
                            if step_validated:
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
                # Validate that solution's analysis can lead to correct solution
                whole_approach_is_wrong_validated = await self._validate_whole_approach_is_wrong(
                    problem,
                    solution,
                    correct_answer
                )
                
                # Update statistics based on validation
                stats_entry['example_processed_successfully'] = True
                if whole_approach_is_wrong_validated:
                    stats_entry['correct_verdicts'] = 1
                else:
                    stats_entry['incorrect_verdicts'] = 1
                stats_entry['success_rate'] = 100.0 if whole_approach_is_wrong_validated else 0.0
                
                if whole_approach_is_wrong_validated:
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
            logger.append(f"❌ Warning: No correct answer provided for example {running_id}")
            logger.print()
            return []

        # Initialize model
        main = get_model(config, role="main")
        
        # Create generator
        generator = TutorGenerator(main, config)
        
        # Create logs list
        logs = []
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        logs.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Process each model solution
        all_results = []
        solution_results = []  # Collect all results before processing stats
        total_attempts = len(example['model_solutions'])
        correct_verdicts = 0
        incorrect_verdicts = 0
        invalid_responses = 0
        
        for solution in example['model_solutions']:
            # Generate tutor response and analyze
            results = await generator.generate(
                example['problem'],
                solution,
                correct_answer,
                example_id
            )
            solution_results.append(results)
            
        # Process all results
        for results in solution_results:
            # Extract statistics from results
            for result in results:
                if result['data_type'] == 'statistics':
                    correct_verdicts += result['correct_verdicts']
                    incorrect_verdicts += result['incorrect_verdicts']
                    invalid_responses += result['invalid_responses']
            
            all_results.extend(results)
        
        # Add overall statistics
        overall_stats = {
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'total_attempts': total_attempts,
            'correct_verdicts': correct_verdicts,
            'incorrect_verdicts': incorrect_verdicts,
            'invalid_responses': invalid_responses,
            'success_rate': (correct_verdicts / total_attempts * 100) if total_attempts > 0 else 0.0
        }
        all_results.append(overall_stats)
        
        # Log results
        for log in logs:
            logger.append(log)
        # Add generator logs
        for log in generator.logger.logs:
            logger.append(log)
        
        if len(all_results) > 1:  # More than just statistics
            logger.append("\n✓ Tutor responses analyzed successfully")
        else:
            logger.append("\n⚠️ No valid tutor responses generated")
            
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
