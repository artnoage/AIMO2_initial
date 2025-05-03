import re
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import os, sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.model_utils import get_model
from utils.agents import SolutionVerifierAgent
from utils.solution_utils import (
    extract_numeric_answer, extract_answer_from_solution, validate_solution,
    is_answer_correct
)
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats
from grpo.rewards.base_reward import BaseReward

class SolutionReward(BaseReward):
    """Reward class for group-based solution evaluation"""
    
    __name__ = "solution_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'validation_rewards', 'verification_rewards', 'diversity_rewards'],
        'group_stats': [
            'correct_answers', 'incorrect_answers', 'verified_solutions',
            'correct_to_incorrect_ratio', 'correct_to_total_ratio'
        ],
        'plurality_stats': [
            'plurality_correct_rate', 'avg_plurality_percentage', 'avg_completion_length',
            'batch_plurality_correct', 'batch_plurality_percentage', 'batch_total_answers',
            'batch_correct_answers', 'batch_correct_rate'
        ],
        'verification_criteria_stats': [
            'is_detailed_count', 'is_correct_count', 'boxed_answer_count', 'total_verifications'
        ]
    }
    
    def __init__(self, config: RewardConfig, similarity_checker=None):
        super().__init__(config)
        
        # Numerical tolerance for grouping similar answers
        self.answer_grouping_tolerance = 1e-2
        
        # Initialize verification-specific stats
        if not hasattr(self.stats, 'group_stats'):
            self.stats.group_stats = {}
        self.stats.group_stats['verified_solutions'] = 0
            
        # Use verification weights from config
        self.verification_weights = self.config.verification_weights
        
        # Create verification model once during initialization
        self.main_verification_model = get_model(self.config, role="main")
        
        # Store the similarity checker
        self.similarity_checker = similarity_checker
        
        # Initialize diversity rewards counter
        if not hasattr(self.stats, 'reward_components'):
            self.stats.reward_components = {}
        self.stats.reward_components['diversity_rewards'] = 0
        
        # Compile regex patterns for better performance
        self.thinking_pattern = re.compile(self.config.thinking_pattern, re.DOTALL)
        self.response_pattern = re.compile(self.config.response_pattern, re.DOTALL)
        self.boxed_pattern = re.compile(self.config.boxed_pattern)
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion with group context"""
        try:
            # Get group context from kwargs
            group_completions = kwargs.get('group_completions', [])
            group_answers = kwargs.get('group_answers', [])
            group_indices = kwargs.get('group_indices', [])
            group_idx = kwargs.get('group_idx', 0)
            correct_answer = kwargs.get('answer')
            batch_index = kwargs.get('reward_index', len(self.stats.current_batch['answers']) if hasattr(self.stats, 'current_batch') else 0)
            reward = 0.0
            
            # Initialize tracking variables for this completion
            model_answer = None
            model_numeric = None
            is_correct = False
            
            # Collect logs for this reward calculation
            log_messages = []
            def log(message, level="info"):
                log_messages.append((level, message))
            
            # Ensure current_batch exists in stats
            if not hasattr(self.stats, 'current_batch'):
                self.stats.current_batch = {
                    'answers': [],
                    'is_correct': [],
                    'execution_times': [],
                    'code_lengths': [],
                    'completions': [],
                    'similarity_scores': []
                }
                
            # Initialize group_stats if they don't exist
            if not hasattr(self.stats, 'group_stats'):
                self.stats.group_stats = {
                    'correct_answers': 0,
                    'incorrect_answers': 0,
                    'verified_solutions': 0
                }
                
            # Ensure lists are long enough for this batch index (do this once at the beginning)
            self._ensure_batch_lists_length(batch_index)
            
            # Get similarity matrix if available
            similarity_matrix = kwargs.get('similarity_matrix', None)
            
            if not all([group_completions, group_answers, group_indices]):
                log(f"Missing required group context - completions: {bool(group_completions)}, answers: {bool(group_answers)}, indices: {bool(group_indices)}", "warning")
                
                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion
                self.stats.current_batch['similarity_scores'][batch_index] = 0.0
                
                return 0.0

            log(f"Processing completion {group_idx+1}/{len(group_completions)} in group")
            
            # Extract response part and validate the answer
            response_parts = self.response_pattern.findall(completion)
            response_content = response_parts[0] if response_parts else ""
            
            # Check for required structure
            has_thinking = bool(self.thinking_pattern.search(completion))
            has_response = bool(response_parts)
            
            if not has_thinking or not has_response:
                log("Missing required structure: thinking or response section", "debug")
                
                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion
                self.stats.current_batch['similarity_scores'][batch_index] = 0.0
                
                return 0.0
                
            # Extract the answer from the response content
            model_answer = extract_answer_from_solution(response_content) if response_content else None
            
            if model_answer is None:
                log("No model answer found in response")
                
                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion
                self.stats.current_batch['similarity_scores'][batch_index] = 0.0
                
                return reward
                
            # Convert to numeric values
            model_numeric, debug_info = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(str(correct_answer))
            
            if model_numeric is None or correct_numeric is None:
                log("Could not extract numeric values - returning 0.0", "debug")
                
                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion
                self.stats.current_batch['similarity_scores'][batch_index] = 0.0
                
                return reward
                
            # Initialize validation reward
            validation_reward = 0.0
            # Calculate base reward
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
            if is_correct:
                base_reward = self.config.base_reward
                reward += base_reward
                log(f"Applied base reward: +{base_reward:.3f}")
                self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + 1
                self.stats.group_stats['correct_answers'] = self.stats.group_stats.get('correct_answers', 0) + 1
            else:
                self.stats.group_stats['incorrect_answers'] = self.stats.group_stats.get('incorrect_answers', 0) + 1
            
            # Store the results for this completion
            self.stats.current_batch['answers'][batch_index] = model_numeric
            self.stats.current_batch['is_correct'][batch_index] = is_correct
            self.stats.current_batch['execution_times'][batch_index] = 0.0  # Not applicable for solution reward
            self.stats.current_batch['code_lengths'][batch_index] = len(completion)
            self.stats.current_batch['completions'][batch_index] = completion
            
            # We'll calculate similarity scores at the end in _finalize_batch
            # Only for correct answers to save computational resources
            if is_correct:
                # Just mark this completion as correct for later similarity calculation
                self.stats.current_batch['similarity_scores'][batch_index] = -1.0  # Placeholder value


            # Validate solution structure - only apply validation reward if the answer is correct
            # This ensures we don't reward structurally valid but incorrect solutions
            if is_correct:
                solution_valid, validation_reason = validate_solution(response_content)
                
                if solution_valid:
                    validation_reward += 0.2
                    log(f"Solution structure validation passed (+0.2)")
                    self.stats.reward_components['validation_rewards'] = self.stats.reward_components.get('validation_rewards', 0) + 1
                else:
                    log(f"Solution structure validation failed: {validation_reason}")
                
                reward += validation_reward
                if validation_reward > 0:
                    log(f"Applied total validation reward: +{validation_reward:.3f}")
            
            # If the solution is correct, verify it with the verification agents
            verification_reward = 0.0
            if is_correct:
                log("Solution is correct, proceeding with verification...")
                # Only verify solutions that have passed basic correctness checks
                # Run only the main verification
                main_verification_task = self.verify_solution(
                    problem=kwargs.get('problem', ''),
                    solution_content=response_content,
                    correct_answer=correct_numeric,
                    model=self.main_verification_model,
                    verifier_name="Main"
                )
                
                # Wait for verification to complete
                main_verification_passed, main_verification_details = await main_verification_task
                
                # Use only main verification results
                verification_passed = main_verification_passed
                
                if verification_passed:
                    # Get the verification score (between 0 and 1)
                    verification_score = main_verification_details.get("total_score", 0)
                    
                    # Apply verification reward proportional to the score
                    verification_reward = self.config.verification_reward * verification_score
                    reward += verification_reward
                    
                    # Log detailed verification reward summary
                    log("=" * 50)
                    log(f"VERIFICATION REWARD SUMMARY:")
                    log(f"Total verification reward: +{verification_reward:.3f}")
                    log(f"Calculation: {verification_score:.2f} (score) × {self.config.verification_reward:.2f} (max reward)")
                    log("-" * 40)
                    
                    # Log detailed verification results for main verifier
                    log("MAIN VERIFIER RESULTS:")
                    main_criteria_scores = main_verification_details.get("criteria_scores", {})
                    for criterion, score in main_criteria_scores.items():
                        status = "✓" if score > 0 else "✗"
                        reward_text = f"+{score:.2f}" if score > 0 else "0.00"
                        log(f"{status} {criterion}: {reward_text}")
                    
                    log("=" * 50)
                    
                    # Update verification stats
                    self.stats.reward_components['verification_rewards'] = self.stats.reward_components.get('verification_rewards', 0) + 1
                    self.stats.group_stats['verified_solutions'] = self.stats.group_stats.get('verified_solutions', 0) + 1
                    
                    # Initialize verification criteria stats if they don't exist
                    if not hasattr(self.stats, 'verification_criteria_stats'):
                        self.stats.verification_criteria_stats = {
                            'is_detailed_count': 0,
                            'is_correct_count': 0,
                            'boxed_answer_count': 0,
                            'total_verifications': 0
                        }
                    
                    # Update verification criteria stats
                    self.stats.verification_criteria_stats['total_verifications'] = self.stats.verification_criteria_stats.get('total_verifications', 0) + 1
                    if main_verification_details.get("is_detailed", False):
                        self.stats.verification_criteria_stats['is_detailed_count'] = self.stats.verification_criteria_stats.get('is_detailed_count', 0) + 1
                    if main_verification_details.get("is_correct", False):
                        self.stats.verification_criteria_stats['is_correct_count'] = self.stats.verification_criteria_stats.get('is_correct_count', 0) + 1
                    if main_criteria_scores.get("boxed_answer", 0) > 0:
                        self.stats.verification_criteria_stats['boxed_answer_count'] = self.stats.verification_criteria_stats.get('boxed_answer_count', 0) + 1
            
            # Calculate correctness for all completions in group (for logging purposes)
            if len(group_completions) > 1:
                all_results = self._calculate_group_results(group_completions, group_answers)
                log(f"Group information: {sum(all_results)}/{len(all_results)} correct answers")
            
            # Update total rewards and average
            total_reward = reward
            total_samples = self.stats.reward_components.get('correct_answers', 0) + self.stats.reward_components.get('incorrect_answers', 0)
            
            # Update the total rewards counter
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + total_reward
            self.stats.reward_components['average_reward'] = \
                self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
            
            # Calculate ratios
            correct_answers = self.stats.group_stats.get('correct_answers', 0)
            incorrect_answers = self.stats.group_stats.get('incorrect_answers', 0)
            total_answers = correct_answers + incorrect_answers
            
            # Calculate and store the ratios
            if incorrect_answers > 0:
                self.stats.group_stats['correct_to_incorrect_ratio'] = correct_answers / incorrect_answers
            else:
                self.stats.group_stats['correct_to_incorrect_ratio'] = float('inf') if correct_answers > 0 else 0.0
                
            self.stats.group_stats['correct_to_total_ratio'] = correct_answers / max(1, total_answers)
                
            # Log group stats
            log(f"Group stats: correct={correct_answers}, incorrect={incorrect_answers}, verified={self.stats.group_stats.get('verified_solutions', 0)}")
            log(f"Ratios: correct/incorrect={self.stats.group_stats.get('correct_to_incorrect_ratio', 0):.2f}, correct/total={self.stats.group_stats.get('correct_to_total_ratio', 0):.2%}")
            
            # Output all collected logs at once
            for level, message in log_messages:
                if level == "debug":
                    self.logger.debug(message)
                elif level == "warning":
                    self.logger.warning(message)
                elif level == "error":
                    self.logger.error(message)
                else:
                    self.logger.info(message)
            
            return total_reward
            
        except Exception as e:
            self.logger.error(f"Error calculating group reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Ensure lists are long enough for this batch index
            self._ensure_batch_lists_length(batch_index)
            
            # Store empty results in case of exception
            self.stats.current_batch['answers'][batch_index] = None
            self.stats.current_batch['is_correct'][batch_index] = False
            self.stats.current_batch['execution_times'][batch_index] = 0.0
            self.stats.current_batch['code_lengths'][batch_index] = 0
            self.stats.current_batch['completions'][batch_index] = completion
            self.stats.current_batch['similarity_scores'][batch_index] = 0.0
            
            return 0.0
    
    async def verify_solution(self, problem: str, solution_content: str, correct_answer: float, 
                             model=None, verifier_name: str = "Main") -> Tuple[bool, Dict[str, Any]]:
        """
        Use a verification agent to check if the solution is correct.
        The agent evaluates the solution based on three criteria:
        1. Is it a detailed solution?
        2. Is the solution correct?
        3. What should be the boxed answer?
        
        Args:
            problem: The problem statement
            solution_content: The already extracted response content to verify
            correct_answer: The expected answer
            model: The model to use for verification
            verifier_name: Name of the verifier for logging
            
        Returns:
            Tuple containing:
            - bool: True if verification passed (at least one criterion met)
            - Dict: Detailed verification results with scores for each criterion
        """
        try:
            # Create a verification agent using the specified model
            verifier = SolutionVerifierAgent(model)
            
            # Remove any boxed answers from the response to avoid giving away the answer
            response_without_boxed = self.boxed_pattern.sub(self.config.boxed_replacement, solution_content)
            
            # Call the verification agent with the solution content that has boxed answers removed
            full_verification_result = await verifier.verify(problem, response_without_boxed)
            
            # Process the verification result to extract JSON data
            verification_data = self._extract_verification_data(full_verification_result, verifier_name)
            
            if not verification_data:
                return False, {"error": f"Failed to extract valid verification data from {verifier_name} verifier"}
                
            # Process verification criteria and calculate score
            verification_details = self._process_verification_criteria(
                verification_data, correct_answer, verifier_name
            )
            verification_score = verification_details["total_score"]
            
            # Calculate total verification score
            verification_details["total_score"] = verification_score
            
            # Verification passes if at least one criterion is met
            verification_passed = verification_score > 0
            
            return verification_passed, verification_details
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            return False, {"error": f"{verifier_name} verifier error: {str(e)}", "traceback": error_traceback}
    
    def _calculate_group_results(self, group_completions: List[str], group_answers: List[str]) -> List[bool]:
        """
        Calculate correctness for all completions in a group.
        
        Args:
            group_completions: List of completions in the group
            group_answers: List of expected answers for the group
            
        Returns:
            List of boolean values indicating correctness of each completion
        """
        all_results = []
        for comp, ans in zip(group_completions, group_answers):
            comp_response_parts = self.response_pattern.findall(comp)
            if not comp_response_parts:
                all_results.append(False)
                continue
                
            comp_answer = extract_answer_from_solution(comp_response_parts[0])
            if comp_answer is None:
                all_results.append(False)
                continue
                
            comp_numeric, _ = extract_numeric_answer(comp_answer)
            ans_numeric, _ = extract_numeric_answer(ans)
            if comp_numeric is None or ans_numeric is None:
                all_results.append(False)
                continue
                
            all_results.append(abs(comp_numeric - ans_numeric) <= self.config.numeric_tolerance)
        
        return all_results
    
    def _process_verification_criteria(self, verification_data: Dict[str, Any], correct_answer: Any, 
                                      verifier_name: str = "Main") -> Dict[str, Any]:
        """
        Process verification criteria and calculate verification score.
        
        Args:
            verification_data: The parsed verification data from the agent
            correct_answer: The expected correct answer
            verifier_name: Name of the verifier for logging
            
        Returns:
            Dict containing verification details and scores
        """
        # Extract the verification criteria
        is_detailed = verification_data.get('is_detailed', False)
        is_correct = verification_data.get('is_correct', False)
        boxed_answer = verification_data.get('boxed_answer', None)
        
        # Initialize verification details
        verification_details = {
            "is_detailed": is_detailed,
            "is_correct": is_correct,
            "boxed_answer": boxed_answer,
            "criteria_scores": {},
            "total_score": 0,
            "verifier_name": verifier_name
        }
        
        # Process each criterion
        criteria = [
            ('is_detailed', is_detailed, f"{verifier_name} verifier: Solution is detailed", None),
            ('is_correct', is_correct, f"{verifier_name} verifier: Solution approach is correct", None),
            ('boxed_answer', boxed_answer is not None, f"{verifier_name} verifier: Boxed answer is correct", 
             lambda: self._compare_answers(boxed_answer, correct_answer) if boxed_answer is not None else False)
        ]
        
        verification_score = 0
        
        for criterion_name, criterion_met, success_message, additional_check in criteria:
            # Apply additional check if provided
            if additional_check is not None:
                criterion_met = criterion_met and additional_check()
            
            weight = self.verification_weights[criterion_name]
            
            if criterion_met:
                verification_score += weight
                verification_details["criteria_scores"][criterion_name] = weight
            else:
                verification_details["criteria_scores"][criterion_name] = 0
        
        verification_details["total_score"] = verification_score
        return verification_details
    
    def _extract_verification_data(self, verification_result: str, verifier_name: str = "Main") -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON data from verification agent response.
        
        Args:
            verification_result: The raw response from the verification agent
            verifier_name: Name of the verifier for logging
            
        Returns:
            Dict containing the parsed verification data, or None if parsing failed
        """
        try:
            # Extract just the response section containing the JSON
            response_match = self.response_pattern.search(verification_result)
            if response_match:
                json_text = response_match.group(1).strip()
            else:
                # Fallback to the full response if no response tags are found
                json_text = verification_result
            
            # Clean up the JSON string to handle potential formatting issues
            # Remove any markdown code block markers
            json_text = re.sub(r'```json|```', '', json_text).strip()
            
            # Try to find a valid JSON object in the response
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                return None
            
            # Parse the JSON data
            verification_data = json.loads(json_text)
            return verification_data
            
        except json.JSONDecodeError:
            return None
        except Exception:
            return None
    
    def _compare_answers(self, agent_answer: Any, correct_answer: Any) -> bool:
        """
        Compare the agent's answer with the correct answer, using extract_numeric_answer for robust parsing.
        
        Args:
            agent_answer: The answer provided by the verification agent
            correct_answer: The expected correct answer
            
        Returns:
            bool: True if the answers match within tolerance, False otherwise
        """
        # Use extract_numeric_answer to handle LaTeX and other formats
        agent_numeric, agent_debug = extract_numeric_answer(str(agent_answer), debug=True)
        correct_numeric, correct_debug = extract_numeric_answer(str(correct_answer), debug=True)
        
        if agent_numeric is not None and correct_numeric is not None:
            # Use the is_answer_correct helper function with our tolerance
            result = is_answer_correct(agent_numeric, correct_numeric, self.config.numeric_tolerance)
            self.logger.info(f"Comparing numeric answers: agent={agent_numeric}, correct={correct_numeric}, tolerance={self.config.numeric_tolerance}")
            self.logger.info(f"Numeric comparison result: {result}, difference: {abs(agent_numeric - correct_numeric) if agent_numeric is not None and correct_numeric is not None else 'N/A'}")
            
            if agent_debug or correct_debug:
                self.logger.debug(f"Agent answer parsing: {agent_debug}")
                self.logger.debug(f"Correct answer parsing: {correct_debug}")
                
            return result
        else:
            # If numeric conversion failed, fall back to string comparison
            agent_str = str(agent_answer).strip()
            correct_str = str(correct_answer).strip()
            is_string_match = agent_str == correct_str
            
            self.logger.info(f"Numeric conversion failed, comparing as strings: agent='{agent_str}', correct='{correct_str}'")
            if agent_debug or correct_debug:
                self.logger.debug(f"Agent answer parsing failed: {agent_debug}")
                self.logger.debug(f"Correct answer parsing failed: {correct_debug}")
                
            return is_string_match
    
    def _ensure_batch_lists_length(self, index):
        """Ensure all batch lists are long enough to store data at the given index"""
        for key in ['answers', 'is_correct', 'execution_times', 'code_lengths', 'completions', 'similarity_scores']:
            while len(self.stats.current_batch[key]) <= index:
                if key == 'is_correct':
                    self.stats.current_batch[key].append(False)
                elif key in ['execution_times', 'code_lengths', 'similarity_scores']:
                    self.stats.current_batch[key].append(0.0)
                else:
                    self.stats.current_batch[key].append(None)
