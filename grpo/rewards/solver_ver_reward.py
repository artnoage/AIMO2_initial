import re
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import os, sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Union
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.solution_utils import (
    extract_numeric_answer, extract_answer_from_solution, validate_solution
)
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats
from grpo.rewards.base_reward import BaseReward

class SolverVerReward(BaseReward):
    """Reward class for solution verification evaluation"""

    __name__ = "solver_ver_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'validation_rewards', 'correct_reflections', 'incorrect_reflections'],
        'group_stats': [
            'correct_answers', 'incorrect_answers'
        ],
        'plurality_stats': [
            'plurality_correct_rate', 'avg_plurality_percentage', 'avg_completion_length',
            'batch_plurality_correct', 'batch_plurality_percentage', 'batch_total_answers',
            'batch_correct_answers', 'batch_correct_rate'
        ],
        'reflection_stats': [
            'total_reflections', 'correct_self_assessments', 'incorrect_self_assessments',
            'self_assessment_accuracy', 'correct_answers_assessed_correct',
            'correct_answers_assessed_incorrect', 'incorrect_answers_assessed_correct',
            'incorrect_answers_assessed_incorrect'
        ]
    }

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Numerical tolerance for grouping similar answers
        self.answer_grouping_tolerance = 1e-2
        
        # Initialize reflection statistics with classification terminology
        self.reflection_stats = {
            'total_reflections': 0,
            'correct_self_assessments': 0,  # Model correctly assessed its answer (right or wrong)
            'incorrect_self_assessments': 0,  # Model incorrectly assessed its answer
            'self_assessment_accuracy': 0.0,  # Percentage of correct assessments
            'correct_answers_assessed_correct': 0,  # True Positives: Correct answers that model thought were correct
            'correct_answers_assessed_incorrect': 0,  # False Negatives: Correct answers that model thought were incorrect
            'incorrect_answers_assessed_correct': 0,  # False Positives: Incorrect answers that model thought were correct
            'incorrect_answers_assessed_incorrect': 0,  # True Negatives: Incorrect answers that model thought were incorrect
        }

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

            # Ensure current_batch exists in stats
            if not hasattr(self.stats, 'current_batch'):
                self.stats.current_batch = {
                    'answers': [],
                    'is_correct': [],
                    'execution_times': [],
                    'code_lengths': [],
                    'completions': []
                }

            if not all([group_completions, group_answers, group_indices]):
                self.logger.warning(f"Missing required group context - completions: {bool(group_completions)}, answers: {bool(group_answers)}, indices: {bool(group_indices)}")

                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)

                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion

                return 0.0

            self.logger.info(f"Processing completion {group_idx+1}/{len(group_completions)} in group")

            # Extract think section
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)

            # Extract and validate the answer
            model_answer = extract_answer_from_solution(completion)
            if model_answer is None:
                self.logger.info("There is no model_answer")

                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)

                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion

                return reward

            # Convert to numeric values
            model_numeric, debug_info = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(str(correct_answer))
            if model_numeric is None or correct_numeric is None:
                self.logger.debug("Could not extract numeric values - returning 0.0")

                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)

                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion

                return reward

            # Structure validation rewards
            has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            has_reflection = bool(re.search(r'<reflection>.*?</reflection>', completion, re.DOTALL))

            if not has_think or not has_response or not has_reflection:
                self.logger.debug(f"Missing required sections - think: {has_think}, response: {has_response}, reflection: {has_reflection}")
                return reward

            validation_reward = 0.0

            # Calculate base reward
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance

            # Extract reflection content to check if model thinks answer is correct
            reflection_match = re.search(r'<reflection>(.*?)</reflection>', completion, re.DOTALL)
            reflection_content = reflection_match.group(1) if reflection_match else ""

            # Check if reflection indicates the answer is correct
            thinks_correct = "answer is correct" in reflection_content.lower()
            thinks_incorrect = "answer may not be correct" in reflection_content.lower() or "answer is not correct" in reflection_content.lower()

            # Update reflection statistics
            self.reflection_stats['total_reflections'] += 1

            # Initialize reflection reward
            reflection_reward = 0.0

            # Log the reflection assessment
            self.logger.info(f"Reflection assessment - Answer is actually {is_correct}, model thinks {'correct' if thinks_correct else 'incorrect' if thinks_incorrect else 'unclear'}")

            if is_correct:
                base_reward = self.config.base_reward

                # If answer is correct but model thinks it's incorrect, subtract 1 point
                if thinks_incorrect:
                    reflection_reward = -1
                    self.logger.info(f"Answer is correct but model thinks it's incorrect: {reflection_reward:.1f}")
                    self.stats.reward_components['incorrect_reflections'] = self.stats.reward_components.get('incorrect_reflections', 0) + 1
                    self.reflection_stats['incorrect_self_assessments'] += 1
                    self.reflection_stats['correct_answers_assessed_incorrect'] += 1
                elif thinks_correct:
                    self.logger.info(f"Answer is correct and model correctly identified it")
                    self.stats.reward_components['correct_reflections'] = self.stats.reward_components.get('correct_reflections', 0) + 1
                    self.reflection_stats['correct_self_assessments'] += 1
                    self.reflection_stats['correct_answers_assessed_correct'] += 1
                else:
                    self.logger.info(f"Answer is correct but model's assessment is unclear")

                reward += base_reward + reflection_reward
                self.logger.info(f"Applied base reward: +{base_reward:.3f}, reflection reward: {reflection_reward:.1f}")
                self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + 1
                self.stats.reward_components['correct_answers'] = self.stats.reward_components.get('correct_answers', 0) + 1

            else:
                self.stats.reward_components['incorrect_answers'] = self.stats.reward_components.get('incorrect_answers', 0) + 1

                # If answer is incorrect and model correctly identifies it, add 1 point
                if thinks_incorrect:
                    reflection_reward = 1.0
                    self.logger.info(f"Answer is incorrect and model correctly identifies it: +{reflection_reward:.1f}")
                    self.stats.reward_components['correct_reflections'] = self.stats.reward_components.get('correct_reflections', 0) + 1
                    self.reflection_stats['correct_self_assessments'] += 1
                    self.reflection_stats['incorrect_answers_assessed_incorrect'] += 1
                    reward += reflection_reward
                elif thinks_correct:
                    self.logger.info(f"Answer is incorrect but model thinks it's correct")
                    self.reflection_stats['incorrect_self_assessments'] += 1
                    self.reflection_stats['incorrect_answers_assessed_correct'] += 1
                else:
                    self.logger.info(f"Answer is incorrect and model's assessment is unclear")

            # Ensure lists are long enough for this batch index
            self._ensure_batch_lists_length(batch_index)

            # Store the results for this completion
            self.stats.current_batch['answers'][batch_index] = model_numeric
            self.stats.current_batch['is_correct'][batch_index] = is_correct
            self.stats.current_batch['execution_times'][batch_index] = 0.0  # Not applicable for solution reward
            self.stats.current_batch['code_lengths'][batch_index] = len(completion)
            self.stats.current_batch['completions'][batch_index] = completion

            # Extract response part and validate solution structure
            response_parts = re.findall(r'<response>(.*?)</response>', completion, re.DOTALL)
            if response_parts:
                # Use the validate_solution method to check solution structure
                solution_valid, validation_reason = validate_solution(response_parts[0])

                if solution_valid:
                    validation_reward += 0.2
                    self.logger.info(f"Solution structure validation passed (+0.2)")
                else:
                    self.logger.info(f"Solution structure validation failed: {validation_reason}")

            reward += validation_reward
            if validation_reward > 0:
                self.stats.reward_components['validation_rewards'] = self.stats.reward_components.get('validation_rewards', 0) + 1
                self.logger.info(f"Applied total validation reward: +{validation_reward:.3f}")

            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.reward_components.get('correct_answers', 0) + self.stats.reward_components.get('incorrect_answers', 0)
            self.stats.reward_components['average_reward'] = \
                self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)

            # Update reflection accuracy statistics
            total_assessments = self.reflection_stats['correct_self_assessments'] + self.reflection_stats['incorrect_self_assessments']
            if total_assessments > 0:
                self.reflection_stats['self_assessment_accuracy'] = self.reflection_stats['correct_self_assessments'] / total_assessments

            # Log reflection statistics
            self.logger.info(f"Reflection stats - Accuracy: {self.reflection_stats['self_assessment_accuracy']:.2%}, "
                            f"Correct assessments: {self.reflection_stats['correct_self_assessments']}/{total_assessments}")

            # Calculate correctness for all completions in group
            all_results = []
            for comp, ans in zip(group_completions, group_answers):
                comp_answer = extract_answer_from_solution(comp)
                if comp_answer is None:
                    all_results.append(False)
                    continue

                comp_numeric, _ = extract_numeric_answer(comp_answer)
                ans_numeric, _ = extract_numeric_answer(ans)
                if comp_numeric is None or ans_numeric is None:
                    all_results.append(False)
                    continue

                all_results.append(abs(comp_numeric - ans_numeric) <= self.config.numeric_tolerance)


            # Update group-specific statistics
            if is_correct:
                self.stats.group_stats['correct_answers'] = self.stats.group_stats.get('correct_answers', 0) + 1
            else:
                self.stats.group_stats['incorrect_answers'] = self.stats.group_stats.get('incorrect_answers', 0) + 1


            return reward

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

            return 0.0

    def _ensure_batch_lists_length(self, index):
        """Ensure all batch lists are long enough to store data at the given index"""
        for key in ['answers', 'is_correct', 'execution_times', 'code_lengths', 'completions']:
            while len(self.stats.current_batch[key]) <= index:
                if key == 'is_correct':
                    self.stats.current_batch[key].append(False)
                elif key in ['execution_times', 'code_lengths']:
                    self.stats.current_batch[key].append(0.0)
                else:
                    self.stats.current_batch[key].append(None)
