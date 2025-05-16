import re
import asyncio
import logging
import json
import math
import torch
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
from utils.similarity_checker import SolutionSimilarityChecker
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats
from grpo.rewards.base_reward import BaseReward
from grpo.rewards.solution_reward import SolutionReward

class SolutionEmbeddingReward(SolutionReward):
    """
    Reward class that extends SolutionReward to incorporate model_solution embeddings for comparison.
    This reward not only checks if an answer is correct but also rewards based on the similarity
    between the generated solution and a reference model solution.
    """
    
    __name__ = "solution_embedding_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'validation_rewards', 'verification_rewards', 'diversity_rewards', 'embedding_similarity_rewards'],
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
        ],
        'embedding_stats': [
            'avg_similarity_score', 'high_similarity_count', 'total_similarity_comparisons'
        ]
    }
    
    def __init__(self, config: RewardConfig, similarity_checker=None):
        # Initialize the BaseReward parent class directly, skipping SolutionReward's __init__
        # to avoid creating the verification model
        BaseReward.__init__(self, config)
        
        # Numerical tolerance for grouping similar answers
        self.answer_grouping_tolerance = 1e-2
        
        # Initialize verification-specific stats
        if not hasattr(self.stats, 'group_stats'):
            self.stats.group_stats = {}
        self.stats.group_stats['verified_solutions'] = 0
            
        # Use verification weights from config
        self.verification_weights = self.config.verification_weights
        
        # Do NOT create verification model - this is what causes the connection error
        # self.main_verification_model = get_model(self.config, role="main")
        
        # Store the similarity checker
        self.similarity_checker = similarity_checker
        
        # Initialize diversity rewards counter
        if not hasattr(self.stats, 'reward_components'):
            self.stats.reward_components = {}
        self.stats.reward_components['diversity_rewards'] = 0
        
        # Compile regex patterns for better performance
        self.thinking_pattern = re.compile(self.config.think_pattern, re.DOTALL)
        self.response_pattern = re.compile(self.config.response_pattern, re.DOTALL)
        self.boxed_pattern = re.compile(self.config.boxed_pattern)
        
        # Initialize embedding-specific stats
        if not hasattr(self.stats, 'embedding_stats'):
            self.stats.embedding_stats = {
                'avg_similarity_score': 0.0,
                'high_similarity_count': 0,
                'total_similarity_comparisons': 0
            }
        
        # Initialize embedding similarity rewards counter
        self.stats.reward_components['embedding_similarity_rewards'] = 0
        
        # Create similarity checker if not provided
        if similarity_checker is None:
            self.similarity_checker = SolutionSimilarityChecker(config)
        else:
            self.similarity_checker = similarity_checker
        
        # Similarity threshold for high similarity
        self.high_similarity_threshold = 0.8
        
        # Maximum embedding similarity reward
        self.embedding_similarity_max_reward = 1.0
        
        # Ensure current_batch has similarity_scores field
        if not hasattr(self.stats, 'current_batch'):
            self.stats.current_batch = {}
        if 'similarity_scores' not in self.stats.current_batch:
            self.stats.current_batch['similarity_scores'] = []
            
    
    async def _calculate_embedding_similarity(self, completion_content: str, model_solution_content: str) -> float:
        """
        Calculate the embedding similarity between completion and model_solution.
        
        Args:
            completion_content: The extracted response content from completion
            model_solution_content: The extracted response content from model_solution
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Get embeddings for both texts
            embeddings = self.similarity_checker.get_embeddings([completion_content, model_solution_content])
            
            # Calculate cosine similarity
            if embeddings.shape[0] >= 2:
                completion_embedding = embeddings[0].unsqueeze(0)
                model_solution_embedding = embeddings[1].unsqueeze(0)
                
                # Calculate cosine similarity
                similarity = torch.matmul(completion_embedding, model_solution_embedding.t()).item()
                
                # Ensure similarity is between 0 and 1
                similarity = max(0.0, min(1.0, similarity))
                
                return similarity
            else:
                self.logger.warning("Failed to get valid embeddings for similarity calculation")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating embedding similarity: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0
    
    def _update_embedding_stats(self, similarity_score: float):
        """
        Update embedding statistics with the new similarity score.
        
        Args:
            similarity_score: The similarity score between completion and model_solution
        """
        # Initialize embedding stats if they don't exist
        if not hasattr(self.stats, 'embedding_stats'):
            self.stats.embedding_stats = {
                'avg_similarity_score': 0.0,
                'high_similarity_count': 0,
                'total_similarity_comparisons': 0
            }
        
        # Update total comparisons
        self.stats.embedding_stats['total_similarity_comparisons'] += 1
        
        # Update high similarity count if score exceeds threshold
        if similarity_score >= self.high_similarity_threshold:
            self.stats.embedding_stats['high_similarity_count'] += 1
        
        # Update average similarity score
        prev_avg = self.stats.embedding_stats['avg_similarity_score']
        prev_count = self.stats.embedding_stats['total_similarity_comparisons'] - 1
        
        if prev_count > 0:
            self.stats.embedding_stats['avg_similarity_score'] = (
                (prev_avg * prev_count + similarity_score) / 
                self.stats.embedding_stats['total_similarity_comparisons']
            )
        else:
            self.stats.embedding_stats['avg_similarity_score'] = similarity_score
            
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """
        Calculate reward for a single completion with group context and model_solution comparison.
        This overrides the parent class's implementation to avoid using self.main_verification_model.
        """
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
            
            # Ensure current_batch exists in stats with all required keys
            if not hasattr(self.stats, 'current_batch'):
                self.stats.current_batch = {
                    'answers': [],
                    'is_correct': [],
                    'execution_times': [],
                    'code_lengths': [],
                    'completions': [],
                    'similarity_scores': []
                }
            else:
                # Ensure all required keys exist in current_batch
                for key in ['answers', 'is_correct', 'execution_times', 'code_lengths', 'completions', 'similarity_scores']:
                    if key not in self.stats.current_batch:
                        self.stats.current_batch[key] = []
                
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
            
            # Skip verification since we don't have a verification model
            verification_reward = 0.0
            log("Skipping verification since no verification model is available")
            
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
            
            # Now add the embedding similarity reward for all solutions
            embedding_similarity_reward = await self._calculate_embedding_similarity_reward(completion, **kwargs)
            total_reward += embedding_similarity_reward
            
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
    
    async def _calculate_embedding_similarity_reward(self, completion: str, **kwargs) -> float:
        """
        Calculate the embedding similarity reward component.
        
        Args:
            completion: The completion to calculate the reward for
            **kwargs: Additional arguments
            
        Returns:
            float: The embedding similarity reward
        """
        try:
            # Get model_solution from kwargs
            model_solution = kwargs.get('solution', '')
            batch_index = kwargs.get('reward_index', len(self.stats.current_batch['answers']) if hasattr(self.stats, 'current_batch') else 0)
            
            # Collect logs for this reward calculation
            log_messages = []
            def log(message, level="info"):
                log_messages.append((level, message))
            
            # Initialize embedding similarity reward
            embedding_similarity_reward = 0.0
            
            # Extract response parts from completion and model_solution
            # We only want to compare the response sections, not the thinking sections
            completion_response_parts = self.response_pattern.findall(completion)
            completion_content = completion_response_parts[0] if completion_response_parts else ""
            
            model_solution_response_parts = self.response_pattern.findall(model_solution)
            model_solution_content = model_solution_response_parts[0] if model_solution_response_parts else ""
            
            log(f"Extracted response content from completion: {len(completion_content)} chars")
            log(f"Extracted response content from model_solution: {len(model_solution_content)} chars")
            
            # Only calculate embedding similarity if both completion and model_solution have content
            if completion_content and model_solution_content:
                # Calculate embedding similarity
                similarity_score = await self._calculate_embedding_similarity(completion_content, model_solution_content)
                
                # Store the similarity score
                self._ensure_batch_lists_length(batch_index)
                self.stats.current_batch['similarity_scores'][batch_index] = similarity_score
                
                # Calculate reward based on similarity score with a highly aggressive non-linear transformation
                # Using the function 1/|(1-x)| with a cutoff at 5
                # This function grows extremely quickly as similarity approaches 1
                if similarity_score >= 0.999:
                    # Avoid division by zero by capping at 0.999
                    transformed_similarity = 5.0
                else:
                    # Calculate 1/|(1-x)| with a maximum of 5.0
                    transformed_similarity = min(5.0, 1.0 /  math.sqrt(abs(1.0 - similarity_score)))
                
                                
                
                # Normalize to [0, 1] range by dividing by the square root of the maximum value (sqrt(5.0))
                normalized_similarity = transformed_similarity
                
                # Apply the normalized similarity to the maximum reward
                embedding_similarity_reward = normalized_similarity * self.embedding_similarity_max_reward
                
                # Log the transformation details
                log(f"Raw similarity: {similarity_score:.4f}, Transformed: {transformed_similarity:.4f}, Normalized: {normalized_similarity:.4f}")
                
                # Update embedding stats
                self._update_embedding_stats(similarity_score)
                
                # Log the similarity score and reward
                log(f"Embedding similarity score: {similarity_score:.4f}")
                log(f"Embedding similarity reward: +{embedding_similarity_reward:.4f}")
                
                # Update embedding similarity rewards counter
                self.stats.reward_components['embedding_similarity_rewards'] = self.stats.reward_components.get('embedding_similarity_rewards', 0) + 1
            else:
                log("Missing content for embedding similarity calculation", "warning")
                
                # Ensure the similarity score is stored
                self._ensure_batch_lists_length(batch_index)
                self.stats.current_batch['similarity_scores'][batch_index] = 0.0
            
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
            
            return embedding_similarity_reward
            
        except Exception as e:
            self.logger.error(f"Error calculating embedding similarity reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0
    
    async def verify_solution(self, problem: str, solution_content: str, correct_answer: float, 
                             model=None, verifier_name: str = "Main") -> Tuple[bool, Dict[str, Any]]:
        """
        Override the verify_solution method to avoid using the verification model.
        Instead, we'll just return a default verification result.
        
        Args:
            problem: The problem statement
            solution_content: The already extracted response content to verify
            correct_answer: The expected answer
            model: The model to use for verification (ignored)
            verifier_name: Name of the verifier for logging (ignored)
            
        Returns:
            Tuple containing:
            - bool: Always False since we're not actually verifying
            - Dict: Default verification results
        """
        self.logger.info("Skipping verification since no verification model is available")
        
        # Return a default verification result
        return False, {
            "error": "Verification skipped - no verification model available",
            "total_score": 0,
            "criteria_scores": {
                "is_detailed": 0,
                "is_correct": 0,
                "boxed_answer": 0
            }
        }
