import re
import asyncio
import logging
import json
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
            
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """
        Calculate reward for a single completion with group context and model_solution comparison.
        Extends the base SolutionReward by adding embedding similarity comparison.
        """
        try:
            # Get the base reward from the parent class
            base_reward = await super().calculate_reward(completion, **kwargs)
            
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
                
                # Calculate reward based on similarity score
                embedding_similarity_reward = similarity_score * self.embedding_similarity_max_reward
                
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
            
            # Add embedding similarity reward to base reward
            total_reward = base_reward + embedding_similarity_reward
            
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
            self.logger.error(f"Error calculating embedding reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Get batch_index safely
            try:
                # Try to get batch_index from kwargs
                batch_index = kwargs.get('reward_index', 0)
                
                # Ensure current_batch exists and has similarity_scores
                if not hasattr(self.stats, 'current_batch'):
                    self.stats.current_batch = {}
                if 'similarity_scores' not in self.stats.current_batch:
                    self.stats.current_batch['similarity_scores'] = []
                
                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)
                
                # Store empty results in case of exception
                self.stats.current_batch['similarity_scores'][batch_index] = 0.0
            except Exception as inner_e:
                self.logger.error(f"Error handling exception in calculate_reward: {str(inner_e)}")
                self.logger.error(traceback.format_exc())
            
            # Return the base reward without embedding similarity
            return base_reward if 'base_reward' in locals() else 0.0
    
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
