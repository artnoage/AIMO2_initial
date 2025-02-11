import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
from .reward_base import BaseReward, RewardConfig

class GroupReward(BaseReward):
    """Reward class for group-based solution evaluation"""
    
    def __init__(self, config: RewardConfig, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__(config)
        # Initialize embedding model for similarity checking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze embedding model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Reward settings
        self.base_reward = 3.0
        self.diversity_bonus = 0.3
        self.majority_bonus = 0.2
        
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for a list of texts"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return F.normalize(embeddings, p=2, dim=1)
            
    def compute_similarity_matrix(self, solutions: List[str]) -> torch.Tensor:
        """Compute pairwise similarities between solutions"""
        embeddings = self.get_embeddings(solutions)
        return torch.mm(embeddings, embeddings.t())
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion within its group context"""
        group = kwargs.get('group', {})
        if not group:
            self.logger.warning("No group context provided")
            return 0.0
            
        # Extract group information
        completions = group.get('completions', [])
        correct_answer = group.get('correct_answer')
        group_index = group.get('index', 0)
        
        if not completions or not correct_answer:
            return 0.0
            
        # Calculate correctness for all completions
        results = []
        for comp in completions:
            model_answer = extract_answer_from_solution(comp)
            if model_answer is None:
                results.append(False)
                continue
                
            model_numeric, _ = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(correct_answer)
            
            if model_numeric is None or correct_numeric is None:
                results.append(False)
                continue
                
            results.append(abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance)
            
        # Calculate similarity matrix
        similarity_matrix = self.compute_similarity_matrix(completions)
        
        # Calculate reward components
        is_correct = results[group_index]
        reward = self.base_reward if is_correct else 0.0
        
        # Majority bonus
        correct_count = sum(results)
        is_in_majority = (is_correct and correct_count > len(completions) / 2) or \
                        (not is_correct and (len(completions) - correct_count) > len(completions) / 2)
        if is_in_majority:
            reward += self.majority_bonus if is_correct else self.majority_bonus * 0.1
            
        # Diversity bonus
        similarities = similarity_matrix[group_index]
        similarities[group_index] = 0  # Remove self-similarity
        avg_similarity = similarities.mean().item()
        
        if avg_similarity < 0.7:  # Unique solution
            reward += self.diversity_bonus if is_correct else self.diversity_bonus * 0.1
        elif avg_similarity > 0.9:  # Very similar to others
            reward -= self.diversity_bonus / 2 if is_correct else self.diversity_bonus * 0.05
            
        # Update statistics
        self.stats.reward_components = getattr(self.stats, 'reward_components', {})
        self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + (1 if is_correct else 0)
        self.stats.reward_components['majority_bonuses'] = self.stats.reward_components.get('majority_bonuses', 0) + (1 if is_in_majority else 0)
        self.stats.reward_components['diversity_bonuses'] = self.stats.reward_components.get('diversity_bonuses', 0) + (1 if avg_similarity < 0.7 else 0)
        
        return reward
