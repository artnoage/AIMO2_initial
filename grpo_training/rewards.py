import re
import json
import asyncio
import torch
import logging
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import os, sys
from typing import List, Optional, Tuple, Any, Union
from transformers import AutoTokenizer, AutoModel
from config import GRPOConfig
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer, validate_solution
from utils.agents import CompletionAgent
from abc import ABC, abstractmethod

@dataclass 
class RewardConfig:
    """Base configuration for reward calculation"""
    model_type: str
    numeric_tolerance: float = 1e-6
    logging_dir: str = "logs"
    stats_dir: str = "statistics"
    max_retries: int = 3
    timeout: int = 300
    
    # Common reward components
    base_reward: float = 2.0
    validation_reward: float = 0.2
    length_penalty_factor: float = 0.0001
    
    # Group-specific settings
    group_base_reward: float = 3.0
    group_diversity_bonus: float = 0.3
    group_majority_bonus: float = 0.2
    group_similarity_threshold_low: float = 0.7
    group_similarity_threshold_high: float = 0.9
    
    # Tutor-specific settings
    tutor_structure_base_reward: float = 0.2
    tutor_analysis_reward: float = 0.2
    tutor_substitution_reward: float = 0.4
    tutor_single_step_bonus: float = 0.2
    tutor_multiple_step_penalty: float = 0.4
    tutor_full_reward: float = 5.0
    tutor_analysis_length_cost: float = 0.0001
    tutor_substitution_length_cost: float = 0.0001
    tutor_redundant_substitution_penalty: float = 0.1

class RewardStats:
    """Base class for tracking reward statistics"""
    def __init__(self, config: RewardConfig):
        self.config = config
        self.total_batches = 0
        self.total_rewards = 0
        self.reward_distribution = {}
        self.start_time = datetime.now()
        
        # Track section-level stats
        self.section_stats = {
            'missing_analysis': 0,
            'missing_verdict': 0,
            'missing_substitution': 0,
            'invalid_step_number': 0,
            'polar_verdict_with_substitution': 0,
            'step_verdict_without_substitution': 0,
            'multiple_steps_in_substitution': 0
        }
        
        # Track reward components
        self.reward_components = {
            'base_rewards': 0,
            'analysis_rewards': 0,
            'substitution_rewards': 0,
            'step_bonuses': 0,
            'step_penalties': 0,
            'total_analysis_length_penalty': 0.0,
            'total_substitution_length_penalty': 0.0,
            'redundant_substitution_penalties': 0,
            'wrong_boxed_answer_penalties': 0,
            'majority_bonuses': 0,
            'diversity_bonuses': 0,
            'improvement_bonuses': {
                '0.1': 0,  # 10-40% completions
                '0.2': 0,  # 40-70% completions
                '0.3': 0,  # >70% completions
                'total': 0  # Total count of improvement bonuses
            }
        }
        
        # Track group-specific stats
        self.group_stats = {
            'unique_solutions': 0,
            'similar_solutions': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'total_similarity': 0.0
        }
        
        # Track full reward reasons
        self.full_reward_reasons = {
            'correct_answer': 0,
            'wrong_approach': 0,
            'step_correction': 0,
            'final_step_correct': 0
        }
        
    def update(self, rewards: List[float], **kwargs):
        """Update statistics with new rewards"""
        self.total_batches += 1
        for r in rewards:
            self.total_rewards += r
            r_rounded = round(r, 6)
            self.reward_distribution[r_rounded] = self.reward_distribution.get(r_rounded, 0) + 1
            
        # Update section stats if provided
        completion = kwargs.get('completion')
        if completion:
            if 'analysis' not in completion.lower():
                self.section_stats['missing_analysis'] += 1
            if 'verdict' not in completion.lower():
                self.section_stats['missing_verdict'] += 1
            if 'substitution' not in completion.lower():
                self.section_stats['missing_substitution'] += 1
                
        # Update group stats if provided
        similarity = kwargs.get('similarity')
        if similarity is not None:
            self.group_stats['total_similarity'] += float(similarity)
            if similarity < 0.7:
                self.group_stats['unique_solutions'] += 1
            elif similarity > 0.9:
                self.group_stats['similar_solutions'] += 1
            
    def save_statistics(self, output_dir: str) -> None:
        """Save current statistics to JSON
        
        Args:
            output_dir: Directory to save statistics files
            
        The statistics are saved with timestamps and include:
        - Total number of batches processed
        - Total rewards accumulated
        - Distribution of rewards
        - Training duration
        - Component-specific statistics
        """
        try:
            stats_dir = Path(output_dir) / self.config.stats_dir
            stats_dir.mkdir(exist_ok=True)
            
            stats = {
                'total_batches': self.total_batches,
                'total_rewards': self.total_rewards,
                'reward_distribution': {str(k): v for k, v in self.reward_distribution.items()},
                'training_duration': str(datetime.now() - self.start_time),
                'section_stats': self.section_stats,
                'reward_components': self.reward_components,
                'group_stats': self.group_stats,
                'full_reward_reasons': self.full_reward_reasons
            }
            
            stats_file = stats_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save statistics: {str(e)}")
            
    def get_summary(self) -> str:
        """Get a human-readable summary of statistics"""
        total_samples = sum(self.reward_distribution.values())
        if total_samples == 0:
            return "No samples processed yet"
            
        elapsed = datetime.now() - self.start_time
        avg_reward = self.total_rewards / total_samples if total_samples > 0 else 0
        
        # Sort rewards for better readability
        sorted_rewards = sorted(self.reward_distribution.items())
        reward_dist_str = "\n".join(
            f"  {reward:.6f}: {count} samples" 
            for reward, count in sorted_rewards
        )
        
        return (
            f"Training time: {elapsed}\n"
            f"Processed {self.total_batches} batches\n"
            f"Average reward: {avg_reward:.6f}\n"
            f"Total samples: {total_samples}\n"
            f"\nReward Distribution:\n{reward_dist_str}\n"
            f"\nSection Issues:\n"
            f"  Missing analysis: {self.section_stats['missing_analysis']}\n"
            f"  Missing verdict: {self.section_stats['missing_verdict']}\n"
            f"  Step verdict without substitution: {self.section_stats['step_verdict_without_substitution']}\n"
            f"  Polar verdict with substitution: {self.section_stats['polar_verdict_with_substitution']}\n"
            f"  Multiple steps in substitution: {self.section_stats['multiple_steps_in_substitution']}\n"
            f"\nReward Components:\n"
            f"  Base rewards: {self.reward_components['base_rewards']}\n"
            f"  Analysis rewards: {self.reward_components['analysis_rewards']}\n"
            f"  Substitution rewards: {self.reward_components['substitution_rewards']}\n"
            f"  Step bonuses: {self.reward_components['step_bonuses']}\n"
            f"  Step penalties: {self.reward_components['step_penalties']}\n"
            f"\nPenalties:\n"
            f"  Analysis length: {self.reward_components['total_analysis_length_penalty']:.6f}\n"
            f"  Substitution length: {self.reward_components['total_substitution_length_penalty']:.6f}\n"
            f"  Wrong boxed answers: {self.reward_components['wrong_boxed_answer_penalties']}\n"
            f"  Redundant substitutions: {self.reward_components['redundant_substitution_penalties']}\n"
            f"\nGroup Statistics:\n"
            f"  Majority bonuses: {self.group_stats['majority_bonuses']}\n"
            f"  Diversity bonuses: {self.group_stats['diversity_bonuses']}\n"
            f"  Unique solutions: {self.group_stats['unique_solutions']}\n"
            f"  Similar solutions: {self.group_stats['similar_solutions']}\n"
            f"  Average similarity: {self.group_stats['total_similarity']/total_samples if total_samples else 0:.3f}\n"
            f"\nFull Reward Reasons:\n"
            f"  Correct answer: {self.full_reward_reasons['correct_answer']}\n"
            f"  Wrong approach: {self.full_reward_reasons['wrong_approach']}\n"
            f"  Step correction: {self.full_reward_reasons['step_correction']}\n"
            f"  Final step correct: {self.full_reward_reasons['final_step_correct']}"
        )


class BaseReward(ABC):
    """Base class for reward calculation"""
    
    __name__ = "base_reward"
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.stats = RewardStats(config)
        self.logger = self._setup_logger()
        
    @abstractmethod
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion
        
        Args:
            completion: The model completion to evaluate
            **kwargs: Additional context needed for reward calculation
            
        Returns:
            float: The calculated reward value
        """
        raise NotImplementedError
        
    @abstractmethod
    async def calculate_reward_async(self, completion: str, **kwargs) -> float:
        """Async version of calculate_reward
        
        Args:
            completion: The model completion to evaluate
            **kwargs: Additional context needed for reward calculation
            
        Returns:
            float: The calculated reward value
        """
        raise NotImplementedError
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.config.logging_dir) / self.config.model_type
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger(f'reward_{self.config.model_type}')
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(
            log_dir / f"reward_{timestamp}.log"
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())
        return logger
        
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Calculate rewards for a batch of completions"""
        answers = kwargs.get('answer', [])
        print(answers)
        correct_answers=kwargs.get('answer', [])
        prompts = kwargs.get('prompts', [])
        print(correct_answers)
        # Handle single answer/prompt case
        if isinstance(answers, str):
            answers = [answers]
        if isinstance(prompts, str):
            prompts = [prompts]
            
        if not answers:
            self.logger.warning("No answers provided")
            return [0.0] * len(completions)
            
        self.logger.info(f"\n{'='*50}\nProcessing new batch of {len(completions)} completions")
        self.logger.info(f"Number of answers: {len(answers)}")
        
        # Group completions by answer
        completion_groups = {}
        for i, (comp, ans) in enumerate(zip(completions, answers)):
            if ans not in completion_groups:
                self.logger.info(f"\nCreating new group for answer: {ans}")
                print(f"Processing answer: {ans}")
                print(f"Answer type: {type(ans)}")
                completion_groups[ans] = {
                    'completions': [],
                    'indices': [],
                    'answer': ans,
                    'correct_answer': ans  # Ensure both fields are present
                }
            completion_groups[ans]['completions'].append(comp)
            completion_groups[ans]['indices'].append(i)

        # Calculate rewards for each group
        rewards = [0.0] * len(completions)
        for group_idx, group in enumerate(completion_groups.values()):
            self.logger.info(f"\n{'='*30}\nProcessing group {group_idx+1}/{len(completion_groups)}")
            batch_completions = group['completions']
            correct_answer = group.get('correct_answer') or group.get('answer')
            print(f"Group correct_answer: {correct_answer}")
            
            # Process each completion in the group
            for i, comp_idx in enumerate(group['indices']):
                self.logger.info(f"\nProcessing completion {i+1}/{len(batch_completions)} (global index {comp_idx})")
                group_context = {
                    'completions': batch_completions,
                    'correct_answer': correct_answer,
                    'index': i
                }
                reward = self.calculate_reward(batch_completions[i], group=group_context)
                rewards[comp_idx] = reward

        # Update statistics with the rewards
        self.stats.update(rewards)

    async def __call_async__(self, completions: List[str], **kwargs) -> List[float]:
        """Async version of __call__
        
        Args:
            completions: List of model completions to evaluate
            **kwargs: Additional context needed for reward calculation
            
        Returns:
            List[float]: List of reward values for each completion
        """
        rewards = await asyncio.gather(*[
            self.calculate_reward_async(comp, **kwargs) 
            for comp in completions
        ])
        self.stats.update(rewards, **kwargs)
        return rewards

class SolutionReward(BaseReward):
    """Reward class for basic solution evaluation"""
    
    __name__ = "solution_reward"
    
    def __init__(self, config: GRPOConfig):
        super().__init__(config)
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion
        
        Args:
            completion: The model completion to evaluate
            **kwargs: Additional context including:
                - correct_answer: Expected answer string
                - problem: Original problem text
                - solution: Complete solution attempt
                
        Returns:
            float: Calculated reward value
            
        The reward is calculated based on:
        1. Correctness of the answer
        2. Validity of the solution
        3. Length penalties
        """
        try:
            reward = 0.0
            
            print("\n=== Starting reward calculation ===")
            print(f"Completion (first 100 chars): {completion[:100]}...")
            
            # Extract and validate the answer
            model_answer = extract_answer_from_solution(completion)
            if model_answer is None:
                self.logger.debug("No boxed answer found")
                print("No boxed answer found - returning 0.0")
                return 0.0
                
            # Convert to numeric values
            model_numeric, debug_info = extract_numeric_answer(model_answer)
            print(f"Model numeric value: {model_numeric}")
            print(f"Debug info: {debug_info}")
            
            correct_answer = kwargs.get('correct_answer') or kwargs.get('answer')
            print(f"Raw correct_answer: {correct_answer}")
            if not correct_answer:
                self.logger.warning("No correct answer or answer provided")
                print("No correct answer found - returning 0.0")
                return 0.0
            
            # Handle different input formats
            print(f"Correct answer type: {type(correct_answer)}")
            if isinstance(correct_answer, (list, tuple)):
                if not correct_answer:  # Empty list/tuple
                    self.logger.warning("Empty correct answer list")
                    print("Empty correct answer list - returning 0.0")
                    return 0.0
                correct_answer = correct_answer[0]
                print(f"Using first element from list: {correct_answer}")
            elif isinstance(correct_answer, dict):
                correct_answer = str(correct_answer.get('answer', ''))
                print(f"Extracted answer from dict: {correct_answer}")
            
            # Convert to string if needed
            correct_answer = str(correct_answer)
            print(f"Final correct answer string: {correct_answer}")
            
            self.logger.info(f"Processing answer extraction:")
            self.logger.info(f"Model answer: {model_answer}")
            self.logger.info(f"Correct answer: {correct_answer}")
            
            correct_numeric, correct_debug = extract_numeric_answer(correct_answer, debug=True)
            print(f"Extracted correct numeric: {correct_numeric}")
            print(f"Correct debug info: {correct_debug}")
            
            self.logger.info(f"Model numeric: {model_numeric} ({debug_info})")
            self.logger.info(f"Correct numeric: {correct_numeric} ({correct_debug})")
            
            if model_numeric is None:
                print("Model numeric is None - returning 0.0")
                return 0.0
            if correct_numeric is None:
                print("Correct numeric is None - returning 0.0") 
                return 0.0
            
            if model_numeric is None or correct_numeric is None:
                self.logger.debug(f"Could not extract numeric values - Model: {model_answer}, Correct: {correct_answer}")
                return 0.0
                
            # Check correctness
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
            print(f"\nCorrectness check:")
            print(f"Difference: {abs(model_numeric - correct_numeric)}")
            print(f"Tolerance: {self.config.numeric_tolerance}")
            print(f"Is correct: {is_correct}")
            
            if is_correct:
                reward += self.config.base_reward
                print(f"Added base reward: +{self.config.base_reward}")
                
            # Add validation reward
            is_valid, validation_msg = validate_solution(completion)
            print(f"\nValidation check:")
            print(f"Is valid: {is_valid}")
            print(f"Validation message: {validation_msg}")
            
            if is_valid:
                reward += self.config.validation_reward
                print(f"Added validation reward: +{self.config.validation_reward}")
                
            # Apply length penalty
            length_penalty = len(completion) * self.config.length_penalty_factor
            reward -= length_penalty
            print(f"\nLength penalty:")
            print(f"Completion length: {len(completion)}")
            print(f"Penalty factor: {self.config.length_penalty_factor}")
            print(f"Total penalty: -{length_penalty}")
            
            # Update statistics
            self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + (1 if reward >= self.config.base_reward else 0)
            self.stats.reward_components['validation_rewards'] = self.stats.reward_components.get('validation_rewards', 0) + (1 if is_valid else 0)
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            print(f"\nFinal reward calculation:")
            print(f"Total reward: {reward}")
            print("=== End reward calculation ===\n")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0
            
    async def calculate_reward_async(self, completion: str, **kwargs) -> float:
        """Async version of calculate_reward"""
        return self.calculate_reward(completion, **kwargs)
            

class SolutionSimilarityChecker:
    """Handles embedding and similarity computation for solutions"""
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model_name)
        self.model = AutoModel.from_pretrained(config.embedding_model_name)
        
        # Move model to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Ensure all parameters are frozen and on correct device
        with torch.no_grad():
            for param in self.model.parameters():
                param.requires_grad_(False)  # Use requires_grad_ method
                if param.device != self.device:
                    param.data = param.data.to(self.device)

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.embedding_max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return F.normalize(embeddings, p=2, dim=1)

    def compute_similarity_matrix(self, solutions: List[str]) -> torch.Tensor:
        embeddings = self.get_embeddings(solutions)
        return torch.mm(embeddings, embeddings.t())

class GroupReward(BaseReward):
    """Reward class for group-based solution evaluation"""
    
    __name__ = "group_reward"
    
    def __init__(self, config: GRPOConfig, similarity_checker: SolutionSimilarityChecker):
        super().__init__(config)
        self.similarity_checker = similarity_checker
        
    async def calculate_reward_async(self, completion: str, **kwargs) -> float:
        """Async version of calculate_reward"""
        return self.calculate_reward(completion, **kwargs)
        
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
            
        self.logger.info(f"Processing completion {group_index+1}/{len(completions)}")
        self.logger.info(f"Correct answer: {correct_answer}")
        
        # Calculate correctness for all completions in group
        results = []
        for comp in completions:
            try:
                model_answer = extract_answer_from_solution(comp)
                if model_answer is None:
                    self.logger.info(f"\nCompletion {len(results)+1}:")
                    self.logger.info("FAILED: No boxed answer found")
                    self.logger.info(f"First 200 chars: {comp[:200]}")
                    results.append(False)
                    continue
                
                model_numeric, model_debug = extract_numeric_answer(model_answer, debug=True)
                correct_numeric, correct_debug = extract_numeric_answer(correct_answer, debug=True)
                
                # Detailed debug prints
                self.logger.info(f"\nCompletion {len(results)+1} numeric extraction:")
                self.logger.info(f"Raw completion (first 200 chars): {comp[:200]}")
                self.logger.info(f"Extracted boxed answer: {model_answer}")
                self.logger.info(f"Model numeric: {model_numeric} ({model_debug})")
                self.logger.info(f"Correct answer: {correct_answer}")
                self.logger.info(f"Correct numeric: {correct_numeric} ({correct_debug})")
                
                if model_numeric is None or correct_numeric is None:
                    self.logger.info(f"Could not extract numeric values for completion {len(results)+1}")
                    results.append(False)
                    continue
                
                is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
                self.logger.info(f"Completion {len(results)+1} correct: {is_correct}")
                results.append(is_correct)
                
            except Exception as e:
                self.logger.error(f"Error processing completion {len(results)+1}: {str(e)}")
                results.append(False)
                
        # Calculate similarity matrix for group
        similarity_matrix = self.similarity_checker.compute_similarity_matrix(completions)
        
        # Calculate reward components
        is_correct = results[group_index]
        reward = self.config.group_base_reward if is_correct else 0.0
        
        # Log correctness results
        self.logger.info(f"Correctness results: {results}")
        self.logger.info(f"Current completion correct: {is_correct}")
        
        # Majority bonus
        correct_count = sum(results)
        is_in_majority = (is_correct and correct_count > len(completions) / 2) or \
                        (not is_correct and (len(completions) - correct_count) > len(completions) / 2)
        majority_bonus = self.config.group_majority_bonus if is_correct else self.config.group_majority_bonus * 0.1
        if is_in_majority:
            reward += majority_bonus
            
        # Diversity bonus
        similarities = similarity_matrix[group_index]
        similarities[group_index] = 0  # Remove self-similarity
        avg_similarity = similarities.mean().item()
        
        self.logger.info(f"Average similarity: {avg_similarity:.3f}")
        
        diversity_bonus = 0
        if avg_similarity < self.config.group_similarity_threshold_low:  # Unique solution
            diversity_bonus = self.config.group_diversity_bonus if is_correct else self.config.group_diversity_bonus * 0.1
            reward += diversity_bonus
        elif avg_similarity > self.config.group_similarity_threshold_high:  # Very similar to others
            diversity_bonus = -(self.config.group_diversity_bonus / 2 if is_correct else self.config.group_diversity_bonus * 0.05)
            reward += diversity_bonus
            
        # Update group-specific statistics
        self.stats.group_stats['correct_answers'] += 1 if is_correct else 0
        self.stats.group_stats['incorrect_answers'] += 0 if is_correct else 1
        self.stats.group_stats['majority_bonuses'] += 1 if is_in_majority else 0
        self.stats.group_stats['diversity_bonuses'] += 1 if diversity_bonus > 0 else 0
        self.stats.group_stats['unique_solutions'] += 1 if avg_similarity < self.config.group_similarity_threshold_low else 0
        self.stats.group_stats['similar_solutions'] += 1 if avg_similarity > self.config.group_similarity_threshold_high else 0
        self.stats.group_stats['total_similarity'] += avg_similarity
        
        self.logger.info(f"Final reward: {reward:.3f}")
        
        return reward

class TutorReward(BaseReward):
    """Reward class for tutor response evaluation"""
    
    __name__ = "tutor_reward"
    
    def __init__(self, config: GRPOConfig):
        super().__init__(config)
        # Initialize completion agent for validation
        self.completion_agent = CompletionAgent(
            port=config.completion_port,
            model=config.completion_model_name,
            logger=self.logger
        )
        
    def extract_sections(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract Analysis, Verdict and Substitution sections"""
        analysis_match = re.search(r'</Analysis>\s*(.*?)\s*<Analysis>', response, re.DOTALL)
        verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
        substitution_match = re.search(r'</Substitution>\s*(.*?)\s*<Substitution>', response, re.DOTALL)
        
        return (
            analysis_match.group(1).strip() if analysis_match else None,
            verdict_match.group(1).strip() if verdict_match else None,
            substitution_match.group(1).strip() if substitution_match else None
        )
        
    def split_into_steps(self, solution: str) -> List[str]:
        """Split solution into analysis and numbered steps"""
        parts = solution.split("Step")
        if not parts:
            return []
            
        steps = []
        if "analysis" in parts[0].lower():
            steps.append(parts[0].strip())
            
        for step in parts[1:]:
            if step.strip():
                steps.append(("Step" + step).strip())
                
        return steps
        
    async def _validate_completions(
        self,
        problem: str,
        partial_solution: str,
        correct_answer: str,
        num_attempts: int = 10
    ) -> Tuple[int, int]:
        """Try completions in parallel until finding a successful one."""
        async def try_completion():
            try:
                completion = await self.completion_agent.generate(problem, partial_solution)
                complete_solution = partial_solution + completion
                
                model_answer = extract_answer_from_solution(complete_solution)
                if model_answer is None:
                    self.logger.debug("No boxed answer found in completion")
                    return False
                    
                numeric_answer, debug_info = extract_numeric_answer(model_answer, debug=True)
                correct_numeric, _ = extract_numeric_answer(correct_answer)
                
                if numeric_answer is None:
                    self.logger.debug(f"Could not extract numeric answer: {debug_info}")
                    return False
                    
                if correct_numeric is None:
                    self.logger.debug(f"Could not extract correct numeric answer from: {correct_answer}")
                    return False
                    
                is_correct = abs(numeric_answer - correct_numeric) <= self.config.numeric_tolerance
                
                return is_correct
                
            except Exception as e:   
                self.logger.warning(f"Completion attempt failed: {str(e)}")
                return False
    
        # Run all completion attempts in parallel
        results = await asyncio.gather(*[try_completion() for _ in range(num_attempts)])
        successful = sum(1 for r in results if r)
        return successful, len(results)

    async def _validate_whole_approach_is_wrong(
        self,
        problem: str,
        solution: str,
        correct_answer: str
    ) -> bool:
        """Validate that the analysis section alone can lead to correct completions"""
        # Split solution into steps and get the analysis part
        steps = self.split_into_steps(solution)
        if not steps:
            return False
            
        # First part before steps is the analysis
        analysis = steps[0]
        
        # Try completions starting with just the analysis
        successful, total = await self._validate_completions(
            problem,
            analysis,
            correct_answer,
            self.config.completion_attempts
        )
        
        return successful == 0 and total == self.config.completion_attempts

    async def _validate_step_identification(
        self,
        problem: str,
        steps: List[str],
        step_num: int,
        substitution: str,
        correct_answer: str,
        original_step: str
    ) -> Tuple[bool, float]:
        """Validate step identification and correction in parallel.
        Returns (is_valid, improvement_bonus)"""
        
        # Run both validations in parallel
        wrong_partial = "".join(steps[:step_num+1])  # Include the wrong step
        corrected_partial = "".join(steps[:step_num]) + substitution  # Replace wrong step
        
        wrong_check, fixed_check = await asyncio.gather(
            self._validate_completions(problem, wrong_partial, correct_answer, self.config.completion_attempts),
            self._validate_completions(problem, corrected_partial, correct_answer, self.config.completion_attempts)
        )

        successful_wrong, total_wrong = wrong_check
        successful_fixed, total_fixed = fixed_check

        # Calculate improvement bonus based on relative success rate
        improvement_bonus = 0.0
        if successful_wrong == 0:  # Only reward if original step had no successful completions
            success_rate = successful_fixed / total_fixed
            if 0.1 < success_rate <= 0.4:  # 10-40%
                improvement_bonus = 0.1
            elif 0.4 < success_rate <= 0.7:  # 40-70%
                improvement_bonus = 0.2
            elif success_rate > 0.7:         # >70%
                improvement_bonus = 0.3
        
        is_valid = successful_wrong == 0 and successful_fixed > 0
        return is_valid, improvement_bonus

    async def calculate_reward_async(self, tutor_response: str, **kwargs) -> float:
        """Calculate reward for a tutor's evaluation of a solution"""
        # First verify if model solution is correct
        model_solution = kwargs.get('solution', '')
        model_answer = extract_answer_from_solution(model_solution)
        if model_answer is None:
            self.logger.warning(f"No boxed answer found in model solution: {model_solution[:100]}...")
            return 0.0

        model_numeric, _ = extract_numeric_answer(model_answer)
        correct_numeric, _ = extract_numeric_answer(kwargs.get('correct_answer', ''))
        
        if model_numeric is None or correct_numeric is None:
            self.logger.warning(f"Could not extract numeric values - Model: {model_answer}, Correct: {kwargs.get('correct_answer', '')}")
            return 0.0
        
        is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance

        # Extract sections from tutor's response
        analysis, verdict, substitution = self.extract_sections(tutor_response)
        
        if verdict is None:
            self.logger.debug(f"Missing verdict section in tutor response: {tutor_response[:100]}...")
            return 0.0
            
        # Get problem and student solution from kwargs
        problem = kwargs.get('problem')
        student_solution = kwargs.get('solution')
        correct_answer = kwargs.get('correct_answer')
        
        if not all([problem, student_solution, correct_answer]):
            self.logger.warning("Missing required context (problem, solution, or correct_answer)")
            return 0.0
            
        polar_verdicts = ["The answer is correct", "The whole approach is wrong"]
        reward = 0.0
        
        # Basic structure reward
        if verdict in polar_verdicts:
            reward = self.config.tutor_structure_base_reward
            if substitution:
                reward -= self.config.tutor_redundant_substitution_penalty
                self.stats.reward_components['redundant_substitution_penalties'] += 1
        elif verdict.startswith("Step "):
            try:
                step_num = int(verdict.split()[1])
                if step_num < 0:
                    self.stats.section_stats['invalid_step_number'] += 1
                    return 0.0
            except (ValueError, IndexError):
                self.stats.section_stats['invalid_step_number'] += 1
                return 0.0
                
            if not substitution:
                self.stats.section_stats['step_verdict_without_substitution'] += 1
                return 0.0
                
            reward = self.config.tutor_structure_base_reward
        else:
            return 0.0
            
        # Analysis reward
        if analysis:
            length_penalty = len(analysis) * self.config.tutor_analysis_length_cost
            reward += self.config.tutor_analysis_reward - length_penalty
            self.stats.reward_components['analysis_rewards'] += 1
            self.stats.reward_components['total_analysis_length_penalty'] += length_penalty
            
        # Verify tutor's verdict using completion agent
        if verdict == "The answer is correct":
            # Check if student solution is actually correct
            student_answer = extract_answer_from_solution(student_solution)
            if student_answer:
                student_numeric, _ = extract_numeric_answer(student_answer)
                correct_numeric, _ = extract_numeric_answer(correct_answer)
                if student_numeric is not None and correct_numeric is not None:
                    if abs(student_numeric - correct_numeric) <= self.config.numeric_tolerance:
                        reward = self.config.tutor_full_reward
                        self.stats.full_reward_reasons['correct_answer'] += 1
                    else:
                        # Tutor incorrectly said answer was correct
                        return 0.0
                        
        elif verdict == "The whole approach is wrong":
            if not analysis:
                return reward
                
            # Verify by trying to complete solution from analysis
            try:
                completion = await self.completion_agent.generate(problem, analysis)
                completion_answer = extract_answer_from_solution(completion)
                if completion_answer:
                    completion_numeric, _ = extract_numeric_answer(completion_answer)
                    correct_numeric, _ = extract_numeric_answer(correct_answer)
                    if completion_numeric is not None and correct_numeric is not None:
                        if abs(completion_numeric - correct_numeric) <= self.config.numeric_tolerance:
                            # Tutor incorrectly said approach was wrong
                            return 0.0
                        else:
                            # Tutor correctly identified wrong approach
                            reward = self.config.tutor_full_reward
                            self.stats.full_reward_reasons['wrong_approach'] += 1
            except Exception as e:
                self.logger.warning(f"Error during completion validation: {str(e)}")
                return reward
                
        elif verdict.startswith("Step "):
            solution_steps = self.split_into_steps(student_solution)
            if step_num >= len(solution_steps):
                return reward
                
            # Check if substitution has multiple steps
            substitution_steps = self.split_into_steps(substitution)
            if len(substitution_steps) > 1:
                reward -= self.config.tutor_multiple_step_penalty
                self.stats.reward_components['step_penalties'] += 1
            else:
                reward += self.config.tutor_single_step_bonus
                self.stats.reward_components['step_bonuses'] += 1
                
            # Try completing from original solution up to wrong step
            partial_solution = "".join(solution_steps[:step_num])
            try:
                # Try completing with tutor's substitution
                completion_with_sub = await self.completion_agent.generate(
                    problem, 
                    partial_solution + substitution
                )
                
                # Try completing with original step
                completion_original = await self.completion_agent.generate(
                    problem,
                    partial_solution + solution_steps[step_num]
                )
                
                # Extract and compare answers
                sub_answer = extract_answer_from_solution(completion_with_sub)
                orig_answer = extract_answer_from_solution(completion_original)
                
                if sub_answer and orig_answer:
                    sub_numeric, _ = extract_numeric_answer(sub_answer)
                    orig_numeric, _ = extract_numeric_answer(orig_answer)
                    correct_numeric, _ = extract_numeric_answer(correct_answer)
                    
                    if all(x is not None for x in [sub_numeric, orig_numeric, correct_numeric]):
                        sub_correct = abs(sub_numeric - correct_numeric) <= self.config.numeric_tolerance
                        orig_correct = abs(orig_numeric - correct_numeric) <= self.config.numeric_tolerance
                        
                        if sub_correct and not orig_correct:
                            # Tutor's substitution leads to correct answer while original doesn't
                            reward = self.config.tutor_full_reward
                            self.stats.full_reward_reasons['step_correction'] += 1
                        elif orig_correct:
                            # Original step was actually correct
                            return 0.0
                            
            except Exception as e:
                self.logger.warning(f"Error during step validation: {str(e)}")
                return reward
                
        # Update base statistics
        self.stats.reward_components['base_rewards'] += 1
        if substitution:
            length_penalty = len(substitution) * self.config.tutor_substitution_length_cost
            reward += self.config.tutor_substitution_reward - length_penalty
            self.stats.reward_components['substitution_rewards'] += 1
            self.stats.reward_components['total_substitution_length_penalty'] += length_penalty
            
        return reward
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Synchronous wrapper for async reward calculation"""
        return asyncio.run(self.calculate_reward_async(completion, **kwargs))

    async def __call_async__(self, completions: List[str], prompts: List[str], problem: List[str], model_solution: List[str], correct_answer: List[str], **kwargs) -> List[float]:
        """Process a batch of examples in parallel"""
        tasks = [
            self.calculate_reward_async(comp, problem=prob, solution=sol, correct_answer=ans)
            for comp, prob, sol, ans in zip(completions, problem, model_solution, correct_answer)
        ]
        rewards = await asyncio.gather(*tasks)
        self.stats.update(rewards, completion=completions[0] if completions else None)
        return rewards
