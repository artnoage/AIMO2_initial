import re
import asyncio
import os, sys
from typing import List, Dict, Tuple, Optional, Any, Union
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.model_utils import *
from utils.solution_utils import *
from utils.similarity_checker import SolutionSimilarityChecker
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats
from grpo.rewards import BaseReward, SolutionReward, FinalizationReward, ProgrammingReward, TutorReward

class DynamicReward(BaseReward):
    """A reward class that dynamically selects between SolutionReward and CompletionReward based on context"""
    
    __name__ = "dynamic_reward"
    
    def __init__(self, config: RewardConfig, similarity_checker: SolutionSimilarityChecker = None):
        """
        Initialize with configuration and similarity checker
        
        Args:
            config: RewardConfig for rewards
            similarity_checker: Optional similarity checker for rewards that need it
        """
        super().__init__(config)
        self.similarity_checker = similarity_checker
        
        # Create instances of all possible reward functions
        self.solution_reward = SolutionReward(config, similarity_checker)
        self.finalization_reward = FinalizationReward(config, similarity_checker)
        self.programming_reward = ProgrammingReward(config)
        self.tutor_reward = TutorReward(config)
        
        # Share the same stats object across all reward functions
        self.solution_reward.stats = self.stats
        self.finalization_reward.stats = self.stats
        self.programming_reward.stats = self.stats
        self.tutor_reward.stats = self.stats
        
        # Collect relevant stats from all possible rewards
        self.relevant_stats = {}
        for reward in [self.solution_reward, self.finalization_reward, self.programming_reward, self.tutor_reward]:
            if hasattr(reward, 'relevant_stats'):
                for category, stats in reward.relevant_stats.items():
                    if category not in self.relevant_stats:
                        self.relevant_stats[category] = []
                    self.relevant_stats[category].extend(stats)
        
        # Add dynamic reward specific stats
        if 'reward_components' not in self.relevant_stats:
            self.relevant_stats['reward_components'] = []
        self.relevant_stats['reward_components'].extend(['solution_reward_uses', 'finalization_reward_uses', 'programming_reward_uses', 'tutor_reward_uses'])
    
    def _extract_example_types(self, batch_kwargs: Dict) -> List[str]:
        """
        Extract and normalize example types from batch kwargs
        
        Args:
            batch_kwargs: Keyword arguments for the batch
            
        Returns:
            List of normalized example type strings
        """
        example_types = []
        raw_types = batch_kwargs.get('example_type', [])
        
        # Handle different formats of example_types
        if isinstance(raw_types, list):
            for et in raw_types:
                if isinstance(et, list) and len(et) > 0:
                    # Handle nested list case
                    example_types.append(et[0])
                elif isinstance(et, str):
                    example_types.append(et)
        elif isinstance(raw_types, str):
            # Handle string case
            example_types.append(raw_types)
            
        # Count the different types
        type_counts = {}
        for et in example_types:
            type_counts[et] = type_counts.get(et, 0) + 1
            
        self.logger.info(f"Extracted example types: {type_counts}")
        return example_types
        
    def _select_reward_type(self, example_types: List[str]) -> str:
        """
        Select which reward type to use for the entire batch based on example_type
        
        Args:
            example_types: List of normalized example type strings
            
        Returns:
            String indicating which reward to use: 'solution', 'finalization', 'programming', or 'tutor'
        """
        # Count the different types in the batch
        finalization_count = sum(1 for et in example_types if et == 'finalization')
        solution_count = sum(1 for et in example_types if et == 'solution')
        programming_count = sum(1 for et in example_types if et == 'programming')
        tutor_count = sum(1 for et in example_types if et == 'tutor')
        
        self.logger.info(f"Type counts in batch: finalization={finalization_count}, solution={solution_count}, programming={programming_count}, tutor={tutor_count}")
        
        # Determine the majority type
        if tutor_count > 0:
            self.logger.info("Selected tutor reward (priority type)")
            return 'tutor'
        elif programming_count > 0 and programming_count >= finalization_count and programming_count >= solution_count:
            self.logger.info("Selected programming reward (majority type)")
            return 'programming'
        elif finalization_count > solution_count:
            self.logger.info("Selected finalization reward (majority type)")
            return 'finalization'
        else:
            # Default to solution reward
            self.logger.info("Selected solution reward (majority type or default)")
            return 'solution'
    
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward using the selected reward function"""
        try:
            # Get the reward type from kwargs (set by __call__)
            reward_type = kwargs.get('reward_type', 'solution')
            
            # Check if answer exists
            answer = kwargs.get('answer') or kwargs.get('correct_answer')
            if not answer:
                self.logger.warning("Missing answer in example, returning zero reward")
                return 0.0
            
            # Verify group context is available
            group_completions = kwargs.get('group_completions', [])
            group_answers = kwargs.get('group_answers', [])
            group_indices = kwargs.get('group_indices', [])
            
            if not all([group_completions, group_answers, group_indices]):
                self.logger.warning(f"Missing group context in DynamicReward - completions: {bool(group_completions)}, answers: {bool(group_answers)}, indices: {bool(group_indices)}")
                # We'll continue anyway as the underlying reward functions should handle this
            
            # Get the example type for this specific completion
            example_type = kwargs.get('example_type', '')
            
            # Select the appropriate reward function
            if reward_type == 'finalization':
                reward_func = self.finalization_reward
                self.stats.reward_components['finalization_reward_uses'] = self.stats.reward_components.get('finalization_reward_uses', 0) + 1
            elif reward_type == 'programming':
                reward_func = self.programming_reward
                self.stats.reward_components['programming_reward_uses'] = self.stats.reward_components.get('programming_reward_uses', 0) + 1
            elif reward_type == 'tutor':
                reward_func = self.tutor_reward
                self.stats.reward_components['tutor_reward_uses'] = self.stats.reward_components.get('tutor_reward_uses', 0) + 1
            else:
                reward_func = self.solution_reward
                self.stats.reward_components['solution_reward_uses'] = self.stats.reward_components.get('solution_reward_uses', 0) + 1
            
            # Log the example type being processed
            self.logger.info(f"Processing example type: {example_type} with {reward_func.__name__}")
            
            # Calculate the reward using the selected function
            reward = await reward_func.calculate_reward(completion, **kwargs)
            
            # Log which reward function was used
            self.logger.info(f"Used {reward_func.__name__} with result: {reward:.4f}")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error in dynamic reward calculation: {str(e)}")
            return 0.0
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Override to handle batch processing with consistent reward selection"""
        # Validate inputs
        prompts = kwargs.get('prompts', [])
        answers = kwargs.get('answer') or kwargs.get('correct_answer', [])
        
        if len(completions) != len(prompts) or len(completions) != len(answers):
            self.logger.error(f"Mismatched lengths: completions={len(completions)}, prompts={len(prompts)}, answers={len(answers)}")
            self.logger.error(f"kwargs keys: {list(kwargs.keys())}")
            return [0.0] * len(completions)
        
        # Log all kwargs keys for debugging
        self.logger.info(f"Available kwargs: {list(kwargs.keys())}")
        
        # Check if example_type exists and log its format
        if 'example_type' in kwargs:
            example_type_value = kwargs['example_type']
            self.logger.info(f"example_type found: {example_type_value} (type: {type(example_type_value)})")
            if isinstance(example_type_value, list):
                self.logger.info(f"example_type list length: {len(example_type_value)}")
                if len(example_type_value) > 0:
                    self.logger.info(f"First element: {example_type_value[0]} (type: {type(example_type_value[0])})")
        else:
            self.logger.warning("No example_type found in kwargs")
        
        # Extract and normalize example_types (this now also checks prompts for wait examples)
        example_types = self._extract_example_types(kwargs)
        
        # Select which reward type to use for the entire batch
        reward_type = self._select_reward_type(example_types)
        self.logger.info(f"Using {reward_type} reward for entire batch of {len(completions)} examples")
        
        # Update stats with example types
        if hasattr(self, 'stats'):
            self.stats.update([], example_type=example_types)
            
            # Update the reward type usage
            if reward_type == 'solution':
                self.stats.reward_components['solution_reward_uses'] = self.stats.reward_components.get('solution_reward_uses', 0) + 1
            elif reward_type == 'finalization':
                self.stats.reward_components['finalization_reward_uses'] = self.stats.reward_components.get('finalization_reward_uses', 0) + 1
            elif reward_type == 'programming':
                self.stats.reward_components['programming_reward_uses'] = self.stats.reward_components.get('programming_reward_uses', 0) + 1
            elif reward_type == 'tutor':
                self.stats.reward_components['tutor_reward_uses'] = self.stats.reward_components.get('tutor_reward_uses', 0) + 1
        
        # Group completions by prompt for group context
        prompt_groups = {}
        for idx, (completion, prompt, ans) in enumerate(zip(completions, prompts, answers)):
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {
                    'completions': [],
                    'answers': [],
                    'indices': []
                }
            prompt_groups[prompt]['completions'].append(completion)
            prompt_groups[prompt]['answers'].append(str(ans))
            prompt_groups[prompt]['indices'].append(idx)
            
        # Process each completion with the selected reward type
        async def process_batch():
            tasks = []
            
            # Extract problems, solutions, and partial solutions from kwargs if present
            problems = kwargs.get('problem', [''] * len(prompts))
            solutions = kwargs.get('model_solution', [''] * len(prompts))
            partial_solutions = kwargs.get('partial_solution', [''] * len(prompts))
            example_types_list = self._extract_example_types(kwargs)
            
            # Ensure example_types_list has the right length
            if len(example_types_list) != len(completions):
                self.logger.warning(f"Example types list length ({len(example_types_list)}) doesn't match completions length ({len(completions)})")
                # Fill with the selected reward_type if lengths don't match
                example_types_list = [reward_type] * len(completions)
            
            for prompt, group in prompt_groups.items():
                # Process each completion in group
                for group_idx, (completion, ans, idx) in enumerate(zip(
                    group['completions'], 
                    group['answers'], 
                    group['indices']
                )):
                    # Get the example type for this specific completion
                    example_type = example_types_list[idx] if idx < len(example_types_list) else reward_type
                    
                    # Extract additional tutor-specific parameters
                    wrong_step = kwargs.get('wrong_step', [None] * len(prompts))[idx] if idx < len(kwargs.get('wrong_step', [])) else None
                    is_correct = kwargs.get('is_correct', [False] * len(prompts))[idx] if idx < len(kwargs.get('is_correct', [])) else False
                    full_solution = kwargs.get('full_solution', [''] * len(prompts))[idx] if idx < len(kwargs.get('full_solution', [])) else ''
                    
                    # Create kwargs with group context and original kwargs
                    task_kwargs = {
                        **kwargs,  # Base kwargs first
                        'prompt': prompt,
                        'problem': problems[idx] if idx < len(problems) else '',
                        'solution': solutions[idx] if idx < len(solutions) else '',
                        'partial_solution': partial_solutions[idx] if idx < len(partial_solutions) else '',
                        'answer': str(ans),
                        'reward_index': idx,
                        'reward_type': reward_type,  # Batch-level reward type
                        'example_type': example_type,  # Individual example type
                        'group_idx': group_idx,
                        'group_completions': group['completions'],
                        'group_answers': group['answers'], 
                        'group_indices': group['indices'],
                        'wrong_step': wrong_step,
                        'is_correct': is_correct,
                        'full_solution': full_solution
                    }
                    
                    task = self.calculate_reward(completion, **task_kwargs)
                    tasks.append(task)
                    
            return await asyncio.gather(*tasks)
            
        # Run async code in event loop - with Jupyter notebook compatibility
        try:
            # Check if we're in IPython/Jupyter
            import sys
            is_jupyter = 'ipykernel' in sys.modules
            
            if is_jupyter:
                self.logger.debug("Detected Jupyter environment, using nest_asyncio")
                # Use nest_asyncio to allow nested event loops in Jupyter
                import nest_asyncio
                nest_asyncio.apply()
                
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                self.logger.debug("Using existing event loop")
            except RuntimeError:
                self.logger.debug("No event loop found - creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the coroutine
            if is_jupyter:
                # In Jupyter, we can directly await the coroutine
                rewards = asyncio.run(process_batch())
            else:
                # In regular Python, use run_until_complete
                rewards = loop.run_until_complete(process_batch())
                
        except Exception as e:
            self.logger.error(f"Error during batch processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            rewards = [0.0] * len(completions)
        
        # Apply normalization if needed (same as in BaseReward)
        if len(rewards) > 1:
            self.logger.info(f"Rewards before: {rewards}")
            
            # If mean is negative, clip all rewards from below by zero
            mean_reward = sum(rewards) / len(rewards)
            if mean_reward < 0:
                self.logger.info(f"Mean reward is negative ({mean_reward:.6f}), clipping all rewards to non-negative values")
                rewards = [max(0.0, r) for r in rewards]
                self.logger.info(f"Rewards after clipping: {rewards}")
        
        # Update stats and print batch summary
        self.stats.update(rewards, completions=completions, example_type=example_types)
        
        # Print reward-specific statistics summary every batch
        self.logger.info("\nReward Statistics Summary:")
        stats_summary = self.stats.get_summary(self.relevant_stats)
        self.logger.info(stats_summary)
        
        return rewards
