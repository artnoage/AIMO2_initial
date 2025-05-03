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

from abc import ABC, abstractmethod
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats

class BaseReward(ABC):
    """Base class for reward calculation
    
    All reward classes handle batches of completions with their corresponding metadata.
    Input format:
    - completions: List[str] - The model outputs to evaluate
    - kwargs: Dict containing lists aligned with completions:
        - prompts: List[str] - The prompts that generated each completion
        - answer: List[str] - The expected answers (may also be 'correct_answer')
        
    The reward calculation process:
    1. Add reward_index to track original completion order
    2. Extract required lists from kwargs (prompts, answers)
    3. Process completions in batches, maintaining order
    4. Return rewards in same order as input completions
    
    Note: The dataset may provide 'answer' or 'correct_answer' - we handle both cases.
          'prompts' is pluralized but other fields maintain dataset names.
    """
    
    __name__ = "base_reward"
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.stats = RewardStats(config)
        self.logger = self._setup_logger()
        
    @abstractmethod
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion
        
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
        
        logger_name = f'reward_{self.config.model_type}'
        logger = logging.getLogger(logger_name)
        
        # Clear any existing handlers to prevent duplicate logging
        if logger.handlers:
            logger.handlers.clear()
            
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
        """Synchronous batch processing that runs async code in event loop"""
        # Validate inputs
        prompts = kwargs.get('prompts', [])
        answers = kwargs.get('answer') or kwargs.get('correct_answer', [])
        
        if len(completions) != len(prompts) or len(completions) != len(answers):
            self.logger.error(f"Mismatched lengths: completions={len(completions)}, prompts={len(prompts)}, answers={len(answers)}")
            return [0.0] * len(completions)
            
        # Reset current batch data for this new batch
        self.stats.current_batch = {
            'answers': [],
            'is_correct': [],
            'execution_times': [],
            'code_lengths': [],
            'completions': []
        }
            
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
            
        # Process completions in parallel using event loop
        async def process_batch():
            tasks = []
            
            # Extract problems, solutions, and partial solutions from kwargs if present
            problems = kwargs.get('problem', [''] * len(prompts))
            solutions = kwargs.get('model_solution', [''] * len(prompts))
            partial_solutions = kwargs.get('partial_solution', [''] * len(prompts))
            
            # Extract tutor-specific parameters if present
            wrong_steps = kwargs.get('wrong_step', [None] * len(prompts))
            is_corrects = kwargs.get('is_correct', [False] * len(prompts))
            full_solutions = kwargs.get('full_solution', [''] * len(prompts))
            
            for prompt, group in prompt_groups.items():
                # Process each completion in group
                for group_idx, (completion, ans, idx) in enumerate(zip(
                    group['completions'], 
                    group['answers'], 
                    group['indices']
                )):
                    # Create kwargs with group context and original kwargs
                    task_kwargs = {
                        **kwargs,  # Base kwargs first
                        'prompt': prompt,
                        'problem': problems[idx] if idx < len(problems) else '',
                        'solution': solutions[idx] if idx < len(solutions) else '',
                        'partial_solution': partial_solutions[idx] if idx < len(partial_solutions) else '',
                        'answer': str(ans),
                        'group_idx': group_idx,
                        'reward_index': idx,
                        'group_completions': group['completions'],
                        'group_answers': group['answers'], 
                        'group_indices': group['indices'],
                        'wrong_step': wrong_steps[idx] if idx < len(wrong_steps) else None,
                        'is_correct': is_corrects[idx] if idx < len(is_corrects) else False,
                        'full_solution': full_solutions[idx] if idx < len(full_solutions) else ''
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
        
        self.logger.info(f"Rewards before: {rewards}")
        
        # Update stats and print batch summary
        self.stats.update(rewards, completions=completions, example_type=kwargs.get('example_type', []))
        
        # Process the batch results to calculate plurality metrics
        self._finalize_batch()
        
        # Print reward-specific statistics summary every batch
        self.logger.info("\nReward Statistics Summary:")
        self.logger.info(self.stats.get_summary(getattr(self, 'relevant_stats', None)))
        
        return rewards
        
    def _finalize_batch(self):
        """Calculate batch-level statistics including plurality voting with numerical grouping"""
        # Skip if no results
        if not self.stats.current_batch['answers']:
            return
        
        # Filter out None values
        valid_results = [(ans, correct) for ans, correct in 
                         zip(self.stats.current_batch['answers'], 
                             self.stats.current_batch['is_correct']) 
                         if ans is not None]
        
        if not valid_results:
            return
        
        # Group similar answers using the tolerance
        grouped_answers = defaultdict(list)
        
        for idx, (ans, correct) in enumerate(valid_results):
            # Find if this answer belongs to an existing group
            found_group = False
            for group_key in grouped_answers:
                if abs(ans - group_key) <= self.answer_grouping_tolerance:
                    # Add to existing group
                    grouped_answers[group_key].append((idx, ans, correct))
                    found_group = True
                    break
            
            if not found_group:
                # Create new group
                grouped_answers[ans].append((idx, ans, correct))
        
        # Find the plurality winner (most common answer group)
        if grouped_answers:
            # Get the group with the most answers
            plurality_group, plurality_indices = max(grouped_answers.items(), 
                                                    key=lambda x: len(x[1]))
            
            # Calculate what percentage of valid answers this represents
            plurality_count = len(plurality_indices)
            plurality_percentage = plurality_count / len(valid_results)
            
            # Check if the plurality answer is correct
            plurality_correct = any(correct for _, _, correct in plurality_indices)
            
            # Calculate average completion length
            avg_completion_length = sum(len(comp) for comp in self.stats.current_batch['completions'] if comp) / len(self.stats.current_batch['completions']) if self.stats.current_batch['completions'] else 0
            
            # Store batch results
            batch_result = {
                'plurality_answer': plurality_group,
                'plurality_count': plurality_count,
                'plurality_percentage': plurality_percentage,
                'plurality_correct': plurality_correct,
                'total_answers': len(valid_results),
                'correct_answers': sum(1 for _, _, correct in plurality_indices if correct),
                'avg_code_length': 0,  # Not applicable for solution reward
                'avg_execution_time': 0,  # Not applicable for solution reward
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to batch history
            if not hasattr(self.stats, 'batch_results'):
                self.stats.batch_results = []
            self.stats.batch_results.append(batch_result)
            
            # Update plurality statistics
            if not hasattr(self.stats, 'plurality_stats'):
                self.stats.plurality_stats = {
                    'plurality_correct_count': 0,
                    'total_batches': 0,
                    'plurality_correct_rate': 0.0,
                    'avg_plurality_percentage': 0.0,
                    'avg_completion_length': 0.0
                }
                
            self.stats.plurality_stats['plurality_correct_count'] += int(plurality_correct)
            self.stats.plurality_stats['total_batches'] += 1
            
            if self.stats.plurality_stats['total_batches'] > 0:
                self.stats.plurality_stats['plurality_correct_rate'] = (
                    self.stats.plurality_stats['plurality_correct_count'] / 
                    self.stats.plurality_stats['total_batches']
                )
            
            # Update average plurality percentage (how dominant is the most common answer)
            prev_avg = self.stats.plurality_stats['avg_plurality_percentage']
            prev_batches = self.stats.plurality_stats['total_batches'] - 1
            
            if prev_batches > 0:
                self.stats.plurality_stats['avg_plurality_percentage'] = (
                    (prev_avg * prev_batches + plurality_percentage) / 
                    self.stats.plurality_stats['total_batches']
                )
            else:
                self.stats.plurality_stats['avg_plurality_percentage'] = plurality_percentage
            
            # Update average completion length
            prev_avg_length = self.stats.plurality_stats['avg_completion_length']
            
            if prev_batches > 0:
                self.stats.plurality_stats['avg_completion_length'] = (
                    (prev_avg_length * prev_batches + avg_completion_length) / 
                    self.stats.plurality_stats['total_batches']
                )
            else:
                self.stats.plurality_stats['avg_completion_length'] = avg_completion_length
            
            # Log the results
            self.logger.info(
                f"Batch plurality results: answer={plurality_group}, " +
                f"count={plurality_count}/{len(valid_results)} ({plurality_percentage:.2%}), " +
                f"correct={plurality_correct}, " +
                f"overall rate={self.stats.plurality_stats['plurality_correct_rate']:.2%}"
            )
            
            # Log the answer groups for debugging
            group_info = []
            for group_key, indices in grouped_answers.items():
                correct_in_group = any(correct for _, _, correct in indices)
                group_info.append(f"{group_key}: {len(indices)} answers, correct={correct_in_group}")
            
            self.logger.info(f"Answer groups (tolerance={self.answer_grouping_tolerance}):")
            for info in group_info:
                self.logger.info(f"  {info}")
