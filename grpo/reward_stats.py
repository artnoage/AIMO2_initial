import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class RewardStats:
    """Base class for tracking reward statistics"""
    def __init__(self, config):
        self.config = config
        self.total_batches = 0
        self.total_examples = 0
        self.total_rewards = 0
        self.reward_distribution = {}
        self.start_time = datetime.now()
        
        # Track step validation stats
        self.step_stats = {
            'correct_step_numbering': 0,
            'incorrect_step_numbering': 0,
            'total_steps_completed': 0
        }
        
        # Track reward components
        self.reward_components = {
            'base_rewards': 0,
            'step_continuity_rewards': 0,
            'diversity_bonuses': 0,
            'similarity_penalties': 0,
            'validation_rewards': 0,
            'total_length_penalty': 0.0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'total_rewards': 0.0,
            'average_reward': 0.0,
            'solution_reward_uses': 0,
            'completion_reward_uses': 0
        }
        
        # Track group-specific stats
        self.group_stats = {
            'unique_solutions': 0,
            'similar_solutions': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'total_similarity': 0.0,
            'diversity_bonuses': 0,
            'similarity_penalties': 0
        }
        
        # Track similarity stats
        self.similarity_stats = {
            'unique_completions': 0,
            'similar_completions': 0,
            'total_similarity': 0.0
        }
        
        # Create a logger for this instance
        self.logger = logging.getLogger(f'reward_stats_{config.model_type}')
        
    def update(self, rewards: List[float], **kwargs):
        """Update statistics with new rewards"""
        self.total_batches += 1
        self.total_examples += len(rewards)
        
        for r in rewards:
            self.total_rewards += r
            r_rounded = round(r, 6)
            self.reward_distribution[r_rounded] = self.reward_distribution.get(r_rounded, 0) + 1
        
        # Update reward type usage stats
        reward_type = kwargs.get('reward_type')
        if reward_type:
            if reward_type == 'solution':
                self.reward_components['solution_reward_uses'] += 1
            elif reward_type == 'completion':
                self.reward_components['completion_reward_uses'] += 1
        
        # Initialize example type tracking if not already present
        if not hasattr(self, 'example_types'):
            self.example_types = {
                'solution': 0,
                'completion': 0,
                'wait': 0,
                'unknown': 0
            }
        
        # Initialize reward type usage tracking if not already present
        if not hasattr(self, 'reward_type_usage'):
            self.reward_type_usage = {
                'solution': 0,
                'completion': 0
            }
        
        # Track the reward type that was used
        reward_type = kwargs.get('reward_type')
        if reward_type and reward_type in self.reward_type_usage:
            self.reward_type_usage[reward_type] += 1
                
        # Track example types if provided
        example_types = kwargs.get('example_type', [])
        
        # Count the different example types
        if isinstance(example_types, list):
            for et in example_types:
                if isinstance(et, list) and len(et) > 0:
                    et = et[0]  # Handle nested lists
                elif not isinstance(et, str):
                    continue
                    
                if et in self.example_types:
                    self.example_types[et] += 1
                else:
                    self.example_types['unknown'] += 1
        elif isinstance(example_types, str):
            if example_types in self.example_types:
                self.example_types[example_types] += 1
            else:
                self.example_types['unknown'] += 1
        
        # Log the current distribution
        self.logger.info(f"Example type distribution: {self.example_types}")
        if hasattr(self, 'reward_type_usage'):
            self.logger.info(f"Reward type usage: {self.reward_type_usage}")
            
        # Update completions if provided
        completions = kwargs.get('completions', [])
        if completions:
            self.logger.info(f"Processing {len(completions)} completions for stats")
            
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
            
    def get_summary(self, relevant_stats=None) -> str:
        """Get a human-readable summary of statistics
        
        Args:
            relevant_stats: Dictionary mapping stat categories to lists of relevant stat names
        """
        total_samples = self.total_examples
        if total_samples == 0:
            return "No samples processed yet"
            
        elapsed = datetime.now() - self.start_time
        avg_reward = self.total_rewards / total_samples if total_samples > 0 else 0
        
        # Always show basic stats
        summary = [
            f"Training time: {elapsed}",
            f"Processed {self.total_batches} batches ({total_samples} examples)",
            f"Average reward: {avg_reward:.6f}"
        ]
        
        # If no relevant stats specified, use default categories
        if not relevant_stats:
            relevant_stats = {
                'reward_components': list(self.reward_components.keys()),
                'group_stats': list(self.group_stats.keys()),
                'step_stats': list(self.step_stats.keys()),
                'similarity_stats': list(self.similarity_stats.keys())
            }
            
            # Add example_types and reward_type_usage if available
            if hasattr(self, 'example_types'):
                relevant_stats['example_types'] = list(self.example_types.keys())
            
            if hasattr(self, 'reward_type_usage'):
                relevant_stats['reward_type_usage'] = list(self.reward_type_usage.keys())
        
        # Build sections based on relevant stats
        for category, stat_names in relevant_stats.items():
            if not stat_names:
                continue
            
            # Get the stats dictionary for this category
            if category == 'reward_components':
                stats_dict = self.reward_components
            elif category == 'group_stats':
                stats_dict = self.group_stats
            elif category == 'step_stats':
                stats_dict = self.step_stats
            elif category == 'similarity_stats':
                stats_dict = self.similarity_stats
            elif category == 'example_types' and hasattr(self, 'example_types'):
                stats_dict = self.example_types
            elif category == 'reward_type_usage' and hasattr(self, 'reward_type_usage'):
                stats_dict = self.reward_type_usage
            else:
                # Skip if category doesn't exist
                continue
                
            # Add section header
            section_name = category.replace('_', ' ').title()
            summary.append(f"\n{section_name}:")
            
            # Add relevant stats
            for stat_name in stat_names:
                if stat_name not in stats_dict:
                    continue
                    
                value = stats_dict[stat_name]
                # Format special cases
                if isinstance(value, float):
                    if 'penalty' in stat_name or 'reward' in stat_name or 'total' in stat_name:
                        formatted_value = f"{value:.6f}"
                    else:
                        formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                
                # Format stat name for display
                display_name = stat_name.replace('_', ' ').title()
                summary.append(f"  {display_name}: {formatted_value}")
        
        return "\n".join(summary)
