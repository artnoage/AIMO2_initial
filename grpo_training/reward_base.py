import os
import re
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass 
class RewardConfig:
    """Base configuration for reward calculation"""
    model_type: str
    numeric_tolerance: float = 1e-6
    logging_dir: str = "logs"
    stats_dir: str = "statistics"

class RewardStats:
    """Base class for tracking reward statistics"""
    def __init__(self, config: RewardConfig):
        self.config = config
        self.total_batches = 0
        self.total_rewards = 0
        self.reward_distribution = {}
        self.start_time = datetime.now()
        
    def update(self, rewards: List[float], **kwargs):
        """Update statistics with new rewards"""
        self.total_batches += 1
        for r in rewards:
            self.total_rewards += r
            r_rounded = round(r, 6)
            self.reward_distribution[r_rounded] = self.reward_distribution.get(r_rounded, 0) + 1
            
    def save_statistics(self, output_dir: str):
        """Save current statistics to JSON"""
        stats_dir = Path(output_dir) / self.config.stats_dir
        stats_dir.mkdir(exist_ok=True)
        
        stats = {
            'total_batches': self.total_batches,
            'total_rewards': self.total_rewards,
            'reward_distribution': {str(k): v for k, v in self.reward_distribution.items()},
            'training_duration': str(datetime.now() - self.start_time)
        }
        
        stats_file = stats_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def get_summary(self) -> str:
        """Get a human-readable summary of statistics"""
        total_samples = sum(self.reward_distribution.values())
        if total_samples == 0:
            return "No samples processed yet"
            
        elapsed = datetime.now() - self.start_time
        avg_reward = self.total_rewards / total_samples if total_samples > 0 else 0
        
        return (
            f"Training time: {elapsed}\n"
            f"Processed {self.total_batches} batches\n"
            f"Average reward: {avg_reward:.6f}\n"
            f"Total samples: {total_samples}"
        )

class BaseReward:
    """Base class for reward calculation"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.stats = RewardStats(config)
        self.logger = self._setup_logger()
        
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
        
    def extract_numeric_answer(self, answer: str, debug: bool = False) -> Tuple[Optional[float], Optional[str]]:
        """Extract numeric value from answer string"""
        # Implementation from existing code...
        pass
        
    def validate_solution(self, solution: str) -> Tuple[bool, str]:
        """Validate solution format and structure"""
        # Implementation from existing code...
        pass
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion"""
        raise NotImplementedError("Subclasses must implement calculate_reward")
        
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Calculate rewards for a batch of completions"""
        rewards = [self.calculate_reward(comp, **kwargs) for comp in completions]
        self.stats.update(rewards, **kwargs)
        return rewards

    async def calculate_reward_async(self, completion: str, **kwargs) -> float:
        """Async version of calculate_reward"""
        return self.calculate_reward(completion, **kwargs)
        
    async def __call_async__(self, completions: List[str], **kwargs) -> List[float]:
        """Async version of __call__"""
        rewards = await asyncio.gather(*[
            self.calculate_reward_async(comp, **kwargs) 
            for comp in completions
        ])
        self.stats.update(rewards, **kwargs)
        return rewards
