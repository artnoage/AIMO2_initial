"""
Reward modules for evaluating model completions.

This package contains various reward classes for different types of evaluation:
- BaseReward: Abstract base class for all reward implementations
- SolutionReward: Evaluates mathematical solution quality
- ProgrammingReward: Evaluates programming solution quality
- SolverReward: Evaluates solution quality with detailed tracking
- SolverVerReward: Evaluates solution verification capabilities
"""

from grpo.rewards.base_reward import BaseReward
from grpo.rewards.solution_reward import SolutionReward
from grpo.rewards.programming_reward import ProgrammingReward
from grpo.rewards.solver_ver_reward import SolverVerReward

# Import the original SolverReward from the parent directory
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from grpo.solver_ref_reward import SolverReward

__all__ = [
    'BaseReward',
    'SolutionReward',
    'ProgrammingReward',
    'SolverReward',
    'SolverVerReward',
]
