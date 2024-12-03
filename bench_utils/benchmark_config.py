"""Configuration for benchmark scripts"""
from dataclasses import dataclass
from argparse import ArgumentParser
from enum import Enum

class ModelOption(Enum):
    """Enum class representing different model options for chat completion.
    
    Each enum value corresponds to a specific model endpoint that can be used
    with either OpenRouter API, SambaNova API, or local deployment.
    """
    CLAUDE = "anthropic/claude-3.5-sonnet:beta"
    GEMINI_PRO_FREE = "google/gemini-pro-1.5-exp"
    GEMINI_FLASH_FREE="google/gemini-flash-1.5-exp"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GEMINI_FLASH="google/gemini-flash-1.5"
    GPT = "openai/gpt-4o"
    GPT_MINI="openai/gpt-4o-mini"
    MASTER = "openai/o1-preview-2024-09-12"
    MASTER_MINI="openai/o1-mini"
    LOCAL = "/Home/stat/laschos/AIMO2_initial/models/20241130_144413"
    #LOCAL ="artnoage/metastral"
    NEMOTRON= "nvidia/llama-3.1-nemotron-70b-instruct"
    CODER="qwen/qwen-2.5-coder-32b-instruct"

@dataclass
class BenchmarkConfig:
    """Unified configuration for benchmarking with optional numeric verification"""
    # Solver settings
    solver: str
    temperature: float = 0.7
    
    # Dataset settings
    dataset: str = 'filtered'
    split: str = 'train'
    split_slice: slice = None
    source: str = 'all'
    exclude: str = None
    
    # Execution settings
    max_concurrent: int = 256
    best_of: int = 5
    initial_steps: int = 1
    
    # Verification settings
    verification_type: str = 'numeric'  # 'numeric', 'answer', or 'solution'
    verifier: str = 'GEMINI_FLASH'
    second_verifier: str = 'CODER'
    verifier_temp: float = 0
    tolerance: float = 1e-6
    
    @classmethod
    def from_args(cls, description: str) -> 'BenchmarkConfig':
        parser = ArgumentParser(description=description)
        
        # Solver arguments
        parser.add_argument('--solver', type=str, 
                          choices=[model.name for model in ModelOption],
                          default='LOCAL', help='Model to use for solving problems')
        parser.add_argument('--temperature', type=float, default=0.7,
                          help='Temperature for model generation (default: 0.7)')
                          
        # Dataset arguments
        parser.add_argument('--dataset', type=str,
                          choices=['original', 'filtered', 'aime'],
                          default='filtered', help='Dataset to use')
        parser.add_argument('--split', type=str, default='train',
                          help='Dataset split to use (train/validation/test)')
        parser.add_argument('--source', type=str, default='all',
                          help='Filter problems by source (default: all)')
        parser.add_argument('--exclude', type=str,
                          help='JSON file containing IDs to exclude from processing')
                          
        # Execution arguments
        parser.add_argument('--max-concurrent', type=int, default=256,
                          help='Maximum number of concurrent problems (default: 256)')
        parser.add_argument('--best-of', type=int, default=5,
                          help='Number of attempts per problem (default: 5)')
        parser.add_argument('--initial-steps', type=int, default=1,
                          help='Number of initial steps before completion (default: 1)')
                          
        # Verification arguments
        parser.add_argument('--verifier', type=str,
                          choices=[model.name for model in ModelOption],
                          default='GEMINI_FLASH', help='Model to use for verification')
        parser.add_argument('--second-verifier', type=str,
                          choices=[model.name for model in ModelOption],
                          default='CODER', help='Second model to use for verification')
        parser.add_argument('--verifier-temp', type=float, default=0,
                          help='Temperature for verifier models')
        parser.add_argument('--verification-type', type=str,
                          choices=['numeric', 'answer', 'solution'],
                          default='numeric',
                          help='Type of verification to use')
        parser.add_argument('--tolerance', type=float, default=1e-6,
                          help='Tolerance for numeric answer comparison (only used with --numeric)')
                          
        args = parser.parse_args()
        return cls(**vars(args))
