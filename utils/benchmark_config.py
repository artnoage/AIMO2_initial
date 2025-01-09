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
    LOCAL ="/Home/stat/laschos/AIMO2_initial/models/20250108_182212"
    NEMOTRON= "nvidia/llama-3.1-nemotron-70b-instruct"
    CODER="qwen/qwen-2.5-coder-32b-instruct"

@dataclass
class BenchmarkConfig:
    """Unified configuration for benchmarking with optional numeric verification"""
    # Solver settings
    solver: str
    port: int = 8000
    temperature: float = 0.9
    
    # Dataset settings
    dataset: str = 'filtered'
    split: str = 'train'
    split_slice: slice = None
    source: str = 'all'
    exclude: str = None
    seed: int = 42  # Seed for dataset operations
    
    # Execution settings
    max_concurrent: int = 256
    best_of: int = 8
    completions: int = 30
    
    # Verification settings
    tolerance: float = 1e-6  # Tolerance for numeric answer comparison
    
    # Output settings
    produce_statistics: bool = True
    stats_update_freq: int = 100  # How often to update statistics (number of examples)
    create_dataset: bool = False  # Whether to create a HuggingFace dataset
    upload_dataset: bool = False  # Whether to upload the dataset to HuggingFace Hub
    
    @classmethod
    def from_args(cls, description: str) -> 'BenchmarkConfig':
        parser = ArgumentParser(description=description)
        
        # Solver arguments
        parser.add_argument('--solver', type=str, 
                          choices=[model.name for model in ModelOption],
                          default='LOCAL', help='Model to use for solving problems')
        parser.add_argument('--port', type=int, default=8000,
                          help='Port for local model server (default: 8000)')
        parser.add_argument('--temperature', type=float, default=0.9,
                          help='Temperature for model generation (default: 0.9)')
                          
        # Dataset arguments
        parser.add_argument('--dataset', type=str,
                          default='Metaskepsis/Numina',
                          help='HuggingFace dataset to use (default: Metaskepsis/Numina)')
        parser.add_argument('--split', type=str, default='train',
                          help='Dataset split to use (train/validation/test)')
        parser.add_argument('--source', type=str, default='all',
                          help='Filter problems by source (default: all)')
        parser.add_argument('--exclude', type=str,
                          help='JSON file containing IDs to exclude from processing')
        parser.add_argument('--seed', type=int, default=42,
                          help='Seed for dataset operations (default: 42)')
                          
        # Execution arguments
        parser.add_argument('--max-concurrent', type=int, default=64,
                          help='Maximum number of concurrent problems (default: 64)')
        parser.add_argument('--best-of', type=int, default=10,
                          help='Number of attempts per problem (default: 5)')
        parser.add_argument('--completions', type=int, default=30,
                          help='Number of completions to try per path (default: 15)')
                          
        # Verification arguments
        parser.add_argument('--tolerance', type=float, default=1e-6,
                          help='Tolerance for numeric answer comparison')
        
        # Output settings
        parser.add_argument('--produce-statistics', action='store_true', default=True,
                          help='Generate detailed statistics file (default: True)')
        parser.add_argument('--stats-update-freq', type=int, default=100,
                          help='How often to update statistics (number of examples)')
        parser.add_argument('--create-dataset', action='store_true',
                          help='Create a HuggingFace dataset from results')
        parser.add_argument('--upload-dataset', action='store_true',
                          help='Upload the created dataset to HuggingFace Hub')
        
        args = parser.parse_args()
        return cls(**vars(args))
