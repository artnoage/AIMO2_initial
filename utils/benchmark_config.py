"""Configuration for benchmark scripts"""
from dataclasses import dataclass
from argparse import ArgumentParser
from enum import Enum


class ModelOption(Enum):
    """Enum class representing different model options for chat completion.
    
    Each enum value corresponds to a specific model endpoint that can be used
    with either OpenRouter API, SambaNova API, or local deployment.
    """
    CLAUDE = "anthropic/claude-3.5-sonnet"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GEMINI_FLASH="google/gemini-2.0-flash-001"
    GPT = "openai/gpt-4o"
    GPT_MINI="openai/gpt-4o-mini"
    MASTER = "openai/o1-preview-2024-09-12"
    MASTER_MINI="openai/o1-mini"
    LOCAL_0 ="/Home/stat/laschos/math/AIMO2_initial/models/waitA"
    LOCAL_1="/Home/stat/laschos/math/AIMO2_initial/models/waitB"
    LOCAL_2="/Home/stat/laschos/math/AIMO2_initial/models/wait_2/20250302_203326"
    LOCAL_3="/Home/stat/laschos/math/AIMO2_initial/models/completion/20250302_211638"
    LOCAL_4="/Home/stat/laschos/math/AIMO2_initial/models/qwen_sft/20250303_224627"
    NEMOTRON= "nvidia/llama-3.1-nemotron-70b-instruct"
    CODER="qwen/qwen-2.5-coder-32b-instruct"
    DEEP="deepseek/deepseek-chat"
    QWEN="qwen/qwq-32b-preview"
@dataclass
class BenchmarkConfig:
    """Unified configuration for benchmarking with optional numeric verification"""
    # Model settings
    main: str = "LOCAL_0"
    auxiliary: str = "LOCAL_1"  # If None, uses same as main
    auxiliary2: str = "LOCAL_2"  # Third model option
    main_port: int = 8000
    auxiliary_port: int = 6000
    auxiliary2_port: int = 6000
    main_temp: float = 0.7
    auxiliary_temp: float = 0.7
    auxiliary2_temp: float = 0.0
    
    # Enable "wait a second" pattern for main model
    use_wait_pattern: bool = True  # Default to False
    
    # Dataset settings
    dataset: str = 'filtered'
    split: str = 'train'
    split_slice: slice = None
    source: str = 'all'
    exclude: str = None
    seed: int = 42  # Seed for dataset operations
    
    # Execution settings
    max_concurrent: int = 1
    best_of: int = 1
    completions: int = 2
    
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
        
        # Model arguments
        parser.add_argument('--main', type=str, 
                          choices=[model.name for model in ModelOption],
                          default='LOCAL_0', help='Main model to use for solving problems')
        parser.add_argument('--auxiliary', type=str,
                          choices=[model.name for model in ModelOption],
                          default='LOCAL_1', help='Auxiliary model to use for judging problems')
        parser.add_argument('--auxiliary2', type=str,
                          choices=[model.name for model in ModelOption],
                          default='LOCAL_2', help='Second auxiliary model (optional)')
        parser.add_argument('--use-wait-pattern', action='store_true',
                          help='Enable "wait a second" pattern for main model')
        parser.add_argument('--main-port', type=int, default=8000,
                          help='Port for main model server (default: 8000)')
        parser.add_argument('--auxiliary-port', type=int, default=6000,
                          help='Port for auxiliary model server (default: 6000)')
        parser.add_argument('--auxiliary2-port', type=int, default=7000,
                          help='Port for second auxiliary model server (default: 7000)')
        parser.add_argument('--main-temp', type=float, default=0.7,
                          help='Temperature for main model generation (default: 0.9)')
        parser.add_argument('--auxiliary-temp', type=float, default=0.7,
                          help='Temperature for auxiliary model generation (default: 0.7)')
        parser.add_argument('--auxiliary2-temp', type=float, default=0.0,
                          help='Temperature for second auxiliary model generation (default: 0.0)')
                          
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
        parser.add_argument('--best-of', type=int, default=1,
                          help='Number of attempts per problem (default: 5)')
        parser.add_argument('--completions', type=int, default=3,
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
        # Convert args to dictionary
        args_dict = vars(args)
        
        # Create config instance
        config = cls(**args_dict)
        
        
        return config
