"""Shared configuration for benchmark scripts"""
from dataclasses import dataclass
from argparse import ArgumentParser
from utils.utils import ModelOption

@dataclass
class BaseConfig:
    """Base configuration shared between benchmark types"""
    solver: str
    dataset: str = 'filtered'
    split: str = 'train'
    split_slice: slice = None
    source: str = 'all'
    max_concurrent: int = 256
    best_of: int = 5
    temperature: float = 0.7
    exclude: str = None

    @classmethod
    def add_base_args(cls, parser: ArgumentParser):
        parser.add_argument('--solver', type=str, 
                          choices=[model.name for model in ModelOption],
                          default='LOCAL', help='Model to use for solving problems')
        parser.add_argument('--exclude', type=str,
                          help='JSON file containing IDs to exclude from processing')
        parser.add_argument('--dataset', type=str,
                          choices=['original', 'filtered', 'aime'],
                          default='filtered', help='Dataset to use')
        parser.add_argument('--split', type=str, default='train',
                          help='Dataset split to use (train/validation/test)')
        parser.add_argument('--source', type=str, default='all',
                          help='Filter problems by source (default: all)')
        parser.add_argument('--max-concurrent', type=int, default=256,
                          help='Maximum number of concurrent problems (default: 256)')
        parser.add_argument('--best-of', type=int, default=5,
                          help='Number of attempts per problem (default: 5)')
        parser.add_argument('--temperature', type=float, default=0.7,
                          help='Temperature for model generation (default: 0.7)')

@dataclass
class BenchmarkConfig(BaseConfig):
    """Configuration for standard benchmark with verifier"""
    verifier: str = 'GEMINI_FLASH'
    verifier_temp: float = 0
    
    @classmethod
    def from_args(cls, description: str) -> 'BenchmarkConfig':
        parser = ArgumentParser(description=description)
        cls.add_base_args(parser)
        parser.add_argument('--verifier', type=str,
                          choices=[model.name for model in ModelOption],
                          default='GEMINI_FLASH', help='Model to use for verification')
        parser.add_argument('--verifier-temp', type=float, default=0,
                          help='Temperature for verifier model')
        args = parser.parse_args()
        return cls(**vars(args))

@dataclass
class NumericConfig(BaseConfig):
    """Configuration for numeric benchmark with tolerance"""
    tolerance: float = 1e-6
    
    @classmethod
    def from_args(cls, description: str) -> 'NumericConfig':
        parser = ArgumentParser(description=description)
        cls.add_base_args(parser)
        parser.add_argument('--tolerance', type=float, default=1e-6,
                          help='Tolerance for numeric answer comparison')
        args = parser.parse_args()
        return cls(**vars(args))
