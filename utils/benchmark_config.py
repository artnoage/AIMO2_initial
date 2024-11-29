"""Shared configuration for benchmark scripts"""
from dataclasses import dataclass
from argparse import ArgumentParser
from utils.utils import ModelOption

@dataclass
class BenchmarkConfig:
    solver: str
    split: str = 'train'
    split_slice: slice = None
    source: str = 'all'
    max_concurrent: int = 256
    best_of: int = 5
    temperature: float = 0.7
    
    @classmethod
    def from_args(cls, description: str) -> 'BenchmarkConfig':
        parser = ArgumentParser(description=description)
        parser.add_argument('--solver', type=str, 
                          choices=[model.name for model in ModelOption],
                          default='LOCAL', help='Model to use for solving problems')
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
        args = parser.parse_args()
        return cls(**vars(args))
