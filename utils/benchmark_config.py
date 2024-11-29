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
                          help='Dataset split to use (train/validation/test), optionally with slice notation (e.g. train[:1000])')
        parser.add_argument('--source', type=str, default='all',
                          help='Filter problems by source (default: all)')
        parser.add_argument('--max-concurrent', type=int, default=256,
                          help='Maximum number of concurrent problems (default: 256)')
        parser.add_argument('--best-of', type=int, default=5,
                          help='Number of attempts per problem (default: 5)')
        parser.add_argument('--temperature', type=float, default=0.7,
                          help='Temperature for model generation (default: 0.7)')
        args = parser.parse_args()
        # Parse split argument for slice notation
        split_parts = args.split.split('[')
        config_args = vars(args)
        
        if len(split_parts) > 1:
            # Has slice notation
            config_args['split'] = split_parts[0]
            slice_str = split_parts[1].rstrip(']')
            
            # Parse slice notation
            slice_parts = slice_str.split(':')
            start = int(slice_parts[0]) if slice_parts[0] else None
            stop = int(slice_parts[1]) if len(slice_parts) > 1 and slice_parts[1] else None
            step = int(slice_parts[2]) if len(slice_parts) > 2 and slice_parts[2] else None
            
            config_args['split_slice'] = slice(start, stop, step)
        else:
            config_args['split_slice'] = None
            
        return cls(**config_args)
