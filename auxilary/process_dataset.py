import os
import json
import argparse
import signal
from contextlib import contextmanager
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from huggingface_hub import HfApi
import re 
from typing import Optional
from tqdm import tqdm
from latex2sympy2 import latex2sympy
import sympy
from typing import Optional, Dict, List, Callable, Tuple, TypeVar, Any
# Convert to Hugging Face dataset format with explicit schema
from datasets import Features, Value
from utils.solution_utils import extract_numeric_answer, extract_answer_from_solution

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def count_boxed_answers(solution: str) -> int:
    """Count number of \boxed{...} occurrences in solution using compiled regex"""
    return len(BOXED_PATTERN.findall(solution))

# Compile regex patterns once
MULTIPLE_CHOICE_PATTERN = re.compile(r'(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*')
BOXED_PATTERN = re.compile(r'\\boxed\{')

# Define character ranges
CHINESE_CHARS = frozenset(chr(i) for i in range(0x4e00, 0x9fff + 1))
RUSSIAN_CHARS = frozenset(chr(i) for i in range(0x0400, 0x04FF + 1))

def contains_invalid_content(text: str) -> Tuple[bool, str]:
    """Check if text contains http links or non-Latin characters"""
    if 'http' in text.lower():
        return True, 'http'
    
    # Use set intersection for faster character checking
    text_chars = set(text)
    if text_chars & CHINESE_CHARS:
        return True, 'chinese'
    if text_chars & RUSSIAN_CHARS:
        return True, 'russian'
    
    return False, ''

def is_numeric_answer(answer: str) -> bool:
    """Check if the answer represents a number using extract_numeric_answer"""
    if not answer or not answer.strip():
        return False
    
    try:
        numeric_value, _ = extract_numeric_answer(answer)
        return numeric_value is not None
    except Exception:
        return False

def is_multiple_choice(problem: str) -> bool:
    """Check if the problem contains multiple choice indicators (A,B,C,D)"""
    # Look for patterns like "(A)", "A)", "A.", etc followed by another option
    pattern = r'(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*'
    return bool(re.search(pattern, problem))

def main():
    # Initialize Hugging Face API
    api = HfApi()
    parser = argparse.ArgumentParser(description='Process local dataset for olympiads with valid answers')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to local dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Name of output directory under local_datasets/')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--repo-name', type=str, default='artnoage/Numina',
                       help='HuggingFace repository name')
    parser.add_argument('--only-numbers', action='store_true',
                       help='Only keep problems where the answer is a number')
    parser.add_argument('--exclude', type=str,
                       help='JSON file containing problems to exclude')
    parser.add_argument('--include', type=str,
                       help='JSON file containing problems to include (keep only these)')
    parser.add_argument('--exclude-multiple-choice', action='store_true', default=False,
                       help='Exclude multiple choice problems (default: keep them)')
    args = parser.parse_args()

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Metadata validation was skipped")
    warnings.filterwarnings("ignore", message="Found cached dataset")
    
    # Load the local dataset
    try:
        print(f"Loading local dataset from {args.dataset}...")
        dataset = load_from_disk(args.dataset)[args.split]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"\nOriginal dataset size: {len(dataset)}")

    # Load exclude/include lists if provided
    excluded_problems = set()
    included_problems = set()
    
    if args.exclude and os.path.exists(args.exclude):
        try:
            with open(args.exclude, 'r') as f:
                exclude_data = json.load(f)
                excluded_problems = {item['problem'] for item in exclude_data if 'problem' in item}
            print(f"Loaded {len(excluded_problems)} problems to exclude")
        except Exception as e:
            print(f"Error loading exclude file: {e}")
            return

    if args.include and os.path.exists(args.include):
        try:
            with open(args.include, 'r') as f:
                include_data = json.load(f)
                included_problems = {item['problem'] for item in include_data if 'problem' in item}
            print(f"Loaded {len(included_problems)} problems to include")
        except Exception as e:
            print(f"Error loading include file: {e}")
            return

    # Apply include/exclude filters
    if included_problems:
        dataset = dataset.filter(lambda x: any(x['problem'] == inc['problem'] for inc in include_data))
        print(f"After including only specified problems: {len(dataset)}")
    if excluded_problems:
        dataset = dataset.filter(lambda x: not any(x['problem'] == exc['problem'] for exc in exclude_data))
        print(f"After excluding problems: {len(dataset)}")

    # Filter out multiple choice problems if requested
    if args.exclude_multiple_choice:
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: not is_multiple_choice(x['problem']))
        removed_count = original_size - len(dataset)
        print(f"Removed {removed_count} multiple choice problems")

    # Initialize detailed statistics
    stats = {
        'original': len(dataset),
        'removed_source': 0,
        'removed_no_boxed': 0,
        'removed_multiple_boxed': 0,
        'removed_http_problem': 0,
        'removed_http_solution': 0,
        'removed_non_latin_problem': 0,
        'removed_non_latin_solution': 0,
        'removed_invalid_answer': 0,
        'removed_non_numeric': 0
    }

    # Filter by source if specified
    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
        stats['removed_source'] = stats['original'] - len(dataset)
        print(f"After filtering for {args.source}: {len(dataset)}")
    
    # Filter for solutions containing exactly one boxed answer
    def has_valid_answer(example: Dict[str, str]) -> bool:
        if 'solution' not in example:
            stats['removed_no_boxed'] += 1
            return False
            
        # Check for exactly one boxed answer
        boxed_count = count_boxed_answers(example['solution'])
        if boxed_count == 0:
            stats['removed_no_boxed'] += 1
            return False
        elif boxed_count > 1:
            stats['removed_multiple_boxed'] += 1
            return False
            
        # Combined check for HTTP and non-Latin characters
        has_invalid, invalid_type = contains_invalid_content(example['problem'])
        if has_invalid:
            if invalid_type == 'http':
                stats['removed_http_problem'] += 1
            else:
                stats['removed_non_latin_problem'] += 1
            return False
            
        has_invalid, invalid_type = contains_invalid_content(example['solution'])
        if has_invalid:
            if invalid_type == 'http':
                stats['removed_http_solution'] += 1
            else:
                stats['removed_non_latin_solution'] += 1
            return False
            
        # Verify answer extraction
        answer = extract_answer_from_solution(example['solution'])
        if answer is None or answer.strip() == "":
            stats['removed_invalid_answer'] += 1
            return False
            
        if args.only_numbers and not is_numeric_answer(answer):
            stats['removed_non_numeric'] += 1
            return False
            
        return True

    # Apply filters with progress bar
    filtered_examples = []
    for example in tqdm(dataset, desc="Filtering dataset"):
        if has_valid_answer(example):
            filtered_examples.append(example)
    
    filtered_dataset = Dataset.from_dict({
        k: [example[k] for example in filtered_examples]
        for k in dataset.features
    })
    
    # Calculate detailed statistics
    stats['final'] = len(filtered_examples)
    stats['removed_invalid'] = len(dataset) - len(filtered_dataset)
    
    # Print detailed statistics
    print("\nDetailed Filtering Statistics:")
    print(f"Original dataset size: {stats['original']}")
    if args.source.lower() != 'all':
        print(f"Removed due to source filter: {stats['removed_source']}")
    print("\nRemoved due to:")
    print(f"- Missing boxed answer: {stats['removed_no_boxed']}")
    print(f"- Multiple boxed answers: {stats['removed_multiple_boxed']}")
    print(f"- HTTP links in problem: {stats['removed_http_problem']}")
    print(f"- HTTP links in solution: {stats['removed_http_solution']}")
    print(f"- Non-Latin chars in problem: {stats['removed_non_latin_problem']}")
    print(f"- Non-Latin chars in solution: {stats['removed_non_latin_solution']}")
    print(f"- Invalid/empty answer: {stats['removed_invalid_answer']}")
    if args.only_numbers:
        print(f"- Non-numeric answer: {stats['removed_non_numeric']}")
    print(f"\nFinal dataset size: {stats['final']}")
    print(f"Total reduction: {((stats['original'] - stats['final'])/stats['original'])*100:.1f}%")


    
    features = Features({
        'id': Value('int64'),
        'problem': Value('string'),
        'solution': Value('string'), 
        'source': Value('string'),
        'answer': Value('string')
    })
    
    # Create dataset with IDs and explicit schema
    filtered_dataset_dict = DatasetDict({
        args.split: Dataset.from_dict(
            {
                'id': list(range(len(filtered_dataset))),
                'problem': filtered_dataset['problem'],
                'solution': filtered_dataset['solution'],
                'source': filtered_dataset['source'],
                'answer': [extract_answer_from_solution(sol) for sol in filtered_dataset['solution']]
            },
            features=features
        )
    })

    # Save in local_datasets directory
    output_path = os.path.join('local_datasets', args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Create dataset_info.json
    # Convert features to JSON-serializable format
    features_json = {
        'id': {'dtype': 'int64', 'id': None},
        'problem': {'dtype': 'string', 'id': None},
        'solution': {'dtype': 'string', 'id': None},
        'source': {'dtype': 'string', 'id': None},
        'answer': {'dtype': 'string', 'id': None}
    }
    
    dataset_info = {
        "description": "Filtered dataset with valid answers",
        "features": features_json,
        "splits": {
            args.split: {
                "name": args.split,
                "num_examples": len(filtered_dataset)
            }
        }
    }
    
    # Save dataset_info.json
    with open(os.path.join(output_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save the dataset
    filtered_dataset_dict.save_to_disk(output_path)
    print(f"\nDataset saved locally to: {output_path}")


if __name__ == "__main__":
    main()
