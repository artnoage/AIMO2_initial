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


def extract_numeric_answer(answer: str, debug: bool = False) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value from a LaTeX answer string.
    First tries to evaluate using sympy, then falls back to direct float conversion.
    Returns float if found, None otherwise.
    """
    if not answer:
        return None, "No answer provided" if debug else (None, None)
        
    # Clean the answer string
    clean_answer = answer.strip()
    clean_answer = re.sub(r'\\textbf{([^}]*)}', r'\1', clean_answer)  # Remove \textbf{} first   
    clean_answer = re.sub(r'\\text{[^}]*}', '', clean_answer)
    clean_answer = clean_answer.replace('\\,', '')
    clean_answer = clean_answer.replace('^\\circ', '')  # Remove degree symbol
    
    # Only split on = or \approx if there's a single term before it
    def has_single_term(text: str) -> bool:
        """Check if text has only a single term (no operators outside brackets)"""
        bracket_level = 0
        for char in text:
            if char == '{':
                bracket_level += 1
            elif char == '}':
                bracket_level -= 1
            elif bracket_level == 0 and char in '+-*/^':
                return False
        return True

    # Handle = and \approx separately
    if '=' in clean_answer:
        eq_pos = clean_answer.rfind('=')
        before_eq = clean_answer[:eq_pos].strip()
        if has_single_term(before_eq):
            clean_answer = clean_answer[eq_pos + 1:].strip()
    
    if '\\approx' in clean_answer:
        approx_pos = clean_answer.rfind('\\approx')
        before_approx = clean_answer[:approx_pos].strip()
        if has_single_term(before_approx):
            clean_answer = clean_answer[approx_pos + 8:].strip()
                
    if not clean_answer:
        return None, "Empty answer after cleaning" if debug else (None, None)
    try:
        with time_limit(10):  # 10 second timeout
            # Parse LaTeX to sympy-compatible format
            latex_expr = latex2sympy(clean_answer)
            # Convert to sympy expression and evaluate
            expr = sympy.sympify(latex_expr)
            # Handle both single values and lists/matrices
            if hasattr(expr, 'evalf'):
                result = float(expr.evalf())
            elif isinstance(expr, list) or isinstance(expr, tuple) or (
                hasattr(expr, 'is_Matrix') and expr.is_Matrix
            ):
                return (None, f"Rejected list/matrix answer: {expr}") if debug else (None, None)
            else:
                result = float(expr)
            return (result, f"Sympy success: {clean_answer} -> {latex_expr} -> {expr} -> {result}") if debug else (result, None)
    except TimeoutException:
        return (None, f"Timeout error: Processing took more than 10 seconds for input: {clean_answer}") if debug else (None, None)
    except (sympy.SympifyError, TypeError, ValueError) as e:
        return (None, f"Sympy error: {str(e)} on input: {clean_answer}") if debug else (None, None)
    
def extract_answer_from_solution(solution: str) -> Optional[str]:
    """
    Extract the first boxed answer from the solution text by searching for LaTeX boxed answers: \boxed{X}.
    Returns the raw answer string with LaTeX notation preserved, or None if no boxed answer is found.
    """
    def find_matching_brace(s: str, start: int) -> int:
        """
        Find the index of the matching closing brace for an opening brace at the given start position.
        
        Args:
            s (str): The string to search.
            start (int): The index of the opening brace '{'.
        
        Returns:
            int: The index of the matching closing brace '}', or -1 if not found.
        """
        count = 1  # Initialize brace count
        i = start + 1  # Start searching after the opening brace
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Pattern to find all occurrences of \boxed{ with proper escaping
    pattern = re.compile(r'\\boxed\{')
    for match in pattern.finditer(solution):
        start = match.end() - 1  # Position of the opening brace '{'
        end = find_matching_brace(solution, start)
        if end != -1:
            # Extract content between the braces
            content = solution[start + 1:end].strip()
            return content  # Return the first found boxed content

    return None  # Return None if no boxed content is found

def count_boxed_answers(solution: str) -> int:
    """Count number of \boxed{...} occurrences in solution"""
    count = 0
    pos = 0
    while True:
        pos = solution.find('\\boxed{', pos)
        if pos == -1:
            break
        count += 1
        pos += 1
    return count

def contains_http(text: str) -> bool:
    """Check if text contains http links"""
    return 'http' in text.lower()

def is_numeric_answer(answer: str) -> bool:
    """Check if the answer represents a number using extract_numeric_answer"""
    if not answer or not answer.strip():
        return False
    
    try:
        numeric_value, _ = extract_numeric_answer(answer)
        return numeric_value is not None
    except Exception:
        return False

def contains_non_latin(text: str) -> bool:
    """Check if text contains Chinese or Russian characters"""
    for char in text:
        # Check for Chinese characters
        if '\u4e00' <= char <= '\u9fff':
            return True
        # Check for Russian characters
        if '\u0400' <= char <= '\u04FF':
            return True
    return False

def is_multiple_choice(problem: str) -> bool:
    """Check if the problem contains multiple choice indicators (A,B,C,D)"""
    # Look for patterns like "(A)", "A)", "A.", etc followed by another option
    pattern = r'(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*'
    return bool(re.search(pattern, problem))

def main():
    # Initialize Hugging Face API
    api = HfApi()
    parser = argparse.ArgumentParser(description='Process dataset for olympiads with valid answers')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g. "AI-MO/NuminaMath-CoT") or path to local dataset')
    parser.add_argument('--local', action='store_true', default=False,
                       help='Load dataset from local disk instead of Hugging Face Hub')
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
    parser.add_argument('--exclude-multiple-choice', action='store_true', default=False,
                       help='Exclude multiple choice problems (default: keep them)')
    args = parser.parse_args()

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Metadata validation was skipped")
    warnings.filterwarnings("ignore", message="Found cached dataset")
    
    # Load the dataset
    try:
        if args.local:
            print(f"Loading local dataset from {args.dataset}...")
            dataset = load_from_disk(args.dataset)[args.split]
        else:
            print(f"Loading dataset {args.dataset} from Hugging Face Hub...")
            dataset = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"\nOriginal dataset size: {len(dataset)}")

    # Load exclude list if provided
    excluded_problems = set()
    if args.exclude and os.path.exists(args.exclude):
        try:
            with open(args.exclude, 'r') as f:
                exclude_data = json.load(f)
                excluded_problems = {item['problem'] for item in exclude_data if 'problem' in item}
            print(f"Loaded {len(excluded_problems)} problems to exclude")
        except Exception as e:
            print(f"Error loading exclude file: {e}")
            return

    # Filter out excluded problems
    if excluded_problems:
        dataset = dataset.filter(lambda x: x['problem'] not in excluded_problems)
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
    def has_valid_answer(example):
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
            
        # Check for HTTP links
        if contains_http(example['problem']):
            stats['removed_http_problem'] += 1
            return False
        if contains_http(example['solution']):
            stats['removed_http_solution'] += 1
            return False
            
        # Check for non-Latin characters
        if contains_non_latin(example['problem']):
            stats['removed_non_latin_problem'] += 1
            return False
        if contains_non_latin(example['solution']):
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

    # Save locally first
    output_dir = "../numina_olympiads"
    os.makedirs(output_dir, exist_ok=True)
    
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
        "description": "Filtered NuminaMath-CoT dataset containing only olympiads problems with valid answers in LaTeX format",
        "citation": "@misc{numina2024,\n  author={AI-MO},\n  title={NuminaMath-CoT Dataset},\n  year={2024},\n  howpublished={\\url{https://huggingface.co/datasets/AI-MO/NuminaMath-CoT}}\n}",
        "homepage": "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT",
        "license": "mit",
        "features": features_json,
        "splits": {
            args.split: {
                "name": args.split,
                "num_bytes": None,
                "num_examples": len(filtered_dataset),
                "dataset_name": "Numina-Olympiads"
            }
        },
        "tags": [
            "mathematics",
            "olympiads", 
            "problem-solving",
            "latex",
            "mathematical-reasoning",
            "math-word-problems",
            "olympiad-math"
        ],
        "task_categories": [
            "text-generation",
            "mathematical-reasoning"
        ],
        "task_ids": [
            "math-word-problems",
            "olympiad-math"  
        ],
        "metrics": [
            {
                "name": "filtered_ratio",
                "type": "ratio",
                "value": len(filtered_dataset) / len(dataset),
                "description": "Ratio of filtered dataset size to original dataset size"
            }
        ],
        "paper_authors": ["AI-MO"],
        "dataset_size": None,
        "config_name": args.split
    }
    
    # Save dataset_info.json
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save the dataset
    filtered_dataset_dict.save_to_disk(output_dir)
    print(f"\nDataset saved locally to: {output_dir}")

    # Try to push to Hugging Face Hub
    try:
        # Get the username from huggingface-cli
        username = api.whoami()["name"]
        repo_id = args.repo_name
        
        # Create or get the repository
        try:
            api.create_repo(
                repo_id=repo_id,
                private=True,
                repo_type="dataset"
            )
        except Exception as repo_error:
            print(f"Note: Repository may already exist: {repo_error}")
        
        # Push the dataset
        filtered_dataset_dict.push_to_hub(repo_id)
        
        # Update the dataset card
        readme_content = f"""---
annotations_creators:
  - expert-generated
language:
  - en
language_creators:
  - expert-generated
license: mit
multilinguality:
  - monolingual
pretty_name: Numina-Olympiads
size_categories:
  - 1K<n<10K
source_datasets:
  - AI-MO/NuminaMath-CoT
task_categories:
  - text-generation
  - mathematical-reasoning
task_ids:
  - math-word-problems
  - olympiad-math
paperswithcode_id: numina-olympiads
tags:
  - mathematics
  - olympiads
  - problem-solving
  - latex
  - mathematical-reasoning
  - math-word-problems
  - olympiad-math
metrics:
  - name: filtered_ratio
    type: ratio 
    value: {len(filtered_dataset) / len(dataset):.3f}
    description: Ratio of filtered dataset size to original dataset size
---

# Numina-Olympiads

Filtered NuminaMath-CoT dataset containing only olympiads problems with valid answers.

## Dataset Information
- Split: {args.split}
- Original size: {len(dataset)}
- Filtered size: {len(filtered_dataset)}
- Source: olympiads
- All examples contain valid boxed answers

## Dataset Description
This dataset is a filtered version of the NuminaMath-CoT dataset, containing only problems from olympiad sources that have valid boxed answers. Each example includes:
- A mathematical word problem
- A detailed solution with step-by-step reasoning
- A boxed final answer in LaTeX format

## Usage
The dataset is particularly useful for:
- Training and evaluating math problem-solving models
- Studying olympiad-style mathematical reasoning
- Testing model capabilities on complex word problems
"""
        with open("README.md", "w") as f:
            f.write(readme_content)
            
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("\nSuccessfully pushed dataset to Hugging Face Hub")
    except Exception as e:
        print(f"\nFailed to push to Hugging Face Hub: {e}")
        print("You can still use the locally saved dataset")


if __name__ == "__main__":
    main()
