import os
import argparse
import signal
from contextlib import contextmanager
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import re
from typing import Optional, Dict, List, Tuple, Any
from tqdm import tqdm
import sys

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.solution_utils import extract_numeric_answer

# Compile regex patterns once
MULTIPLE_CHOICE_PATTERN = re.compile(r'(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*(?:[(\s]|^)[A-D][\s\)\.][^A-D]*')
BOXED_PATTERN = re.compile(r'\\boxed\{')

# Define character ranges
CHINESE_CHARS = frozenset(chr(i) for i in range(0x4e00, 0x9fff + 1))
RUSSIAN_CHARS = frozenset(chr(i) for i in range(0x0400, 0x04FF + 1))

def log_info(message):
    print(f"[INFO] {message}")

def count_boxed_answers(solution: str) -> int:
    """Count number of \boxed{...} occurrences in solution using compiled regex"""
    return len(BOXED_PATTERN.findall(solution))

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

def is_multiple_choice(problem: str) -> bool:
    """Check if the problem contains multiple choice indicators (A,B,C,D)"""
    return bool(MULTIPLE_CHOICE_PATTERN.search(problem))

def extract_answer_from_solution(solution: str) -> Optional[str]:
    """
    Extract the first boxed answer from the solution text by searching for LaTeX boxed answers: \boxed{X}.
    Returns the raw answer string with LaTeX notation preserved, or None if no boxed answer is found.
    """
    def find_matching_brace(s: str, start: int) -> int:
        """Find the index of the matching closing brace for an opening brace at the given start position."""
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

def filter_dataset(repo_name: str, output_dir: str = None, max_length: int = 2000, exclude_multiple_choice: bool = True):
    """
    Download a dataset from HuggingFace, filter it, and save it locally.
    
    Args:
        repo_name: HuggingFace repository name
        output_dir: Directory to save the filtered dataset (default: repo_name's last part)
        max_length: Maximum character length for problems
        exclude_multiple_choice: Whether to exclude multiple choice problems
    """
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Metadata validation was skipped")
    warnings.filterwarnings("ignore", message="Found cached dataset")
    
    # Set default output directory if not provided
    if not output_dir:
        output_dir = repo_name.split('/')[-1] + "_filtered"
    
    # Initialize statistics
    stats = {
        'original': 0,
        'removed_length': 0,
        'removed_no_boxed': 0,
        'removed_multiple_boxed': 0,
        'removed_http_problem': 0,
        'removed_http_solution': 0,
        'removed_non_latin_problem': 0,
        'removed_non_latin_solution': 0,
        'removed_invalid_answer': 0,
        'removed_non_numeric': 0,
        'removed_multiple_choice': 0,
        'final': 0
    }
    
    try:
        # Try to load the dataset
        log_info(f"Loading dataset from {repo_name}...")
        
        # First try with 'train' split
        try:
            dataset = load_dataset(repo_name, split="train")
            split_used = "train"
        except Exception:
            # If 'train' fails, try without specifying a split
            try:
                dataset_dict = load_dataset(repo_name)
                # Use the first available split
                split_used = list(dataset_dict.keys())[0]
                dataset = dataset_dict[split_used]
            except Exception as e:
                log_info(f"Error loading dataset: {e}")
                return
        
        log_info(f"Successfully loaded dataset with {len(dataset)} examples from split '{split_used}'")
        stats['original'] = len(dataset)
        
        # Normalize column names
        def normalize_example(example):
            # Create a normalized example with consistent field names
            normalized = {}
            
            # Map problem/question field
            if 'problem' in example:
                normalized['problem'] = example['problem']
            elif 'question' in example:
                normalized['problem'] = example['question']
            else:
                # If neither field exists, skip this example
                return None
            
            # Map solution field
            if 'solution' in example:
                normalized['solution'] = example['solution']
            else:
                # If no solution field, skip this example
                return None
            
            # Map source field (optional)
            if 'source' in example:
                normalized['source'] = example['source']
            else:
                normalized['source'] = repo_name
            
            # Ensure all fields are strings
            for key in normalized:
                if normalized[key] is not None and not isinstance(normalized[key], str):
                    normalized[key] = str(normalized[key])
            
            return normalized
        
        # Normalize the dataset
        normalized_examples = []
        for example in dataset:
            norm_example = normalize_example(example)
            if norm_example:
                normalized_examples.append(norm_example)
        
        log_info(f"Normalized {len(normalized_examples)} examples")
        
        # Filter the examples
        filtered_examples = []
        for example in tqdm(normalized_examples, desc="Filtering dataset"):
            # Check problem length
            if len(example['problem']) > max_length:
                stats['removed_length'] += 1
                continue
                
            # Check for multiple choice if enabled
            if exclude_multiple_choice and is_multiple_choice(example['problem']):
                stats['removed_multiple_choice'] += 1
                continue
                
            # Check for exactly one boxed answer
            boxed_count = count_boxed_answers(example['solution'])
            if boxed_count == 0:
                stats['removed_no_boxed'] += 1
                continue
            elif boxed_count > 1:
                stats['removed_multiple_boxed'] += 1
                continue
                
            # Check for invalid content in problem
            has_invalid, invalid_type = contains_invalid_content(example['problem'])
            if has_invalid:
                if invalid_type == 'http':
                    stats['removed_http_problem'] += 1
                else:
                    stats['removed_non_latin_problem'] += 1
                continue
                
            # Check for invalid content in solution
            has_invalid, invalid_type = contains_invalid_content(example['solution'])
            if has_invalid:
                if invalid_type == 'http':
                    stats['removed_http_solution'] += 1
                else:
                    stats['removed_non_latin_solution'] += 1
                continue
                
            # Extract and verify answer
            answer = extract_answer_from_solution(example['solution'])
            if answer is None or answer.strip() == "":
                stats['removed_invalid_answer'] += 1
                continue
                
            # Check if answer has a valid numeric value
            numeric_value, _ = extract_numeric_answer(answer)
            if numeric_value is None:
                stats['removed_non_numeric'] += 1
                continue
                
            # Add the answer to the example
            example['answer'] = answer
            example['numeric_value'] = numeric_value
            
            # If we got here, the example passed all filters
            filtered_examples.append(example)
        
        stats['final'] = len(filtered_examples)
        
        # Create the filtered dataset
        if filtered_examples:
            # Create a dataset with consistent schema
            filtered_dataset = Dataset.from_dict({
                'id': list(range(len(filtered_examples))),
                'problem': [ex['problem'] for ex in filtered_examples],
                'solution': [ex['solution'] for ex in filtered_examples],
                'source': [ex['source'] for ex in filtered_examples],
                'answer': [ex['answer'] for ex in filtered_examples],
                'numeric_value': [ex['numeric_value'] for ex in filtered_examples]
            })
            
            # Save the dataset
            output_path = os.path.join('local_datasets', output_dir)
            os.makedirs(output_path, exist_ok=True)
            filtered_dataset.save_to_disk(output_path)
            
            log_info(f"Filtered dataset saved to {output_path}")
        else:
            log_info("No examples passed the filters!")
        
        # Print statistics
        log_info("\nFiltering Statistics:")
        log_info(f"Original dataset size: {stats['original']}")
        log_info("\nRemoved due to:")
        log_info(f"- Problem too long (>{max_length} chars): {stats['removed_length']}")
        if exclude_multiple_choice:
            log_info(f"- Multiple choice problems: {stats['removed_multiple_choice']}")
        log_info(f"- Missing boxed answer: {stats['removed_no_boxed']}")
        log_info(f"- Multiple boxed answers: {stats['removed_multiple_boxed']}")
        log_info(f"- HTTP links in problem: {stats['removed_http_problem']}")
        log_info(f"- HTTP links in solution: {stats['removed_http_solution']}")
        log_info(f"- Non-Latin chars in problem: {stats['removed_non_latin_problem']}")
        log_info(f"- Non-Latin chars in solution: {stats['removed_non_latin_solution']}")
        log_info(f"- Invalid/empty answer: {stats['removed_invalid_answer']}")
        log_info(f"- Non-numeric answer: {stats['removed_non_numeric']}")
        log_info(f"\nFinal dataset size: {stats['final']}")
        
        if stats['original'] > 0:
            reduction_pct = ((stats['original'] - stats['final'])/stats['original'])*100
            log_info(f"Total reduction: {reduction_pct:.1f}%")
        
        return filtered_dataset
        
    except Exception as e:
        log_info(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Filter a HuggingFace dataset for valid numeric answers')
    parser.add_argument('--repo-name', type=str, required=True,
                       help='HuggingFace repository name (e.g., "Metaskepsis/Olympiads_hard")')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Name of output directory under local_datasets/ (default: repo name + "_filtered")')
    parser.add_argument('--max-length', type=int, default=2000,
                       help='Maximum character length for problems (default: 2000)')
    parser.add_argument('--include-multiple-choice', action='store_true',
                       help='Include multiple choice problems (default: exclude them)')
    
    args = parser.parse_args()
    
    filter_dataset(
        repo_name=args.repo_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        exclude_multiple_choice=not args.include_multiple_choice
    )

if __name__ == "__main__":
    main()
