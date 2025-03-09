import re
import random
import logging
from typing import Dict, Optional, Tuple
from datasets import concatenate_datasets, Dataset
from utils.solution_utils import (
    extract_response_section, split_into_steps, get_partial_solutions,
    has_response_section, has_thinking_section, extract_thinking_section
)

# Setup logging
logger = logging.getLogger('data_preparation')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def prepare_solution_data(data: Dataset, system_prompt: str) -> Dataset:
    """Create examples for full solution tasks"""
    logger.info("Creating solution examples...")
    solution_data = data.map(lambda x: {
        'prompt': "<|im_start|>system<|im_sep|>" + system_prompt + "<|im_end|><|im_start|>user<|im_sep|>" + x['problem'] + "<|im_end|><|im_start|>assistant<|im_sep|>",
        'answer': x.get('answer', x.get('correct_answer', '')),  # Try both answer and correct_answer
        'partial_solution': '',  # Empty partial solution indicates full solution task
        'example_type': 'solution'  # Add type for tracking
    })
    return solution_data

def prepare_programming_data(data: Dataset, system_prompt: str) -> Dataset:
    """Create examples for programming tasks using the programming-specific system prompt"""
    logger.info("Creating programming examples...")
    programming_data = data.map(lambda x: {
        'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + x['problem'] + '<|im_end|><|im_start|>assistant<|im_sep|>',
        'answer': x.get('answer', x.get('correct_answer', '')),
        'partial_solution': '',
        'example_type': 'programming'
    })
    return programming_data

def prepare_finalization_data(data: Dataset, system_prompt: str, finalization_system_prompt: str, 
                           tokenizer=None, max_prompt_tokens: int = 1500) -> Dataset:
    """Create examples for finalization tasks"""
    logger.info("Creating finalization examples...")
    
    # Function to count tokens if tokenizer is provided
    def count_tokens(text):
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text) // 4  # Rough estimate if no tokenizer provided
    
    # Calculate token counts for system prompts if tokenizer is provided
    finalization_prompt_tokens = count_tokens(finalization_system_prompt) if tokenizer else 0
    
    def create_partial_solution(example):
        try:
            # Only process examples that have model_solutions with proper steps
            if 'model_solution' not in example or not example['model_solution']:
                return {
                    'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|><|im_start|>assistant<|im_sep|>',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'example_type': 'solution'
                }
            
            # Extract response section using solution_utils
            response = extract_response_section(example['model_solution'])
            if not response:
                return {
                    'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|><|im_start|>assistant<|im_sep|>',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'example_type': 'solution'
                }
            
            # Extract steps using solution_utils
            steps = split_into_steps(response)
            
            # Need at least 2 steps to create a partial solution
            if len(steps) < 2:
                return {
                    'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|><|im_start|>assistant<|im_sep|>',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'example_type': 'solution'
                }
            
            # Use a deterministic approach based on example ID
            example_id = example.get('id', hash(example.get('problem', '')))
            if isinstance(example_id, str):
                example_id = hash(example_id)
            
            # Always use exactly half of the steps for consistency
            split_point = max(1, len(steps) // 2)
            
            # Create partial solutions using solution_utils
            partial_steps = steps[:split_point]
            partial_solutions = get_partial_solutions(partial_steps)
            if not partial_solutions:
                return {
                    'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|><|im_start|>assistant<|im_sep|>',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'example_type': 'solution'
                }
            
            partial_solution = partial_solutions[-1]  # Get the last partial solution (with all steps)
            
            # Check token count for the finalization prompt if tokenizer is provided
            if tokenizer:
                finalization_text = f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}"
                total_tokens = finalization_prompt_tokens + count_tokens(finalization_text)
                
                # If token count is too high, reduce the number of steps
                if total_tokens >= max_prompt_tokens:
                    # Try with just one step
                    if len(steps) > 0:
                        one_step_solutions = get_partial_solutions(steps[:1])
                        if one_step_solutions:
                            partial_solution = one_step_solutions[-1]
                            finalization_text = f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}"
                            total_tokens = finalization_prompt_tokens + count_tokens(finalization_text)
                    
                    # If still too long, return as full solution
                    if total_tokens >= max_prompt_tokens:
                        logger.info(f"Finalization prompt too long ({total_tokens} tokens), converting to full solution")
                        return {
                            'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|><|im_start|>assistant<|im_sep|>',
                            'answer': example.get('answer', example.get('correct_answer', '')),
                            'partial_solution': '',
                            'example_type': 'solution'
                        }
            
            # Format the finalization prompt with the partial solution in the user section
            formatted_prompt = '<|im_start|>system<|im_sep|>' + finalization_system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + \
                f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}<|im_end|><|im_start|>assistant<|im_sep|>"
            
            #logger.info(f"Created finalization example with {split_point} steps out of {len(steps)}")
            return {
                'prompt': formatted_prompt,
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': partial_solution,
                'example_type': 'finalization'
            }
                
        except Exception as e:
            logger.warning(f"Error creating partial solution: {str(e)}")
            # Return as a full solution example on error
            return {
                'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|><|im_start|>assistant<|im_sep|>',
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': '',
                'example_type': 'solution'
            }
    
    # Process all examples for finalization tasks
    finalization_data = data.map(create_partial_solution)
    
    # Filter to only keep actual finalization examples
    finalization_data = finalization_data.filter(lambda x: x['example_type'] == 'finalization')
    
    # Log finalization data details
    finalization_count = len(finalization_data)
    logger.info(f"Found {finalization_count} finalization examples after filtering")
    
    return finalization_data

# Wait data preparation function removed

def prepare_detailed_finalization_data(data: Dataset, system_prompt: str, tokenizer=None) -> Dataset:
    """Create examples for finalization tasks with detailed validation"""
    logger.info("Creating detailed finalization examples...")
    
    def prepare_finalization_example(example):
        try:
            # First, check if we have the required problem field
            if 'problem' not in example or not example['problem']:
                return {
                    'valid': False,
                    'prompt': '',
                    'problem': '',
                    'partial_solution': '',
                    'full_solution': '',
                    'answer': ''
                }
                
            # If example is explicitly marked as correct, consider it valid
            if example.get('is_correct', False) == True:
                # Still need to extract data but skip validation checks
                response = ''
                steps = []
                
                # Try to extract response section if model_solution exists
                if 'model_solution' in example and example['model_solution']:
                    response = extract_response_section(example['model_solution'])
                    if response:
                        steps = split_into_steps(response)
                
                # Create partial solution with at least one step if available
                partial_solution = ''
                full_solution = ''
                if steps:
                    split_point = min(1, len(steps) - 1)
                    partial_solutions = get_partial_solutions(steps[:split_point])
                    full_solutions = get_partial_solutions(steps)
                    
                    if partial_solutions:
                        partial_solution = partial_solutions[-1]
                    if full_solutions:
                        full_solution = full_solutions[-1]
                
                # Get the answer
                answer = example.get('answer', '')
                if not answer and 'correct_answer' in example:
                    answer = example['correct_answer']
                
                # Create the prompt with the data we have
                prompt = '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + \
                        f"Problem: {example.get('problem', '')}\n\nPartial Solution: {partial_solution}<|im_end|><|im_start|>assistant<|im_sep|>"
                
                return {
                    'valid': True,
                    'prompt': prompt,
                    'problem': example.get('problem', ''),
                    'partial_solution': partial_solution,
                    'full_solution': full_solution,
                    'answer': answer
                }
            
            # Skip if no model_solution or if it's empty
            if 'model_solution' not in example or not example['model_solution']:
                return {
                    'valid': False,
                    'prompt': '',
                    'problem': example.get('problem', ''),
                    'partial_solution': '',
                    'full_solution': '',
                    'answer': ''
                }
            
            # Check if solution has a response section
            if not has_response_section(example['model_solution']):
                return {
                    'valid': False,
                    'prompt': '',
                    'problem': example.get('problem', ''),
                    'partial_solution': '',
                    'full_solution': '',
                    'answer': ''
                }
            
            # Extract response section
            response = extract_response_section(example['model_solution'])
            if not response:
                return {
                    'valid': False,
                    'prompt': '',
                    'problem': example.get('problem', ''),
                    'partial_solution': '',
                    'full_solution': '',
                    'answer': ''
                }
            
            # Check if response has step tags
            if '<step>' not in response or '</step>' not in response:
                return {
                    'valid': False,
                    'prompt': '',
                    'problem': example.get('problem', ''),
                    'partial_solution': '',
                    'full_solution': '',
                    'answer': ''
                }
            
            # Split into steps
            steps = split_into_steps(response)
            
            if len(steps) < 2:  # Need at least 2 steps to create a partial solution
                return {
                    'valid': False,
                    'prompt': '',
                    'problem': example.get('problem', ''),
                    'partial_solution': '',
                    'full_solution': '',
                    'answer': ''
                }
            
            # Verify step numbering
            step_numbers = []
            for step in steps:
                # Look for step numbers like "Step 1:", "Step 2:" etc.
                number_match = re.search(r'Step\s+(\d+):', step)
                if not number_match:
                    return {
                        'valid': False,
                        'prompt': '',
                        'problem': example.get('problem', ''),
                        'partial_solution': '',
                        'full_solution': '',
                        'answer': ''
                    }
                
                try:
                    step_num = int(number_match.group(1))
                    step_numbers.append(step_num)
                except ValueError:
                    return {
                        'valid': False,
                        'prompt': '',
                        'problem': example.get('problem', ''),
                        'partial_solution': '',
                        'full_solution': '',
                        'answer': ''
                    }
            
            # Check if step numbers are sequential
            expected_numbers = list(range(1, len(steps) + 1))
            if step_numbers != expected_numbers:
                return {
                    'valid': False,
                    'prompt': '',
                    'problem': example.get('problem', ''),
                    'partial_solution': '',
                    'full_solution': '',
                    'answer': ''
                }
            
            # Randomly decide how many steps to include in partial solution
            random.seed(hash(example.get('id', 0)) % 10000)  # Deterministic but varied
            split_point = random.randint(1, len(steps) - 1)  # At least 1 step, leave at least 1 step
            
            # Create partial solution with the first 'split_point' steps
            partial_solutions = get_partial_solutions(steps[:split_point])
            full_solutions = get_partial_solutions(steps)
            
            partial_solution = partial_solutions[-1] if partial_solutions else ''
            full_solution = full_solutions[-1] if full_solutions else ''
            
            # Get the answer from the example or extract from solution
            answer = example.get('answer', '')
            if not answer and 'correct_answer' in example:
                answer = example['correct_answer']
            
            # Create the prompt with all required fields
            prompt = '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + \
                    f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}<|im_end|><|im_start|>assistant<|im_sep|>"
            return {
                'valid': True,
                'prompt': prompt,
                'problem': example['problem'],
                'partial_solution': partial_solution,
                'full_solution': full_solution,
                'answer': answer
            }
        except Exception as e:
            logger.warning(f"Error processing example: {str(e)}, example ID: {example.get('id', 'unknown')}")
            return {
                'valid': False,
                'prompt': '',
                'problem': example.get('problem', ''),
                'partial_solution': '',
                'full_solution': '',
                'answer': ''
            }
    
    # Process all examples
    processed_data = data.map(prepare_finalization_example)
    
    # Add token count to each example if tokenizer is provided
    if tokenizer:
        def count_tokens(example):
            if not example['valid'] or not example['prompt']:
                return {'token_count': 0}
            return {'token_count': len(tokenizer.encode(example['prompt']))}
        
        processed_data = processed_data.map(count_tokens)
        
        # Log token count statistics
        token_counts = [ex['token_count'] for ex in processed_data if ex['valid']]
        if token_counts:
            logger.info(f"Token count statistics:")
            logger.info(f"  Min: {min(token_counts)}")
            logger.info(f"  Max: {max(token_counts)}")
            logger.info(f"  Mean: {sum(token_counts)/len(token_counts):.2f}")
            logger.info(f"  Examples > 2000 tokens: {sum(1 for t in token_counts if t > 2000)}")
        
        # Filter valid examples and those with token count <= 2000
        valid_data = processed_data.filter(lambda x: x['valid'] and x['token_count'] <= 2000)
    else:
        # If no tokenizer, just filter valid examples
        valid_data = processed_data.filter(lambda x: x['valid'])
    
    # Log validation results
    logger.info(f"Total examples in dataset: {len(data)}")
    logger.info(f"Valid examples after processing: {len(processed_data.filter(lambda x: x['valid']))}")
    logger.info(f"Final valid examples: {len(valid_data)}")
    
    return valid_data

def prepare_tutor_data(data: Dataset, system_prompt: str) -> Dataset:
    """Create examples for tutor tasks using the tutor-specific system prompt"""
    logger.info("Creating tutor examples...")
    
    def create_tutor_example(example):
        try:
            # Check if we have the required fields
            if 'problem' not in example or not example['problem']:
                return None
                
            if 'model_solution' not in example or not example['model_solution']:
                return None
                
            # Extract is_correct flag
            is_correct = example.get('is_correct', False)
            if isinstance(is_correct, str):
                is_correct = is_correct.lower() == 'true'
                
            # Extract wrong_step if available
            wrong_step = example.get('wrong_step', None)
            if wrong_step is not None:
                try:
                    wrong_step = int(wrong_step)
                except (ValueError, TypeError):
                    wrong_step = None
            
            # Format the prompt with the problem and model solution
            formatted_prompt = '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|><|im_start|>user<|im_sep|>' + \
                f"Here is a mathematical problem and a proposed solution:\n\n" + \
                f"Problem:\n{example['problem']}\n\n" + \
                f"Proposed Solution:\n{example['model_solution']}<|im_end|><|im_start|>assistant<|im_sep|>"
            
            return {
                'prompt': formatted_prompt,
                'answer': example.get('answer', example.get('correct_answer', '')),
                'model_solution': example['model_solution'],
                'is_correct': is_correct,
                'wrong_step': wrong_step,
                'example_type': 'tutor'
            }
        except Exception as e:
            logger.warning(f"Error creating tutor example: {str(e)}")
            return None
    
    # Process all examples
    tutor_data = data.map(create_tutor_example)
    
    # Filter out None values
    tutor_data = tutor_data.filter(lambda x: x is not None)
    
    logger.info(f"Created {len(tutor_data)} tutor examples")
    return tutor_data

def prepare_combined_data(data: Dataset, system_prompt: str, finalization_system_prompt: str, 
                          programming_system_prompt: str, tutor_system_prompt: str = None,
                         tokenizer=None, distribution: Dict[str, float] = None) -> Dataset:
    """
    Load and format dataset with multiple example types based on the specified distribution.
    Default distribution:
    - 30% solution examples
    - 30% programming examples
    - 20% finalization examples
    - 20% tutor examples (if tutor_system_prompt is provided)
    """
    # Default distribution if not provided
    if distribution is None:
        distribution = {
            'solution': 0.30,
            'programming': 0.30,
            'finalization': 0.20,
            'tutor': 0.20
        }
    
    # Check if we have model_solutions in the dataset
    has_model_solutions = sum(1 for x in data if 'model_solution' in x and x['model_solution'])
    logger.info(f"Dataset has {has_model_solutions} examples with model_solutions")
    
    # Check how many model solutions have valid steps
    valid_steps = 0
    for example in data:
        if 'model_solution' in example and example['model_solution']:
            response_match = re.search(r'<response>(.*?)</response>', example['model_solution'], re.DOTALL)
            if response_match:
                response = response_match.group(1).strip()
                steps = re.findall(r'<step>(.*?)</step>', response, re.DOTALL)
                if len(steps) >= 2:
                    valid_steps += 1
    
    logger.info(f"Found {valid_steps} examples with valid steps (2+ steps)")
    
    # Create examples for each type using the separate methods
    solution_data = prepare_solution_data(data, system_prompt)
    programming_data = prepare_programming_data(data, programming_system_prompt)
    finalization_data = prepare_finalization_data(data, system_prompt, finalization_system_prompt, tokenizer, max_prompt_tokens=1500)
    
    # Create tutor examples if tutor_system_prompt is provided
    tutor_data = None
    if tutor_system_prompt:
        from utils.agents import TUTOR_SYSTEM_PROMPT
        tutor_data = prepare_tutor_data(data, tutor_system_prompt or TUTOR_SYSTEM_PROMPT)
    
    # Calculate the target number of examples for each type
    total_examples = len(data)
    solution_target = int(total_examples * distribution['solution'])
    programming_target = int(total_examples * distribution['programming'])
    finalization_target = int(total_examples * distribution['finalization'])
    tutor_target = int(total_examples * distribution.get('tutor', 0)) if tutor_data else 0
    
    # Function to count example types in a dataset
    def count_types(dataset):
        type_counts = {}
        for example in dataset:
            example_type = example.get('example_type', 'unknown')
            type_counts[example_type] = type_counts.get(example_type, 0) + 1
        return type_counts
    
    # Log the counts before shuffling
    logger.info(f"Created {len(solution_data)} full solution examples (target: {solution_target})")
    logger.info(f"Created {len(programming_data)} programming examples (target: {programming_target})")
    logger.info(f"Created {len(finalization_data)} finalization examples (target: {finalization_target})")
    if tutor_data:
        logger.info(f"Created {len(tutor_data)} tutor examples (target: {tutor_target})")
    
    # Shuffle and select examples for each type
    solution_data = solution_data.shuffle(seed=42)
    programming_data = programming_data.shuffle(seed=43)
    finalization_data = finalization_data.shuffle(seed=44)
    if tutor_data:
        tutor_data = tutor_data.shuffle(seed=45)
    
    solution_data = solution_data.select(range(min(solution_target, len(solution_data))))
    programming_data = programming_data.select(range(min(programming_target, len(programming_data))))
    finalization_data = finalization_data.select(range(min(finalization_target, len(finalization_data))))
    if tutor_data:
        tutor_data = tutor_data.select(range(min(tutor_target, len(tutor_data))))
    
    # Log type distribution before combining
    logger.info("Dataset type distribution before combining:")
    logger.info(f"Solution dataset: {count_types(solution_data)}")
    logger.info(f"Programming dataset: {count_types(programming_data)}")
    logger.info(f"Finalization dataset: {count_types(finalization_data)}")
    if tutor_data:
        logger.info(f"Tutor dataset: {count_types(tutor_data)}")
    
    # Combine all datasets
    datasets_to_combine = [solution_data, programming_data, finalization_data]
    if tutor_data:
        datasets_to_combine.append(tutor_data)
    
    combined_data = concatenate_datasets(datasets_to_combine)
    
    # Count types in the combined dataset
    combined_types = count_types(combined_data)
    logger.info(f"Combined dataset types: {combined_types}")
    
    # Calculate percentages
    total = sum(combined_types.values())
    percentages = {k: f"{v/total*100:.1f}%" for k, v in combined_types.items()}
    logger.info(f"Type percentages: {percentages}")
    
    return combined_data
