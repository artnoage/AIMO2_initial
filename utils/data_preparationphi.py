import re
import random
import logging
from typing import Dict
from datasets import concatenate_datasets, Dataset
from utils.solution_utils import (
    extract_response_section, split_into_steps, get_partial_solutions,
    has_thinking_section, extract_thinking_section, has_response_section
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
        'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + 
                 '<|im_start|>user<|im_sep|>' + x['problem'] + '<|im_end|>' + 
                 '<|im_start|>assistant<|im_sep|>',
        'answer': x.get('answer', x.get('correct_answer', '')),  # Try both answer and correct_answer
        'partial_solution': '',  # Empty partial solution indicates full solution task
        'example_type': 'solution'  # Add type for tracking
    })
    return solution_data

def prepare_programming_data(data: Dataset, system_prompt: str) -> Dataset:
    """Create examples for programming tasks"""
    logger.info("Creating programming examples...")
    programming_data = data.map(lambda x: {
        'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + 
                 '<|im_start|>user<|im_sep|>' + x['problem'] + '<|im_end|>' + 
                 '<|im_start|>assistant<|im_sep|>',
        'answer': x.get('answer', x.get('correct_answer', '')),
        'partial_solution': '',
        'example_type': 'programming'
    })
    return programming_data

def prepare_completion_data(data: Dataset, system_prompt: str, completion_system_prompt: str, 
                           tokenizer=None, max_prompt_tokens: int = 2000) -> Dataset:
    """Create examples for completion tasks"""
    logger.info("Creating completion examples...")
    
    # Function to count tokens if tokenizer is provided
    def count_tokens(text):
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text) // 4  # Rough estimate if no tokenizer provided
    
    # Calculate token counts for system prompts if tokenizer is provided
    completion_prompt_tokens = count_tokens(completion_system_prompt) if tokenizer else 0
    
    def create_partial_solution(example):
        try:
            # Only process examples that have model_solutions with proper steps
            if 'model_solution' not in example or not example['model_solution']:
                return {
                    'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + 
                             '<|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|>' + 
                             '<|im_start|>assistant<|im_sep|>',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'example_type': 'solution'
                }
            
            # Extract response section using solution_utils
            response = extract_response_section(example['model_solution'])
            if not response:
                return {
                    'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + 
                             '<|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|>' + 
                             '<|im_start|>assistant<|im_sep|>',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'example_type': 'solution'
                }
            
            # Extract steps using solution_utils
            steps = split_into_steps(response)
            
            # Need at least 2 steps to create a partial solution
            if len(steps) < 2:
                return {
                    'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + 
                             '<|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|>' + 
                             '<|im_start|>assistant<|im_sep|>',
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
                    'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'example_type': 'solution'
                }
            
            partial_solution = partial_solutions[-1]  # Get the last partial solution (with all steps)
            
            # Check token count for the completion prompt if tokenizer is provided
            if tokenizer:
                completion_text = f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}"
                total_tokens = completion_prompt_tokens + count_tokens(completion_text)
                
                # If token count is too high, reduce the number of steps
                if total_tokens >= max_prompt_tokens:
                    # Try with just one step
                    if len(steps) > 0:
                        one_step_solutions = get_partial_solutions(steps[:1])
                        if one_step_solutions:
                            partial_solution = one_step_solutions[-1]
                            completion_text = f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}"
                            total_tokens = completion_prompt_tokens + count_tokens(completion_text)
                    
                    # If still too long, return as full solution
                    if total_tokens >= max_prompt_tokens:
                        logger.info(f"Completion prompt too long ({total_tokens} tokens), converting to full solution")
                        return {
                            'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                            'answer': example.get('answer', example.get('correct_answer', '')),
                            'partial_solution': '',
                            'example_type': 'solution'
                        }
            
            # Format the completion prompt with the partial solution in the user section
            formatted_prompt = '<|im_start|>system<|im_sep|>' + completion_system_prompt + '<|im_end|>' + \
                '<|im_start|>user<|im_sep|>' + \
                f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}" + \
                '<|im_end|>' + '<|im_start|>assistant<|im_sep|>'
            
            logger.info(f"Created completion example with {split_point} steps out of {len(steps)}")
            return {
                'prompt': formatted_prompt,
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': partial_solution,
                'example_type': 'completion'
            }
                
        except Exception as e:
            logger.warning(f"Error creating partial solution: {str(e)}")
            # Return as a full solution example on error
            return {
                'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + 
                         '<|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|>' + 
                         '<|im_start|>assistant<|im_sep|>',
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': '',
                'example_type': 'solution'
            }
    
    # Process all examples for completion tasks
    completion_data = data.map(create_partial_solution)
    
    # Filter to only keep actual completion examples
    completion_data = completion_data.filter(lambda x: x['example_type'] == 'completion')
    
    # Log completion data details
    completion_count = len(completion_data)
    logger.info(f"Found {completion_count} completion examples after filtering")
    
    return completion_data

def prepare_wait_data(data: Dataset, system_prompt: str) -> Dataset:
    """Create examples for wait-a-second tasks"""
    logger.info("Creating wait examples...")
    
    def create_wait_example(example):
        try:
            # Only create wait examples for examples with model_solution
            if 'model_solution' in example and example['model_solution']:
                # Check if the solution is incorrect (if is_correct field exists)
                is_correct = example.get('is_correct', None)
                
                # If is_correct is explicitly True, return as regular solution
                if is_correct == True:
                    return {
                        'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                        'answer': example.get('answer', example.get('correct_answer', '')),
                        'partial_solution': '',
                        'example_type': 'solution'
                    }
                
                # Check if solution has a thinking section using solution_utils
                if has_thinking_section(example['model_solution']):
                    # Extract the thinking section using solution_utils
                    thinking_content = extract_thinking_section(example['model_solution'])
                    if thinking_content:
                        # Modify the thinking section with "wait a second"
                        modified_thinking = thinking_content + "...no wait a second."
                        
                        # Create prompt with the modified thinking section
                        prompt = (
                            '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' +
                            '<|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|>' +
                            '<|im_start|>assistant<|im_sep|>' +
                            '<thinking>' + modified_thinking
                        )
                        
                        return {
                            'prompt': prompt,
                            'answer': example.get('answer', example.get('correct_answer', '')),
                            'partial_solution': '',
                            'example_type': 'wait'
                        }
            
            # If we couldn't create a wait example, return as regular solution
            return {
                'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + 
                         '<|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|>' + 
                         '<|im_start|>assistant<|im_sep|>',
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': '',
                'example_type': 'solution'
            }
            
        except Exception as e:
            logger.warning(f"Error creating wait example: {str(e)}")
            # Return as a regular solution example on error
            return {
                'prompt': '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + 
                         '<|im_start|>user<|im_sep|>' + example['problem'] + '<|im_end|>' + 
                         '<|im_start|>assistant<|im_sep|>',
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': '',
                'example_type': 'solution'
            }
    
    # Process all examples for wait tasks
    wait_data = data.map(create_wait_example)
    
    # Filter out any wait examples that didn't actually get the wait modification
    wait_data = wait_data.filter(lambda x: x['example_type'] == 'wait' and "...no wait a second." in x['prompt'])
    
    logger.info(f"Found {len(wait_data)} wait examples after filtering")
    
    return wait_data

def prepare_detailed_completion_data(data: Dataset, system_prompt: str, tokenizer=None) -> Dataset:
    """Create examples for completion tasks with detailed validation"""
    logger.info("Creating detailed completion examples...")
    
    def prepare_completion_example(example):
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
                prompt = '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + \
                        '<|im_start|>user<|im_sep|>' + \
                        f"Problem: {example.get('problem', '')}\n\nPartial Solution: {partial_solution}" + \
                        '<|im_end|>' + '<|im_start|>assistant<|im_sep|>'
                
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
            prompt = '<|im_start|>system<|im_sep|>' + system_prompt + '<|im_end|>' + \
                    '<|im_start|>user<|im_sep|>' + \
                    f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}" + \
                    '<|im_end|>' + '<|im_start|>assistant<|im_sep|>'
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
    processed_data = data.map(prepare_completion_example)
    
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

def prepare_combined_data(data: Dataset, system_prompt: str, completion_system_prompt: str, 
                         tokenizer=None, distribution: Dict[str, float] = None) -> Dataset:
    """
    Load and format dataset with multiple example types based on the specified distribution.
    Default distribution:
    - 35% solution examples
    - 35% programming examples
    - 15% completion examples
    - 15% wait examples
    """
    # Default distribution if not provided
    if distribution is None:
        distribution = {
            'solution': 0.35,
            'programming': 0.35,
            'completion': 0.15,
            'wait': 0.15
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
    programming_data = prepare_programming_data(data, system_prompt)
    completion_data = prepare_completion_data(data, system_prompt, completion_system_prompt, tokenizer)
    wait_data = prepare_wait_data(data, system_prompt)
    
    # Calculate the target number of examples for each type
    total_examples = len(data)
    solution_target = int(total_examples * distribution['solution'])
    programming_target = int(total_examples * distribution['programming'])
    completion_target = int(total_examples * distribution['completion'])
    wait_target = int(total_examples * distribution['wait'])
    
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
    logger.info(f"Created {len(completion_data)} completion examples (target: {completion_target})")
    logger.info(f"Created {len(wait_data)} wait examples (target: {wait_target})")
    
    # Shuffle and select examples for each type
    solution_data = solution_data.shuffle(seed=42)
    programming_data = programming_data.shuffle(seed=43)
    completion_data = completion_data.shuffle(seed=44)
    wait_data = wait_data.shuffle(seed=45)
    
    solution_data = solution_data.select(range(min(solution_target, len(solution_data))))
    programming_data = programming_data.select(range(min(programming_target, len(programming_data))))
    completion_data = completion_data.select(range(min(completion_target, len(completion_data))))
    wait_data = wait_data.select(range(min(wait_target, len(wait_data))))
    
    # Log type distribution before combining
    logger.info("Dataset type distribution before combining:")
    logger.info(f"Solution dataset: {count_types(solution_data)}")
    logger.info(f"Programming dataset: {count_types(programming_data)}")
    logger.info(f"Completion dataset: {count_types(completion_data)}")
    logger.info(f"Wait dataset: {count_types(wait_data)}")
    
    # Combine all datasets
    combined_data = concatenate_datasets([solution_data, programming_data, completion_data, wait_data])
    
    # Count types in the combined dataset
    combined_types = count_types(combined_data)
    logger.info(f"Combined dataset types: {combined_types}")
    
    # Calculate percentages
    total = sum(combined_types.values())
    percentages = {k: f"{v/total*100:.1f}%" for k, v in combined_types.items()}
    logger.info(f"Type percentages: {percentages}")
    
    return combined_data
