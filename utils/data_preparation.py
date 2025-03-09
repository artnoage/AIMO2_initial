import re
import random
import logging
from typing import Dict
from datasets import concatenate_datasets, Dataset
from utils.solution_utils import (extract_response_section, split_into_steps, get_partial_solutions, has_response_section)

# Setup logging
logger = logging.getLogger('data_preparation')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def prepare_solution_data(data: Dataset, system_prompt: str) -> Dataset:
    """Create examples for full solution tasks"""
    logger.info("Creating solution examples...")
    solution_data = data.map(lambda x: {
        'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
        'answer': x.get('answer', x.get('correct_answer', '')),  # Try both answer and correct_answer
        'partial_solution': '',  # Empty partial solution indicates full solution task
        'model_solution': '',
        'is_correct': '',
        'wrong_step': '',
        'example_type': 'solution'  # Add type for tracking
    })
    return solution_data

def prepare_programming_data(data: Dataset, system_prompt: str) -> Dataset:
    """Create examples for programming tasks using the programming-specific system prompt"""
    logger.info("Creating programming examples...")
    programming_data = data.map(lambda x: {
        'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
        'answer': x.get('answer', x.get('correct_answer', '')),
        'partial_solution': '',
        'model_solution': '',
        'is_correct': '',
        'wrong_step': '',
        'example_type': 'programming'
    })
    return programming_data

def prepare_tutor_data(data: Dataset, system_prompt: str, tokenizer=None, max_prompt_tokens: int = 1500) -> Dataset:
    """Create examples for tutor tasks using the tutor-specific system prompt"""
    logger.info("Creating tutor examples...")
    
    # Function to count tokens if tokenizer is provided
    def count_tokens(text):
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text) // 4  # Rough estimate if no tokenizer provided
    
    def create_tutor_example(example):
        try:
            # Check if we have the required fields
            if 'problem' not in example or not example['problem']:
                return None
                
            if 'model_solution' not in example or not example['model_solution']:
                return None
                
            # Extract response part from model_solution
            response_match = re.search(r'<response>(.*?)</response>', example['model_solution'], re.DOTALL)
            full_solution = response_match.group(1).strip() if response_match else None
            
            # If no response part found, return None
            if full_solution is None:
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
            
            # Format the prompt with the problem and full solution
            formatted_prompt = '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + \
                f"Here is a mathematical problem and a proposed solution:\n\n" + \
                f"Problem:\n{example['problem']}\n\n" + \
                f"Proposed Solution:\n{full_solution}<|im_end|>\\n<|im_start|>assistant\\n"
            
            # Check token count if tokenizer is provided
            if tokenizer:
                prompt_tokens = count_tokens(formatted_prompt)
                if prompt_tokens > max_prompt_tokens:
                    logger.info(f"Tutor prompt too long ({prompt_tokens} tokens), skipping")
                    return None
            
            return {
                'prompt': formatted_prompt,
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': '',
                'full_solution': full_solution,
                'model_solution': '',  # Empty as we're using full_solution instead
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

def prepare_finalization_data(data: Dataset, system_prompt: str, finalization_system_prompt: str, 
                           tokenizer=None, max_prompt_tokens: int = 1500) -> Dataset:
    """Create examples for finalization tasks with detailed validation"""
    logger.info("Creating finalization examples...")
    
    # Function to count tokens if tokenizer is provided
    def count_tokens(text):
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text) // 4  # Rough estimate if no tokenizer provided
    
    # Calculate token counts for system prompts if tokenizer is provided
    finalization_prompt_tokens = count_tokens(finalization_system_prompt) if tokenizer else 0
    
    def prepare_finalization_example(example):
        try:
            # First, check if we have the required problem field
            if 'problem' not in example or not example['problem']:
                return {
                    'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example.get('problem', '') + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'full_solution': '',
                    'model_solution': '',
                    'is_correct': '',
                    'wrong_step': '',
                    'example_type': 'solution',
                    'valid': False
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
                formatted_prompt = '<|im_start|>system\\n' + finalization_system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + \
                        f"Problem: {example.get('problem', '')}\n\nPartial Solution: {partial_solution}<|im_end|>\\n<|im_start|>assistant\\n"
                
                return {
                    'prompt': formatted_prompt,
                    'answer': answer,
                    'partial_solution': partial_solution,
                    'full_solution': full_solution,
                    'model_solution': example.get('model_solution', ''),
                    'is_correct': example.get('is_correct', ''),
                    'wrong_step': example.get('wrong_step', ''),
                    'example_type': 'finalization',
                    'valid': True
                }
            
            # Skip if no model_solution or if it's empty
            if 'model_solution' not in example or not example['model_solution']:
                return {
                    'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'full_solution': '',
                    'model_solution': '',
                    'is_correct': '',
                    'wrong_step': '',
                    'example_type': 'solution',
                    'valid': False
                }
            
            # Check if solution has a response section
            if not has_response_section(example['model_solution']):
                return {
                    'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'full_solution': '',
                    'model_solution': '',
                    'is_correct': '',
                    'wrong_step': '',
                    'example_type': 'solution',
                    'valid': False
                }
            
            # Extract response section
            response = extract_response_section(example['model_solution'])
            if not response:
                return {
                    'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'full_solution': '',
                    'model_solution': '',
                    'is_correct': '',
                    'wrong_step': '',
                    'example_type': 'solution',
                    'valid': False
                }
            
            # Extract steps using solution_utils
            steps = split_into_steps(response)
            
            # Need at least 2 steps to create a partial solution
            if len(steps) < 2:
                return {
                    'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'full_solution': '',
                    'model_solution': '',
                    'is_correct': '',
                    'wrong_step': '',
                    'example_type': 'solution',
                    'valid': False
                }
            
            # Verify step numbering
            step_numbers = []
            for step in steps:
                # Look for step numbers like "Step 1:", "Step 2:" etc.
                number_match = re.search(r'Step\s+(\d+):', step)
                if not number_match:
                    return {
                        'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                        'answer': example.get('answer', example.get('correct_answer', '')),
                        'partial_solution': '',
                        'full_solution': '',
                        'model_solution': '',
                        'is_correct': '',
                        'wrong_step': '',
                        'example_type': 'solution',
                        'valid': False
                    }
                
                try:
                    step_num = int(number_match.group(1))
                    step_numbers.append(step_num)
                except ValueError:
                    return {
                        'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                        'answer': example.get('answer', example.get('correct_answer', '')),
                        'partial_solution': '',
                        'full_solution': '',
                        'model_solution': '',
                        'is_correct': '',
                        'wrong_step': '',
                        'example_type': 'solution',
                        'valid': False
                    }
            
            # Check if step numbers are sequential
            expected_numbers = list(range(1, len(steps) + 1))
            if step_numbers != expected_numbers:
                return {
                    'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'full_solution': '',
                    'model_solution': '',
                    'is_correct': '',
                    'wrong_step': '',
                    'example_type': 'solution',
                    'valid': False
                }
            
            # Use a deterministic approach based on example ID
            example_id = example.get('id', hash(example.get('problem', '')))
            if isinstance(example_id, str):
                example_id = hash(example_id)
            
            # Deterministically decide how many steps to include in partial solution
            random.seed(example_id % 10000)  # Deterministic but varied
            split_point = random.randint(1, len(steps) - 1)  # At least 1 step, leave at least 1 step
            
            # Create partial solution with the first 'split_point' steps
            partial_steps = steps[:split_point]
            partial_solutions = get_partial_solutions(partial_steps)
            full_solutions = get_partial_solutions(steps)
            
            if not partial_solutions:
                return {
                    'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example.get('answer', example.get('correct_answer', '')),
                    'partial_solution': '',
                    'full_solution': '',
                    'model_solution': '',
                    'is_correct': '',
                    'wrong_step': '',
                    'example_type': 'solution',
                    'valid': False
                }
            
            partial_solution = partial_solutions[-1]  # Get the last partial solution (with all steps)
            full_solution = full_solutions[-1] if full_solutions else ''
            
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
                    
                    # If still too long, return as invalid
                    if total_tokens >= max_prompt_tokens:
                        logger.info(f"Finalization prompt too long ({total_tokens} tokens), marking as invalid")
                        return {
                            'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                            'answer': example.get('answer', example.get('correct_answer', '')),
                            'partial_solution': '',
                            'full_solution': '',
                            'model_solution': '',
                            'is_correct': '',
                            'wrong_step': '',
                            'example_type': 'solution',
                            'valid': False
                        }
            
            # Format the finalization prompt with the partial solution in the user section
            formatted_prompt = '<|im_start|>system\\n' + finalization_system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + \
                f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}<|im_end|>\\n<|im_start|>assistant\\n"
            
            # Get the answer from the example or extract from solution
            answer = example.get('answer', '')
            if not answer and 'correct_answer' in example:
                answer = example['correct_answer']
            
            return {
                'prompt': formatted_prompt,
                'answer': answer,
                'partial_solution': partial_solution,
                'full_solution': full_solution,
                'model_solution': example.get('model_solution', ''),
                'is_correct': example.get('is_correct', ''),
                'wrong_step': example.get('wrong_step', ''),
                'example_type': 'finalization',
                'valid': True
            }
                
        except Exception as e:
            logger.warning(f"Error creating finalization example: {str(e)}")
            # Return as invalid on error
            return {
                'prompt': '<|im_start|>system\\n' + system_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example.get('problem', '') + '<|im_end|>\\n<|im_start|>assistant\\n',
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': '',
                'full_solution': '',
                'model_solution': '',
                'is_correct': '',
                'wrong_step': '',
                'example_type': 'solution',
                'valid': False
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
        finalization_data = processed_data.filter(lambda x: x['valid'] and x['token_count'] <= 2000)
    else:
        # If no tokenizer, just filter valid examples
        finalization_data = processed_data.filter(lambda x: x['valid'])
    
    # Log validation results
    logger.info(f"Total examples in dataset: {len(data)}")
    logger.info(f"Valid examples after processing: {len(processed_data.filter(lambda x: x['valid']))}")
    logger.info(f"Final finalization examples: {len(finalization_data)}")
    
    return finalization_data

def prepare_combined_data(data: Dataset, system_prompt: str, finalization_system_prompt: str, 
                          programming_system_prompt: str, tutor_system_prompt: str = None,
                         tokenizer=None, distribution: Dict[str, float] = None, max_prompt_tokens: int = 1500) -> Dataset:
    """
    Load and format dataset with multiple example types based on the specified distribution.
    Default distribution:
    - 30% solution examples
    - 30% programming examples
    - 20% finalization examples
    - 20% tutor examples
    
    If any distribution value is 0, no examples of that type will be generated.
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
    
    # Create examples for each type only if their distribution is non-zero
    solution_data = None
    if distribution.get('solution', 0) > 0:
        solution_data = prepare_solution_data(data, system_prompt)
        
    programming_data = None
    if distribution.get('programming', 0) > 0:
        programming_data = prepare_programming_data(data, programming_system_prompt)
        
    finalization_data = None
    if distribution.get('finalization', 0) > 0:
        finalization_data = prepare_finalization_data(data, system_prompt, finalization_system_prompt, tokenizer, max_prompt_tokens=1500)
    
    # Create tutor examples if tutor_system_prompt is provided and distribution is non-zero
    tutor_data = None
    if tutor_system_prompt and distribution.get('tutor', 0) > 0:
        from utils.agents import TUTOR_SYSTEM_PROMPT
        tutor_data = prepare_tutor_data(data, tutor_system_prompt or TUTOR_SYSTEM_PROMPT, tokenizer, max_prompt_tokens)
    
    # Calculate the target number of examples for each type
    total_examples = len(data)
    solution_target = int(total_examples * distribution.get('solution', 0))
    programming_target = int(total_examples * distribution.get('programming', 0))
    finalization_target = int(total_examples * distribution.get('finalization', 0))
    tutor_target = int(total_examples * distribution.get('tutor', 0)) if tutor_data else 0
    
    # Function to count example types in a dataset
    def count_types(dataset):
        if dataset is None:
            return {}
        type_counts = {}
        for example in dataset:
            example_type = example.get('example_type', 'unknown')
            type_counts[example_type] = type_counts.get(example_type, 0) + 1
        return type_counts
    
    # Log the counts before shuffling
    if solution_data:
        logger.info(f"Created {len(solution_data)} full solution examples (target: {solution_target})")
    if programming_data:
        logger.info(f"Created {len(programming_data)} programming examples (target: {programming_target})")
    if finalization_data:
        logger.info(f"Created {len(finalization_data)} finalization examples (target: {finalization_target})")
    if tutor_data:
        logger.info(f"Created {len(tutor_data)} tutor examples (target: {tutor_target})")
    
    # Shuffle and select examples for each type
    if solution_data:
        solution_data = solution_data.shuffle(seed=42)
        solution_data = solution_data.select(range(min(solution_target, len(solution_data))))
    
    if programming_data:
        programming_data = programming_data.shuffle(seed=43)
        programming_data = programming_data.select(range(min(programming_target, len(programming_data))))
    
    if finalization_data:
        finalization_data = finalization_data.shuffle(seed=44)
        finalization_data = finalization_data.select(range(min(finalization_target, len(finalization_data))))
    
    if tutor_data:
        tutor_data = tutor_data.shuffle(seed=45)
        tutor_data = tutor_data.select(range(min(tutor_target, len(tutor_data))))
    
    # Log type distribution before combining
    logger.info("Dataset type distribution before combining:")
    if solution_data:
        logger.info(f"Solution dataset: {count_types(solution_data)}")
    if programming_data:
        logger.info(f"Programming dataset: {count_types(programming_data)}")
    if finalization_data:
        logger.info(f"Finalization dataset: {count_types(finalization_data)}")
    if tutor_data:
        logger.info(f"Tutor dataset: {count_types(tutor_data)}")
    
    # Combine all datasets
    datasets_to_combine = []
    if solution_data:
        datasets_to_combine.append(solution_data)
    if programming_data:
        datasets_to_combine.append(programming_data)
    if finalization_data:
        datasets_to_combine.append(finalization_data)
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
