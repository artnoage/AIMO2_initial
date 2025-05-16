import re
import random
import logging
from typing import Dict, List, Union
from datasets import concatenate_datasets, Dataset
from utils.solution_utils import (extract_response_section, split_into_steps, get_partial_solutions, has_response_section, validate_solution)

# Setup logging
logger = logging.getLogger('data_preparation')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def prepare_solution_data(data: Dataset, system_prompts: Union[str, List[str]]) -> Dataset:
    """Create examples for full solution tasks"""
    logger.info("Creating solution examples...")
    
    # Convert single prompt to list for consistent handling
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    
    def map_function(x):
        # Randomly select a prompt from the list
        selected_prompt = random.choice(system_prompts)
        return {
            'prompt': '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x.get('answer', x.get('correct_answer', '')),  # Try both answer and correct_answer
            'model_solution':x.get('model_solution',''),
            'partial_solution': '',  # Empty partial solution indicates full solution task
            'full_solution': '',
            'is_correct': None,  # Use None instead of empty string for consistency
            'wrong_step': None,  # Use None instead of empty string for consistency
            'example_type': 'solution'  # Add type for tracking
        }
    
    solution_data = data.map(map_function)
    return solution_data

def prepare_programming_data(data: Dataset, system_prompts: Union[str, List[str]]) -> Dataset:
    """Create examples for programming tasks using the programming-specific system prompt"""
    logger.info("Creating programming examples...")
    
    # Convert single prompt to list for consistent handling
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    
    def map_function(x):
        # Randomly select a prompt from the list
        selected_prompt = random.choice(system_prompts)
        return {
            'prompt': '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x.get('answer', x.get('correct_answer', '')),
            'partial_solution': '',
            'full_solution': '',
            'is_correct': None,  # Use None instead of empty string for consistency
            'wrong_step': None,  # Use None instead of empty string for consistency
            'example_type': 'programming'
        }
    
    programming_data = data.map(map_function)
    return programming_data

def prepare_test_programming_data(data: Dataset, system_prompts: Union[str, List[str]]) -> Dataset:
    """Create examples for test programming tasks using the test-specific system prompt"""
    logger.info("Creating test programming examples...")
    
    # Convert single prompt to list for consistent handling
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    
    def map_function(x):
        # Randomly select a prompt from the list
        selected_prompt = random.choice(system_prompts)
        return {
            'prompt': '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x['problem'],
            'partial_solution': '',
            'full_solution': '',
            'is_correct': None,
            'wrong_step': None,
            'example_type': 'test_programming'
        }
    
    test_programming_data = data.map(map_function)
    return test_programming_data

def prepare_dual_proof_data(data: Dataset, system_prompts: Union[str, List[str]]) -> Dataset:
    """Create examples for dual proof tasks using the dual-proof-specific system prompt"""
    logger.info("Creating dual proof examples...")
    
    # Convert single prompt to list for consistent handling
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    
    def map_function(x):
        # Randomly select a prompt from the list
        selected_prompt = random.choice(system_prompts)
        return {
            'prompt': '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x['problem'],
            'partial_solution': '',
            'full_solution': '',
            'is_correct': None,
            'wrong_step': None,
            'example_type': 'dual_proof'
        }
    
    dual_proof_data = data.map(map_function)
    return dual_proof_data

def prepare_test_driven_programmer_data(data: Dataset, system_prompts: Union[str, List[str]]) -> Dataset:
    """Create examples for test-driven programmer tasks using the test-driven-programmer-specific system prompt"""
    logger.info("Creating test-driven programmer examples...")
    
    # Convert single prompt to list for consistent handling
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    
    def map_function(x):
        # Randomly select a prompt from the list
        selected_prompt = random.choice(system_prompts)
        return {
            'prompt': '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x['problem'],
            'partial_solution': '',
            'full_solution': '',
            'is_correct': None,
            'wrong_step': None,
            'example_type': 'test_driven_programmer'
        }
    
    test_driven_programmer_data = data.map(map_function)
    return test_driven_programmer_data

def prepare_architect_data(data: Dataset, system_prompts: Union[str, List[str]]) -> Dataset:
    """Create examples for architect/engineering tasks using the architect-specific system prompt"""
    logger.info("Creating architect examples...")
    
    # Convert single prompt to list for consistent handling
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    
    def map_function(x):
        # Randomly select a prompt from the list
        selected_prompt = random.choice(system_prompts)
        return {
            'prompt': '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x['problem'],
            'partial_solution': '',
            'full_solution': '',
            'is_correct': None,
            'wrong_step': None,
            'example_type': 'architect'
        }
    
    architect_data = data.map(map_function)
    return architect_data

def prepare_tutor_data(data: Dataset, system_prompts: Union[str, List[str]], tokenizer=None, max_prompt_tokens: int = 1500) -> Dataset:
    """Create examples for tutor tasks using the tutor-specific system prompt"""
    logger.info("Creating tutor examples...")
    
    # Function to count tokens if tokenizer is provided
    def count_tokens(text):
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text) // 4  # Rough estimate if no tokenizer provided
        
    # Convert single prompt to list for consistent handling
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    
    # Default invalid example template - mark as solution type instead of using valid flag
    def create_invalid_example(example):
        # Randomly select a prompt from the list
        selected_prompt = random.choice(system_prompts)
        return {
            'prompt': '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example.get('problem', '') + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': example.get('answer', example.get('correct_answer', '')),
            'partial_solution': '',
            'full_solution': '',
            'is_correct': None,
            'wrong_step': None,
            'example_type': 'solution'  # Mark as solution type
        }
    
    def create_tutor_example(example):
        try:
            # Check if we have the required fields
            if 'problem' not in example or not example['problem']:
                return create_invalid_example(example)
                
            if 'model_solution' not in example or not example['model_solution']:
                return create_invalid_example(example)
                
            # Extract response part from model_solution
            response_match = re.search(r'<response>(.*?)</response>', example['model_solution'], re.DOTALL)
            full_solution = response_match.group(1).strip() if response_match else None
            
            # If no response part found, return invalid example
            if full_solution is None:
                return create_invalid_example(example)
                
            # Check if the solution contains the string "Begin with the first calculation or operation"
            if "Begin with the first calculation or operation" in example['model_solution']:
                logger.info(f"Skipping tutor example with 'Begin with the first calculation or operation' instruction")
                return create_invalid_example(example)
                
            # Extract is_correct flag - default to None if not available
            is_correct = example.get('is_correct', None)
            if isinstance(is_correct, str):
                if is_correct.lower() == 'true':
                    is_correct = True
                elif is_correct.lower() == 'false':
                    is_correct = False
                else:
                    is_correct = None
                
            # Extract wrong_step if available
            wrong_step = example.get('wrong_step', None)
            if wrong_step is not None:
                try:
                    wrong_step = int(wrong_step)
                except (ValueError, TypeError):
                    wrong_step = None
            
            # Skip examples that are not correct and don't have a wrong step
            if is_correct is not True and (wrong_step is None or wrong_step == ''):
                logger.info(f"Skipping tutor example that is not correct and has no wrong_step")
                return create_invalid_example(example)
                
            # For correct examples, keep only 10% as tutor examples, convert the rest to solution type
            if is_correct is True and wrong_step is None:
                # Use a deterministic approach based on example hash
                example_hash = hash(str(example.get('problem', '')))
                if example_hash % 20 != 0:  # Keep only 10% (when hash mod 10 equals 0)
                    logger.info(f"Converting correct tutor example to solution type (90% filter)")
                    return create_invalid_example(example)
            
            # Randomly select a prompt from the list
            selected_prompt = random.choice(system_prompts)
            
            # Format the prompt with the problem and full solution
            formatted_prompt = '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + \
                f"Here is a mathematical problem and a proposed solution:\n\n" + \
                f"Problem:\n{example['problem']}\n\n" + \
                f"Proposed Solution:\n{full_solution}<|im_end|>\\n<|im_start|>assistant\\n"
            
            # Check token count if tokenizer is provided
            if tokenizer:
                prompt_tokens = count_tokens(formatted_prompt)
                if prompt_tokens > max_prompt_tokens:
                    logger.info(f"Tutor prompt too long ({prompt_tokens} tokens), skipping")
                    return create_invalid_example(example)
            
            return {
                'prompt': formatted_prompt,
                'answer': example.get('answer', example.get('correct_answer', '')),
                'partial_solution': '',
                'full_solution': full_solution,
                'is_correct': is_correct,
                'wrong_step': wrong_step,
                'example_type': 'tutor'
            }
        except Exception as e:
            logger.warning(f"Error creating tutor example: {str(e)}")
            # Return as invalid on error
            return create_invalid_example(example)
    
    # Process all examples
    processed_data = data.map(create_tutor_example)
    
    # Filter to keep only tutor examples
    tutor_data = processed_data.filter(lambda x: x.get('example_type') == 'tutor')
    
    # Log validation results
    logger.info(f"Total examples in dataset: {len(data)}")
    logger.info(f"Tutor examples after processing: {len(tutor_data)}")
    
    return tutor_data

def prepare_finalization_data(data: Dataset, system_prompts: Union[str, List[str]], finalization_system_prompts: Union[str, List[str]], 
                           tokenizer=None, max_prompt_tokens: int = 1500) -> Dataset:
    """Create examples for finalization tasks with detailed validation"""
    logger.info("Creating finalization examples...")
    
    # Function to count tokens if tokenizer is provided
    def count_tokens(text):
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text) // 4  # Rough estimate if no tokenizer provided
    
    # Convert single prompts to lists for consistent handling
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    if isinstance(finalization_system_prompts, str):
        finalization_system_prompts = [finalization_system_prompts]
    
    # Calculate token counts for system prompts if tokenizer is provided
    # Use the longest prompt for estimation
    longest_prompt = max(finalization_system_prompts, key=len) if finalization_system_prompts else ""
    finalization_prompt_tokens = count_tokens(longest_prompt) if tokenizer else 0
    
    # Default invalid example template - mark as solution type instead of using valid flag
    def create_invalid_example(example):
        # Randomly select a prompt from the list
        selected_prompt = random.choice(system_prompts)
        return {
            'prompt': '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + example.get('problem', '') + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': example.get('answer', example.get('correct_answer', '')),
            'partial_solution': '',
            'full_solution': '',
            'is_correct': None,
            'wrong_step': None,
            'example_type': 'solution'  # Mark as solution type instead of using valid flag
        }
    
    def prepare_finalization_example(example):
        try:
            # First, check if we have the required problem field
            if 'problem' not in example or not example['problem']:
                return create_invalid_example(example)
                
            # Skip if no model_solution or if it's empty
            if 'model_solution' not in example or not example['model_solution']:
                return create_invalid_example(example)
            
            # Check if solution has a response section
            if not has_response_section(example['model_solution']):
                return create_invalid_example(example)
            
            # Extract response section
            response = extract_response_section(example['model_solution'])
            if not response:
                return create_invalid_example(example)
                
            # Check if the solution contains the string "Begin with the first calculation or operation"
            if "Begin with the first calculation or operation" in example['model_solution']:
                logger.info(f"Skipping example with 'Begin with the first calculation or operation' instruction")
                return create_invalid_example(example)
                
            # Check if the solution is marked as correct
            is_correct = example.get('is_correct', None)
            if isinstance(is_correct, str):
                if is_correct.lower() == 'true':
                    is_correct = True
                elif is_correct.lower() == 'false':
                    is_correct = False
                else:
                    is_correct = None
            
            # Only use examples that are explicitly marked as correct
            if is_correct is not True:
                logger.info(f"Skipping example that is not explicitly marked as correct")
                return create_invalid_example(example)
                
            # Validate the solution structure
            is_valid_solution, validation_reason = validate_solution(response)
            if not is_valid_solution:
                logger.info(f"Invalid solution structure: {validation_reason}")
                return create_invalid_example(example)
            
            # Extract steps using solution_utils (now returns steps with tags)
            steps = split_into_steps(response)
            
            # Need at least 2 steps to create a partial solution
            if len(steps) < 2:
                return create_invalid_example(example)
            
            # Use a deterministic approach based on example ID
            example_id = example.get('id', hash(example.get('problem', '')))
            if example_id is None:
                example_id = hash(str(random.random()))  # Fallback if both id and problem are None
            elif isinstance(example_id, str):
                example_id = hash(example_id)
            
            # Deterministically decide how many steps to include in partial solution
            random.seed(example_id % 10000)  # Deterministic but varied
            split_point = random.randint(1, len(steps) - 1)  # At least 1 step, leave at least 1 step
            
            # Create partial solution with the first 'split_point' steps
            partial_steps = steps[:split_point]
            partial_solutions = get_partial_solutions(partial_steps)
            full_solutions = get_partial_solutions(steps)
            
            if not partial_solutions:
                return create_invalid_example(example)
            
            partial_solution = partial_solutions[-1]  # Get the last partial solution (with all steps)
            full_solution = full_solutions[-1] if full_solutions else ''
            
            # Validate the partial solution structure
            is_valid_partial, validation_reason = validate_solution(partial_solution, start_step=0)
            if not is_valid_partial:
                logger.info(f"Invalid partial solution: {validation_reason}")
                return create_invalid_example(example)
                
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
                        return create_invalid_example(example)
            
            # Randomly select a prompt from the list
            selected_prompt = random.choice(finalization_system_prompts)
            
            # Format the finalization prompt with the partial solution in the user section
            formatted_prompt = '<|im_start|>system\\n' + selected_prompt + '<|im_end|>\\n<|im_start|>user\\n' + \
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
                'is_correct': example.get('is_correct', None),
                'wrong_step': example.get('wrong_step', None),
                'example_type': 'finalization'
            }
                
        except Exception as e:
            logger.warning(f"Error creating finalization example: {str(e)}")
            # Return as invalid on error
            return create_invalid_example(example)
    
    # Process all examples
    processed_data = data.map(prepare_finalization_example)
    
    # Add token count to each example if tokenizer is provided
    if tokenizer:
        def count_tokens(example):
            if example.get('example_type') != 'finalization' or not example.get('prompt'):
                return {'token_count': 0}
            return {'token_count': len(tokenizer.encode(example['prompt']))}
        
        processed_data = processed_data.map(count_tokens)
        
        # Log token count statistics
        token_counts = [ex['token_count'] for ex in processed_data if ex.get('example_type') == 'finalization']
        if token_counts:
            logger.info(f"Token count statistics:")
            logger.info(f"  Min: {min(token_counts)}")
            logger.info(f"  Max: {max(token_counts)}")
            logger.info(f"  Mean: {sum(token_counts)/len(token_counts):.2f}")
            logger.info(f"  Examples > 1500 tokens: {sum(1 for t in token_counts if t > 1500)}")
        
        # Filter finalization examples and those with token count <= 1500
        finalization_data = processed_data.filter(lambda x: x.get('example_type') == 'finalization' and x.get('token_count', 0) <= 1500)
    else:
        # If no tokenizer, just filter finalization examples
        finalization_data = processed_data.filter(lambda x: x.get('example_type') == 'finalization')
    
    # Log validation results
    logger.info(f"Total examples in dataset: {len(data)}")
    logger.info(f"Finalization examples after processing: {len(processed_data.filter(lambda x: x.get('example_type') == 'finalization'))}")
    logger.info(f"Final finalization examples: {len(finalization_data)}")
    
    return finalization_data

def prepare_combined_data(data: Dataset, 
                          system_prompts: Union[str, List[str]], 
                          finalization_system_prompts: Union[str, List[str]], 
                          programming_system_prompts: Union[str, List[str]], 
                          tutor_system_prompts: Union[str, List[str]],
                          test_programming_system_prompts: Union[str, List[str]], 
                          architect_system_prompts: Union[str, List[str]],
                          dual_proof_system_prompts: Union[str, List[str]], 
                          test_driven_programmer_system_prompts: Union[str, List[str]],
                          tokenizer=None, distribution: Dict[str, float] = None, max_prompt_tokens: int = 2000) -> Dataset:
    """
    Load and format dataset with multiple example types based on the specified distribution.
    Default distribution:
    - 20% solution examples
    - 15% programming examples
    - 15% finalization examples
    - 10% tutor examples
    - 10% test programming examples
    - 10% architect examples
    - 10% dual proof examples
    - 10% test-driven programmer examples
    
    If any distribution value is 0, no examples of that type will be generated.
    """
    # Default distribution if not provided
    if distribution is None:
        distribution = {
            'solution': 0.20,
            'programming': 0.15,
            'finalization': 0.15,
            'tutor': 0.10,
            'test_programming': 0.10,
            'architect': 0.10,
            'dual_proof': 0.10,
            'test_driven_programmer': 0.10
        }
    
    # Check if we have model_solutions in the dataset
    has_model_solutions = sum(1 for x in data if 'model_solution' in x and x['model_solution'])
    logger.info(f"Dataset has {has_model_solutions} examples with model_solutions")

    
    # Create examples for each type only if their distribution is non-zero
    solution_data = None
    if distribution.get('solution', 0) > 0:
        solution_data = prepare_solution_data(data, system_prompts)
        
    programming_data = None
    if distribution.get('programming', 0) > 0:
        programming_data = prepare_programming_data(data, programming_system_prompts)
        
    finalization_data = None
    if distribution.get('finalization', 0) > 0:
        finalization_data = prepare_finalization_data(data, system_prompts, finalization_system_prompts, tokenizer, max_prompt_tokens=1500)
    
    # Create tutor examples if tutor_system_prompts is provided and distribution is non-zero
    tutor_data = None
    if tutor_system_prompts and distribution.get('tutor', 0) > 0:
        tutor_data = prepare_tutor_data(data, tutor_system_prompts, tokenizer, max_prompt_tokens)
    
    # Create test programming examples if test_programming_system_prompts is provided and distribution is non-zero
    test_programming_data = None
    if test_programming_system_prompts and distribution.get('test_programming', 0) > 0:
        test_programming_data = prepare_test_programming_data(data, test_programming_system_prompts)
    
    # Create architect examples if architect_system_prompts is provided and distribution is non-zero
    architect_data = None
    if architect_system_prompts and distribution.get('architect', 0) > 0:
        architect_data = prepare_architect_data(data, architect_system_prompts)
    
    # Create dual proof examples if dual_proof_system_prompts is provided and distribution is non-zero
    dual_proof_data = None
    if dual_proof_system_prompts and distribution.get('dual_proof', 0) > 0:
        dual_proof_data = prepare_dual_proof_data(data, dual_proof_system_prompts)
    
    # Create test-driven programmer examples if test_driven_programmer_system_prompts is provided and distribution is non-zero
    test_driven_programmer_data = None
    if test_driven_programmer_system_prompts and distribution.get('test_driven_programmer', 0) > 0:
        test_driven_programmer_data = prepare_test_driven_programmer_data(data, test_driven_programmer_system_prompts)
    
    # Calculate the target number of examples for each type
    total_examples = len(data)
    solution_target = int(total_examples * distribution.get('solution', 0))
    programming_target = int(total_examples * distribution.get('programming', 0))
    finalization_target = int(total_examples * distribution.get('finalization', 0))
    tutor_target = int(total_examples * distribution.get('tutor', 0)) if tutor_data else 0
    test_programming_target = int(total_examples * distribution.get('test_programming', 0)) if test_programming_data else 0
    architect_target = int(total_examples * distribution.get('architect', 0)) if architect_data else 0
    dual_proof_target = int(total_examples * distribution.get('dual_proof', 0)) if dual_proof_data else 0
    test_driven_programmer_target = int(total_examples * distribution.get('test_driven_programmer', 0)) if test_driven_programmer_data else 0
    
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
    if test_programming_data:
        logger.info(f"Created {len(test_programming_data)} test programming examples (target: {test_programming_target})")
    if architect_data:
        logger.info(f"Created {len(architect_data)} architect examples (target: {architect_target})")
    if dual_proof_data:
        logger.info(f"Created {len(dual_proof_data)} dual proof examples (target: {dual_proof_target})")
    if test_driven_programmer_data:
        logger.info(f"Created {len(test_driven_programmer_data)} test-driven programmer examples (target: {test_driven_programmer_target})")
    
    # Shuffle and select examples for each type
    if solution_data:
        solution_data = solution_data.shuffle(seed=142)
        solution_data = solution_data.select(range(min(solution_target, len(solution_data))))
    
    if programming_data:
        programming_data = programming_data.shuffle(seed=143)
        programming_data = programming_data.select(range(min(programming_target, len(programming_data))))
    
    if finalization_data:
        finalization_data = finalization_data.shuffle(seed=144)
        finalization_data = finalization_data.select(range(min(finalization_target, len(finalization_data))))
    
    if tutor_data:
        tutor_data = tutor_data.shuffle(seed=145)
        tutor_data = tutor_data.select(range(min(tutor_target, len(tutor_data))))
    
    if test_programming_data:
        test_programming_data = test_programming_data.shuffle(seed=146)
        test_programming_data = test_programming_data.select(range(min(test_programming_target, len(test_programming_data))))
    
    if architect_data:
        architect_data = architect_data.shuffle(seed=147)
        architect_data = architect_data.select(range(min(architect_target, len(architect_data))))
    
    if dual_proof_data:
        dual_proof_data = dual_proof_data.shuffle(seed=148)
        dual_proof_data = dual_proof_data.select(range(min(dual_proof_target, len(dual_proof_data))))
    
    if test_driven_programmer_data:
        test_driven_programmer_data = test_driven_programmer_data.shuffle(seed=149)
        test_driven_programmer_data = test_driven_programmer_data.select(range(min(test_driven_programmer_target, len(test_driven_programmer_data))))
    
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
    if test_programming_data:
        logger.info(f"Test programming dataset: {count_types(test_programming_data)}")
    if architect_data:
        logger.info(f"Architect dataset: {count_types(architect_data)}")
    if dual_proof_data:
        logger.info(f"Dual proof dataset: {count_types(dual_proof_data)}")
    if test_driven_programmer_data:
        logger.info(f"Test-driven programmer dataset: {count_types(test_driven_programmer_data)}")
    
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
    if test_programming_data:
        datasets_to_combine.append(test_programming_data)
    if architect_data:
        datasets_to_combine.append(architect_data)
    if dual_proof_data:
        datasets_to_combine.append(dual_proof_data)
    if test_driven_programmer_data:
        datasets_to_combine.append(test_driven_programmer_data)
    
    combined_data = concatenate_datasets(datasets_to_combine)
    
    # Count types in the combined dataset
    combined_types = count_types(combined_data)
    logger.info(f"Combined dataset types: {combined_types}")
    
    # Calculate percentages
    total = sum(combined_types.values())
    percentages = {k: f"{v/total*100:.1f}%" for k, v in combined_types.items()}
    logger.info(f"Type percentages: {percentages}")
    
    return combined_data
