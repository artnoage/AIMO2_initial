import os
import wandb
import logging
import json
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import sys
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import re
import time
from time import time
import random

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from rewards import CompletionReward, SolutionSimilarityChecker

SYSTEM_PROMPT = """You will be given a mathematical problem and a partial solution. Your task is to complete the solution.

Your response MUST include both a <thinking> section and a <response> section.

<thinking>
First, analyze the problem and the partial solution carefully.
Understand what has been done so far and determine the next logical steps.
Identify the step numbering pattern and continue from there.
Make sure you understand the mathematical concepts involved.
Work through the solution mentally to ensure your approach is correct.
</thinking>

<response>
Continue the solution from where it left off, maintaining the same step numbering and style.
The partial solution will only contain the beginning of the response section with some steps.
You must continue with the next step number in sequence.

IMPORTANT: Each step must be properly enclosed in <step> and </step> tags.

For example, if the partial solution ends with Step 2, you should start with:

<step>Step 3: [Description of the step]
[Mathematical work for this step]
</step>

Continue with additional steps as needed:

<step>Step 4: [Description of the step]
[Mathematical work for this step]
</step>

In your final step, include your answer in a LaTeX boxed environment:
\\boxed{your final answer}

Make sure all your steps follow logically from the partial solution and that each step has both opening and closing tags.
</response>
"""

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('completion_grpo')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(
        f"{log_dir}/training_{timestamp}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    return logger

class LoggingCallback(TrainerCallback):
    """Callback for logging training metrics"""
    def __init__(self, reward_func, logger, save_frequency=100):
        self.reward_func = reward_func
        self.save_frequency = save_frequency
        self.step = 0
        self.logger = logger
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        
        if logs and 'rewards/0' in logs and hasattr(self.reward_func, 'stats'):
            # Key performance metrics for wandb
            wandb_stats = {
                'reward': logs['rewards/0'],
                'average_reward': self.reward_func.stats.reward_components.get('average_reward', 0.0),
                'correct_answers': self.reward_func.stats.reward_components.get('correct_answers', 0),
                'step_continuity': self.reward_func.stats.step_stats.get('correct_step_numbering', 0) / 
                                  max(1, self.reward_func.stats.step_stats.get('correct_step_numbering', 0) + 
                                     self.reward_func.stats.step_stats.get('incorrect_step_numbering', 0))
            }
            
            # Detailed stats for local logging only
            local_stats = {
                'reward_components': {
                    'base_rewards': self.reward_func.stats.reward_components.get('base_rewards', 0) - getattr(self, '_last_base_rewards', 0),
                    'step_continuity_rewards': self.reward_func.stats.reward_components.get('step_continuity_rewards', 0) - getattr(self, '_last_step_continuity_rewards', 0),
                    'length_penalties': self.reward_func.stats.reward_components.get('total_length_penalty', 0.0) - getattr(self, '_last_length_penalties', 0.0)
                },
                'step_stats': {
                    'correct_step_numbering': self.reward_func.stats.step_stats.get('correct_step_numbering', 0) - getattr(self, '_last_correct_step_numbering', 0),
                    'incorrect_step_numbering': self.reward_func.stats.step_stats.get('incorrect_step_numbering', 0) - getattr(self, '_last_incorrect_step_numbering', 0),
                    'total_steps_completed': self.reward_func.stats.step_stats.get('total_steps_completed', 0) - getattr(self, '_last_total_steps_completed', 0)
                }
            }
            
            # Store current values for next round
            self._last_base_rewards = self.reward_func.stats.reward_components.get('base_rewards', 0)
            self._last_step_continuity_rewards = self.reward_func.stats.reward_components.get('step_continuity_rewards', 0)
            self._last_length_penalties = self.reward_func.stats.reward_components.get('total_length_penalty', 0.0)
            self._last_correct_step_numbering = self.reward_func.stats.step_stats.get('correct_step_numbering', 0)
            self._last_incorrect_step_numbering = self.reward_func.stats.step_stats.get('incorrect_step_numbering', 0)
            self._last_total_steps_completed = self.reward_func.stats.step_stats.get('total_steps_completed', 0)
            
            # Update logs with our metrics
            logs.update(wandb_stats)

def main():
    # Configuration
    model_type = "completion"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/qwen_sft/20250303_224627"
    dataset_name = "/Home/stat/laschos/math/AIMO2_initial/local_datasets/20250301_141300"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Log dataset information
    logger.info(f"Using dataset: {dataset_name}")
    
    # Initialize config
    reward_config = RewardConfig(model_type=model_type)
    reward_config.step_continuity_reward = 1.0  # Reward for correctly continuing step numbering
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandbname = f"{model_type}, {model_name}, {dataset_name}, {timestamp}"
    
    # Initialize wandb
    wandb.init(
        project="grpo",
        name=wandbname,
        config={
            "model_type": reward_config.model_type,
            "dataset": dataset_name,
            "base_reward": reward_config.base_reward,
            "step_continuity_reward": reward_config.step_continuity_reward,
            "length_penalty_factor": reward_config.length_penalty_factor
        }
    )
    
    # Initialize similarity checker first
    similarity_checker = SolutionSimilarityChecker(reward_config)
    
    # Initialize reward function with similarity checker
    reward_func = CompletionReward(reward_config, similarity_checker)
    logger.info("\nInitialized CompletionReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization= 0.6,
        max_lora_rank=64)
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    
    # We don't need to prepare partial solutions as they're already in the dataset
    
    def prepare_completion_data(example):
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
                    response_pattern = re.compile(r'<response>(.*?)</response>', re.DOTALL)
                    response_match = response_pattern.search(example['model_solution'])
                    if response_match:
                        response = response_match.group(1).strip()
                        step_pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
                        steps = step_pattern.findall(response)
                
                # Create partial solution with at least one step if available
                partial_solution = ''
                full_solution = ''
                if steps:
                    split_point = min(1, len(steps) - 1)
                    partial_steps = steps[:split_point]
                    partial_solution = '\n\n'.join([f'<step>{step}</step>' for step in partial_steps])
                    full_solution = '\n\n'.join([f'<step>{step}</step>' for step in steps])
                
                # Get the answer
                answer = example.get('answer', '')
                if not answer and 'correct_answer' in example:
                    answer = example['correct_answer']
                
                # Create the prompt with the data we have
                prompt = '<|im_start|>system\n' + SYSTEM_PROMPT + '<|im_end|>\n<|im_start|>user\n' + \
                        f"Problem: {example.get('problem', '')}\n\nPartial Solution: {partial_solution}<|im_end|>\n<|im_start|>assistant\n"
                
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
            
            # Extract response section
            response_pattern = re.compile(r'<response>(.*?)</response>', re.DOTALL)
            response_match = response_pattern.search(example['model_solution'])
            
            if not response_match:
                return {
                    'valid': False,
                    'prompt': '',
                    'problem': example.get('problem', ''),
                    'partial_solution': '',
                    'full_solution': '',
                    'answer': ''
                }
            
            response = response_match.group(1).strip()
            
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
            step_pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
            steps = step_pattern.findall(response)
            
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
            partial_steps = steps[:split_point]
            partial_solution = '\n\n'.join([f'<step>{step}</step>' for step in partial_steps])
            # Create full solution for reference
            full_solution = '\n\n'.join([f'<step>{step}</step>' for step in steps])
            
            # Get the answer from the example or extract from solution
            answer = example.get('answer', '')
            if not answer and 'correct_answer' in example:
                answer = example['correct_answer']
            
            # Create the prompt with all required fields
            prompt = '<|im_start|>system\n' + SYSTEM_PROMPT + '<|im_end|>\n<|im_start|>user\n' + \
                    f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}<|im_end|>\n<|im_start|>assistant\n"
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
            # Create a more detailed error log for debugging
            try:
                error_details = {
                    'error': str(e),
                    'has_problem': 'problem' in example and bool(example.get('problem')),
                    'has_model_solution': 'model_solution' in example and bool(example.get('model_solution')),
                    'has_answer': 'answer' in example and bool(example.get('answer')) or 
                                'correct_answer' in example and bool(example.get('correct_answer')),
                    'example_keys': list(example.keys())
                }
                logger.warning(f"Error details: {json.dumps(error_details)}")
            except:
                pass  # If we can't log details, just continue
                
            return {
                'valid': False,
                'prompt': '',
                'problem': example.get('problem', ''),
                'partial_solution': '',
                'full_solution': '',
                'answer': ''
            }
    
    # Apply the transformation and filter out invalid results
    data = load_from_disk(dataset_name)
    
    # Log dataset structure before processing
    logger.info(f"Original dataset columns: {data.column_names}")
    logger.info(f"Sample example keys: {list(data[0].keys()) if len(data) > 0 else 'No examples'}")
    
    # Process the data
    processed_data = data.map(prepare_completion_data)
    
    # Add token count to each example
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
    
    # Log validation results
    logger.info(f"Total examples in dataset: {len(data)}")
    logger.info(f"Valid examples after processing: {len(processed_data.filter(lambda x: x['valid']))}")
    logger.info(f"Valid examples with ≤ 2000 tokens: {len(valid_data)}")
    
    # Check if we have enough valid examples
    if len(valid_data) < 10:
        logger.error(f"Not enough valid examples found! Only {len(valid_data)} valid examples.")
        # Log some invalid examples to help diagnose the issue
        invalid_data = processed_data.filter(lambda x: not x['valid']).select(range(min(5, len(processed_data))))
        for i, example in enumerate(invalid_data):
            logger.error(f"Invalid example {i}:")
            logger.error(f"  Keys: {list(example.keys())}")
            logger.error(f"  Problem exists: {'problem' in example and bool(example['problem'])}")
            logger.error(f"  Original keys: {list(data[i].keys()) if i < len(data) else 'unknown'}")
        raise ValueError(f"Not enough valid examples found! Only {len(valid_data)} valid examples.")
    
    # Shuffle and select examples
    valid_data = valid_data.shuffle(seed=11)
    max_examples = min(2000, len(valid_data))
    valid_data = valid_data.select(range(max_examples))
    
    # More debug information
    logger.info(f"Using {len(valid_data)} examples for training")
    logger.info(f"Dataset columns: {valid_data.column_names}")
    logger.info(f"First example prompt length: {len(valid_data[0]['prompt']) if len(valid_data) > 0 else 'N/A'}")
    
    # Verify all examples have the required fields
    missing_fields = []
    for i in range(min(10, len(valid_data))):  # Check first 10 examples
        example = valid_data[i]
        if isinstance(example, dict):
            if not example.get('prompt') or not example.get('problem') or not example.get('partial_solution'):
                missing_fields.append(i)
        else:
            logger.error(f"Example at index {i} is not a dictionary but a {type(example)}")
            missing_fields.append(i)
    
    if missing_fields:
        logger.warning(f"Some examples are missing required fields: {missing_fields}")
        for i in missing_fields:
            if i < len(valid_data):
                example = valid_data[i]
                if isinstance(example, dict):
                    logger.warning(f"Example {i} fields: {list(example.keys())}")
                else:
                    logger.warning(f"Example {i} is not a dictionary: {type(example)}")
    
    
    # GRPO specific training arguments
    training_args = GRPOConfig(
        torch_empty_cache_steps=1,
        learning_rate=6e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=9,
        gradient_accumulation_steps=4,
        num_generations=9,
        max_prompt_length=2048,
        max_completion_length=2048,
        num_train_epochs=1,
        save_steps=50,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
    )
    
    # Initialize trainer with reward function
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=valid_data,
        callbacks=[LoggingCallback(reward_func=reward_func, logger=logger, save_frequency=10)]
    )
    
    # Train
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        wandb.finish()
        raise
        
    # Save model
    try:
        models_dir = "models"
        os.makedirs(os.path.join(models_dir, reward_config.model_type), exist_ok=True)
        model_output_dir = os.path.join(models_dir, reward_config.model_type, timestamp)
        model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"Merged model saved to {model_output_dir}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
