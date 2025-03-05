import os
import wandb
import logging
import torch
import re
import asyncio
import tempfile
import subprocess
import sys
from datasets import load_dataset, Dataset
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional, Any, Union

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from rewards import BaseReward
from reward_stats import RewardStats
from utils.solution_utils import extract_numeric_answer
from utils.model_utils import time_limit

SYSTEM_PROMPT = """You will be given a mathematical problem. Your task is to write Python code that solves this problem.

<thinking>
First, analyze the problem carefully and determine the mathematical concepts involved.
Break down the problem into steps that can be implemented in code.
Consider edge cases and potential numerical issues.
Plan your approach before writing any code.
</thinking>

<response>
Write a complete, self-contained Python program that solves the problem.
Your code must:
1. Be syntactically correct and runnable with standard Python libraries (numpy, sympy, scipy are allowed)
2. Include clear comments explaining your approach
3. Print the final answer as a single float value (or integer if appropriate)
4. Handle potential errors gracefully
5. Be efficient and not use excessive resources

DO NOT include explanations outside of code comments. Your response should ONLY contain valid Python code.

Example format:
```python
# Solution for the problem
import math

# Step 1: Parse the problem
# [explanation comment]
...

# Step 2: Solve using appropriate method
# [explanation comment]
...

# Calculate and print the final answer
result = ...
print(result)  # Just the number, no text
```
</response>"""

class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('programming_grpo')
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
                'programming_reward': logs['rewards/0'],
                'average_reward': self.reward_func.stats.reward_components.get('average_reward', 0.0),
                'correct_solutions': self.reward_func.stats.reward_components.get('correct_solutions', 0),
                'syntax_valid_solutions': self.reward_func.stats.reward_components.get('syntax_valid_solutions', 0),
                'execution_valid_solutions': self.reward_func.stats.reward_components.get('execution_valid_solutions', 0)
            }
            
            # Detailed stats for local logging only
            local_stats = {
                'reward_components': {
                    'structure_rewards': self.reward_func.stats.reward_components.get('structure_rewards', 0) - getattr(self, '_last_structure_rewards', 0),
                    'syntax_rewards': self.reward_func.stats.reward_components.get('syntax_rewards', 0) - getattr(self, '_last_syntax_rewards', 0),
                    'execution_rewards': self.reward_func.stats.reward_components.get('execution_rewards', 0) - getattr(self, '_last_execution_rewards', 0),
                    'correctness_rewards': self.reward_func.stats.reward_components.get('correctness_rewards', 0) - getattr(self, '_last_correctness_rewards', 0),
                    'length_penalties': self.reward_func.stats.reward_components.get('total_length_penalty', 0.0) - getattr(self, '_last_length_penalties', 0.0)
                }
            }
            
            # Store current values for next round
            self._last_structure_rewards = self.reward_func.stats.reward_components.get('structure_rewards', 0)
            self._last_syntax_rewards = self.reward_func.stats.reward_components.get('syntax_rewards', 0)
            self._last_execution_rewards = self.reward_func.stats.reward_components.get('execution_rewards', 0)
            self._last_correctness_rewards = self.reward_func.stats.reward_components.get('correctness_rewards', 0)
            self._last_length_penalties = self.reward_func.stats.reward_components.get('total_length_penalty', 0.0)
            
            # Update logs with our metrics
            logs.update(wandb_stats)

def extract_code_from_response(response: str) -> str:
    """Extract code from the model's response"""
    # First try to extract code from ```python blocks
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # If no code blocks, try to extract from <response> section
    response_match = re.search(r'<response>\s*(.*?)\s*</response>', response, re.DOTALL)
    if response_match:
        response_content = response_match.group(1)
        # Check if there are code blocks within the response section
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response_content, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        # If no code blocks in response section, assume the entire response section is code
        return response_content
    
    # If no structured format, assume the entire response is code
    return response

def check_code_quality(code: str) -> Tuple[bool, str]:
    """Check code for syntax errors and basic linting issues"""
    # First check for syntax errors
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    
    # Check for basic issues without requiring pylint
    issues = []
    
    # Check for potentially dangerous operations
    dangerous_patterns = [
        (r'os\.system', 'Contains potentially unsafe system call'),
        (r'subprocess\.', 'Contains potentially unsafe subprocess call'),
        (r'exec\s*\(', 'Contains potentially unsafe exec call'),
        (r'eval\s*\(', 'Contains potentially unsafe eval call'),
        (r'__import__', 'Contains potentially unsafe dynamic import'),
        (r'open\s*\(.+,\s*[\'"]w', 'Contains file write operation'),
        (r'import\s+requests', 'Contains network request library'),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, code):
            issues.append(message)
    
    # If there are issues, return them
    if issues:
        return False, "Linting issues: " + "; ".join(issues)
    
    return True, "Code passed quality checks"

def run_code_safely(code: str, timeout: int = 5) -> Tuple[bool, Optional[float], str]:
    """Run the code in a safe environment with timeout and capture the output"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))
    
    try:
        # Run the code with timeout
        with time_limit(timeout):
            # Use subprocess to run the code
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return False, None, f"Execution error: {result.stderr}"
            
            # Try to parse the output as a float
            output = result.stdout.strip()
            try:
                answer = float(output)
                return True, answer, "Success"
            except ValueError:
                return False, None, f"Output is not a valid number: '{output}'"
    
    except TimeoutException:
        return False, None, "Code execution timed out"
    except Exception as e:
        return False, None, f"Error running code: {str(e)}"
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

class ProgrammingReward(BaseReward):
    """Reward class for programming solution evaluation"""
    
    __name__ = "programming_reward"
    relevant_stats = {
        'reward_components': [
            'structure_rewards', 'syntax_rewards', 'execution_rewards', 'correctness_rewards',
            'total_length_penalty', 'correct_solutions', 'syntax_valid_solutions', 
            'execution_valid_solutions', 'total_rewards', 'average_reward'
        ]
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a programming solution"""
        try:
            # Get problem and correct answer
            problem = kwargs.get('problem', '')
            correct_answer = kwargs.get('answer', '')
            
            if not all([problem, correct_answer]):
                self.logger.warning("Missing required inputs for programming reward calculation")
                return 0.0
            
            # Initialize reward
            reward = 0.0
            
            # 1. Check for thinking and response sections (structure reward)
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if has_thinking and has_response:
                structure_reward = self.config.structure_reward
                reward += structure_reward
                self.stats.reward_components['structure_rewards'] = self.stats.reward_components.get('structure_rewards', 0) + 1
                self.logger.info(f"Applied structure reward: +{structure_reward:.3f}")
            else:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
            
            # Extract code from the completion
            code = extract_code_from_response(completion)
            if not code:
                self.logger.info("No code found in completion")
                return reward
            
            # 2. Check code quality (syntax reward)
            code_quality_passed, quality_message = check_code_quality(code)
            if code_quality_passed:
                syntax_reward = self.config.syntax_reward
                reward += syntax_reward
                self.stats.reward_components['syntax_rewards'] = self.stats.reward_components.get('syntax_rewards', 0) + 1
                self.stats.reward_components['syntax_valid_solutions'] = self.stats.reward_components.get('syntax_valid_solutions', 0) + 1
                self.logger.info(f"Applied syntax reward: +{syntax_reward:.3f}")
            else:
                self.logger.info(f"Code quality check failed: {quality_message}")
                return reward  # Return early if syntax is invalid
            
            # 3. Run the code and check if it produces a valid output (execution reward)
            execution_success, result, error_message = run_code_safely(code, timeout=self.config.timeout)
            if execution_success and result is not None:
                execution_reward = self.config.execution_reward
                reward += execution_reward
                self.stats.reward_components['execution_rewards'] = self.stats.reward_components.get('execution_rewards', 0) + 1
                self.stats.reward_components['execution_valid_solutions'] = self.stats.reward_components.get('execution_valid_solutions', 0) + 1
                self.logger.info(f"Applied execution reward: +{execution_reward:.3f}")
            else:
                self.logger.info(f"Code execution failed: {error_message}")
                return reward  # Return early if execution fails
            
            # 4. Check if the result matches the correct answer (correctness reward)
            # Convert correct_answer to float if it's not already
            try:
                if isinstance(correct_answer, str):
                    numeric_answer, _ = extract_numeric_answer(correct_answer)
                    if numeric_answer is not None:
                        correct_answer = numeric_answer
                    else:
                        correct_answer = float(correct_answer)
                else:
                    correct_answer = float(correct_answer)
            except (ValueError, TypeError):
                self.logger.info(f"Could not convert correct answer to float: {correct_answer}")
                return reward
            
            # Compare with tolerance
            is_correct = abs(correct_answer - result) <= self.config.numeric_tolerance
            if is_correct:
                correctness_reward = self.config.correctness_reward
                reward += correctness_reward
                self.stats.reward_components['correctness_rewards'] = self.stats.reward_components.get('correctness_rewards', 0) + 1
                self.stats.reward_components['correct_solutions'] = self.stats.reward_components.get('correct_solutions', 0) + 1
                self.logger.info(f"Applied correctness reward: +{correctness_reward:.3f}")
            else:
                self.logger.info(f"Incorrect answer: expected {correct_answer}, got {result}")
            
            # Apply length penalty
            length_penalty = len(code) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating programming reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0

def main():
    # Configuration
    model_type = "programming_0"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/qwen_sft/20250303_224627"
    dataset_name = "Metaskepsis/Olympiads_medium"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Initialize config with reward values
    reward_config = RewardConfig(model_type=model_type)
    # Reward values are already set in the config class
    
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
            "structure_reward": reward_config.structure_reward,
            "syntax_reward": reward_config.syntax_reward,
            "execution_reward": reward_config.execution_reward,
            "correctness_reward": reward_config.correctness_reward
        }
    )
    
    # Initialize reward function
    reward_func = ProgrammingReward(reward_config)
    logger.info("\nInitialized ProgrammingReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=0.6,
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
    
    def get_questions(split="train") -> Dataset:
        # Load dataset
        data = load_dataset(dataset_name, split=split)
        data = data.map(lambda x: {
            'prompt': '<|im_start|>system\\n' + SYSTEM_PROMPT + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x.get('answer', x.get('correct_answer', ''))
        })
        return data
    
    formatted_dataset = get_questions()
    formatted_dataset = formatted_dataset.shuffle(seed=42)
    # Use a smaller dataset for programming training
    formatted_dataset = formatted_dataset.select(range(1000))
    
    # Verify first few entries
    for i in range(min(3, len(formatted_dataset))):
        entry = formatted_dataset[i]
        print(f"\nEntry {i} verification:")
        print(f"Answer: {entry.get('answer')}")
        print(f"Correct answer: {entry.get('correct_answer')}")
    
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,  # Fewer generations for programming tasks
        max_prompt_length=800,
        max_completion_length=3296,
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
        train_dataset=formatted_dataset,
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
