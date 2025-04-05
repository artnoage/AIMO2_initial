import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, 
    TrainerCallback, TrainerState, TrainerControl
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DPOTrainer, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from contextlib import contextmanager
import signal
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.data_preparation import prepare_test_driven_programmer_data
from utils.agents import TEST_DRIVEN_PROGRAMMER_SYSTEM_PROMPT
from grpo.rewards import TestDrivenProgrammerReward
from grpo.config import RewardConfig

class TimeoutException(Exception):
    pass

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / model_type
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logger = logging.getLogger(model_type)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
        
    # Add file handler
    file_handler = logging.FileHandler(
        log_dir / f"{model_type}_{timestamp}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    return logger

class LoggingCallback(TrainerCallback):
    """Callback for logging training metrics"""
    def __init__(self, reward_func, logger, save_frequency=100):
        self.reward_func = reward_func
        self.logger = logger
        self.save_frequency = save_frequency
        self.step = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.save_frequency == 0:
            self.logger.info(f"Step {self.step}: Saving model checkpoint")
            control.should_save = True
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info(f"Step {state.global_step}: {logs}")
            
            # Log reward statistics if available
            if hasattr(self.reward_func, 'stats'):
                self.logger.info("\nReward Statistics Summary:")
                self.logger.info(self.reward_func.stats.get_summary())

def main():
    # Configuration
    model_type = "test_driven_programmer_0"
    base_model = "Qwen/Qwen1.5-7B"
    dataset_name = "Metaskepsis/math_dataset_filtered"
    output_dir = f"models/{model_type}"
    
    # Create logger
    logger = setup_logging(model_type)
    logger.info(f"Starting training for {model_type} using {base_model}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Define reward config
    reward_config = RewardConfig(
        model_type=model_type,
        base_reward=2.0,
        syntax_reward=0.5,
        execution_reward=0.5,
        correctness_reward=1.0,
        syntax_penalty=0.2,
        length_penalty_factor=0.0001,
        numeric_tolerance=1e-2,
        timeout=30,
        logging_dir="logs",
        stats_dir="stats"
    )
    
    # Create reward function
    reward_func = TestDrivenProgrammerReward(reward_config)
    
    def get_questions(split="train") -> Dataset:
        """Load and prepare dataset"""
        logger.info(f"Loading {split} dataset from {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Prepare dataset for test-driven programmer tasks
        test_driven_programmer_data = prepare_test_driven_programmer_data(
            dataset, 
            TEST_DRIVEN_PROGRAMMER_SYSTEM_PROMPT
        )
        logger.info(f"Prepared {len(test_driven_programmer_data)} test-driven programmer examples")
        
        return test_driven_programmer_data
    
    # Load training dataset
    train_dataset = get_questions("train")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        learning_rate=2e-5,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="prompt",
        max_seq_length=4096,
        callbacks=[LoggingCallback(reward_func, logger)]
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model")
    trainer.save_model(output_dir)
    
    logger.info("Training complete")

if __name__ == "__main__":
    main()
