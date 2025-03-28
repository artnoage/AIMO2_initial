import os
import wandb
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from trl import GRPOTrainer, GRPOConfig

from config import RewardConfig
from dynamic_reward import DynamicReward
from utils.similarity_checker import SolutionSimilarityChecker
from utils.data_preparation import prepare_combined_data
from utils.agents import (
    FULLSOLUTION_SYSTEM_PROMPT,
    FINALIZATION_SYSTEM_PROMPT,
    PROGRAMMER_SYSTEM_PROMPT,
    TUTOR_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
    ARCHITECT_SYSTEM_PROMPT
)

# Initialize the accelerator
accelerator = Accelerator()
load_dotenv()

def is_bfloat16_supported():
    # Check for bfloat16 support
    return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

def main():
    model_type = "dynamic_0"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/dynamic_2/20250324_215025"
    dataset_name = "Metaskepsis/Olympiads_medium"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{model_type}/{timestamp}"

    # Initialize Weights and Biases
    wandb.init(
        project="grpo",
        name=f"{model_type}_{timestamp}",
        config={"model_type": model_type, "dataset": dataset_name}
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto"
    )
    for name, module in model.named_modules():
        if "Block" in str(type(module)):
            print(name, type(module))
    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Don't use auto_mapping which tries to find TransformerBlock
        modules_to_save=None
    )

    # Initialize reward function components
    reward_config = RewardConfig(model_type=model_type)
    similarity_checker = SolutionSimilarityChecker(reward_config)
    reward_func = DynamicReward(reward_config, similarity_checker)

    # Load and prepare datasets
    data1 = load_dataset(dataset_name, split="train").shuffle(seed=141).select(range(2500))
    data2 = load_dataset("Metaskepsis/Olympiads_hard", split="train").shuffle(seed=141).select(range(500))
    data = concatenate_datasets([data1, data2]).shuffle(seed=141)

    distribution = {
        'solution': 0.25,
        'programming': 0.25,
        'finalization': 0,
        'tutor': 0,
        'test_programming': 0.25,
        'architect': 0.25
    }

    dataset = prepare_combined_data(
        data,
        FULLSOLUTION_SYSTEM_PROMPT,
        FINALIZATION_SYSTEM_PROMPT,
        PROGRAMMER_SYSTEM_PROMPT,
        TUTOR_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT,
        ARCHITECT_SYSTEM_PROMPT,
        tokenizer,
        distribution
    )

    # Define training arguments
    training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=2e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    save_steps=50,
    report_to="wandb",
    num_generations=8,
    max_prompt_length=1800,
    max_completion_length=5200)

    # Initialize the trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        train_dataset=dataset,
        # Disable FSDP which is causing the TransformerBlock error
        fsdp_config=None
    )

    # Start training
    trainer.train()
    model.save_pretrained(output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
