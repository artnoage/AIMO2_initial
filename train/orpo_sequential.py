import os
import torch
from datasets import load_from_disk, Dataset
from datetime import datetime
from trl import ORPOTrainer, ORPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
PatchDPOTrainer()
from trl import ORPOTrainer
from transformers import logging
import re

def _strip_prefix(s, pattern):
    # Use re.escape to escape any special characters in the pattern
    return re.sub(f"^{re.escape(pattern)}", "", s)

def main():
    logging.set_verbosity_info()

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=4096,
        load_in_4bit=False)

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens",],
        lora_alpha=64,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None)

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True)
    
    # Load dataset from disk
    dataset = load_from_disk("/Home/stat/laschos/AIMO2_initial/local_datasets/20241208_165400")

    def formatting_func(example):
        example["prompt"] = tokenizer.apply_chat_template([example["prompt"]], tokenize=False)
        example["chosen"] = tokenizer.apply_chat_template([example["chosen"]], tokenize=False)
        example["rejected"] = tokenizer.apply_chat_template([example["rejected"]], tokenize=False)
        example["chosen"] = _strip_prefix(example["chosen"], "<s>")
        example["rejected"] = _strip_prefix(example["rejected"], "<s>")
        # Ensure scores are present
        if "score_chosen" not in example or "score_rejected" not in example:
            raise ValueError("Dataset must include score_chosen and score_rejected fields")
        return example

    # Load and format dataset
    formatted_dataset = dataset.map(
        formatting_func,
        desc="Applying chat template"
    )

    # Split dataset into two parts and duplicate each to create four datasets
    dataset_size = len(formatted_dataset)
    chunk_size = dataset_size // 2
    # Create two datasets first
    base_datasets = [
        Dataset.from_dict(formatted_dataset[i:i+chunk_size])
        for i in range(0, dataset_size, chunk_size)
    ]
    # Duplicate each dataset to create four total datasets
    mini_datasets = base_datasets + base_datasets

    # Training configuration
    batch_size = 2
    grad_accum = 8
    base_training_args = {
        "max_length": 4096,
        "max_prompt_length": 2048,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "optim": "adafactor",
        "learning_rate": 4e-6,
        "lr_scheduler_type": "constant",
        "seed": 42,
        "bf16": True,
        "beta": 0.1,
        "weight_decay": 0.1
    }

    # Train on each mini dataset sequentially
    timestamp = None  # Will store the final timestamp
    for idx, mini_dataset in enumerate(mini_datasets):
        print(f"Starting training on chunk {idx+1}/4 (Dataset {(idx % 2) + 1}, Pass {(idx // 2) + 1})...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"train_results/{timestamp}"
        
        training_args = ORPOConfig(
            output_dir=output_dir,
            **base_training_args
        )

        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=mini_dataset,
            tokenizer=tokenizer
        )
        trainer.train()

    # Save merged model and LoRA weights using final timestamp
    models_dir = "models"
    loras_dir = "loras"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(loras_dir, exist_ok=True)
    
    model_output_dir = os.path.join(models_dir, timestamp)
    lora_output_dir = os.path.join(loras_dir, timestamp)
    
    # Save the merged model
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Final merged model saved to {model_output_dir}")
    
    # Save the LoRA weights
    model.save_pretrained_merged(lora_output_dir, tokenizer, save_method="lora")
    print(f"LoRA weights saved to {lora_output_dir}")

if __name__ == "__main__":
    main()
