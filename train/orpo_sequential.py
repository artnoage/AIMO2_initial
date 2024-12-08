import os
import torch
from datasets import load_dataset, Dataset
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
    
    # Load dataset - adjust path as needed
    dataset = load_dataset("artnoage/orpo", split="train")

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

    # Split dataset into two parts for sequential training
    dataset_size = len(formatted_dataset)
    split_point = dataset_size // 2
    first_dataset = Dataset.from_dict(formatted_dataset[:split_point])
    second_dataset = Dataset.from_dict(formatted_dataset[split_point:])

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"train_results/{timestamp}"
    
    # Calculate total steps for both phases
    batch_size = 2
    grad_accum = 32
    effective_batch_size = batch_size * grad_accum
    steps_per_epoch_phase1 = len(first_dataset) // effective_batch_size
    steps_per_epoch_phase2 = len(second_dataset) // effective_batch_size
    total_steps = steps_per_epoch_phase1 + steps_per_epoch_phase2

    # Training configuration with steps-based scheduling
    base_training_args = {
        "max_length": 4096,
        "max_prompt_length": 2048,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_steps": total_steps,  # Use steps instead of epochs
        "learning_rate": 6e-6,
        "logging_steps": 1,
        "optim": "adamw_torch",
        "seed": 42,
        "bf16": True,
        "weight_decay": 0.1,
        "lr_scheduler_type": "linear",
        "warmup_steps": total_steps // 10,  # 10% warmup of total steps
        "beta": 0.1
    }

    # Phase 1: Train on first dataset
    print("Starting Phase 1 training...")
    phase1_output_dir = os.path.join(base_output_dir, "phase1")
    training_args_phase1 = ORPOConfig(
        output_dir=phase1_output_dir,
        **base_training_args
    )

    trainer_phase1 = ORPOTrainer(
        model=model,
        args=training_args_phase1,
        train_dataset=first_dataset,
        tokenizer=tokenizer
    )
    trainer_phase1.train(resume_from_checkpoint=None)

    # Phase 2: Train on second dataset, continuing from phase 1's progress
    print("Starting Phase 2 training...")
    phase2_output_dir = os.path.join(base_output_dir, "phase2")
    training_args_phase2 = ORPOConfig(
        output_dir=phase2_output_dir,
        **base_training_args
    )

    trainer_phase2 = ORPOTrainer(
        model=model,  # Continue with the same model
        args=training_args_phase2,
        train_dataset=second_dataset,
        tokenizer=tokenizer
    )
    # Continue training from where phase 1 left off
    trainer_phase2.train(resume_from_checkpoint=True)

    # Save the final merged model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    final_output_dir = os.path.join(models_dir, timestamp)
    
    # Save the merged model using unsloth's method
    model.save_pretrained_merged(final_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Final merged model saved to {final_output_dir}")

if __name__ == "__main__":
    main()
