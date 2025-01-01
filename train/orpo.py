import os
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
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
        model_name="mistralai/Mathstral-7B-v0.1",
        max_seq_length=4096,
        load_in_4bit=False)

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens",],
        lora_alpha=128,
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
    #dataset = load_dataset("local_dataset/20241208_111257", split="train")
    dataset = load_from_disk("local_datasets/20241231_191017")
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
    
    # Create a shuffled copy with seed 42
    shuffled_dataset = formatted_dataset.shuffle(seed=42)
    shuffled_dataset2=shuffled_dataset.shuffle(seed=42)
    shuffled_dataset3=shuffled_dataset2.shuffle(seed=42)
    # Concatenate original and shuffled datasets
    formatted_dataset = concatenate_datasets([formatted_dataset, shuffled_dataset,shuffled_dataset2,shuffled_dataset3])

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{timestamp}"


    # ORPO specific training arguments
    training_args = ORPOConfig(
        max_length=4096,
        max_prompt_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=4e-6,
        logging_steps=1,
        optim = "adafactor",
        seed=42,
        bf16=True,
        weight_decay=0.1,
        lr_scheduler_type="constant",
        output_dir=output_dir,
        beta=0.1)

    # Initialize ORPO trainer
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save both merged model and LoRA weights
    models_dir = "models"
    loras_dir = "loras"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(loras_dir, exist_ok=True)
    
    model_output_dir = os.path.join(models_dir, timestamp)
    
    # Save the merged model
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {model_output_dir}")
    

if __name__ == "__main__":
    main()
