from datasets import load_dataset
from datetime import datetime
import os
import torch.distributed as dist
from trl import DPOTrainer, DPOConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers import logging
import torch
from peft import LoraConfig, get_peft_model


def main():
    logging.set_verbosity_info()

    # Load the base model and tokenizer
    model_name = "artnoage/metastral"
    
    # Initialize tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens",],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    def formatting_func(examples):
        formatted = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # Apply chat template to each message
            formatted["prompt"].append(tokenizer.apply_chat_template([prompt], tokenize=False))
            formatted["chosen"].append(tokenizer.apply_chat_template([prompt,chosen], tokenize=False))
            formatted["rejected"].append(tokenizer.apply_chat_template([prompt,rejected], tokenize=False))
            
        return formatted

    # Load and format dataset
    dataset = load_dataset("artnoage/dpo_full", split="train")

    formatted_dataset = dataset.map(
        formatting_func,
        batched=True,
        desc="Applying chat template"
    )
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{timestamp}"

    # Training configuration
    training_args = DPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=64,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=1,
        optim="adamw_8bit",
        seed=42,
        bf16=True,
        gradient_checkpointing=True,
        max_length=4096,
        max_prompt_length=2048,
        output_dir=output_dir,
        ddp_find_unused_parameters=False,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        deepspeed=None,  # Let DPOTrainer handle DDP
    )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
