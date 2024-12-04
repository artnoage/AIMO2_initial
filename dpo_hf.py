from datasets import load_dataset
from datetime import datetime
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
    tokenizer.chat_template = tokenizer.default_chat_template  # Ensure chat template is loaded
    
    # Initialize model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "lm_head", "embed_tokens"],
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
            # Format messages using the chat template with proper roles
            formatted["prompt"].append(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False))
            formatted["chosen"].append(tokenizer.apply_chat_template([{"role": "assistant", "content": chosen}], tokenize=False))
            formatted["rejected"].append(tokenizer.apply_chat_template([{"role": "assistant", "content": rejected}], tokenize=False))
            
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        num_train_epochs=1,
        learning_rate=4e-6,
        logging_steps=1,
        optim="adamw_torch",
        seed=42,
        fp16=True,
        output_dir=output_dir
    )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_length=4096,
        max_prompt_length=1024
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
