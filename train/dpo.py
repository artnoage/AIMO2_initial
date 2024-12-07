import torch
from datasets import load_dataset
from datetime import datetime
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
PatchDPOTrainer()
from trl import DPOTrainer
from transformers import logging
import re

def _strip_prefix(s, pattern):
    # Use re.escape to escape any special characters in the pattern
    return re.sub(f"^{re.escape(pattern)}", "", s)

def main():
    logging.set_verbosity_info()


    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/Home/stat/laschos/AIMO2_initial/models/20241206_112933",
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
        loftq_config = None)

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True)
    dataset = load_dataset(path="artnoage/orpo2", split="train")


    def formatting_func(example):
        example["prompt"]=tokenizer.apply_chat_template([example["prompt"]],tokenize=False, add_generation_prompt=True)
        example["chosen"]=tokenizer.apply_chat_template([example["chosen"]], tokenize=False)
        example["rejected"]=tokenizer.apply_chat_template([example["rejected"]], tokenize=False)
        example["chosen"] = _strip_prefix(example["chosen"], "<s>")
        example["rejected"] = _strip_prefix(example["rejected"], "<s>")    
        return example

    # Load and format dataset
    
    formatted_dataset = dataset.map(
        formatting_func,
        desc="Applying chat template"
    )
    print(formatted_dataset[0]["prompt"])
    print(formatted_dataset[0]["chosen"])
    print(formatted_dataset[0]["rejected"])
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{timestamp}"

    # Print maximum weight value before training
    max_weight = max([torch.max(param).item() for param in model.parameters()])
    print(f"Maximum weight value before training: {max_weight}")
    training_args = DPOConfig(
         max_length=4096,
        max_prompt_length=1024,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 32,
        num_train_epochs = 2,
        learning_rate = 4e-6,
        logging_steps = 1,
        optim = "adamw_torch",
        seed=42,
        bf16=True,
        weight_decay=0.1,
        lr_scheduler_type = "linear",
        warmup_ratio = 0.1,
        output_dir = output_dir)
    # Initialize DPO trainer
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
       )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
