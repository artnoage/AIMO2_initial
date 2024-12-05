import torch
from datasets import load_dataset
from datetime import datetime
from trl import ORPOTrainer, ORPOConfig
from unsloth import FastLanguageModel, PatchORPOTrainer
from unsloth.chat_templates import get_chat_template
PatchORPOTrainer()
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
    dataset = load_dataset("artnoage/orpo_full", split="train")

    def formatting_func(example):
        example["prompt"] = tokenizer.apply_chat_template([example["prompt"]], tokenize=False, add_generation_prompt=True)
        # Format responses with their scores
        example["responses"] = [
            {
                "response": tokenizer.apply_chat_template([r["response"]], tokenize=False),
                "score": r["score"]
            }
            for r in example["responses"]
        ]
        # Strip prefix from responses
        for resp in example["responses"]:
            resp["response"] = _strip_prefix(resp["response"], "<s>")
        return example

    # Load and format dataset
    formatted_dataset = dataset.map(
        formatting_func,
        desc="Applying chat template"
    )

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/orpo_{timestamp}"

    # Print maximum weight value before training
    max_weight = max([torch.max(param).item() for param in model.parameters()])
    print(f"Maximum weight value before training: {max_weight}")

    # ORPO specific training arguments
    training_args = ORPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        num_train_epochs=1,
        learning_rate=6e-6,
        logging_steps=1,
        optim="adamw_8bit",
        seed=42,
        bf16=True,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        output_dir=output_dir,
        # ORPO specific parameters
        beta=0.1,  # Controls the strength of the ORPO regularization
        desirable_score=1.0,  # Target score for responses
        undesirable_score=0.0  # Minimum acceptable score
    )

    # Initialize ORPO trainer
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        max_length=4096,
        max_prompt_length=1024
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
