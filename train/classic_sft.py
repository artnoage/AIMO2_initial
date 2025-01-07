from datasets import load_dataset 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import torch
from datetime import datetime
from pathlib import Path


def main():
    # Print initial GPU state
    

    # Load model and tokenizer
    model_name = "mistralai/Mathstral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with DeepSpeed (no device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
    )


    # Add padding token
    new_pad_token = "[control_748]"
    tokenizer.add_special_tokens({"pad_token": new_pad_token})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Dataset preparation
    def formatting_prompts_func(examples):
        texts = [f"<s>[INST] {conv[0]['content']} [/INST]\n{conv[1]['content']}</s>"
                 for conv in examples["conversations"]]
        return {"text": texts}

    dataset = load_dataset("Metaskepsis/sft", split="train")
    dataset = dataset.shuffle(seed=42)
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

    # Tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=4096,
            padding="max_length",
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset.column_names
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"train_results/classic_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Batch size per device
        gradient_accumulation_steps=32,  # Accumulate gradients
        learning_rate=4e-6,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,  # Use bfloat16 for better performance
        gradient_checkpointing=True,
        deepspeed="ds_config.json",  # DeepSpeed config
        tf32=True,
        weight_decay=0.01,
        warmup_steps=1000
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=default_data_collator,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
