import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=2048,
        dtype=None,  # Will use best dtype available
        load_in_4bit=True,
    )

    # Load dataset
    dataset = load_dataset("artnoage/sft")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
    )

    # Initialize trainer
    trainer = FastLanguageModel.get_trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        dataset_text_field="text",  # Adjust this based on your dataset's structure
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
