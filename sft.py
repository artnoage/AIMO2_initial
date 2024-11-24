import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq
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

    # Load and prepare the dataset
    dataset = load_dataset("artnoage/sft")
    
    def format_conversation(example):
        """Format conversation into a chat format with roles and content."""
        formatted_text = ""
        for message in example['conversations']:
            role = message['role']
            content = message['content']
            formatted_text += f"{role}: {content}\n\n"
        return {"text": formatted_text.strip()}
    
    # Apply the formatting to the dataset
    formatted_dataset = dataset.map(format_conversation, remove_columns=dataset["train"].column_names)
    
    print("Dataset sample:")
    print(formatted_dataset["train"][0]["text"])
    
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
        train_dataset=formatted_dataset["train"],
        dataset_text_field="text",
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
