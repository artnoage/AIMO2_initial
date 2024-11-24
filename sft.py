import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import bitsandbytes as bnb
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=8192,
        load_in_8bit=True,
    )

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "human", "assistant": "assistant"},
        map_eos_token=True,
    )

    # Load and prepare the dataset
    dataset = load_dataset("artnoage/sft", split="train")
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    # Apply the formatting to the dataset
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    print("Raw conversations sample (index 5):")
    print(dataset[5]["conversations"])
    print("\nFormatted text sample (index 5):")
    print(formatted_dataset[5]["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./train_results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        optim="adamw_bnb_8bit",
    )

    # Initialize trainer
    trainer = FastLanguageModel.get_trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
