from datasets import load_dataset
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from accelerate import dispatch_model, infer_auto_device_map
import torch
from datetime import datetime
import os
from typing import Dict, Sequence
import GPUtil

def print_gpu_utilization():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f'\nGPU ID: {gpu.id} ({gpu.name})')
        print(f'GPU load: {gpu.load*100:.1f}%')
        print(f'GPU memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB')
        print(f'GPU memory free: {gpu.memoryFree}MB')
    if torch.cuda.is_available():
        print(f'\nPyTorch GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB')
        print(f'PyTorch GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB')


def main():
    print("\n=== Initial GPU State ===")
    print_gpu_utilization()

    # Load model and tokenizer
    model_name = "artnoage/metastral"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with automatic device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # This enables model parallelism
        max_memory={i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / 1024**3)}GiB" for i in range(torch.cuda.device_count())},
    )
    
    print("\n=== After Model Load ===")
    print_gpu_utilization()

    # Setup Mistral chat template with start/end tokens and proper spacing
    tokenizer.chat_template = "<s>[INST] {{ messages[0]['content'] }} [/INST]\n{{ messages[1]['content'] }}</s>"

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False) 
                for convo in convos]
        return {"text": texts}

    # Load, shuffle and format dataset
    dataset = load_dataset("Metaskepsis/sft", split="train")
    dataset = dataset.shuffle(seed=42)  # Add deterministic shuffling
    
    print("\nFirst conversation before formatting:")
    print(json.dumps(dataset[0]["conversations"], indent=2))
    
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    print("\nFirst conversation after formatting:")
    print(json.dumps(formatted_dataset[0]["text"], indent=2))

    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"train_results/classic_{timestamp}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=4e-6,
        logging_steps=1,
        save_strategy="steps", 
        save_steps=200,
        fp16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=formatted_dataset["text"],
        args=training_args,
        data_collator=default_data_collator,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
