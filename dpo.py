from datasets import load_dataset
import json
from datetime import datetime
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
PatchDPOTrainer()
from transformers import TrainingArguments
from trl import DPOTrainer
import os
import torch
import GPUtil
from transformers import logging
from unsloth import is_bfloat16_supported


# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def print_gpu_utilization():
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_gpus:
        visible_ids = [int(x) for x in visible_gpus.split(",")]
        GPUs = [gpu for gpu in GPUtil.getGPUs() if gpu.id in visible_ids]
        for gpu in GPUs:
            print(f'\nGPU ID: {gpu.id} ({gpu.name})')
            print(f'GPU load: {gpu.load*100:.1f}%')
            print(f'GPU memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB')
            print(f'GPU memory free: {gpu.memoryFree}MB')
        if torch.cuda.is_available():
            print(f'\nPyTorch GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB')
            print(f'PyTorch GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB')

def main():
    logging.set_verbosity_info()
    print("\n=== Initial GPU State ===")
    print_gpu_utilization()
    print("\n=== Before Model Load ===")
    print_gpu_utilization()

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=4096,
        dtype="bfloat16",
        load_in_4bit=False)
        
    print("\n=== After Model Load ===")
    print_gpu_utilization()

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",],
        lora_alpha=64,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing=False,  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False)
    
    print("\n=== After LoRA Configuration ===")
    print_gpu_utilization()

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True,
    )

    def formatting_prompts_func(examples):
        # Handle single example case
        if isinstance(examples, dict):
            examples = [examples]
            
        formatted_pairs = []
        for example in examples:
            # Format chosen response
            chosen_text = tokenizer.apply_chat_template(example["chosen"], tokenize=False, add_generation_prompt=False)
            
            # Format rejected response
            rejected_text = tokenizer.apply_chat_template(example["rejected"], tokenize=False, add_generation_prompt=False)
            
            # Use prompt as-is since it's already a plain string
            prompt_text = example["prompt"]
            
            formatted_pairs.append({
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text
            })
        return formatted_pairs

    # Load the DPO dataset
    dataset = load_dataset("artnoage/dpo3", split="train")
    
    # Print original format and structure
    print("\nDataset structure:")
    print("Columns:", dataset.column_names)
    print("\nFirst example before formatting:")
    print(json.dumps(dataset[0], indent=2))
    
    # Apply formatting
    formatted_dataset = dataset.map(
        lambda x: {"formatted": formatting_prompts_func([x])[0]},
        remove_columns=dataset.column_names
    )
    
    # Print formatted example
    print("\nFirst example after formatting:")
    print(json.dumps(formatted_dataset[0]["formatted"], indent=2))

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./train_results/dpo/{timestamp}"
    

    training_args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 5e-6,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = output_dir,
        report_to = "all")
    # Initialize DPO trainer
    
    trainer = DPOTrainer(
        model=model,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
