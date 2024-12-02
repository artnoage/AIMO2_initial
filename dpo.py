from datasets import load_dataset
from datetime import datetime
from tqdm import tqdm
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
PatchDPOTrainer()
from trl import DPOTrainer
import os
import torch
import GPUtil
from transformers import logging
from unsloth import is_bfloat16_supported

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        model_name="/Home/stat/laschos/AIMO2_initial/models/20241129_162714",
        max_seq_length=4096,
        load_in_4bit=False)
        
    print("\n=== After Model Load ===")
    print_gpu_utilization()

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
        use_rslora=False)
    
    print("\n=== After LoRA Configuration ===")
    print_gpu_utilization()

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True,
    )


    def apply_chat_template(example, tokenizer):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["chosen"] = tokenizer.apply_chat_template([example["chosen"]], tokenize=False)
        example["rejected"] = tokenizer.apply_chat_template([example["rejected"]], tokenize=False)
        return example

    # Load the local DPO dataset
    raw_datasets = load_dataset("json", data_files="dpo_dataset.json", split="train")
    print("\nDataset keys before mapping:")
    print(raw_datasets.column_names)
    
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs = {"tokenizer": tokenizer},
        num_proc = 12,
        desc = "Formatting comparisons with prompt template")

    print("\nDataset keys after mapping:")
    print(raw_datasets.column_names)
    # Print formatted example
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{timestamp}"
    

    training_args = DPOConfig(
        per_gpu_train_batch_size = 2,
        gradient_accumulation_steps = 64,
        num_train_epochs = 1,
        learning_rate = 4e-6,
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        output_dir = output_dir)
    # Initialize DPO trainer
    
    trainer = DPOTrainer(
        model=model,
        train_dataset=raw_datasets,
        tokenizer=tokenizer,
        args=training_args,
        max_length = 4096,
        max_prompt_length = 1024,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
