from datasets import load_dataset
import json
from datetime import datetime
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
import re
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

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
        model_name="/Home/stat/laschos/AIMO2_initial/models",
        max_seq_length=8192,
        load_in_4bit=False)
        
    print("\n=== After Model Load ===")
    print_gpu_utilization()

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",],
        lora_alpha=32,
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

    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    def apply_chat_template(example, tokenizer):
        if all(k in example.keys() for k in ("chosen", "rejected")):
                # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
                prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
                # Insert system message
                if example["chosen"][0]["role"] != "system":
                    prompt_messages.insert(0, {"role": "system", "content": ""})
                else:
                    prompt_messages.insert(0, example["chosen"][0])
                chosen_messages = example["chosen"][1:]
                rejected_messages = example["rejected"][1:]
                example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
                example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
                example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
                return example
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )

    # Load the DPO dataset
    raw_datasets = load_dataset("artnoage/dpo3", split="train")
    print("\nDataset keys before mapping:")
    print(raw_datasets.column_names)
    column_names = list(raw_datasets.features)
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs = {"tokenizer": tokenizer},
        num_proc = 12,
        remove_columns = column_names,
        desc = "Formatting comparisons with prompt template")

    print("\nDataset keys after mapping:")
    print(raw_datasets.column_names)

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    raw_datasets = raw_datasets.rename_columns({"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"})
    print("\nDataset keys after mapping:")
    print(raw_datasets.column_names)
    # Print formatted example
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{timestamp}"
    
    print("\nToken counts for all examples:")
    total_tokens = 0
    #for i in range(len(raw_datasets)):
        #row = raw_datasets[i]
        #tokens_prompt = len(tokenizer.encode(row["prompt"]))
        #tokens_chosen = len(tokenizer.encode(row["chosen"]))
        #tokens_rejected = len(tokenizer.encode(row["rejected"]))
        #example_total = tokens_prompt + tokens_chosen + tokens_rejected
        #total_tokens += example_total
        #print(f"\nExample {i}:")
        #print(f"Prompt tokens: {tokens_prompt}")
        #print(f"Chosen tokens: {tokens_chosen}")
        #print(f"Rejected tokens: {tokens_rejected}")
        #print(f"Example total: {example_total}")
    
    print(f"\nTotal tokens across all examples: {total_tokens}")
    print(f"Average tokens per example: {total_tokens / len(raw_datasets):.1f}")


    training_args = DPOConfig(
        per_gpu_train_batch_size = 1,
        gradient_accumulation_steps = 64,
        num_train_epochs = 1,
        learning_rate = 5e-6,
        logging_steps = 1,
        optim = "adamw_torch",
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
        max_length = 8192,
        max_prompt_length = 1024,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
