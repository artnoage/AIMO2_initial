from datasets import load_dataset
from datetime import datetime
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
PatchDPOTrainer()
from trl import DPOTrainer
import os
from transformers import logging
from unsloth import is_bfloat16_supported

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def main():
    logging.set_verbosity_info()


    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/Home/stat/laschos/AIMO2_initial/models/20241130_144413",
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
        use_rslora=False)

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True)


    def formatting_func(examples):
        formatted = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # Apply chat template to each message
            formatted["prompt"].append(tokenizer.apply_chat_template([prompt], tokenize=False))
            formatted["chosen"].append(tokenizer.apply_chat_template([chosen], tokenize=False))
            formatted["rejected"].append(tokenizer.apply_chat_template([rejected], tokenize=False))
            
        return formatted

    # Load and format dataset
    dataset = load_dataset("artnoage/dpo_full", split="train")
    formatted_dataset = dataset.map(
        formatting_func,
        batched=True,
        desc="Applying chat template"
    )
    print(formatted_dataset[0])
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{timestamp}"
    

    training_args = DPOConfig(
        per_device_train_batch_size = 2,
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
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_length=4096,
        max_prompt_length=1024)

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
