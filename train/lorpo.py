import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datetime import datetime
from trl import ORPOTrainer, ORPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
PatchDPOTrainer()
from trl import ORPOTrainer
from transformers import logging
import re


model_type = "light"
model_name= "/Home/stat/laschos/AIMO2_initial/models/light/20250206_212611"
dataset_name="/Home/stat/laschos/AIMO2_initial/local_datasets/light/20250209_171956"


# Check if model_type is in paths
if model_type not in model_name:
    print("\n" + "!"*80)
    print(f"WARNING: model_type '{model_type}' not found in model_name path!")
    print("!"*80 + "\n")

if model_type not in dataset_name:
    print("\n" + "!"*80)
    print(f"WARNING: model_type '{model_type}' not found in dataset_name path!")
    print("!"*80 + "\n")

def _strip_prefix(s, pattern):
    # Use re.escape to escape any special characters in the pattern
    return re.sub(f"^{re.escape(pattern)}", "", s)

def main():
    # Set training type
    logging.set_verbosity_info()

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=16384,
        load_in_4bit=False)

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=256,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens",],
        lora_alpha=256,
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
    #dataset = load_dataset("Metaskepsis/orpo", split="train")
    dataset = load_from_disk(dataset_name)
    def formatting_func(example):
        # Only keep the required fields
        required_fields = ['prompt', 'chosen', 'rejected', 'score_chosen', 'score_rejected']
        filtered_example = {k: example[k] for k in required_fields if k in example}
        
        # Apply formatting
        filtered_example["prompt"] = tokenizer.apply_chat_template([filtered_example["prompt"]], tokenize=False)
        filtered_example["chosen"] = tokenizer.apply_chat_template([filtered_example["chosen"]], tokenize=False)
        filtered_example["rejected"] = tokenizer.apply_chat_template([filtered_example["rejected"]], tokenize=False)
        filtered_example["chosen"] = _strip_prefix(filtered_example["chosen"], "<s>")
        filtered_example["rejected"] = _strip_prefix(filtered_example["rejected"], "<s>")
        
        # Ensure all required fields are present
        missing_fields = [f for f in required_fields if f not in filtered_example]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        return filtered_example
    # Load and format dataset
    formatted_dataset = dataset.map(
        formatting_func,
        desc="Applying chat template"
    )
    
    # Create a shuffled copy with seed 42
    shuffled_dataset = formatted_dataset.shuffle(seed=42)
    shuffled_dataset2=shuffled_dataset.shuffle(seed=42)
    shuffled_dataset3=shuffled_dataset2.shuffle(seed=42)
    #shuffled_dataset4=shuffled_dataset3.shuffle(seed=42)
    # Concatenate original and shuffled datasets
    formatted_dataset = concatenate_datasets([shuffled_dataset])

    # Create timestamped output directory with model_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{model_type}/{timestamp}"


    # ORPO specific training arguments
    training_args = ORPOConfig(
        max_length=16384,
        max_prompt_length=8192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        learning_rate=2e-6,
        logging_steps=1,
        optim = "adafactor",
        seed=42,
        bf16=True,
        weight_decay=0.05,
        lr_scheduler_type="constant",
        output_dir=output_dir,
        beta=0.1)

    # Initialize ORPO trainer
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save both merged model and LoRA weights
    models_dir = "models"
    
    os.makedirs(os.path.join(models_dir, model_type), exist_ok=True)
    
    
    model_output_dir = os.path.join(models_dir, model_type, timestamp)
    
    # Save the merged model
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {model_output_dir}")
    

if __name__ == "__main__":
    main()
