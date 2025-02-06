import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datetime import datetime
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
PatchDPOTrainer()
from transformers import logging
import re
import sys
sys.path.append(".")  # Add project root to path
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer


model_type = "light"
model_name = "/Home/stat/laschos/AIMO2_initial/models/light/20250205_085918"
dataset_name = "/Home/stat/laschos/AIMO2_initial/local_datasets/light/20250206_082412"


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

def reward_func(completions, **kwargs):
    """Custom reward function that processes completions"""
    # Get correct answers from kwargs
    correct_answers = kwargs.get('correct_answer', [])
    
    # Debug prints
    print("\n" + "="*50)
    print(f"Number of completions: {len(completions)}")
    print(f"Number of correct answers: {len(correct_answers)}")
    print("\nFirst completion:")
    print(f"Type: {type(completions[0])}")
    print(f"Content: {completions[0][:200]}...")  # First 200 chars
    if correct_answers:
        print(f"\nFirst correct answer: {correct_answers[0]}")
    print("="*50 + "\n")
    
    # Process completions and compute rewards
    rewards = []
    for completion, correct_answer in zip(completions, correct_answers):
        # Extract model's answer from completion
        model_answer = extract_answer_from_solution(completion)
        if model_answer is None:
            rewards.append(0.0)
            continue
            
        # Convert both answers to numeric values
        numeric_answer, _ = extract_numeric_answer(model_answer, debug=False)
        correct_numeric, _ = extract_numeric_answer(correct_answer, debug=False)
        
        if numeric_answer is None or correct_numeric is None:
            rewards.append(0.0)
            continue
            
        # Compare with tolerance
        tolerance = 1e-6
        is_correct = abs(numeric_answer - correct_numeric) <= tolerance
        rewards.append(1.0 if is_correct else 0.0)
        
    return rewards

def main():
    # Set training type
    logging.set_verbosity_info()

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
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
    dataset = load_from_disk(dataset_name)
    
    def formatting_func(example):
        # Only keep the required fields
        required_fields = ['prompt', 'correct_answer']
        filtered_example = {k: example[k] for k in required_fields if k in example}
        
        # Apply formatting
        filtered_example["prompt"] = tokenizer.apply_chat_template([filtered_example["prompt"]], tokenize=False)
        filtered_example["prompt"] = _strip_prefix(filtered_example["prompt"], "<s>")
        
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
    shuffled_dataset2 = shuffled_dataset.shuffle(seed=42)
    shuffled_dataset3 = shuffled_dataset2.shuffle(seed=42)
    
    # Concatenate original and shuffled datasets
    formatted_dataset = concatenate_datasets([shuffled_dataset])

    # Create timestamped output directory with model_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{model_type}/{timestamp}"

    # GRPO specific training arguments
    training_args = GRPOConfig(
        max_prompt_length=1024,
        max_completion_length=6192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        learning_rate=2e-6,
        logging_steps=1,
        optim="adafactor",
        seed=42,
        bf16=True,
        weight_decay=0.05,
        lr_scheduler_type="constant",
        output_dir=output_dir,
        beta=0.04,  # KL coefficient for GRPO
        num_generations=8,  # Number of generations per prompt
        temperature=0.9)

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        reward_funcs=reward_func
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
