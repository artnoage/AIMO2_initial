from datasets import load_dataset, load_from_disk, concatenate_datasets
import os
import sys
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import logging
from unsloth import is_bfloat16_supported
from datetime import datetime
import json

# Add project root to path to import from utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.agents import FULLSOLUTION_SYSTEM_PROMPT, PROGRAMMER_SYSTEM_PROMPT
def main():
    # Set training type
    logging.set_verbosity_info()

    

    # Load model from checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/Home/stat/laschos/math/AIMO2_initial/models/sft_M/20250412_170449",
        max_seq_length=8000,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth")
        

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=256,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens",],
        lora_alpha=256,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    

    # Choose between FULLSOLUTION_SYSTEM_PROMPT and PROGRAMMER_SYSTEM_PROMPT based on model_code
    def formatting_prompts_func(examples):
        problems = examples['problem']
        
        # Handle fallback logic
        raw_solutions = examples.get('solution', examples.get('model_solution', []))
        raw_model_codes = examples.get('model_code', [])

        # Get batched inputs safely
        solutions = [
            raw_solutions[i] if i < len(raw_solutions) else None
            for i in range(len(problems))
        ]
        model_codes = [
            raw_model_codes[i] if i < len(raw_model_codes) else None
            for i in range(len(problems))
        ]

        texts = []
        fullsolution_count = 0
        programmer_count = 0

        for i, (problem, solution, model_code) in enumerate(zip(problems, solutions, model_codes)):
            if not solution:
                texts.append("")  # or continue if you want to skip it
                continue

            if model_code and isinstance(model_code, str) and len(model_code.strip()) > 0:
                system_prompt = PROGRAMMER_SYSTEM_PROMPT
                programmer_count += 1
            else:
                system_prompt = FULLSOLUTION_SYSTEM_PROMPT
                fullsolution_count += 1

            formatted_text = (
                '<|im_start|>system\n' + system_prompt + '<|im_end|>\n'
                '<|im_start|>user\n' + problem + '<|im_end|>\n'
                '<|im_start|>assistant\n' + solution + '<|im_end|>'
            )
            texts.append(formatted_text)

        # Comment out print if you don't want it flooding output during batched runs
        print(f"FULL: {fullsolution_count}, PROGRAMMER: {programmer_count}, TOTAL: {len(texts)}")

        return {"text": texts}



    # Load dataset
    dataset = load_from_disk("/Home/stat/laschos/math/AIMO2_initial/local_datasets/20250412_164300")
    dataset = dataset.shuffle(seed=42)  # Keep same shuffle seed for consistency
    # Apply the formatting to the dataset
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    formatted_dataset1 = formatted_dataset.shuffle(seed=42)
    formatted_dataset2= formatted_dataset.shuffle(seed=31)
    formatted_dataset3= formatted_dataset.shuffle(seed=13)
    formatted_dataset= concatenate_datasets([formatted_dataset1,formatted_dataset2,formatted_dataset3])
    #print("\nFirst conversation after formatting:")
    print(json.dumps(formatted_dataset[0]["text"], indent=2))
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Training arguments
    training_args = UnslothTrainingArguments(
        output_dir=f"train_results/{timestamp}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="paged_adamw_8bit",
        logging_first_step=True,
        logging_dir=f"train_results/{timestamp}/logs",
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        max_grad_norm=0.1,
    )

    # Initialize SFT trainer
    trainer = UnslothTrainer(
        model=model,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args)

    #exit()
    # Train the model
    trainer.train()
    models_dir = "models"
    model_type = "sft_M"
    os.makedirs(os.path.join(models_dir, model_type), exist_ok=True)
    
    
    model_output_dir = os.path.join(models_dir, model_type, timestamp)
    
    # Save the merged model
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {model_output_dir}")

if __name__ == "__main__":
    main()
