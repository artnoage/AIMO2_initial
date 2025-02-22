from datasets import load_dataset, load_from_disk, concatenate_datasets
import os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import logging
from unsloth import is_bfloat16_supported
from datetime import datetime
import json
def main():
    # Set training type
    logging.set_verbosity_info()

    

    # Load model from checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/Home/stat/laschos/AIMO2_initial/models/reseted/20250218_001806",
        max_seq_length=8192,
        load_in_4bit=False)
        

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
    model,
    r = 256, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens",],
    lora_alpha = 256,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = False, # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False)
    

    SYSTEM_PROMPT = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.\n\n
    <thinking>
    First, analyze the problem in depth and outline your approach.\n 
    This section should capture your reasoning, including any abstract thoughts or potential strategies.\n  
    Feel free to refine or correct your ideas as you work toward the solution.\n  
    </thinking>
    <response>\n
    <step>Step 1: Begin with the first calculation or operation\n
    Show your work clearly using LaTeX notation</step>\n\n
    <step>Step 2: Continue with the next logical step\n
    Each step should be numbered and self-contained</step>\n\n
    <step>Step N: In your final step, state your conclusion\n
    Put your final answer in \\boxed{}</step>\n
    </response>\n\n"""

    def formatting_prompts_func(examples):
        texts = []
        for example in examples:
            formatted_text = (
                "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
                "<|im_start|>user\n" + example['problem'] + "<|im_end|>\n"
                "<|im_start|>assistant\n" + example['solution'] + "<|im_end|>\n"
            )
            texts.append(formatted_text)
        return {"text": texts}


    # Load dataset
    dataset = load_from_disk("/Home/stat/laschos/AIMO2_initial/local_datasets/from_qwen/20250219_084006")
    dataset = dataset.shuffle(seed=42)  # Keep same shuffle seed for consistency
    # Apply the formatting to the dataset
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    #print("\nFirst conversation after formatting:")
    print(json.dumps(formatted_dataset[0]["text"], indent=2))
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"train_results/{timestamp}",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        logging_steps=10,  # More frequent logging
        save_strategy="steps",
        save_steps=1000,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_torch",  # Enable tensorboard logging
        logging_first_step=True,    # Log the first training step
        logging_dir=f"train_results/{timestamp}/logs",  # Directory for logs
    )

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args)

    #exit()
    # Train the model
    trainer.train()
    models_dir = "models"
    model_type = "reseted_and_from_qwen"
    os.makedirs(os.path.join(models_dir, model_type), exist_ok=True)
    
    
    model_output_dir = os.path.join(models_dir, model_type, timestamp)
    
    # Save the merged model
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {model_output_dir}")

if __name__ == "__main__":
    main()
