from datasets import load_dataset
import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import logging
from unsloth import is_bfloat16_supported
from datetime import datetime


def main():
    logging.set_verbosity_info()

    

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/Home/stat/laschos/AIMO2_initial/models/20250106_092733",
        max_seq_length=4096,
        load_in_4bit=False)
        

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
    model,
    r = 384, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens",],
    lora_alpha = 384,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = False, # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False)
    

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True)
    

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in convos]
        return {"text": texts}


     # Load, shuffle and format dataset
    dataset = load_dataset("Metaskepsis/sft", split="train")
    dataset = dataset.shuffle(seed=42)  # Add deterministic shuffling

    # Print original format
    print("\nFirst conversation before formatting:")
    print(json.dumps(dataset[0]["conversations"], indent=2))
    
    # Apply the formatting to the dataset
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    print("\nFirst conversation after formatting:")
    print(json.dumps(formatted_dataset[0]["text"], indent=2))
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"train_results/{timestamp}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=5e-6,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1000,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_torch",
    )

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args)

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
