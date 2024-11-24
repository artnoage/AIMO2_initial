from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
import bitsandbytes as bnb
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        dtype = "auto",
        max_seq_length=8192,
        load_in_8bit=True)  # Will use default dtype settings
        
    # Configure LoRA
    peft_config = LoraConfig(
        r=64,  # Rank
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA config to the model
    model.add_adapter(peft_config)

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "human", "assistant": "assistant"},
        map_eos_token=True,
    )
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    

    dataset = load_dataset("artnoage/sft", split="train")
    # Apply the formatting to the dataset
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    print("Raw conversations sample (index 5):")
    print(dataset[5]["conversations"])
    print("\nFormatted text sample (index 5):")
    print(formatted_dataset[5]["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./train_results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        learning_rate=5e-6,
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        optim="adamw_hf",
    )

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
