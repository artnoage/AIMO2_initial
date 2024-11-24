from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
import os
import torch
import GPUtil
from transformers import logging
from unsloth import is_bfloat16_supported

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_gpu_utilization():
    GPUs = GPUtil.getGPUs()
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
    
    # Configure 8-bit quantization
    compute_dtype = "float16" if not is_bfloat16_supported() else "bfloat16"
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=compute_dtype,
        bnb_8bit_quant_type="fp8",  # fp8 is more stable for training
        bnb_8bit_use_double_quant=False,  # Disable double quantization for training stability
        llm_int8_skip_modules=None,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_8bit_use_fp8_qdq=True,
    )

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=8192,
        quantization_config=quantization_config)
        
    print("\n=== After Model Load ===")
    print_gpu_utilization()
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = False, # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False)
    
    print("\n=== After LoRA Configuration ===")
    print_gpu_utilization()
    


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
    
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./train_results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        logging_steps=1,
        save_strategy="steps",
        save_steps=200,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
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
