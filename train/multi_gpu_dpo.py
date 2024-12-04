import torch
from datasets import load_dataset
from datetime import datetime
from trl import DPOTrainer, DPOConfig
from transformers import logging, AutoTokenizer
import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

async def run_inference():
    """Run inference on GPU 6 before training"""
    print("\nStarting inference on GPU 6...")
    
    # Initialize vLLM engine for inference
    engine_args = AsyncEngineArgs(
        model="artnoage/metastral",
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        device="cuda:6",
        trust_remote_code=True
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained("artnoage/metastral")
    
    # Sample question for testing
    question = "What is the capital of France?"
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.7,
        top_p=0.95
    )
    
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    async for response in engine.generate(prompt, sampling_params=sampling_params):
        final_output = response
    
    print(f"Inference output: {final_output.outputs[0].text}\n")
    
    # Explicitly delete engine and clear CUDA cache
    del engine
    torch.cuda.empty_cache()

def run_training():
    """Run DPO training on GPU 5"""
    print("Starting DPO training on GPU 5...")
    
    # Set CUDA device
    torch.cuda.set_device(5)
    
    logging.set_verbosity_info()
    
    # Now we can safely import unsloth after setting GPU 5
    from unsloth import FastLanguageModel, PatchDPOTrainer
    from unsloth.chat_templates import get_chat_template
    PatchDPOTrainer()
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=4096,
        load_in_4bit=False
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "lm_head", "embed_tokens",],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True
    )

    def formatting_func(examples):
        formatted = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
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

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{timestamp}"
    
    training_args = DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        learning_rate=4e-6,
        logging_steps=1,
        optim="adamw_torch",
        seed=42,
        bf16=True,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        output_dir=output_dir
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        max_length=4096,
        max_prompt_length=1024
    )

    trainer.train()

async def main():
    # First run inference on GPU 6
    await run_inference()
    
    # Then run training on GPU 5
    run_training()

if __name__ == "__main__":
    asyncio.run(main())
