from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import os
import torch
import asyncio
from typing import Dict
from tqdm import tqdm

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f'GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB')

async def process_example(example: Dict, running_id: int, model, tokenizer) -> Dict:
    """Process a single example with the model"""
    try:
        messages = [
            {"role": "human", "content": example["question"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'id': running_id,
            'question': example["question"],
            'response': response
        }
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    print("\nLoading model...")
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=8192,
        dtype="bfloat16",
        load_in_4bit=True)
    
    # Prepare model for inference
    model = FastLanguageModel.for_inference(model)
    print_gpu_memory()

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "human", "assistant": "assistant"},
        map_eos_token=True,
    )

    # Test examples
    examples = [
        {"question": "What is the capital of France?"},
        {"question": "What is 2+2?"},
        {"question": "Who wrote Romeo and Juliet?"},
        {"question": "What is the speed of light?"},
             {"question": "What is the capital of France?"},
        {"question": "What is 2+2?"},
        {"question": "Who wrote Romeo and Juliet?"},
        {"question": "What is the speed of light?"},
             {"question": "What is the capital of France?"},
        {"question": "What is 2+2?"},
        {"question": "Who wrote Romeo and Juliet?"},
        {"question": "What is the speed of light?"},
             {"question": "What is the capital of France?"},
        {"question": "What is 2+2?"},
        {"question": "Who wrote Romeo and Juliet?"},
        {"question": "What is the speed of light?"},
             {"question": "What is the capital of France?"},
        {"question": "What is 2+2?"},
        {"question": "Who wrote Romeo and Juliet?"},
        {"question": "What is the speed of light?"},
             {"question": "What is the capital of France?"},
        {"question": "What is 2+2?"},
        {"question": "Who wrote Romeo and Juliet?"},
        {"question": "What is the speed of light?"},
             {"question": "What is the capital of France?"},
        {"question": "What is 2+2?"},
        {"question": "Who wrote Romeo and Juliet?"},
        {"question": "What is the speed of light?"},
    ]

    # Create a semaphore to limit concurrency
    max_concurrent = 32  # Adjust based on your GPU memory
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, model, tokenizer)

    # Create tasks for all examples
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(examples)]
    
    # Process all examples with progress bar
    results = []
    progress_bar = tqdm(total=len(examples), desc="Processing examples")
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            print(f"\nProcessed example {result['id']}:")
            print(f"Q: {result['question']}")
            print(f"A: {result['response']}\n")
        progress_bar.update(1)
    
    progress_bar.close()
    print("\nFinal GPU memory:")
    print_gpu_memory()

if __name__ == "__main__":
    asyncio.run(main())
