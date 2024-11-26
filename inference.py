from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import os
import torch
import asyncio
from typing import Dict, Optional
from tqdm import tqdm
from datasets import load_dataset
from utils.utils import ModelOption, get_model, extract_answer_from_solution

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f'GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB')

async def compare_math_answers(model_answer: Optional[str], correct_answer: Optional[str], problem: str, verifier_model) -> bool:
    """Use the model to compare two mathematical answers"""
    if model_answer is None or correct_answer is None:
        return False
        
    comparison_prompt = [
        {"role": "system", "content": "You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."},
        {"role": "user", "content": f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}"}
    ]
    
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await verifier_model.ainvoke(comparison_prompt)
            return response.content.strip().lower() == 'yes'
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Verification failed after {max_retries} attempts")
                return False
            print(f"Connection error during verification. Retrying... ({retry_count}/{max_retries})")
            await asyncio.sleep(1)
    return False

async def process_example(example: Dict, running_id: int, model, tokenizer, verifier_model, best_of: int = 10) -> Dict:
    """Process a single example with multiple attempts"""
    try:
        correct_answer = extract_answer_from_solution(example["solution"])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None
            
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
        for attempt in range(best_of):
            messages = [
                {"role": "human", "content": example["problem"]}
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
            current_solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
            current_answer = extract_answer_from_solution(current_solution)
            
            is_correct = await compare_math_answers(current_answer, correct_answer, example["problem"], verifier_model)
            
            if is_correct:
                correct_count += 1
                if best_solution is None:
                    best_solution = current_solution
                    best_answer = current_answer
            
            solutions.append({
                'solution': current_solution,
                'answer': current_answer,
                'is_correct': is_correct
            })
        
        return {
            'id': running_id,
            'problem': example["problem"],
            'correct_answer': correct_answer,
            'solutions': solutions,
            'best_solution': best_solution or solutions[0]['solution'],
            'best_answer': best_answer or solutions[0]['answer'],
            'success_rate': correct_count / best_of
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

    # Load the filtered dataset
    username = os.environ.get("HF_USERNAME")
    if not username:
        raise ValueError("HF_USERNAME environment variable not set")
    dataset = load_dataset(f"{username}/Numina-Olympiads", split="train")
    dataset = dataset.shuffle(seed=42)
    examples = dataset.select(range(1000))

    # Create a semaphore to limit concurrency
    max_concurrent = 32  # Adjust based on your GPU memory
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, model, tokenizer)

    # Initialize verifier model
    verifier_model = get_model(ModelOption.GEMINI_FLASH)
    
    # Create tasks for all examples
    tasks = [process_with_semaphore(ex, i, model, tokenizer, verifier_model) for i, ex in enumerate(examples)]
    
    # Process all examples with progress bar
    results = []
    progress_bar = tqdm(total=len(examples), desc="Processing examples")
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            print(f"\nProcessed example {result['id']}:")
            print(f"Success rate: {result['success_rate']*100:.1f}%")
            if result['success_rate'] > 0:
                print(f"Best answer: {result['best_answer']}")
        progress_bar.update(1)
    
    progress_bar.close()
    print("\nFinal GPU memory:")
    print_gpu_memory()

if __name__ == "__main__":
    asyncio.run(main())
