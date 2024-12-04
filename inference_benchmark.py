import time
import asyncio
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

async def process_question(model, tokenizer, question, i):
    print(f"Question {i}: {question}")
    
    # Start timing
    start_time = time.time()
    
    # Generate response
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt').to(model.device)
    
    generated_ids = model.generate(
       prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    response = tokenizer.batch_decode(generated_ids)[0]
    
    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Response: {response}")
    print(f"Time taken: {time_taken:.2f} seconds\n")
    
    return time_taken

async def main():
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=4096,
        load_in_4bit=False)  # Using 4-bit quantization for inference
    
    # Optimize model for inference
    model=FastLanguageModel.for_inference(model)
    
    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True)
    
    # Sample questions
    questions = [
        "What is the capital of France?",
        "Explain quantum entanglement briefly.",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
        "How does a computer CPU work?",
        "What causes the seasons on Earth?",
        "Explain the theory of relativity.",
        "What is the difference between DNA and RNA?",
        "How do vaccines work?",
        "What is machine learning?"
    ]
    
    print("\nStarting inference benchmark with 10 questions...\n")
    
    # Create tasks for all questions
    tasks = [
        process_question(model, tokenizer, question, i)
        for i, question in enumerate(questions, 1)
    ]
    
    # Run all tasks concurrently and collect times
    times = await asyncio.gather(*tasks)
    
    total_time = sum(times)
    avg_time = total_time / len(questions)
    
    print(f"Benchmark complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per question: {avg_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
