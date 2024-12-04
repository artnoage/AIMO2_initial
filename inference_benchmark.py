import time
import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

async def process_question(engine, question, i):
    print(f"Question {i}: {question}")
    
    # Start timing
    start_time = time.time()
    
    # Generate response using vLLM
    from vllm.sampling_params import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.7,
        top_p=0.95
    )
    
    # Format prompt with chat template
    prompt = f"[INST] {question} [/INST]"
    
    # Process the async generator
    async for response in engine.generate(prompt, sampling_params=sampling_params, request_id=f"request_{i}"):
        generated_text = response.outputs[0].text
        break  # We only need the first response
    
    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Response: {generated_text}")
    print(f"Time taken: {time_taken:.2f} seconds\n")
    
    return time_taken

async def main():
    # Initialize vLLM engine
    engine_args = AsyncEngineArgs(
        model="artnoage/metastral",
        max_model_len=4096,
        tensor_parallel_size=1,  # Adjust based on number of GPUs
        gpu_memory_utilization=0.90,
        trust_remote_code=True
    )
    
    print("Initializing vLLM engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Engine initialized!")
    
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
        process_question(engine, question, i)
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
