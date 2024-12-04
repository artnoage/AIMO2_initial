import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer, logging as transformers_logging
import logging
from unsloth.chat_templates import get_chat_template
import os

async def process_question(engine, tokenizer, question, i):
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.7,
        top_p=0.95
    )
    
    # Format messages in chat format
    messages = [{"role": "user", "content": question}]
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=True)
    
    # Process the async generator
    async for response in engine.generate(prompt, sampling_params=sampling_params, request_id=f"request_{i}"):
        final_output = response
    return final_output.outputs[0].text
    


async def main():
    
    # Initialize tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained("artnoage/metastral")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True
    )

    # Initialize vLLM engine
    engine_args = AsyncEngineArgs(
        model="artnoage/metastral",
        max_model_len=4096,
        tensor_parallel_size=1,  # Adjust based on number of GPUs
        gpu_memory_utilization=0.90,
        trust_remote_code=True)
    
    print("Initializing engine...", end="", flush=True)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(" done")
    
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
        process_question(engine, tokenizer, question, i)
        for i, question in enumerate(questions, 1)
    ]
    
    # Run all tasks concurrently and collect results
    outputs = await asyncio.gather(*tasks)
    
    # Create results markdown
    with open('benchmark_results.md', 'w') as f:
        f.write("""# Inference Results\n\n""")
        f.write("| Question | Response |\n")
        f.write("|----------|----------|\n")
        
        # Add each question's results
        for q, o in zip(questions, outputs):
            # Clean up response for markdown table
            clean_output = o.replace('\n', ' ').replace('|', '\\|')
            if len(clean_output) > 100:
                clean_output = clean_output[:97] + "..."
            f.write(f"| {q} | {clean_output} |\n")

if __name__ == "__main__":
    asyncio.run(main())
