from vllm import LLM, SamplingParams
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List, Optional

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

app = FastAPI()

# Initialize the model with INT8 quantization
llm = LLM(
    model="Qwen/Qwen2.5-Math-7B-Instruct",
    quantization="awq",  # Enable INT8 quantization
    gpu_memory_utilization=0.90,
    max_model_len=4096,
)

@app.post("/generate")
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )
    
    outputs = await llm.generate_async(
        prompts=[request.prompt],
        sampling_params=sampling_params
    )
    
    generated_text = outputs[0].outputs[0].text
    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
