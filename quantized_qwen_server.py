from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for the model")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling parameter")

class GenerateResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float

app = FastAPI(
    title="Quantized Qwen Math Model API",
    description="API for the quantized Qwen2.5-Math-7B-Instruct model",
    version="1.0.0"
)

# Model configuration
MODEL_PATH = "./qwen-math-7b-W8A8-Dynamic-Per-Token"
MAX_MODEL_LEN = 4096

try:
    # Initialize the model using the quantized version
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_LEN,
        add_bos_token=True,  # Important for quantized models
        trust_remote_code=True
    )
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_path": MODEL_PATH,
        "max_model_length": MAX_MODEL_LEN,
        "quantization": "W8A8",
        "type": "Qwen2.5-Math-7B-Instruct"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text based on the input prompt"""
    try:
        start_time = time.time()
        
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
        
        generation_time = time.time() - start_time
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        response = GenerateResponse(
            generated_text=generated_text,
            prompt_tokens=len(output.prompt_token_ids),
            completion_tokens=len(output.outputs[0].token_ids),
            total_tokens=len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            generation_time=generation_time
        )
        
        logger.info(f"Generated response with {response.total_tokens} tokens in {generation_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
