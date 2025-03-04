import re
import os
import asyncio
import signal
import aiohttp
from openai import AsyncOpenAI
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Dict, List, Callable, Tuple, TypeVar, Any
from utils.benchmark_config import *
T = TypeVar('T')
from langchain_core.messages import HumanMessage

class TimeoutException(Exception): pass

class OpenRouterChat:
    """Chat model that makes direct requests to OpenRouter API"""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0,
        api_key: str = None
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        """Async call to OpenRouter chat completion endpoint"""
        max_tokens = kwargs.get("max_tokens", None)
        # Handle different prompt types
        if hasattr(prompt, 'content'):  # LangChain message object
            messages = [{"role": "user", "content": prompt.content}]
        elif isinstance(prompt, list):  # List of messages
            messages = [{"role": "user", "content": prompt[-1].content}] if prompt else []
        else:  # String or other
            messages = [{"role": "user", "content": str(prompt)}]
            
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Create a new session for each request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"Error from OpenRouter API: {await response.text()}")
                    
                    result = await response.json()
                    return type('Response', (), {
                        'content': result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    })()
            except Exception as e:
                print(f"Exception in OpenRouterChat.ainvoke: {str(e)}")
                raise



class CustomChat:
    """Chat model that makes requests using OpenAI client library"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        temperature: float = 0,
        api_key: str = "EMPTY"
    ):
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        """Async call to chat completion endpoint using OpenAI client"""
        max_tokens = kwargs.get("max_tokens", None)
        print(prompt)
        exit()
        try:
            completion_params = {
                "model": self.model,
                "messages": prompt,
                "temperature": self.temperature
            }
            
            if max_tokens:
                completion_params["max_tokens"] = max_tokens
                
            completion = await self.client.chat.completions.create(**completion_params)
            print(completion)
            exit()
            return type('Response', (), {
                'content': completion.choices[0].message.content
            })()
        except Exception as e:
            print(f"Exception in CustomChat.ainvoke: {str(e)}")
            raise

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def get_model(config: BenchmarkConfig, role: str = "main"):
    """
    Initialize the ChatOpenAI model based on configuration.
    For LOCAL models, it connects to a local endpoint.
    For other models, it uses the OpenRouter API.
    
    Args:
        config: The benchmark configuration
        role: The role of the model (e.g. "main", "auxiliary", etc.)
    """
    model = ModelOption[getattr(config, role)]
    
    name = model.value
    
    if role=="main":
        temp=config.main_temp
    elif role=="auxiliary":
        temp = config.auxiliary_temp
    else:
        temp=config.auxiliary2_temp

    if (model == ModelOption.LOCAL_0) or (model == ModelOption.LOCAL_1) or (model == ModelOption.LOCAL_2):
        port = {
            "main": config.main_port,
            "auxiliary": config.auxiliary_port,
            "auxiliary2": config.auxiliary2_port
        }.get(role, config.main_port)
        
        return CustomChat(
            model=name,
            temperature=temp,
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1")
    else:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        
        return OpenRouterChat(
            model=name,
            temperature=temp,
            api_key=openrouter_api_key)


def async_retry(max_retries: int = 3, timeout: int = 300):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    await asyncio.sleep(1)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    await asyncio.sleep(1)
            raise Exception(f"Failed after {max_retries} retries")
        return wrapper
    return decorator

@async_retry(max_retries=3, timeout=240)
async def get_model_response(model, prompt, max_tokens=None) -> str:
    """Get response from model with retry logic"""
    try:
        if max_tokens==None:
            response = await model.ainvoke(prompt)
        else:
            response = await model.ainvoke(prompt, max_tokens=max_tokens)
        return response.content
    except Exception as e:
        # Add small delay before retry to prevent overwhelming API
        await asyncio.sleep(0.1)
        raise
