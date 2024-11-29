from functools import wraps
import asyncio
from typing import TypeVar, Callable, Any

T = TypeVar('T')

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
                    print(f"Timeout error. Retrying... ({retry_count}/{max_retries})")
                    await asyncio.sleep(1)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    print(f"Error: {str(e)}. Retrying... ({retry_count}/{max_retries})")
                    await asyncio.sleep(1)
            raise Exception(f"Failed after {max_retries} retries")
        return wrapper
    return decorator
