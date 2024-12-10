import asyncio
from bench_utils.benchmark_utils import CustomChat

async def main():
    # Initialize CustomChat with the local model
    chat = CustomChat(
        base_url="http://localhost:8000/v1",
        model="artnoage/metastral",
        temperature=0.7,
        api_key="EMPTY"
    )
    
    # Test prompt
    prompt = "What is 2+2? Answer in one word."
    
    try:
        # Make the request
        print(f"Sending prompt: {prompt}")
        response = await chat.ainvoke(prompt)
        print(f"\nResponse content: {response.content}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
