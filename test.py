import asyncio
from utils.utils import ModelOption, get_model
from langchain_core.messages import SystemMessage, HumanMessage

async def main():
    try:
        # Initialize the LOCAL model
        model = get_model(ModelOption.LOCAL)
        
        # Create messages with system prompt to control behavior
        messages = [
            SystemMessage(content="You are a helpful assistant. Provide direct, concise answers without repeating the conversation history."),
            HumanMessage(content="what is 2+5?")
        ]
        
        # Get response with timeout
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=10.0)
        print("\nModel response:")
        print(response.content)
        
    except asyncio.TimeoutError:
        print("\nError: Model response timed out after 10 seconds")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
