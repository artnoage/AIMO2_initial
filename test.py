import asyncio
from utils.utils import ModelOption, get_model
from langchain_core.messages import SystemMessage, HumanMessage

async def main():
    # Initialize the LOCAL model
    model = get_model(ModelOption.LOCAL)
    
    # Create a simple message
    messages = [
        HumanMessage(content="what is 2+5?")
    ]
    
    # Get response
    response = await model.ainvoke(messages)
    print("\nModel response:")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
