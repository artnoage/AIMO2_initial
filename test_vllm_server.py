from openai import OpenAI

# Configure the client to use local vLLM server
client = OpenAI(
    api_key="EMPTY",  # Not needed for local server
    base_url="http://localhost:8000/v1"
)

def test_completion():
    """Test basic completion endpoint"""
    completion = client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        prompt="San Francisco is a",
        max_tokens=7,
        temperature=0
    )
    print("\nCompletion test:")
    print(f"Prompt: 'San Francisco is a'")
    print(f"Response: {completion.choices[0].text}")

def test_chat():
    """Test chat completion endpoint"""
    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke about programming."}
        ]
    )
    print("\nChat test:")
    print(f"Response: {chat_response.choices[0].message.content}")

if __name__ == "__main__":
    print("Testing vLLM server...")
    test_completion()
    test_chat()
