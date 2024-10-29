from vllm import LLM, SamplingParams

def test_completion():
    """Test basic text completion using vLLM directly"""
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=7
    )
    
    prompt = "San Francisco is a"
    outputs = llm.generate([prompt], sampling_params)
    
    print("\nCompletion test:")
    print(f"Prompt: '{prompt}'")
    print(f"Response: {outputs[0].outputs[0].text}")

def test_chat():
    """Test chat completion using vLLM directly"""
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100
    )
    
    # Format chat messages into a prompt string
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about programming."}
    ]
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    outputs = llm.generate([prompt], sampling_params)
    
    print("\nChat test:")
    print(f"Response: {outputs[0].outputs[0].text}")

if __name__ == "__main__":
    print("Testing vLLM server...")
    test_completion()
    test_chat()
