from openai import OpenAI

def test_math_problem():
    """Test solving a complex math problem"""
    # Configure the client to use local vLLM server
    client = OpenAI(
        api_key="EMPTY",  # Not needed for local server
        base_url="http://localhost:8000/v1"
    )
    
    problem = """Find the number of ordered pairs of integers (a, b) such that the sequence
[3, 4, 5, a, b, 30, 40, 50] is strictly increasing and no set of four 
(not necessarily consecutive) terms forms an arithmetic progression."""
    
    system_prompt = """You are a mathematical problem solver. Analyze the problem step by step:
1. Understand the given conditions
2. Break down the problem into smaller parts
3. Consider all possible cases
4. Calculate systematically
5. Verify the answer matches known solutions
Provide your final answer as a single number."""

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ],
        temperature=0,
        max_tokens=1000
    )
    
    print("\nMath Problem Test:")
    print(f"Problem: {problem}")
    print("\nSolution attempt:")
    print(f"{chat_response.choices[0].message.content}")
    
    # Try to extract the numerical answer
    response_text = chat_response.choices[0].message.content
    try:
        # Look for numbers in the response, focusing on the last one as it's likely the final answer
        import re
        numbers = re.findall(r'\b\d+\b', response_text)
        if numbers:
            final_answer = int(numbers[-1])
            print(f"\nExtracted answer: {final_answer}")
            print(f"Expected answer (from conversation): 228")
            print(f"Match: {'Yes' if final_answer == 228 else 'No'}")
    except Exception as e:
        print(f"Could not extract numerical answer: {e}")

if __name__ == "__main__":
    print("Testing vLLM server...")
    test_math_problem()
