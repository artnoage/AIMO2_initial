import os
import re
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class ModelOption(Enum):
    CLAUDE = "anthropic/claude-3.5-sonnet:beta"
    GEMINI_PRO_FREE = "google/gemini-pro-1.5-exp"
    GEMINI_FLASH_FREE="google/gemini-flash-1.5-exp"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GEMINI_FLASH="google/gemini-flash-1.5"
    GPT = "openai/gpt-4o"
    GPT_MINI="openai/gpt-4o-mini"
    MASTER = "openai/o1-preview-2024-09-12"
    LOCAL = "mistralai/Mathstral-7B-v0.1"
    GROQ = "llama-3.1-70b-versatile"
    NOUS ="nousresearch/hermes-3-llama-3.1-405b:free"

SYSTEM_PROMPT = """You are a Python code generator for mathematical problems. When given a problem and its solution:
1. Analyze the problem and solution carefully
2. Create a Python function that solves the problem
3. The function should:
   - Take no arguments
   - Return a single integer as the answer
   - Use clear variable names and comments
   - Include docstring explaining the approach
4. Format your response as:
   ```python
   def solve():
       '''Your docstring here'''
       # Your code here
       return answer
   ```
Only return the code block, nothing else."""

def get_model(model: ModelOption, temp: float = 0.1):
    """Initialize the model with OpenRouter"""
    if model == ModelOption.LOCAL:
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
    else:
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

def extract_answer_from_solution(solution: str) -> Optional[int]:
    """Extract answer from solution text (looking for \boxed{X})"""
    match = re.search(r'\\boxed{(\d+)}', solution)
    if match:
        return int(match.group(1))
    return None

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract code block from LLM response"""
    match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def save_results(results: list, model_name: str):
    """Save results to a JSON file in benchmark_results directory"""
    os.makedirs('benchmark_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join('benchmark_results', f"TIR_results_{model_name}_{timestamp}.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Python implementations from math problems')
    parser.add_argument('--model', type=str, choices=[model.name for model in ModelOption],
                       default='NOUS', help='Model to use')
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    
    # Filter for olympiad problems
    olympiad_problems = [ex for ex in dataset if ex['source'] == 'olympiads'][:100]
    
    print(f"\nFound {len(olympiad_problems)} olympiad problems")
    
    # Initialize model
    model = get_model(ModelOption[args.model])
    
    results = []
    correct_count = 0
    
    for idx, example in enumerate(olympiad_problems):
        print(f"\nProcessing problem {idx + 1}/100...")
        
        try:
            # Extract ground truth
            ground_truth = extract_answer_from_solution(example['solution'])
            if ground_truth is None:
                print(f"Could not extract answer from problem {idx}, skipping...")
                continue
                
            # Generate code solution
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Problem: {example['problem']}\nSolution: {example['solution']}")
            ]
            
            response = model.invoke(messages)
            code = extract_code_from_response(response.content)
            
            if code is None:
                print("Could not extract code from response")
                continue
                
            # Execute generated code
            try:
                # Create a temporary namespace
                namespace = {}
                exec(code, namespace)
                
                # Call the solve function
                if 'solve' not in namespace:
                    print("No solve() function found in generated code")
                    continue
                    
                model_answer = namespace['solve']()
                
                # Check result
                is_correct = model_answer == ground_truth
                if is_correct:
                    correct_count += 1
                
                result = {
                    'id': idx,
                    'problem': example['problem'],
                    'solution': example['solution'],
                    'generated_code': code,
                    'model_answer': model_answer,
                    'ground_truth': ground_truth,
                    'is_correct': is_correct
                }
                results.append(result)
                
                # Print progress
                print(f"Model Answer: {model_answer}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Correct: {is_correct}")
                print(f"Running Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")
                
            except Exception as e:
                print(f"Error executing generated code: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
    
    # Print final results
    print("\nFinal Results:")
    print(f"Total examples processed: {len(results)}")
    if len(results) > 0:
        print(f"Final Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")
    
    # Save results
    save_results(results, args.model)

if __name__ == "__main__":
    main()
