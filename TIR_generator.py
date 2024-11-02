import os
import re
from enum import Enum
from typing import Optional, Union
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from datetime import datetime

# Load environment variables
load_dotenv()

from utils.utils import ModelOption

SYSTEM_PROMPT = """You are a Python code generator for mathematical problems. When given a problem and its solution:
1. Analyze the problem and solution carefully
2. Explain what concepts you are going to use for getting the answer.
2. Create a Python function that solves the problem. The actuall ground truth should not be given to the code
solving the issue in a trivial manner.
3. The function should:
   - Take no arguments
   - Return a type of answer that matches the real answer. 
   . You can also use symbolic libraries. 
   - Use clear variable names and comments
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

def extract_answer_from_solution(solution: Optional[str], messages: Optional[str] = None) -> Optional[Union[int, str]]:
    """Extract answer from solution text (looking for \boxed{X})"""
    # Handle None or non-string inputs
    if not isinstance(solution, str):
        print(f"Warning: solution is not a string (type: {type(solution)})")
        solution = str(solution) if solution is not None else ""
        
    if messages is not None and not isinstance(messages, str):
        print(f"Warning: messages is not a string (type: {type(messages)})")
        messages = str(messages)
    
    # Try solution first
    try:
        # Look for any content inside \boxed{}
        match = re.search(r'\\boxed{([^}]+)}', solution)
        if match:
            answer = match.group(1).strip()
            # Try converting to int if possible
            try:
                return int(answer)
            except ValueError:
                return answer
    except Exception as e:
        print(f"Error searching solution: {e}")
    
    # If not found and messages provided, try messages
    if messages:
        try:
            match = re.search(r'\\boxed{([^}]+)}', messages)
            if match:
                answer = match.group(1).strip()
                # Try converting to int if possible
                try:
                    return int(answer)
                except ValueError:
                    return answer
        except Exception as e:
            print(f"Error searching messages: {e}")
    
    # Print solution for troubleshooting
    print("Could not find ground truth. Solution text:")
    print(solution)
    if messages:
        print("\nMessages text:")
        print(messages)
    
    return None

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract code block from LLM response"""
    match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def save_results(results: list, model_name: str):
    """Save results to JSON and Markdown files in TIR_data directory"""
    os.makedirs('TIR_data', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"TIR_results_{model_name}_{timestamp}"
    
    # Save JSON
    json_path = os.path.join('TIR_data', f"{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save Markdown
    md_path = os.path.join('TIR_data', f"{base_filename}.md")
    with open(md_path, 'w') as f:
        f.write(f"# TIR Results - {model_name}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        correct = sum(1 for r in results if r.get('is_correct'))
        total = len(results)
        f.write(f"## Summary\n")
        f.write(f"- Total problems processed: {total}\n")
        f.write(f"- Correct solutions: {correct}\n")
        f.write(f"- Accuracy: {correct/total:.2%}\n\n")
        
        # Individual results
        f.write("## Problem Solutions\n\n")
        for result in results:
            f.write(f"### Problem {result['id']}\n")
            f.write(f"**Problem:**\n{result['problem']}\n\n")
            f.write(f"**Generated Code:**\n```python\n{result['generated_code']}\n```\n\n")
            f.write(f"**Model Answer:** {result['model_answer']}\n")
            f.write(f"**Ground Truth:** {result['ground_truth']}\n")
            f.write(f"**Correct:** {result['is_correct']}\n\n")
            
    print(f"\nResults saved to:")
    print(f"- {json_path}")
    print(f"- {md_path}")

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
            ground_truth = extract_answer_from_solution(
                example['solution'],
                example.get('messages')  # Pass messages if available
            )
            if ground_truth is None:
                print(f"Could not extract answer from problem {idx}")
                ground_truth = "ERROR: Could not extract ground truth"
                
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
