import os
import json
import random
import asyncio
import argparse
from typing import Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from utils.utils import ModelOption, get_model
from dotenv import load_dotenv

load_dotenv()

async def verify_solution(problem: str, solution: str, verifier_model) -> bool:
    """Verify if a solution is detailed, correct and coherent"""
    prompt = [
        SystemMessage(content="You are a mathematical solution validator. Given a problem and a proposed solution, respond ONLY with 'yes' if the solution is mathematically correct, detailed and coherent, or 'no' if it contains any errors, lacks detail, or has incoherent reasoning. Just one word, no explanation."),
        HumanMessage(content=f"Problem:\n{problem}\n\nProposed solution:\n{solution}\n\nIs this solution mathematically correct, detailed and coherent?")
    ]
    
    try:
        response = await verifier_model.ainvoke(prompt)
        return response.content.strip().lower() == 'yes'
    except Exception as e:
        print(f"Error during verification: {e}")
        return False

async def process_examples(examples: List[Dict], verifier_model) -> List[Dict]:
    """Process the randomly selected examples"""
    results = []
    for i, example in enumerate(examples, 1):
        try:
            is_correct = await verify_solution(
                example['problem'],
                example['model_response'],
                verifier_model
            )
            
            result = {
                'id': example.get('id', f'example_{i}'),
                'problem': example['problem'],
                'is_correct': is_correct
            }
            results.append(result)
            
            # Print progress
            status = '✓' if is_correct else '✗'
            print(f"Example {i}/100: {status}")
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            
    return results

async def main():
    parser = argparse.ArgumentParser(description='Verify mathematical solutions')
    parser.add_argument('--verifier', type=str, 
                       choices=[model.name for model in ModelOption],
                       required=True,
                       help='Model to use for verification')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file containing problems and solutions')
    args = parser.parse_args()

    # Load and validate input file
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Input file must contain a list of examples")
            return
            
        if len(data) < 100:
            print(f"Warning: Input file contains only {len(data)} examples")
            sample_size = len(data)
        else:
            sample_size = 100
            
        # Randomly select examples
        selected_examples = random.sample(data, sample_size)
        
    except FileNotFoundError:
        print(f"Error: File {args.input} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: File {args.input} is not valid JSON")
        return
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Initialize verifier model
    try:
        verifier_model = get_model(ModelOption[args.verifier])
    except Exception as e:
        print(f"Error initializing verifier model: {e}")
        return

    print(f"\nVerifying {sample_size} randomly selected examples...")
    results = await process_examples(selected_examples, verifier_model)

    # Calculate and display statistics
    if results:
        correct_count = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_count / len(results)) * 100
        print(f"\nResults:")
        print(f"Total examples processed: {len(results)}")
        print(f"Correct solutions: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Save results
        output_filename = f"verification_results_{args.verifier}.json"
        try:
            with open(output_filename, 'w') as f:
                json.dump({
                    'verifier': args.verifier,
                    'total_examples': len(results),
                    'correct_count': correct_count,
                    'accuracy': accuracy,
                    'results': results
                }, f, indent=2)
            print(f"\nResults saved to {output_filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    asyncio.run(main())
