import os
import json
import random
random.seed(42)  # Fixed seed for reproducibility
import asyncio
import argparse
from typing import Dict, List
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from utils.utils import ModelOption, get_model
from dotenv import load_dotenv

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
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

async def process_examples(examples: List[Dict], verifier_model, sample_size: int) -> List[Dict]:
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
                'model_response': example['model_response'],
                'is_correct': is_correct
            }
            results.append(result)
            
            # Print minimal progress
            status = '✓' if is_correct else '✗'
            print(f"Example {i}/{sample_size}: {status}", end='\r')
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            
    return results

async def main():
    parser = argparse.ArgumentParser(description='Verify mathematical solutions')
    parser.add_argument('--verifier', type=str, 
                       choices=[model.name for model in ModelOption],
                       required=True,
                       help='Model to use for verification')
    parser.add_argument('--input', type=str, default='augmented_datasets/synthetic_augmented.json',
                       help='Input JSON file containing problems and solutions')
    parser.add_argument('--remove_incorrect', type=bool, default=False,
                       help='Remove incorrect solutions from the original dataset')
    args = parser.parse_args()

    # Load and validate input file
    try:
        if not os.path.exists(args.input):
            print(f"Error: Dataset file {args.input} not found. Please specify the correct dataset file.")
            return
            
        with open(args.input, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Dataset file must contain a list of examples")
            return
            
        if len(data) == 0:
            print("Dataset is empty. Please provide a dataset with examples to verify.")
            return
            
        sample_size = min(len(data), len(data))
        # Always select the first 10 examples for consistency
        selected_examples = data[:sample_size]
    except json.JSONDecodeError:
        print(f"Error: File {args.input} is not valid JSON")
        return
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Initialize verifier model with temperature 0
    try:
        verifier_model = get_model(ModelOption[args.verifier], temp=0)
    except Exception as e:
        print(f"Error initializing verifier models: {e}")
        return

    print(f"\nVerifying {sample_size} randomly selected examples...")
    results = await process_examples(selected_examples, verifier_model, sample_size)

    # Calculate and display statistics
    if results:
        correct_count = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_count / len(results)) * 100
        print(f"\nResults:")
        print(f"Total examples processed: {len(results)}")
        print(f"Correct solutions: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Save results
        output_filename = "verification_results.json"
        try:
            # Create or load existing results
            if os.path.exists(output_filename):
                with open(output_filename, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {"results": {}}

            # Process current results
            for result in results:
                example_id = str(result['id'])  # Ensure ID is string for consistency
                # Create new entry if it doesn't exist
                if example_id not in existing_data["results"]:
                    existing_data["results"][example_id] = {
                        "problem": result["problem"],
                        "model_response": result["model_response"],
                        "verifications": []
                    }
                
                # Add new verification result
                verification = {
                    "verifier": args.verifier,
                    "timestamp": datetime.now().isoformat(),
                    "is_correct": result["is_correct"]
                }
                existing_data["results"][example_id]["verifications"].append(verification)

            # Save updated results
            with open(output_filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
            print(f"\nResults saved to {output_filename}")
            
            # Remove incorrect solutions from original dataset if requested
            if args.remove_incorrect:
                incorrect_ids = {r['id'] for r in results if not r['is_correct']}
                if incorrect_ids:
                    filtered_data = [ex for ex in data if str(ex['id']) not in incorrect_ids]
                    with open(args.input, 'w') as f:
                        json.dump(filtered_data, f, indent=2)
                    print(f"\nRemoved {len(incorrect_ids)} incorrect solutions from {args.input}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    asyncio.run(main())
