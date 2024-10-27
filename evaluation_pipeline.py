import os
import re
import json
import time
import argparse
import datetime
from typing import Optional, List
import openai
from openai import OpenAI
from datasets import load_dataset
from urllib.parse import urlparse
from tqdm import tqdm
from pydantic import BaseModel, ValidationError

# Model configurations
OPENAI_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
OPENROUTER_MODELS = ["anthropic/claude-3-opus", "anthropic/claude-3-sonnet", "google/gemini-pro","qwen/qwen-2.5-7b-instruct"]

def setup_client(provider: str, model: str):
    """Setup API client based on provider and model selection"""
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        ), model
    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key), model

# Initialize these globally as they'll be set in main()
client = None
MODEL = None

# Constants
MAX_RETRIES = 1
SLEEP_TIME = 30  # Seconds to wait between retries in case of rate limits

# Pydantic model for structured output
class FullResponse(BaseModel):
    reasoning: str
    final_answer: int

def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

OUTPUT_FILE = f"evaluation_results_{timestamp()}.jsonl"

def extract_answer(solution: str) -> Optional[int]:
    """
    Extracts the numerical answer from the solution string.
    """
    try:
        return int(solution.strip())
    except ValueError:
        # Extract the first number from the solution string
        match = re.search(r"\d+", solution)
        if match:
            return int(match.group())
        return None

def get_structured_response(prompt: str, model: str = MODEL) -> Optional[FullResponse]:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise numerical answers to math problems."},
                    {"role": "user", "content": prompt},
                ],
                response_format=FullResponse,  # Directly parse into FullResponse
                temperature=0,  # Set temperature to 0 for deterministic outputs
                timeout=20  # Set timeout to 20 seconds
            )
            message = response.choices[0].message.parsed  # Type: FullResponse
            return message  # This is an instance of FullResponse

        except openai.RateLimitError:
            print(f"Rate limit reached. Retrying in {SLEEP_TIME} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(SLEEP_TIME)
        except openai.APIError as e:
            print(f"API error occurred: {e}. Retrying in {SLEEP_TIME} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(SLEEP_TIME)
        except ValidationError as ve:
            print(f"Validation error: {ve}.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
            return None
    print("Max retries exceeded. Skipping this prompt.")
    return None

def evaluate_model(dataset):
    """
    Evaluates the model on the provided dataset.
    Saves each response with metadata and timestamp.
    """
    results = []

    for example in tqdm(dataset, desc="Evaluating"):
        problem_id = example['id']
        problem = example['problem']
        solution = example['solution']
        answer = example['answer']
        ground_truth = extract_answer(answer)

        if not ground_truth:
            print(f"[Problem ID {problem_id}] Could not extract ground truth answer. Skipping.")
            continue

        prompt = f"Solve the following problem and provide only the numerical answer:\n\n{problem}"

        structured_response = get_structured_response(prompt)

        current_timestamp = timestamp()

        if structured_response:
            model_reasoning = structured_response.reasoning
            model_answer = structured_response.final_answer
            correctness = (model_answer == ground_truth)
            if not correctness:
                print(f"[Problem ID {problem_id}] Incorrect. Model answer: {model_answer}, Ground truth: {ground_truth}")
            refusal = None
        else:
            model_answer = None
            correctness = False
            refusal = "No valid response parsed."
            print(f"[Problem ID {problem_id}] No valid response parsed.")

        result = {
            "problem_id": problem_id,
            "problem": problem,
            "ground_truth": ground_truth,
            "solution": solution,
            "model_answer": model_answer,
            "model_reasoning": model_reasoning,
            "correctness": correctness,
            "refusal": refusal,
            "timestamp": current_timestamp,
            "url": example.get('url', ''),
        }

        results.append(result)

    return results

def save_results(results, filename=OUTPUT_FILE):
    """
    Saves the evaluation results to a JSON Lines file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in results:
            json.dump(entry, f)
            f.write('\n')
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Run evaluation pipeline with selected model and provider')
    parser.add_argument('--provider', choices=['openai', 'openrouter'], default='openrouter',
                      help='API provider to use (default: openai)')
    parser.add_argument('--model', help='Model to use for evaluation')
    args = parser.parse_args()

    # Validate model selection
    if args.provider == 'openai' and args.model not in OPENAI_MODELS:
        print(f"Invalid OpenAI model. Choose from: {', '.join(OPENAI_MODELS)}")
        return
    elif args.provider == 'openrouter' and args.model not in OPENROUTER_MODELS:
        print(f"Invalid OpenRouter model. Choose from: {', '.join(OPENROUTER_MODELS)}")
        return

    # Setup global client and model
    global client, MODEL
    client, MODEL = setup_client(args.provider, args.model)

    print(f"Using {args.provider} with model: {args.model}")
    print("Loading dataset...")
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
    print(f"Loaded {len(dataset)} problems.")

    print("Starting evaluation...")
    results = evaluate_model(dataset)

    print("Saving results...")
    save_results(results)

    total_attempted = len([res for res in results if res["model_answer"] is not None])
    total_correct = len([res for res in results if res["correctness"]])
    accuracy = (total_correct / total_attempted) * 100 if total_attempted > 0 else 0
    print(f"\nEvaluation Results:\nCorrect Answers: {total_correct}\nTotal Attempted: {total_attempted}\nAccuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
