from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import os
import torch
import asyncio
import json
from typing import Dict, Optional
from tqdm import tqdm
from datasets import load_dataset
from utils.utils import ModelOption, get_model, extract_answer_from_solution
from dotenv import load_dotenv
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are a precise mathematical problem solver. You will be given a problem to solve.

DO:
▪ List applicable theorems/techniques upfront
▪ If possible each step must contain a justification. 
▪ Use LaTeX notation

FORMAT:

**Problem Analysis and Approach**:
1. Start by categorizing the problem (e.g., "This is an inequality problem involving algebraic identities" or "This is a combinatorial proof").
2. List specific tools or theorems that will guide your solution (e.g., "AM-GM inequality," "Basic algebraic manipulations").

**PROOF**:
Example format for each step:
Given: \\( a, b, c > 0 \\) and \\( a + b + c = 3 \\). Prove that \\( abc \\leq 1 \\).

Step 1. By the AM-GM inequality, \\( \\frac{a + b + c}{3} \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Apply AM-GM inequality to \\( a, b, c \\)]  
Step 2. Substituting \\( a + b + c = 3 \\), we get \\( 1 \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Replace with given sum condition]  
Step 3. Cube both sides to eliminate the root: \\( 1 \\geq abc \\) \\hspace{10pt} [Cube both sides to solve for \\( abc \\)]  
Step 4. Thus, \\( abc \\leq 1 \\), as required.  

For each step, clearly state the action, use concise LaTeX notation, and provide a justification in brackets.

**ANSWER**:
\\(\\boxed{\\text{result}}\\) """

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
load_dotenv()
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f'GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB')

async def compare_math_answers(model_answer: Optional[str], correct_answer: Optional[str], problem: str, verifier_model) -> bool:
    """Use the model to compare two mathematical answers"""
    if model_answer is None or correct_answer is None:
        return False
        
    comparison_prompt = [
        {"role": "system", "content": "You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."},
        {"role": "user", "content": f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}"}
    ]
    
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await verifier_model.ainvoke(comparison_prompt)
            return response.content.strip().lower() == 'yes'
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Verification failed after {max_retries} attempts")
                return False
            print(f"Connection error during verification. Retrying... ({retry_count}/{max_retries})")
            await asyncio.sleep(1)
    return False

async def process_example(example: Dict, running_id: int, model, tokenizer, verifier_model, best_of: int = 10) -> Dict:
    """Process a single example with parallel attempts"""
    try:
        correct_answer = extract_answer_from_solution(example["solution"])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None
            
        # Create prompt once for all attempts
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "human", "content": example["problem"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Create inputs tensor for all attempts at once
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs = {k: v.repeat(best_of, 1) for k, v in inputs.items()}
        
        # Generate all solutions in parallel
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        # Process all solutions
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
        # Create verification tasks for all solutions
        current_solutions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        print(current_solutions[0])
        current_answers = [extract_answer_from_solution(sol) for sol in current_solutions]
        verification_tasks = [
            compare_math_answers(ans, correct_answer, example["problem"], verifier_model)
            for ans in current_answers
        ]
        
        # Wait for all verifications to complete
        is_correct_list = await asyncio.gather(*verification_tasks)
        
        for i, (current_solution, current_answer, is_correct) in enumerate(zip(current_solutions, current_answers, is_correct_list)):
            if is_correct:
                correct_count += 1
                if best_solution is None:
                    best_solution = current_solution
                    best_answer = current_answer
            
            solutions.append({
                'solution': current_solution,
                'answer': current_answer,
                'is_correct': is_correct,
                'verifier_response': 'yes' if is_correct else 'no'
            })
        
        return {
            'id': running_id,
            'problem': example["problem"],
            'correct_answer': correct_answer,
            'solutions': solutions,
            'best_solution': best_solution or solutions[0]['solution'],
            'best_answer': best_answer or solutions[0]['answer'],
            'success_rate': correct_count / best_of
        }
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    print("\nLoading model...")
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="artnoage/metastral",
        max_seq_length=8192,
        dtype="bfloat16",
        load_in_4bit=True)
    
    # Prepare model for inference
    model = FastLanguageModel.for_inference(model)

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "human", "assistant": "assistant"},
        map_eos_token=True,
    )

    # Load the filtered dataset
    username = os.environ.get("HF_USERNAME")
    if not username:
        raise ValueError("HF_USERNAME environment variable not set")
    dataset = load_dataset(f"{username}/Numina-Olympiads", split="train")
    dataset = dataset.shuffle(seed=42)
    examples = dataset.select(range(1000))

    # Initialize verifier model
    verifier_model = get_model(ModelOption.GEMINI_FLASH)
    
    # Process examples sequentially with progress bar
    results = []
    progress_bar = tqdm(total=len(examples), desc="Processing examples")
    
    for i, example in enumerate(examples):
        result = await process_example(example, i, model, tokenizer, verifier_model)
        if result:
            results.append(result)
            print(f"\nProcessed example {result['id']}:")
            print(f"Success rate: {result['success_rate']*100:.1f}%")
            if result['success_rate'] > 0:
                print(f"Best answer: {result['best_answer']}")
            
            # Save results after each example
            with open('inference.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        progress_bar.update(1)
    
    progress_bar.close()
    print("\nFinal GPU memory:")
    print_gpu_memory()

if __name__ == "__main__":
    asyncio.run(main())
