import os
import re
import json
import asyncio
import argparse
from enum import Enum
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from utils.augmented_data_handler import handle_augmented_data_file, save_augmented_data, get_existing_ids
from utils.utils import ModelOption, get_model
from typing import List, Dict, Optional
from benchmark_numina import compare_math_answers
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from utils.utils import extract_answer_from_solution

# Load environment variables
load_dotenv()
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

TEACHER_SYSTEM_PROMPT = """You are a mathematics teacher. When given a problem:
1. First analyze its key components and solution approach
2. Create a similar problem that uses the same concepts but with different numbers/context
3. Solve your created problem step by step, showing clear work

Format your response as:
SIMILAR PROBLEM: [your created problem]
SOLUTION: [your step-by-step solution]"""

STUDENT_SYSTEM_PROMPT = """You are a student learning to solve mathematical problems by analogy. 
You will be given:
1. An original problem to solve
2. A similar problem with its solution as demonstration

Use the demonstration to understand the solution approach, then apply similar reasoning to solve 
the original problem. Show your work step by step and provide your final answer inside \boxed{}"""

async def get_teacher_demonstration(problem: str, teacher_model) -> Dict:
    """Get a similar problem and solution from the teacher"""
    prompt = [
        SystemMessage(content=TEACHER_SYSTEM_PROMPT),
        HumanMessage(content=problem)
    ]
    
    response = await teacher_model.ainvoke(prompt)
    content = response.content
    
    # Extract similar problem and solution
    problem_match = re.search(r"SIMILAR PROBLEM:(.*?)SOLUTION:", content, re.DOTALL)
    solution_match = re.search(r"SOLUTION:(.*)", content, re.DOTALL)
    
    similar_problem = problem_match.group(1).strip() if problem_match else ""
    solution = solution_match.group(1).strip() if solution_match else ""
    
    return {
        "similar_problem": similar_problem,
        "solution": solution,
        "demonstration_answer": extract_answer_from_solution(solution)
    }

async def get_student_solutions(original_problem: str, demonstration: Dict, student_model) -> Tuple[str, str]:
    """Get student's solutions both before and after seeing the demonstration"""
    # First attempt without demonstration
    initial_prompt = [
        SystemMessage(content=STUDENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""Here is a problem to solve:
{original_problem}

Solve it step by step, showing your work.""")
    ]
    
    initial_response = await student_model.ainvoke(initial_prompt)
    initial_solution = initial_response.content
    
    # Second attempt with demonstration
    demo_prompt = [
        SystemMessage(content=STUDENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""Let's try this problem again with a helpful example.

Original Problem to Solve:
{original_problem}

Similar Problem for Reference:
{demonstration['similar_problem']}

Solution to Similar Problem:
{demonstration['solution']}

Now solve the original problem using similar reasoning.""")
    ]
    
    demo_response = await student_model.ainvoke(demo_prompt)
    demo_solution = demo_response.content
    
    return initial_solution, demo_solution

async def process_example(example: Dict, running_id: int, teacher_model, student_model, verifier_model) -> Optional[Dict]:
    """Process a single example using the teacher-student approach"""
    try:
        print(f"\nProcessing Problem {running_id + 1}")
        
        # Get correct answer
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution")
            return None
            
        # Get teacher's demonstration
        print("Getting teacher's demonstration...")
        demonstration = await get_teacher_demonstration(example['problem'], teacher_model)
        
        # Get student's solutions (both attempts)
        print("Getting student's solutions...")
        initial_solution, demo_solution = await get_student_solutions(
            example['problem'], 
            demonstration,
            student_model
        )
        
        # Extract answers
        initial_answer = extract_answer_from_solution(initial_solution)
        demo_answer = extract_answer_from_solution(demo_solution)
        
        # Use NEMOTRON to verify both answers
        initial_correct = await compare_math_answers(
            initial_answer, correct_answer, example['problem'], verifier_model
        )
        demo_correct = await compare_math_answers(
            demo_answer, correct_answer, example['problem'], verifier_model
        )
        
        # Print results
        print("\nResults:")
        print(f"Initial attempt: {'✓' if initial_correct else '✗'}")
        print(f"After demonstration: {'✓' if demo_correct else '✗'}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Initial Answer: {initial_answer}")
        print(f"Final Answer: {demo_answer}")
        if not initial_correct and demo_correct:
            print("✨ Demonstration helped!")
        print("-" * 80)
        
        return {
            'id': example['id'],
            'problem': example['problem'],
            'teacher_demonstration': demonstration,
            'initial_solution': initial_solution,
            'demo_solution': demo_solution,
            'correct_answer': correct_answer,
            'initial_answer': initial_answer,
            'demo_answer': demo_answer,
            'initial_correct': initial_correct,
            'demo_correct': demo_correct,
            'demonstration_helped': not initial_correct and demo_correct
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Solve math problems by analogy')
    parser.add_argument('--teacher', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL_ORIGINAL_Q', help='Model to use as teacher')
    parser.add_argument('--student', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL_ORIGINAL_Q', help='Model to use as student')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Model to use for verifying answers')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--max-examples', type=int, default=10,
                       help='Maximum number of examples to process')
    args = parser.parse_args()

    # Load dataset
    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina-Olympiads", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize models
    try:
        teacher_model = get_model(ModelOption[args.teacher])
        student_model = get_model(ModelOption[args.student])
        verifier_model = get_model(ModelOption[args.verifier])
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    print(f"\nSolving by analogy with:")
    print(f"Teacher: {args.teacher}")
    print(f"Student: {args.student}")
    
    # Prepare examples
    dataset = dataset.shuffle(seed=42)
    example_data = []
    for example in dataset.select(range(min(args.max_examples, len(dataset)))):
        example_data.append({
            'id': example['id'],
            'problem': example['problem'],
            'solution': example['solution']
        })

    # Process examples
    results = []
    progress_bar = tqdm(total=len(example_data), desc="Processing examples")
    
    for i, example in enumerate(example_data):
        result = await process_example(example, i, teacher_model, student_model, verifier_model)
        if result:
            results.append(result)
        progress_bar.update(1)
    
    progress_bar.close()

    # Calculate statistics
    if results:
        initial_correct = sum(1 for r in results if r['initial_correct'])
        demo_correct = sum(1 for r in results if r['demo_correct'])
        helped_count = sum(1 for r in results if r['demonstration_helped'])
        
        print("\nFinal Results:")
        print(f"Total examples processed: {len(results)}")
        print(f"Initial accuracy: {initial_correct}/{len(results)} = {(initial_correct/len(results))*100:.2f}%")
        print(f"Final accuracy: {demo_correct}/{len(results)} = {(demo_correct/len(results))*100:.2f}%")
        print(f"Demonstration helped in {helped_count} cases ({(helped_count/len(results))*100:.2f}%)")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join('results', 
                              f"analogy_results_{args.teacher}_{args.student}_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump({
                'teacher_model': args.teacher,
                'student_model': args.student,
                'initial_accuracy': (initial_correct/len(results))*100,
                'final_accuracy': (demo_correct/len(results))*100,
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    # Print timing information
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration}")
    if results:
        print(f"Average time per example: {total_duration.total_seconds() / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
