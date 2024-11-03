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
4. Provide your final answer inside \boxed{}

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

async def get_student_solution(original_problem: str, demonstration: Dict, student_model) -> str:
    """Get student's solution using the demonstration"""
    prompt = [
        SystemMessage(content=STUDENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""Original Problem to Solve:
{original_problem}

Similar Problem for Reference:
{demonstration['similar_problem']}

Solution to Similar Problem:
{demonstration['solution']}

Now solve the original problem using similar reasoning.""")
    ]
    
    response = await student_model.ainvoke(prompt)
    return response.content

async def process_example(example: Dict, running_id: int, teacher_model, student_model) -> Optional[Dict]:
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
        
        # Get student's solution
        print("Getting student's solution...")
        student_solution = await get_student_solution(
            example['problem'], 
            demonstration,
            student_model
        )
        
        # Extract student's answer
        student_answer = extract_answer_from_solution(student_solution)
        
        # Determine correctness
        is_correct = student_answer == correct_answer
        
        # Print results
        status = '✓' if is_correct else '✗'
        print(f"\nResult: {status}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Student's Answer: {student_answer}")
        print("-" * 80)
        
        return {
            'id': example['id'],
            'problem': example['problem'],
            'teacher_demonstration': demonstration,
            'student_solution': student_solution,
            'correct_answer': correct_answer,
            'student_answer': student_answer,
            'is_correct': is_correct
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Solve math problems by analogy')
    parser.add_argument('--teacher', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Model to use as teacher')
    parser.add_argument('--student', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL', help='Model to use as student')
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
        result = await process_example(example, i, teacher_model, student_model)
        if result:
            results.append(result)
        progress_bar.update(1)
    
    progress_bar.close()

    # Calculate statistics
    if results:
        correct_count = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_count / len(results)) * 100
        
        print("\nFinal Results:")
        print(f"Total examples processed: {len(results)}")
        print(f"Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join('results', 
                              f"analogy_results_{args.teacher}_{args.student}_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump({
                'teacher_model': args.teacher,
                'student_model': args.student,
                'accuracy': accuracy,
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
