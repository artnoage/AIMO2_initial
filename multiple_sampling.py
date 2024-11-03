import re
import os
import asyncio
import argparse
from functools import partial
from typing import Annotated, TypedDict, Union, List, Dict, Optional
from utils.utils import extract_answer_from_solution
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from utils.librarian import init_conversation_md, append_to_conversation_md
import tiktoken
from utils.utils import ModelOption, get_model


# Load environment variables from .env
load_dotenv()
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Define system prompts as constants
SOLVER_PROMPT_TEMPLATE = """You are a mathematical problem solver. When given a problem, solve it step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

In the end provide your final answer inside \boxed{}"""

JUDGE_PROMPT_TEMPLATE = """You are a mathematical solution judge. You will be given multiple different solutions to the same problem. Your task is to:
1. Review all solutions carefully
2. Identify the most common answer if there is one
3. Evaluate the reasoning in solutions that arrived at this answer
4. Make a final determination of the most likely correct answer

Provide your final answer inside \boxed{}"""


# Define state schema
class AgentState(TypedDict):
    solver_messages: Annotated[List[BaseMessage], add_messages]
    judge_messages: Annotated[List[BaseMessage], add_messages]
    solution: Annotated[Union[int, str, None], "Final numerical answer"]
    md_file: Annotated[str, "Path to markdown file"]
    right_answer_among_all: Annotated[bool, "Whether correct answer appeared in any attempt"]


async def solve(state: AgentState, model_option: ModelOption, num_samples: int = 20) -> Dict:
    """Solver agent function - generates multiple solution attempts"""
    messages = state["solver_messages"]
    all_solutions = []
    individual_answers = []
    right_answer_among_all = False
    
    for attempt in range(num_samples):
        print(f"Solving attempt {attempt + 1}/{num_samples}...")
        
        solver = get_model(model_option, temp=0.2)
        response = await solver.ainvoke(messages)
        solution_content = response.content
        
        append_to_conversation_md(
            state["md_file"],
            f"Solver's Solution (Attempt {attempt + 1})",
            solution_content,
            attempt,
            messages[-1].content
        )
        
        answer = extract_answer_from_solution(solution_content)
        if answer:
            individual_answers.append(answer)
            if 'ground_truth' in state and answer == state['ground_truth']:
                right_answer_among_all = True
        
        all_solutions.append(f"=== Solution {attempt + 1} ===\n\nSolver's reasoning and steps:\n{solution_content}\n")
    
    # After 10 attempts, combine all solutions into one message with a header
    combined_solutions = "Here are all 10 solution attempts:\n\n" + "\n".join(all_solutions)
    
    return {
        "solution": combined_solutions,
        "judge_messages": HumanMessage(content=combined_solutions),
        "right_answer_among_all": right_answer_among_all}

async def judge(state: AgentState, model_option: ModelOption) -> Dict:
    """Judge agent function - evaluates all solutions"""
    messages = state["judge_messages"]
    print("Judge evaluating solutions...")
    
    judge = get_model(model_option, temp=0)
    response = await judge.ainvoke(messages)
    
    append_to_conversation_md(
        state["md_file"],
        "Judge's Evaluation",
        response.content,
        state["attempt_count"],
        messages[-1].content
    )
    
    solution = response.content
    answer = extract_answer_from_solution(solution)
    return {"solution": answer}

async def verify(state: AgentState, model_option: ModelOption) -> Dict:
    """Verify the judge's solution against the ground truth"""
    solution = state["solution"]
    ground_truth = state.get("ground_truth")
    
    if not solution or not ground_truth:
        return {"solution": None, "is_correct": False}
        
    verifier = get_model(model_option, temp=0)
    verification_prompt = [
        SystemMessage(content="You are a mathematical solution verifier. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
        HumanMessage(content=f"Problem:\n{state['solver_messages'][-1].content}\n\nAre these two answers equivalent?\nAnswer 1: {solution}\nAnswer 2: {ground_truth}")
    ]
    
    response = await verifier.ainvoke(verification_prompt)
    is_correct = response.content.strip().lower() == 'yes'
    return {"solution": solution, "is_correct": is_correct}

def decide_next_step(state: AgentState) -> str:
    """Determine if we should continue solving or move to judging"""
    return "judge"

def build_graph(solver_model: ModelOption, judge_model: ModelOption, verifier_model: ModelOption, num_samples: int):
    """Build the workflow graph"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("solver", partial(solve, model_option=solver_model, num_samples=num_samples))
    workflow.add_node("judge", partial(judge, model_option=judge_model))
    workflow.add_node("verifier", partial(verify, model_option=verifier_model))

    # Add edges
    workflow.set_entry_point("solver")
    workflow.add_conditional_edges(
        "solver",
        decide_next_step,
        {
            "solver": "solver",
            "judge": "judge"
        }
    )
    workflow.add_edge("judge", "verifier")
    workflow.add_edge("verifier", END)

    return workflow

async def process_problem(problem_text: str, ground_truth: int, 
                   solver_model: ModelOption,
                   judge_model: ModelOption,
                   verifier_model: ModelOption,
                   md_file: str = None,
                   num_samples: int = 2):
    """Process a single problem through the graph"""
    try:
        # Create initial messages
        initial_state = {
            "solver_messages": [
                SystemMessage(content=SOLVER_PROMPT_TEMPLATE),
                HumanMessage(content=f"Here is the problem: {problem_text}")
            ],
            "judge_messages": [SystemMessage(content=JUDGE_PROMPT_TEMPLATE),HumanMessage(content=f"Here is the problem: {problem_text}"),AIMessage(content="Please provide solutions")],
            "solution": "",
            "md_file": md_file,
            "ground_truth": ground_truth,
            "right_answer_among_all": False
        }
        
        workflow = build_graph(solver_model, judge_model, verifier_model, num_samples)
        app = workflow.compile()
        
        print("Processing problem with Monte Carlo approach...")
        final_state = await app.ainvoke(initial_state, {"recursion_limit": 200})
        return final_state
    except Exception as e:
        print(f"Error processing problem: {e}")
        return {
            "solution": None,
            "attempt_count": 0,
            "solutions": [],
            "right_answer_among_all": False
        }

async def main():
    parser = argparse.ArgumentParser(description='Run math problem solver with Monte Carlo approach')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption], 
                       default='NEMOTRON', help='Solver model to use')
    parser.add_argument('--judge', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Judge model to use')
    parser.add_argument('--both', type=str, choices=[model.name for model in ModelOption],
                       help='Use same model for solver, judge, and verifier')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Verifier model to use')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--max-examples', type=int, default=4,
                       help='Maximum number of examples to process')
    parser.add_argument('--max-concurrent', type=int, default=4,
                       help='Maximum number of concurrent problems (default: 4)')
    parser.add_argument('--samples', type=int, default=2,
                       help='Number of samples to generate per problem (default: 20)')
    args = parser.parse_args()
    
    # Define models
    if args.both:
        SOLVER_MODEL = JUDGE_MODEL = VERIFIER_MODEL = ModelOption[args.both]
    else:
        SOLVER_MODEL = ModelOption[args.solver]
        JUDGE_MODEL = ModelOption[args.judge]
        VERIFIER_MODEL = ModelOption[args.verifier]
    
    # Load dataset
    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina-Olympiads", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # Shuffle and limit examples
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    
    print(f"\n=== Starting Monte Carlo evaluation with {SOLVER_MODEL.value} as solver and {JUDGE_MODEL.value} as judge ===")
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(example):
        async with semaphore:
            problem_id = example['id']
            problem = example['problem']
            print(f"\nProcessing problem {problem_id}...")
            
            ground_truth = int(example['answer']) if example['answer'].isdigit() else None
            
            # Initialize conversation file
            md_file = init_conversation_md(
                problem_id=str(problem_id),
                problem=problem,
                solution=example['solution'],
                solver_model_name=f"{SOLVER_MODEL.name}_MC",
                suffix="",
                directory=f"monte_carlo/{SOLVER_MODEL.name.lower()}"
            )
            
            return await process_problem(
                problem, 
                ground_truth,
                solver_model=SOLVER_MODEL,
                judge_model=JUDGE_MODEL,
                verifier_model=VERIFIER_MODEL,
                md_file=md_file,
                num_samples=args.samples
            )

    # Process examples concurrently
    results = []
    progress_bar = tqdm(total=len(dataset), desc="Processing examples")
    
    # Create tasks for all examples
    tasks = [process_with_semaphore(ex) for ex in dataset]
    
    # Process all examples with progress bar
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            # Store result
            result_entry = {
                'solver_model': SOLVER_MODEL.value,
                'judge_model': JUDGE_MODEL.value,
                'problem_id': result.get('id'),
                'solution': result.get('solution'),
                'ground_truth': result.get('ground_truth'),
                'right_answer_among_all': result.get('right_answer_among_all', False)
            }
        results.append(result_entry)
        
        # Print immediate result for this problem
        is_correct = result_entry['solution'] == result_entry['ground_truth']
        print(f"\nProblem {result_entry['problem_id']} Result:")
        print(f"Model Answer: {result_entry['solution']}")
        print(f"Ground Truth: {result_entry['ground_truth']}")
        print(f"Correct: {is_correct}")
        
        # Print running accuracy
        correct_so_far = sum(1 for r in results if r['solution'] == r['ground_truth'])
        print(f"Running Accuracy: {correct_so_far}/{len(results)} = {correct_so_far/len(results):.2%}")
        print(f"Correct answer was among solutions: {result_entry.get('right_answer_among_all', False)}")
        print("-" * 50)
    
    # Print final summary
    print("\nResults Summary:")
    correct_count = 0
    for result in results:
        print(f"\nProblem {result['problem_id']}:")
        print(f"Model Answer: {result['solution']}")
        print(f"Ground Truth: {result['ground_truth']}")
        is_correct = result['solution'] == result['ground_truth']
        print(f"Correct: {is_correct}")
        if is_correct:
            correct_count += 1
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")
    
    # Analysis of judge performance
    correct_in_solutions = sum(1 for r in results if r.get('right_answer_among_all', False))
    if correct_in_solutions > correct_count:
        missed_opportunities = correct_in_solutions - correct_count
        print(f"\nJudge Performance Analysis:")
        print(f"Problems where correct answer was among solutions: {correct_in_solutions}")
        print(f"Times judge missed picking correct answer: {missed_opportunities}")
        print(f"Judge accuracy when correct answer was available: {correct_count}/{correct_in_solutions} = {correct_count/correct_in_solutions:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
