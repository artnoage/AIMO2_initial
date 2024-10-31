import os
import re
from enum import Enum
from functools import partial
from typing import Annotated, TypedDict, Union, List, Callable
import time
from functools import wraps
from dotenv import load_dotenv
from datasets import load_dataset
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from librarian import init_conversation_md, append_to_conversation_md

def retry_with_delay(max_attempts: int = 3, delay: int = 30):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"Attempt {attempts} failed. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# Load environment variables from .env
load_dotenv()

# Define system prompts as constants
SOLVER_PROMPT_TEMPLATE = """You are a mathematical problem solver. When given a problem, solve it step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

Never ask for confirmation. Just provide your final answer as a number at the end of your 
response prefixed with 'ANSWER: '."""

JUDGE_PROMPT_TEMPLATE = """You are a mathematical solution judge. You will be given 10 different solutions to the same problem. Your task is to:
1. Review all solutions carefully
2. Identify the most common answer if there is one
3. Evaluate the reasoning in solutions that arrived at this answer
4. Make a final determination of the most likely correct answer

Provide your final answer as a number prefixed with 'FINAL_ANSWER: '."""

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

# Define state schema
class AgentState(TypedDict):
    solver_messages: Annotated[List[BaseMessage], add_messages]
    judge_messages: Annotated[List[BaseMessage], add_messages]
    solution: Annotated[Union[int, str, None], "Final numerical answer"]
    attempt_count: Annotated[int, "Counter for solver attempts"]
    md_file: Annotated[str, "Path to markdown file"]
    right_answer_among_all: Annotated[bool, "Whether correct answer appeared in any attempt"]

# Initialize the models
def get_model(model: ModelOption, temp: float = 0.1):
    if model == ModelOption.LOCAL:
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
    else:
        # OpenRouter setup
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        return ChatOpenAI(
            model=model.value,
            temperature=temp,
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

def solve(state: AgentState, model_option: ModelOption):
    """Solver agent function - runs multiple times"""
    messages = state["solver_messages"]
    attempt_count = state["attempt_count"]
    all_solutions = []
    individual_answers = []
    right_answer_among_all = False
    
    while attempt_count < 10:
        print(f"Solving attempt {attempt_count + 1}/10...")
        
        @retry_with_delay(max_attempts=3, delay=30)
        def single_attempt():
            solver = get_model(model_option, temp=0.2)
            response = solver.invoke(messages)
            return response.content
        
        try:
            solution_content = single_attempt()
            
            # Update markdown file
            append_to_conversation_md(
                state["md_file"],
                f"Solver's Solution (Attempt {attempt_count + 1})",
                solution_content,
                attempt_count,
                messages[-1].content
            )
            
            # Extract numerical answer
            answer_match = re.search(r"ANSWER:\s*(\d+)", solution_content)
            if answer_match:
                answer = int(answer_match.group(1))
                individual_answers.append(answer)
                if 'ground_truth' in state and answer == state['ground_truth']:
                    right_answer_among_all = True
            
            # Add this solution to our list with clear labeling
            all_solutions.append(f"=== Solution {attempt_count + 1} ===\n\nSolver's reasoning and steps:\n{solution_content}\n")
            attempt_count += 1
            
        except Exception as e:
            print(f"Failed attempt {attempt_count + 1}: {e}")
            continue
    
    # After 10 attempts, combine all solutions into one message with a header
    combined_solutions = "Here are all 10 solution attempts:\n\n" + "\n".join(all_solutions)
    
    return {
        "solution": combined_solutions,
        "attempt_count": attempt_count,
        "judge_messages": HumanMessage(content=combined_solutions),
        "right_answer_among_all": right_answer_among_all}

@retry_with_delay(max_attempts=3, delay=30)
def judge(state: AgentState, model_option: ModelOption):
    """Judge agent function - evaluates all solutions"""
    messages = state["judge_messages"]
    
    print("Judge evaluating solutions...")
    
    judge = get_model(model_option, temp=0)
    response = judge.invoke(messages)
    
    # Update markdown file
    append_to_conversation_md(
        state["md_file"],
        "Judge's Evaluation",
        response.content,
        state["attempt_count"],
        messages[-1].content
    )
    
    return {
        "solution": response.content
    }

def clean_answer(state: AgentState) -> AgentState:
    """Extract numerical answer from judge's evaluation"""
    solution = state["solution"]
    match = re.search(r"FINAL_ANSWER:\s*(\d+)", solution)
    if match:
        solution = int(match.group(1))
    else:
        match = re.search(r"\d+", solution)
        solution = int(match.group()) if match else None
    
    return {"solution": solution}

def decide_next_step(state: AgentState) -> str:
    """Determine if we should continue solving or move to judging"""
    if state["attempt_count"] < 10:
        return "solver"
    return "judge"

def build_graph(solver_model: ModelOption, judge_model: ModelOption):
    """Build the workflow graph"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("solver", partial(solve, model_option=solver_model))
    workflow.add_node("judge", partial(judge, model_option=judge_model))
    workflow.add_node("cleaner", clean_answer)

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
    workflow.add_edge("judge", "cleaner")
    workflow.add_edge("cleaner", END)

    return workflow

def process_problem(problem_text: str, ground_truth: int, 
                   solver_model: ModelOption,
                   judge_model: ModelOption,
                   md_file: str = None):
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
            "attempt_count": 0,
            "md_file": md_file
        }
        
        workflow = build_graph(solver_model, judge_model)
        app = workflow.compile()
        
        print("Processing problem with Monte Carlo approach...")
        final_state = app.invoke(initial_state, {"recursion_limit": 200})
        return final_state
    except Exception as e:
        print(f"Error processing problem: {e}")
        return {
            "solution": None,
            "attempt_count": 0,
            "solutions": []
        }

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run math problem solver with Monte Carlo approach')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption], 
                       default='LOCAL', help='Solver model to use')
    parser.add_argument('--judge', type=str, choices=[model.name for model in ModelOption],
                       default='GEMINI_PRO_FREE', help='Judge model to use')
    parser.add_argument('--both', type=str, choices=[model.name for model in ModelOption],
                       help='Use same model for both solver and judge')
    args = parser.parse_args()
    
    # Define models
    if args.both:
        SOLVER_MODEL = JUDGE_MODEL = ModelOption[args.both]
    else:
        SOLVER_MODEL = ModelOption[args.solver]
        JUDGE_MODEL = ModelOption[args.judge]
    
    # Load dataset
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train[:10]")
    
    print(f"\n=== Starting Monte Carlo evaluation with {SOLVER_MODEL.value} as solver and {JUDGE_MODEL.value} as judge ===")
    
    results = []
    for example in dataset:
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
        
        result = process_problem(
            problem, 
            ground_truth,
            solver_model=SOLVER_MODEL,
            judge_model=JUDGE_MODEL,
            md_file=md_file
        )
        
        # Store result
        result_entry = {
            'solver_model': SOLVER_MODEL.value,
            'judge_model': JUDGE_MODEL.value,
            'problem_id': problem_id,
            'solution': result['solution'],
            'ground_truth': ground_truth,
            'num_attempts': result['attempt_count']
        }
        results.append(result_entry)
        
        # Print immediate result for this problem
        is_correct = result_entry['solution'] == result_entry['ground_truth']
        print(f"\nProblem {problem_id} Result:")
        print(f"Model Answer: {result_entry['solution']}")
        print(f"Ground Truth: {result_entry['ground_truth']}")
        print(f"Number of Attempts: {result_entry['num_attempts']}")
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
