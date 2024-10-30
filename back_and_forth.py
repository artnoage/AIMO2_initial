import os
import time
from functools import partial, wraps
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
from typing import Annotated, TypedDict, Union, List
from datasets import load_dataset
from langgraph.graph.message import add_messages
import re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from enum import Enum

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1,
    exponential_base: float = 2,
    error_types: tuple = (Exception,)
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    if i == max_retries - 1:  # Last attempt
                        raise  # Re-raise the last exception
                    print(f"Attempt {i + 1} failed with error: {str(e)}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= exponential_base  # Exponential backoff
            return func(*args, **kwargs)  # Final attempt
        return wrapper
    return decorator

class ModelOption(Enum):
    CLAUDE = "anthropic/claude-3.5-sonnet:beta"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GPT = "openai/gpt-4"
    MASTER = "openai/o1-preview-2024-09-12"
    LOCAL = "Qwen/Qwen2.5-Math-7B-Instruct"

# Define state schema
class AgentState(TypedDict):
    solver_messages: Annotated[List[BaseMessage], add_messages]
    verifier_messages: Annotated[List[BaseMessage], add_messages]
    current_solution: Annotated[str, "Current solution being worked on"]
    final_answer: Annotated[Union[int, None], "Final numerical answer"]
    iteration_count: Annotated[int, "Counter for solver-verifier iterations"]
    is_the_answer_correct: Annotated[bool, "Flag indicating if the current answer is correct"]
    md_file: Annotated[str, "Path to markdown file for logging"]
    problem: Annotated[str, "Original problem text"]

# Initialize the models
def get_model(model: ModelOption, temp: float = 0):
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


def create_solver(problem: str, model_option: ModelOption):
    model = get_model(model_option, temp=0.1)
    return model

def create_verifier(problem: str, model_option: ModelOption):
    model = get_model(model_option, temp=0.1)
    return model

def solve(state: AgentState, solver):
    """Solver agent function"""
    messages_text = "\n".join([msg.content for msg in state["solver_messages"]])
    print("Solving...")
    messages = state["solver_messages"]
    
    response = solver.invoke(messages)
    solution_content = response.content
    
    ai_message = AIMessage(content=solution_content)
    human_message = HumanMessage(content=solution_content)
    
    # Update markdown file
    append_to_conversation_md(
        state["md_file"], 
        "Solver's Solution",
        solution_content,
        state["iteration_count"],
        messages_text,
        state["problem"]
    )
    
    return {
        "current_solution": solution_content,
        "solver_messages": [ai_message],
        "verifier_messages": [human_message]
    }

def verify(state: AgentState, verifier):
    """Verifier agent function"""
    messages_text = "\n".join([msg.content for msg in state["verifier_messages"]])
    print("Verifying...")
    response = verifier.invoke(state["verifier_messages"])
    
    ai_message = AIMessage(content=response.content)
    human_message = HumanMessage(content=response.content)
    
    # Update markdown file
    append_to_conversation_md(
        state["md_file"],
        "Verifier's Response",
        response.content,
        state["iteration_count"],
        messages_text,
        state["problem"]
    )
    
    return {
        "solver_messages": [human_message],
        "verifier_messages": [ai_message],
        "iteration_count": state["iteration_count"] + 1
    }

def clean_answer(state: AgentState) -> AgentState:
    """Extract numerical answer from solver's solution"""
    solution = state["current_solution"]
    match = re.search(r"ANSWER:\s*(\d+)", solution)
    if match:
        final_answer = int(match.group(1))
    else:
        match = re.search(r"\d+", solution)
        final_answer = int(match.group()) if match else None
    
    return {"final_answer": final_answer}

def decide_next_step(state: AgentState, ground_truth: int) -> str:
    """Determine if we should continue verification or end"""
    state["is_the_answer_correct"] = (state["final_answer"] == ground_truth)
    
    # End if answer is correct or we've hit iteration limit
    if state["is_the_answer_correct"] or state["iteration_count"] >= 2:
        return END
    
    return "verifier"

def build_graph(solver_chain, verifier_chain, ground_truth: int):
    """Build the workflow graph"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("solver", partial(solve, solver_chain=solver_chain))
    workflow.add_node("verifier", partial(verify, verifier_chain=verifier_chain))
    workflow.add_node("cleaner", clean_answer)

    # Add edges
    workflow.set_entry_point("solver")
    workflow.add_edge("solver", "cleaner")
    workflow.add_conditional_edges(
        "cleaner",
        partial(decide_next_step, ground_truth=ground_truth),
        {
            "verifier": "verifier",
            END: END
        }
    )
    workflow.add_edge("verifier", "solver")

    return workflow

from librarian import init_conversation_md, append_to_conversation_md, format_text_blocks

def process_problem(problem_text: str, ground_truth: int, 
                   solver_model: ModelOption,
                   verifier_model: ModelOption,
                   md_file: str = None):
    """Process a single problem through the graph"""
    try:
        solver = create_solver(problem_text, solver_model)
        verifier = create_verifier(problem_text, verifier_model)
        
        # Create initial system prompts
        solver_prompt = f"""You are a mathematical problem solver. Your goal is to solve this problem:

{problem_text}

Then solve the problem step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

Never ask for confirmation. Just provide your final answer as a number at the end of your 
response prefixed with 'ANSWER: '."""

        verifier_prompt = f"""You are a mathematical solution verifier. For this problem:

{problem_text}

The solver's current answer is INCORRECT. Your job is to analyze their solution and try to isolate the most important 
issue with the solution.

Respond with:
'FEEDBACK: [Explanation of errors found and specific suggestions for improvement]'"""

        initial_state = {
            "solver_messages": [HumanMessage(content=solver_prompt)],
            "verifier_messages": [HumanMessage(content=verifier_prompt)],
            "current_solution": "",
            "final_answer": None,
            "iteration_count": 0,
            "is_the_answer_correct": False,
            "md_file": md_file,
            "problem": problem_text
        }
        
        workflow = build_graph(solver, verifier, ground_truth)
        app = workflow.compile()
        
        print("Solving problem...")
        final_state = app.invoke(initial_state)
        return final_state
    except Exception as e:
        print(f"Error processing problem: {e}")
        return {
            "final_answer": None,
            "iteration_count": 0,
            "is_the_answer_correct": False
        }

if __name__ == "__main__":
    # Define models
    SOLVER_MODEL = ModelOption.LOCAL
    VERIFIER_MODEL = ModelOption.CLAUDE
    
    # Load dataset
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train[14:16]")
    
    print(f"\n=== Starting evaluation with {SOLVER_MODEL.value} as solver and {VERIFIER_MODEL.value} as verifier ===")
    
    results = []
    for example in dataset:
        problem_id = example['id']
        problem = example['problem']
        print(f"\nProcessing problem {problem_id}...")
        
        ground_truth = int(example['answer']) if example['answer'].isdigit() else None
        # Initialize conversation file with problem and dataset solution
        md_file = init_conversation_md(str(problem_id), problem, example['solution'], SOLVER_MODEL.name, "_local")
        
        result = process_problem(
            problem, 
            ground_truth,
            solver_model=SOLVER_MODEL,
            verifier_model=VERIFIER_MODEL,
            md_file=md_file
        )
        
        results.append({
            'solver_model': SOLVER_MODEL.value,
            'verifier_model': VERIFIER_MODEL.value,
            'problem_id': problem_id,
            'final_answer': result['final_answer'],
            'ground_truth': ground_truth,
            'iterations': result['iteration_count']
        })
        
    
    # Print summary
    print("\nResults Summary:")
    correct_count = 0
    total_iterations = 0
    for result in results:
        print(f"\nProblem {result['problem_id']}:")
        print(f"Model Answer: {result['final_answer']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Iterations: {result['iterations']}")
        is_correct = result['final_answer'] == result['ground_truth']
        print(f"Correct: {is_correct}")
        if is_correct:
            correct_count += 1
        total_iterations += result['iterations']
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")
    print(f"Average iterations: {total_iterations/len(results):.1f}")
