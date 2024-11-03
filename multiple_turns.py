import re
from functools import partial
from typing import Annotated, TypedDict, Union, List, Callable
import time
from functools import wraps
from dotenv import load_dotenv
from datasets import load_dataset
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from utils.librarian import init_conversation_md, append_to_conversation_md
import tiktoken
from utils.utils import get_model
from utils.utils import ModelOption

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

VERIFIER_PROMPT_TEMPLATE = """You are a mathematical solution verifier. When given a problem and solution which you know that is INCORRECT. Analyze the solution and try to isolate the most important 
issue with it, giving helpful feedback to improve.

Respond with:
'FEEDBACK: [Explanation of errors found and specific suggestions for improvement]'"""




# Define state schema
class AgentState(TypedDict):
    solver_messages: Annotated[List[BaseMessage], add_messages]
    verifier_messages: Annotated[List[BaseMessage], add_messages] 
    solution: Annotated[Union[int, str, None], "Final numerical answer"]
    iteration_count: Annotated[int, "Counter for iterations"]
    md_file: Annotated[str, "Path to markdown file"]



@retry_with_delay(max_attempts=3, delay=30)
def solve(state: AgentState, model_option: ModelOption):
    """Solver agent function"""
    messages = state["solver_messages"]
    
    # Always keep first two messages and last two if more than 2 messages exist
    if len(messages) > 2:
        messages = messages[:2] + messages[-2:]
    
    messages_text = "\n".join([msg.content for msg in messages])
    print("Solving...")
    
    # Count input tokens
    enc = tiktoken.get_encoding("cl100k_base")
    input_text = "\n".join(msg.content for msg in messages)
    input_tokens = len(enc.encode(input_text))
    print(f"Input tokens to solver: {input_tokens}")
    
    solver = get_model(model_option, temp=0.1)
    response = solver.invoke(messages)
    
    # Count output tokens
    output_tokens = len(enc.encode(response.content))
    print(f"Output tokens from solver: {output_tokens}")
    
    solution_content = response.content
    
    # Create state summary
    state_summary = f"# Current State - Solver Phase\n\n"
    state_summary += f"## Iteration {state['iteration_count']}\n\n"
    state_summary += "### Input Messages:\n"
    for msg in messages:
        state_summary += f"#### {msg.__class__.__name__}\n"
        state_summary += f"{msg.content}\n\n"
    state_summary += "### Solver Response:\n"
    state_summary += f"{solution_content}\n\n"
    
    # Update markdown file with state information
    append_to_conversation_md(
        state["md_file"], 
        "Solver State",
        state_summary,
        state["iteration_count"],
        ""
    )
    
    ai_message = AIMessage(content=solution_content)
    human_message = HumanMessage(content=solution_content)
    
    return {
        "solution": solution_content,
        "solver_messages": [ai_message],
        "verifier_messages": [human_message]
    }

@retry_with_delay(max_attempts=3, delay=30)
def verify(state: AgentState, model_option: ModelOption):
    """Verifier agent function"""
    messages = state["verifier_messages"]
    
    # Keep first three messages and last message if more than 3 messages exist
    if len(messages) > 3:
        messages = messages[:3] + [messages[-1]]
    
    messages_text = "\n".join([msg.content for msg in messages])
    print("Verifying...")
    
    # Count input tokens
    enc = tiktoken.get_encoding("cl100k_base")
    input_text = "\n".join(msg.content for msg in messages)
    input_tokens = len(enc.encode(input_text))
    print(f"Input tokens to verifier: {input_tokens}")
    
    verifier = get_model(model_option, temp=0.1)
    response = verifier.invoke(messages)
    
    # Count output tokens
    output_tokens = len(enc.encode(response.content))
    print(f"Output tokens from verifier: {output_tokens}")
    
    # Create state summary
    state_summary = f"# Current State - Verifier Phase\n\n"
    state_summary += f"## Iteration {state['iteration_count']}\n\n"
    state_summary += "### Input Messages:\n"
    for msg in messages:
        state_summary += f"#### {msg.__class__.__name__}\n"
        state_summary += f"{msg.content}\n\n"
    state_summary += "### Verifier Response:\n"
    state_summary += f"{response.content}\n\n"
    
    # Update markdown file with state information
    append_to_conversation_md(
        state["md_file"],
        "Verifier State",
        state_summary,
        state["iteration_count"],
        ""
    )
    
    ai_message = AIMessage(content=response.content)
    human_message = HumanMessage(content=response.content)
    
    return {
        "solver_messages": [human_message],
        "verifier_messages": [ai_message],
        "iteration_count": state["iteration_count"] + 1
    }

def clean_answer(state: AgentState) -> AgentState:
    """Extract numerical answer from solver's solution"""
    solution = state["solution"]
    match = re.search(r"ANSWER:\s*(\d+)", solution)
    if match:
        solution = int(match.group(1))
    else:
        match = re.search(r"\d+", solution)
        solution = int(match.group()) if match else None
    
    return {"solution": solution}

def decide_next_step(state: AgentState, ground_truth: int) -> str:
    """Determine if we should continue verification or end"""
    state["is_the_answer_correct"] = (state["solution"] == ground_truth)
    
    # End if answer is correct or we've hit iteration limit
    if state["is_the_answer_correct"] or state["iteration_count"] >= 4:
        return END
    
    return "verifier"

def build_graph(solver_model: ModelOption, verifier_model: ModelOption, ground_truth: int):
    """Build the workflow graph"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("solver", partial(solve, model_option=solver_model))
    workflow.add_node("verifier", partial(verify, model_option=verifier_model))
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



def process_problem(problem_text: str, ground_truth: int, 
                   solver_model: ModelOption,
                   verifier_model: ModelOption,
                   md_file: str = None):
    """Process a single problem through the graph"""
    try:
        # Create initial system prompts
        initial_state = {
            "solver_messages": [
                SystemMessage(content=SOLVER_PROMPT_TEMPLATE),
                HumanMessage(content=f"Here is the problem: {problem_text}")
            ],
            "verifier_messages": [
                SystemMessage(content=VERIFIER_PROMPT_TEMPLATE),
                HumanMessage(content=f"Here is the problem: {problem_text}"),
                AIMessage(content="And what is the solution?")
            ],
            "solution": "",
            "iteration_count": 0,
            "md_file": md_file}
        
        workflow = build_graph(solver_model, verifier_model, ground_truth)
        app = workflow.compile()
        
        print("Solving problem...")
        final_state = app.invoke(initial_state,{"recursion_limit": 200})
        return final_state
    except Exception as e:
        print(f"Error processing problem: {e}")
        return {
            "solution": None,
            "iteration_count": 0,
            "is_the_answer_correct": False
        }

if __name__ == "__main__":
    import argparse
    from huggingface_hub import HfApi
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run math problem solver with specified model')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption], 
                       default='NOUS', help='Solver model to use')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='NOUS', help='Verifier model to use')
    parser.add_argument('--both', type=str, choices=[model.name for model in ModelOption],
                       help='Use same model for both solver and verifier')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--max-examples', type=int, default=10,
                       help='Maximum number of examples to process')
    args = parser.parse_args()
    
    # Define models
    if args.both:
        SOLVER_MODEL = VERIFIER_MODEL = ModelOption[args.both]
    else:
        SOLVER_MODEL = ModelOption[args.solver]
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
    
    print(f"\n=== Starting evaluation with {SOLVER_MODEL.value} as solver and {VERIFIER_MODEL.value} as verifier ===")
    
    results = []
    for example in dataset:
        problem_id = example['id']
        problem = example['problem']
        print(f"\nProcessing problem {problem_id}...")
        
        ground_truth = int(example['answer']) if example['answer'].isdigit() else None
        # Initialize conversation file with problem and dataset solution
        md_file = init_conversation_md(
            problem_id=str(problem_id),
            problem=problem,
            solution=example['solution'],
            solver_model_name=SOLVER_MODEL.name,
            suffix="",
            directory=f"conversations/{SOLVER_MODEL.name.lower()}"
        )
        
        result = process_problem(
            problem, 
            ground_truth,
            solver_model=SOLVER_MODEL,
            verifier_model=VERIFIER_MODEL,
            md_file=md_file
        )
        
        # Store result
        result_entry = {
            'solver_model': SOLVER_MODEL.value,
            'verifier_model': VERIFIER_MODEL.value,
            'problem_id': problem_id,
            'solution': result['solution'],
            'ground_truth': ground_truth,
            'iterations': result['iteration_count']
        }
        results.append(result_entry)
        
        # Print immediate result for this problem
        is_correct = result_entry['solution'] == result_entry['ground_truth']
        print(f"\nProblem {problem_id} Result:")
        print(f"Model Answer: {result_entry['solution']}")
        print(f"Ground Truth: {result_entry['ground_truth']}")
        print(f"Iterations: {result_entry['iterations']}")
        print(f"Correct: {is_correct}")
        
        # Print running accuracy
        correct_so_far = sum(1 for r in results if r['solution'] == r['ground_truth'])
        print(f"Running Accuracy: {correct_so_far}/{len(results)} = {correct_so_far/len(results):.2%}")
        print("-" * 50)
    
    # Print final summary
    print("\nResults Summary:")
    correct_count = 0
    total_iterations = 0
    for result in results:
        print(f"\nProblem {result['problem_id']}:")
        print(f"Model Answer: {result['solution']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Iterations: {result['iterations']}")
        is_correct = result['solution'] == result['ground_truth']
        print(f"Correct: {is_correct}")
        if is_correct:
            correct_count += 1
        total_iterations += result['iterations']
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")
    print(f"Average iterations: {total_iterations/len(results):.1f}")
