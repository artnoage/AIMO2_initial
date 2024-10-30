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
    solver1_messages: Annotated[List[BaseMessage], add_messages]
    solver2_messages: Annotated[List[BaseMessage], add_messages]
    verifier_messages: Annotated[List[BaseMessage], add_messages]
    current_solution: Annotated[str, "Current solution being worked on"]
    final_answer: Annotated[Union[int, None], "Final numerical answer"]
    iteration_count: Annotated[int, "Counter for solver-verifier iterations"]
    is_the_answer_correct: Annotated[bool, "Flag indicating if the current answer is correct"]
    md_file: Annotated[str, "Path to markdown file for logging"]
    problem: Annotated[str, "Original problem text"]
    feedback: Annotated[str, "Feedback from verifier"]

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

# Define system prompts
SOLVER1_PROMPT = """You are a mathematical problem solver. Your goal is to solve this problem:

{problem}

Then solve the problem step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

Never ask for confirmation. Just provide your final answer as a number at the end of your 
response prefixed with 'ANSWER: '.

{messages}"""

SOLVER2_PROMPT = """You are a mathematical problem solver. Here is:

1. The original problem:
{problem}

2. A previous solution attempt:
{solution}

3. Feedback on what was wrong:
{feedback}

Your task is to fix the solution based on the feedback. Focus specifically on addressing
the issues mentioned in the feedback while keeping the correct parts of the original solution.

Provide your complete revised solution, ending with your final answer prefixed with 'ANSWER: '.
"""

VERIFIER_PROMPT = """You are a mathematical solution verifier. For this problem:

{problem}

The solver's current answer is INCORRECT. Your job is to analyze their solution and try to isolate the most important 
issue with the solution.

Respond with:
'FEEDBACK: [Explanation of errors found and specific suggestions for improvement]'

{messages}"""

def create_solver1_chain(problem: str, model_option: ModelOption):
    model = get_model(model_option, temp=0.1)
    escaped_problem = problem.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SOLVER1_PROMPT.format(problem=escaped_problem, messages="{messages}"))
    ])
    return prompt | model

def create_solver2_chain(problem: str, model_option: ModelOption):
    model = get_model(model_option, temp=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SOLVER2_PROMPT.format(
            problem="{problem}",
            solution="{solution}",
            feedback="{feedback}")
        )
    ])
    return prompt | model

def create_verifier_chain(problem: str, model_option: ModelOption):
    model = get_model(model_option, temp=0.1)
    escaped_problem = problem.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFIER_PROMPT.format(
            problem=escaped_problem,
            messages="{messages}")
        )
    ])
    return prompt | model

def solve1(state: AgentState, solver1_chain):
    """First solver agent function"""
    messages_text = "\n".join([msg.content for msg in state["solver1_messages"]])
    print("Solver 1...")
    response = solver1_chain.invoke({"messages": messages_text})
    solution_content = response.content
    
    ai_message = AIMessage(content=solution_content)
    human_message = HumanMessage(content=solution_content)
    
    # Update markdown file
    append_to_conversation_md(
        state["md_file"], 
        "Solver 1's Solution",
        solution_content,
        state["iteration_count"],
        messages_text,
        state["problem"]
    )
    
    return {
        "current_solution": solution_content,
        "solver1_messages": [ai_message],
        "verifier_messages": [human_message]
    }

def verify(state: AgentState, verifier_chain):
    """Verifier agent function"""
    messages_text = "\n".join([msg.content for msg in state["verifier_messages"]])
    print("Verifying...")
    response = verifier_chain.invoke({"messages": messages_text})
    
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
    
    # Extract feedback from verifier response
    feedback_match = re.search(r"FEEDBACK:\s*(.*?)(?:\n|$)", response.content, re.DOTALL)
    feedback = feedback_match.group(1) if feedback_match else response.content
    
    return {
        "solver2_messages": [human_message],
        "verifier_messages": [ai_message],
        "iteration_count": state["iteration_count"] + 1,
        "feedback": feedback
    }

def solve2(state: AgentState, solver2_chain):
    """Second solver agent function"""
    print("Solver 2...")
    response = solver2_chain.invoke({
        "problem": state["problem"],
        "solution": state["current_solution"],
        "feedback": state["feedback"]
    })
    solution_content = response.content
    
    # Update markdown file
    append_to_conversation_md(
        state["md_file"], 
        "Solver 2's Solution",
        solution_content,
        state["iteration_count"],
        f"Problem: {state['problem']}\nPrevious solution: {state['current_solution']}\nFeedback: {state['feedback']}",
        state["problem"]
    )
    
    return {
        "current_solution": solution_content,
        "final_answer": extract_answer(solution_content)
    }

def extract_answer(solution: str) -> Union[int, None]:
    """Extract numerical answer from solution text"""
    match = re.search(r"ANSWER:\s*(\d+)", solution)
    if match:
        return int(match.group(1))
    match = re.search(r"\d+", solution)
    return int(match.group()) if match else None

def build_graph(solver1_chain, solver2_chain, verifier_chain):
    """Build the workflow graph"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("solver1", partial(solve1, solver1_chain=solver1_chain))
    workflow.add_node("verifier", partial(verify, verifier_chain=verifier_chain))
    workflow.add_node("solver2", partial(solve2, solver2_chain=solver2_chain))

    # Add edges
    workflow.set_entry_point("solver1")
    workflow.add_edge("solver1", "verifier")
    workflow.add_edge("verifier", "solver2")
    workflow.add_edge("solver2", END)

    return workflow

def init_conversation_md(problem_id: str, problem: str, solution: str, solver_model: ModelOption):
    """Initialize the markdown file with problem details"""
    filename = f"conversation_{problem_id}_squash.md"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Create/overwrite the file with initial content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Problem {problem_id} - Solver: {solver_model.name}\n\n")
        f.write("## Problem Statement\n\n")
        f.write(f"{problem}\n\n")
        f.write("## Dataset Solution\n\n")
        f.write(f"{solution}\n\n")
        f.write("## Conversation History\n\n")
    return filename

def format_text_blocks(text: str, max_line_length: int = 80) -> str:
    """Format text into lines of maximum length while preserving paragraphs"""
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        words = paragraph.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + len(current_line) <= max_line_length:
                current_line.append(word)
                current_length += word_length
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
                
        if current_line:
            lines.append(' '.join(current_line))
            
        formatted_paragraphs.append('\n'.join(lines))
    
    return '\n\n'.join(formatted_paragraphs)

def append_to_conversation_md(filename: str, role: str, content: str, round_num: int, 
                            messages: str = "", problem: str = ""):
    """Append a new message to the conversation markdown file"""
    if not os.path.exists(filename):
        print(f"Warning: Markdown file {filename} not found")
        return
        
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"### Round {round_num + 1}\n\n")
            
            # Add the prompt that was used
            if role == "Solver 1's Solution":
                prompt = SOLVER1_PROMPT.format(problem=problem, messages=messages)
                f.write("#### Input Prompt\n")
                f.write("```\n")
                f.write(format_text_blocks(prompt))
                f.write("\n```\n\n")
            elif role == "Solver 2's Solution":
                prompt = SOLVER2_PROMPT.format(
                    problem=problem,
                    solution=messages.split("Previous solution: ")[1].split("\nFeedback:")[0],
                    feedback=messages.split("Feedback: ")[1]
                )
                f.write("#### Input Prompt\n")
                f.write("```\n")
                f.write(format_text_blocks(prompt))
                f.write("\n```\n\n")
            elif role == "Verifier's Response":
                prompt = VERIFIER_PROMPT.format(problem=problem, messages=messages)
                f.write("#### Input Prompt\n")
                f.write("```\n")
                f.write(format_text_blocks(prompt))
                f.write("\n```\n\n")
            
            # Add the agent's response
            f.write(f"#### {role}\n")
            f.write("```\n")
            f.write(format_text_blocks(content))
            f.write("\n```\n\n")
            f.flush()  # Force write to disk
    except Exception as e:
        print(f"Error appending to markdown file: {e}")

def process_problem(problem_text: str, ground_truth: int, 
                   solver1_model: ModelOption,
                   solver2_model: ModelOption,
                   verifier_model: ModelOption,
                   md_file: str = None):
    """Process a single problem through the graph"""
    try:
        solver1_chain = create_solver1_chain(problem_text, solver1_model)
        solver2_chain = create_solver2_chain(problem_text, solver2_model)
        verifier_chain = create_verifier_chain(problem_text, verifier_model)
        
        # Create initial system prompts
        solver1_prompt = f"""You are a mathematical problem solver. Your goal is to solve this problem:

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
            "solver1_messages": [HumanMessage(content=solver1_prompt)],
            "solver2_messages": [],
            "verifier_messages": [HumanMessage(content=verifier_prompt)],
            "current_solution": "",
            "final_answer": None,
            "iteration_count": 0,
            "is_the_answer_correct": False,
            "md_file": md_file,
            "problem": problem_text,
            "feedback": ""
        }
        
        workflow = build_graph(solver1_chain, solver2_chain, verifier_chain)
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
    SOLVER1_MODEL = ModelOption.LOCAL
    SOLVER2_MODEL = ModelOption.CLAUDE
    VERIFIER_MODEL = ModelOption.GPT
    
    # Load dataset
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train[14:16]")
    
    print(f"\n=== Starting evaluation with:")
    print(f"Solver 1: {SOLVER1_MODEL.value}")
    print(f"Solver 2: {SOLVER2_MODEL.value}")
    print(f"Verifier: {VERIFIER_MODEL.value}")
    
    results = []
    for example in dataset:
        problem_id = example['id']
        problem = example['problem']
        print(f"\nProcessing problem {problem_id}...")
        
        ground_truth = int(example['answer']) if example['answer'].isdigit() else None
        # Initialize conversation file
        md_file = init_conversation_md(str(problem_id), problem, example['solution'], SOLVER1_MODEL)
        
        result = process_problem(
            problem, 
            ground_truth,
            solver1_model=SOLVER1_MODEL,
            solver2_model=SOLVER2_MODEL,
            verifier_model=VERIFIER_MODEL,
            md_file=md_file
        )
        
        results.append({
            'solver1_model': SOLVER1_MODEL.value,
            'solver2_model': SOLVER2_MODEL.value,
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
