import os
import time
from functools import partial, wraps
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
SOLVER_PROMPT = """You are a mathematical problem solver. Your goal is to solve this problem:

{problem}

Before solving, start with a brief analysis:
1. What mathematical concepts is this problem testing?
2. What theoretical tools or formulas might be useful?
3. Are there any tricks or simplifications that could make this problem easier?

Then solve the problem step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

Here is the previous feedback if any:
{messages}

Never ask for confirmation. Just provide your final answer as a number at the end of your 
response prefixed with 'ANSWER: '."""

VERIFIER_PROMPT = """You are a mathematical solution verifier. For this problem:

{problem}

The solver's current answer is INCORRECT. Your job is to analyze their solution and identify specific issues.

Follow these steps:

1. Check the initial approach:
   - Is the chosen method appropriate?
   - Are all necessary concepts being used correctly?

2. Verify calculations:
   - Look for numerical errors
   - Check if formulas are applied correctly
   - Verify intermediate results
   
3. Examine logic:
   - Are all steps properly justified?
   - Are there any gaps in reasoning?
   - Are edge cases considered?
   - Are there any invalid assumptions?

4. Provide specific feedback:
   - Point out exact locations of errors
   - Explain why certain steps are problematic
   - Suggest areas that need more careful consideration
   - DO NOT reveal the correct answer

Here is the solution to verify:
{messages}

Respond with:
'FEEDBACK: [Detailed explanation of errors found and specific suggestions for improvement]'"""

def create_solver_chain(problem: str, model_option: ModelOption):
    model = get_model(model_option, temp=0.1)
    escaped_problem = problem.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SOLVER_PROMPT.format(problem=escaped_problem, messages="{messages}"))
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

def solve(state: AgentState, solver_chain):
    """Solver agent function"""
    messages_text = "\n".join([msg.content for msg in state["solver_messages"]])
    print("Solving...")
    response = solver_chain.invoke({"messages": messages_text})
    solution_content = response.content
    
    ai_message = AIMessage(content=solution_content)
    human_message = HumanMessage(content=solution_content)
    
    return {
        "current_solution": solution_content,
        "solver_messages": [ai_message],
        "verifier_messages": [human_message]
    }

def verify(state: AgentState, verifier_chain):
    """Verifier agent function"""
    messages_text = "\n".join([msg.content for msg in state["verifier_messages"]])
    print("Verifying...")
    response = verifier_chain.invoke({"messages": messages_text})
    
    ai_message = AIMessage(content=response.content)
    human_message = HumanMessage(content=response.content)
    
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
    if state["is_the_answer_correct"] or state["iteration_count"] >= 3:
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

def save_conversation_to_md(state: AgentState, problem_id: str, problem: str, solution: str, solver_model: ModelOption):
    """Save the conversation to a Markdown file"""
    filename = f"conversation_{problem_id}_local.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Problem {problem_id} - Solver: {solver_model.name}\n\n")
        f.write("## Problem Statement\n\n")
        f.write(f"{problem}\n\n")
        f.write("## Dataset Solution\n\n")
        f.write(f"{solution}\n\n")
        f.write("## Conversation History\n\n")
        
        messages = []
        for i in range(len(state["solver_messages"])):
            messages.append(("Solver's Solution", state["solver_messages"][i].content))
            if i < len(state["verifier_messages"]):
                messages.append(("Verifier's Response", state["verifier_messages"][i].content))
                
        for i, (role, content) in enumerate(messages, 1):
            f.write(f"### Round {(i + 1) // 2}\n\n")
            f.write(f"#### {role}\n")
            f.write("```\n")
            f.write(f"{content}\n")
            f.write("```\n\n")

@retry_with_exponential_backoff(max_retries=3, initial_delay=1)
def process_problem(problem_text: str, ground_truth: int, 
                   solver_model: ModelOption,
                   verifier_model: ModelOption):
    """Process a single problem through the graph"""
    solver_chain = create_solver_chain(problem_text, solver_model)
    verifier_chain = create_verifier_chain(problem_text, verifier_model)
    
    initial_state = {
        "solver_messages": [],
        "verifier_messages": [],
        "current_solution": "",
        "final_answer": None,
        "iteration_count": 0,
        "is_the_answer_correct": False
    }
    
    workflow = build_graph(solver_chain, verifier_chain, ground_truth)
    app = workflow.compile()
    
    print("Solving problem...")
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    # Define models
    SOLVER_MODEL = ModelOption.LOCAL
    VERIFIER_MODEL = ModelOption.CLAUDE
    
    # Load dataset
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train[11:14]")
    
    print(f"\n=== Starting evaluation with {SOLVER_MODEL.value} as solver and {VERIFIER_MODEL.value} as verifier ===")
    
    results = []
    for example in dataset:
        problem_id = example['id']
        problem = example['problem']
        print(f"\nProcessing problem {problem_id}...")
        
        ground_truth = int(example['answer']) if example['answer'].isdigit() else None
        result = process_problem(
            problem, 
            ground_truth,
            solver_model=SOLVER_MODEL,
            verifier_model=VERIFIER_MODEL
        )
        
        results.append({
            'solver_model': SOLVER_MODEL.value,
            'verifier_model': VERIFIER_MODEL.value,
            'problem_id': problem_id,
            'final_answer': result['final_answer'],
            'ground_truth': ground_truth,
            'iterations': result['iteration_count']
        })
        
        save_conversation_to_md(
            result, 
            f"{problem_id}", 
            example['problem'], 
            example['solution'],
            SOLVER_MODEL
        )
    
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
