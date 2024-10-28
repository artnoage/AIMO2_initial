import os
from functools import partial
from typing import Annotated, TypedDict, Union, List
from datasets import load_dataset
from langgraph.graph.message import add_messages
import re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

# Setup for OpenRouter
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Both agents using claude-3-sonnet for now
SOLVER_MODEL = "anthropic/claude-3-sonnet"
VERIFIER_MODEL = "anthropic/claude-3-sonnet"

# Define state schema
class AgentState(TypedDict):
    solver_messages: Annotated[List[BaseMessage], add_messages]
    verifier_messages: Annotated[List[BaseMessage], add_messages]
    current_solution: Annotated[str, "Current solution being worked on"]
    final_answer: Annotated[Union[int, None], "Final numerical answer"]
    iteration_count: Annotated[int, "Counter for solver-verifier iterations"]

# Initialize the models
def get_model(model_name: str):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

# Define system prompts
SOLVER_PROMPT = """You are a mathematical problem solver. Here is the problem:

{problem}

Previous messages:
{messages}

Note: If you see variables in double curly braces like {{n}}, treat them as part of the problem text.

Before solving, start with a brief analysis:
1. What mathematical concepts is this problem testing?
2. What theoretical tools or formulas might be useful?
3. Are there any tricks or simplifications that could make this problem easier?

Then solve the problem step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Highlight any key insights or clever observations
- Double-check your work

Provide your final answer as a number at the end of your response prefixed with 'ANSWER: '."""

VERIFIER_PROMPT = """You are a mathematical solution verifier. For this problem:

{problem}

Previous messages:
{messages}

Your job is to rigorously verify the solver's solution. Follow these steps:

1. Check the initial approach:
   - Is the chosen method appropriate?
   - Are all necessary concepts being used correctly?

2. Verify calculations:
   - Redo ALL numerical calculations independently
   - Check for arithmetic errors
   - Verify any algebraic manipulations

3. Examine logic:
   - Are all steps properly justified?
   - Are there any gaps in reasoning?
   - Are edge cases considered?

4. Validate the final answer:
   - Does it make sense in the context?
   - Are the units correct (if applicable)?
   - Is it in the expected range?

Respond with either:
'VERIFIED: [detailed explanation of what you checked and why it's correct]' 
or 
'NEEDS_REVISION: [specific issues found and what needs to be fixed]'"""

# Create the chains
def create_solver_chain(problem: str, model: ChatOpenAI = get_model(SOLVER_MODEL)):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SOLVER_PROMPT.format(problem=problem, messages="{messages}"))
    ])
    return prompt | model

def create_verifier_chain(problem: str, model: ChatOpenAI = get_model(VERIFIER_MODEL)):
    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFIER_PROMPT.format(problem=problem, messages="{messages}"))
    ])
    return prompt | model

def solve(state: AgentState, solver_chain):
    """Solver agent function"""
    # Convert messages list to string
    messages_text = "\n".join([msg.content for msg in state["solver_messages"]])
    
    # Get response from the solver
    response = solver_chain.invoke({"messages": messages_text})
    solution_content = response.content
    
    # Create both AI and Human versions of the response
    ai_message = AIMessage(content=solution_content)
    human_message = HumanMessage(content=solution_content)
    
    return {
        "current_solution": solution_content,
        "solver_messages": [ai_message],
        "verifier_messages": [human_message]
    }

def verify(state: AgentState, verifier_chain):
    """Verifier agent function"""
    
    # Convert messages list to string
    messages_text = "\n".join([msg.content for msg in state["verifier_messages"]])
    
    # Get response from the verifier
    response = verifier_chain.invoke({"messages": messages_text})
    verification_content = response.content
    
    # Create both AI and Human versions of the response
    ai_message = AIMessage(content=verification_content)
    human_message = HumanMessage(content=verification_content)
    
    return {
        "solver_messages": [human_message],
        "verifier_messages": [ai_message],
        "iteration_count": state["iteration_count"] + 1
    }

def should_continue(state: AgentState) -> Union[str, None]:
    """Determine if we need another iteration"""
    last_message = state["verifier_messages"][-1].content
    if "NEEDS_REVISION" in last_message and state["iteration_count"] < 3:
        return "solver"
    return "cleaner"

def clean_answer(state: AgentState) -> AgentState:
    """Extract numerical answer from verifier's response"""
    last_message = state["verifier_messages"][-1].content
    if "VERIFIED" in last_message:
        # Try to extract a number from the solver's solution
        solution = state["current_solution"]
        match = re.search(r"ANSWER:\s*(\d+)", solution)
        if match:
            final_answer = int(match.group(1))
        else:
            # Fallback to any number in the solution
            match = re.search(r"\d+", solution)
            final_answer = int(match.group()) if match else None
    else:
        final_answer = None
    
    # Note: This call will be handled in the main loop instead
    pass
    
    return {
        **state,
        "final_answer": final_answer
    }

def save_conversation_to_md(state: AgentState, problem_id: str, problem: str, solution: str):
    """Save the conversation to a Markdown file"""
    filename = f"conversation_{problem_id}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Problem {problem_id}\n\n")
        f.write("## Problem Statement\n\n")
        f.write(f"{problem}\n\n")
        f.write("## Dataset Solution\n\n")
        f.write(f"{solution}\n\n")
        f.write("## Solver History\n\n")
        for msg in state["solver_messages"]:
            f.write(f"{msg.content}\n\n")

def build_graph(solver_chain, verifier_chain):
    """Build the workflow graph for a specific problem"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("solver", partial(solve, solver_chain=solver_chain))
    workflow.add_node("verifier", partial(verify, verifier_chain=verifier_chain))
    workflow.add_node("cleaner", clean_answer)

    # Add edges
    workflow.set_entry_point("solver")
    workflow.add_edge("solver", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        should_continue,
        {
            "solver": "solver",
            "cleaner": "cleaner"
        }
    )
    workflow.add_edge("cleaner", END)

    return workflow

def process_problem(problem_text: str):
    """Process a single problem through the graph"""
    # Escape any variables in the problem text
    escaped_problem = problem_text.replace("{", "{{").replace("}", "}}")
    
    # Create chains for this specific problem
    solver_chain = create_solver_chain(escaped_problem)
    verifier_chain = create_verifier_chain(escaped_problem)
    
    # Initialize state
    initial_state = {
        "solver_messages": [],
        "verifier_messages": [],
        "current_solution": "",
        "final_answer": None,
        "iteration_count": 0
    }
    
    # Build and compile graph for this problem
    workflow = build_graph(solver_chain, verifier_chain)
    app = workflow.compile()
    
    # Run the graph
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    # Load first 3 problems from AIME dataset
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train[0:1]+train[2:4]")
    
    results = []
    for example in dataset:
        problem_id = example['id']
        problem = example['problem']
        print(f"\nProcessing problem {problem_id}...")
        
        result = process_problem(problem)
        results.append({
            'problem_id': problem_id,
            'final_answer': result['final_answer'],
            'ground_truth': int(example['answer']) if example['answer'].isdigit() else None
        })
        
        # Save conversation after processing
        save_conversation_to_md(result, problem_id, example['problem'], example['solution'])
    
    # Print summary
    print("\nResults Summary:")
    for result in results:
        print(f"\nProblem {result['problem_id']}:")
        print(f"Model Answer: {result['final_answer']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Correct: {result['final_answer'] == result['ground_truth']}")
