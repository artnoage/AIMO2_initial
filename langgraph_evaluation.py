import os
from typing import Annotated, Sequence, TypedDict, Union, List
from datasets import load_dataset
from langgraph.graph.message import add_messages
import re
from typing_extensions import TypeVar
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, MessageGraph
from langgraph.prebuilt.message_graphs import ChatGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Setup for OpenRouter
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Both agents using claude-3-sonnet for now
SOLVER_MODEL = "anthropic/claude-3-sonnet"
VERIFIER_MODEL = "anthropic/claude-3-sonnet"

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_solution: Annotated[str, "Current solution being worked on"]
    problem_id: Annotated[str, "ID of the current problem"]
    final_answer: Annotated[Union[int, None], "Final numerical answer"]

# Initialize the models
def get_model(model_name: str):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

# Setup solver agent
solver = ChatPromptTemplate.from_messages([
    ("system", "You are a mathematical problem solver. Your task is to solve math problems step by step, "
              "showing your work clearly. Provide your final answer as a number at the end of your response "
              "prefixed with 'ANSWER: '. Be thorough but concise."),
    MessagesPlaceholder(variable_name="solver_messages"),
    ("human", "{input}")
])

# Setup verifier agent
verifier = ChatPromptTemplate.from_messages([
    ("system", "You are a mathematical solution verifier. Your job is to check if the solver's solution is correct. "
              "Look for any errors in logic or calculation. Respond with either:\n"
              "'VERIFIED: [explanation]' if the solution looks correct\n"
              "'NEEDS_REVISION: [explanation]' if you find any issues"),
    MessagesPlaceholder(variable_name="verifier_messages"),
    ("human", "Please verify this solution:\n{solution}")
])

def solve(state: AgentState, model: ChatOpenAI = get_model(SOLVER_MODEL)):
    """Solver agent function"""
    # Convert verifier messages to human messages for solver's view
    solver_history = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.content == state["current_solution"]:
            continue
        if isinstance(msg, AIMessage):
            solver_history.append(HumanMessage(content=msg.content))
        else:
            solver_history.append(msg)
    
    prompt = solver.invoke({
        "solver_messages": solver_history,
        "input": state["messages"][-1].content
    })
    response = model.invoke(prompt)
    return {
        "current_solution": response.content,
        "messages": [response]
    }

def verify(state: AgentState, model: ChatOpenAI = get_model(VERIFIER_MODEL)):
    """Verifier agent function"""
    # Convert solver messages to human messages for verifier's view
    verifier_history = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and ("VERIFIED" in msg.content or "NEEDS_REVISION" in msg.content):
            continue
        if isinstance(msg, AIMessage):
            verifier_history.append(HumanMessage(content=msg.content))
        else:
            verifier_history.append(msg)
    
    prompt = verifier.invoke({
        "verifier_messages": verifier_history,
        "solution": state["current_solution"]
    })
    response = model.invoke(prompt)
    return {
        "messages": [response]
    }

def should_continue(state: AgentState) -> Union[str, None]:
    """Determine if we need another iteration"""
    last_message = state["messages"][-1].content
    if "NEEDS_REVISION" in last_message:
        return "solver"
    return "cleaner"

def clean_answer(state: AgentState) -> AgentState:
    """Extract numerical answer from verifier's response"""
    last_message = state["messages"][-1].content
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
    
    # Save conversation to MD file
    save_conversation_to_md(state)
    
    return {
        **state,
        "final_answer": final_answer
    }

def save_conversation_to_md(state: AgentState):
    """Save the conversation to a Markdown file"""
    filename = f"conversation_{state['problem_id']}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Problem {state['problem_id']}\n\n")
        for msg in state["messages"]:
            role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            f.write(f"## {role}\n\n{msg.content}\n\n")

# Build the graph
workflow = Graph()

# Add nodes
workflow.add_node("solver", solve)
workflow.add_node("verifier", verify)
workflow.add_node("cleaner", clean_answer)

# Add edges
workflow.add_edge("solver", "verifier")
workflow.add_conditional_edges(
    "verifier",
    should_continue,
    {
        "solver": "solver",
        "cleaner": "cleaner"
    }
)
workflow.add_edge("cleaner", None)

# Compile the graph
app = workflow.compile()

def process_problem(problem_text: str, problem_id: str):
    """Process a single problem through the graph"""
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=problem_text)],
        "current_solution": "",
        "problem_id": problem_id,
        "final_answer": None
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    # Load first 3 problems from AIME dataset
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train[:3]")
    
    results = []
    for example in dataset:
        problem_id = example['id']
        problem = example['problem']
        print(f"\nProcessing problem {problem_id}...")
        
        result = process_problem(problem, problem_id)
        results.append({
            'problem_id': problem_id,
            'final_answer': result['final_answer'],
            'ground_truth': int(example['answer']) if example['answer'].isdigit() else None
        })
    
    # Print summary
    print("\nResults Summary:")
    for result in results:
        print(f"\nProblem {result['problem_id']}:")
        print(f"Model Answer: {result['final_answer']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Correct: {result['final_answer'] == result['ground_truth']}")
