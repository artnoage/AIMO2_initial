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
    solver_messages: Annotated[List[BaseMessage], add_messages]
    verifier_messages: Annotated[List[BaseMessage], add_messages]
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

def solve(state: AgentState, model: ChatOpenAI = get_model(SOLVER_MODEL)):
    """Solver agent function"""
    # Combine all messages for context
    all_messages = []
    
    # System message
    all_messages.append(("system", "You are a mathematical problem solver. Your task is to solve math problems step by step, "
                      "showing your work clearly. Provide your final answer as a number at the end of your response "
                      "prefixed with 'ANSWER: '. Be thorough but concise."))
    
    # Add solver history
    for msg in state["solver_messages"]:
        all_messages.append((msg.type, msg.content))
    
    # Add verifier history
    for msg in state["verifier_messages"]:
        all_messages.append((msg.type, msg.content))
    
    # Add latest input
    latest_content = (state["solver_messages"][-1].content if state["solver_messages"] 
                     else state["verifier_messages"][-1].content)
    all_messages.append(("human", latest_content))
    
    prompt = ChatPromptTemplate.from_messages(all_messages)
    response = model.invoke(prompt)
    
    # Create both AI and Human versions of the response
    ai_message = AIMessage(content=response.content)
    human_message = HumanMessage(content=response.content)
    
    return {
        "current_solution": response.content,
        "solver_messages": [ai_message],
        "verifier_messages": [human_message]
    }

def verify(state: AgentState, model: ChatOpenAI = get_model(VERIFIER_MODEL)):
    """Verifier agent function"""
    # Combine all messages for context
    all_messages = []
    
    # System message
    all_messages.append(("system", "You are a mathematical solution verifier. Your job is to check if the solver's solution is correct. "
                      "Look for any errors in logic or calculation. Respond with either:\n"
                      "'VERIFIED: [explanation]' if the solution looks correct\n"
                      "'NEEDS_REVISION: [explanation]' if you find any issues"))
    
    # Add solver history
    for msg in state["solver_messages"]:
        all_messages.append((msg.type, msg.content))
    
    # Add verifier history
    for msg in state["verifier_messages"]:
        all_messages.append((msg.type, msg.content))
    
    # Add solution to verify
    all_messages.append(("human", f"Please verify this solution:\n{state['current_solution']}"))
    
    prompt = ChatPromptTemplate.from_messages(all_messages)
    response = model.invoke(prompt)
    
    # Create both AI and Human versions of the response
    ai_message = AIMessage(content=response.content)
    human_message = HumanMessage(content=response.content)
    
    return {
        "verifier_messages": [ai_message],
        "solver_messages": [human_message]
    }

def should_continue(state: AgentState) -> Union[str, None]:
    """Determine if we need another iteration"""
    last_message = state["verifier_messages"][-1].content
    if "NEEDS_REVISION" in last_message:
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
        f.write("## Solver Messages\n\n")
        for msg in state["solver_messages"]:
            role = "Human" if isinstance(msg, HumanMessage) else "Solver"
            f.write(f"### {role}\n\n{msg.content}\n\n")
        f.write("## Verifier Messages\n\n")
        for msg in state["verifier_messages"]:
            role = "Human" if isinstance(msg, HumanMessage) else "Verifier"
            f.write(f"### {role}\n\n{msg.content}\n\n")

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
        "solver_messages": [HumanMessage(content=problem_text)],
        "verifier_messages": [],
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
