# langgraph_cognitive_arch/agent/graphs/brain.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from agent.state import AgentState, BrainAction
from agent.tools.brain_tools import (
    access_temp_knowledge, 
    access_permanent_knowledge, 
    write_to_permanent_knowledge
)

# --- LLMs and Tools ---
brain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    convert_system_message_to_human=True
).with_structured_output(BrainAction)

brain_tools = [access_temp_knowledge, access_permanent_knowledge, write_to_permanent_knowledge]
brain_tool_node = ToolNode(brain_tools)

# --- Nodes for the Brain's Cognitive Loop ---

async def generate_node(state: AgentState) -> dict:
    """Step 1: GENERATE. Generates questions about the task."""
    print("---BRAIN LOOP: Generate---")
    # In this implementation, Generate & Think are combined for efficiency
    return {}

async def think_node(state: AgentState) -> dict:
    """Step 2: THINK. Forms a plan and decides on the next action."""
    print("---BRAIN LOOP: Think---")
    prompt = f"""
You are the "Brain," a deep reasoning engine. Your task is to solve a complex problem.
Analyze the user's request and the current state of knowledge to decide on the next action.

You have the following actions available:
1.  `access_temp_knowledge`: Retrieve information from the short-term working memory.
2.  `access_permanent_knowledge`: Recall long-term memories from the vector store.
3.  `write_to_permanent_knowledge`: Distill and save a new insight to long-term memory.
4.  `respond_to_mind`: You have solved the problem. Formulate a final answer to send back to the Mind.

Current Conversation History:
{state['messages']}

Temp Knowledge:
{state['temp_knowledge']}

Based on this, what is your reasoning and the next action to take?
"""
    brain_action = await brain_llm.ainvoke(prompt)
    print(f"---BRAIN Thought: {brain_action.reasoning} -> Action: {brain_action.action}---")
    return {"brain_action": brain_action}

async def update_node(state: AgentState) -> dict:
    """Step 4: UPDATE. The results of the action are implicitly added to state by the ToolNode."""
    print("---BRAIN LOOP: Update---")
    return {}

async def respond_to_mind_node(state: AgentState) -> dict:
    """The final node, packaging the result for the Mind."""
    print("---BRAIN LOOP: Respond to Mind---")
    return {"messages": [AIMessage(content=state['brain_action'].tool_input, name="Brain")]}

# --- Conditional Router for the Brain ---

def route_brain_action(state: AgentState) -> str:
    action = state['brain_action'].action
    if action == 'respond_to_mind':
        return 'respond_to_mind'
    else: # Any of the tool-using actions
        return 'action'

# --- Define and Compile the Brain's Sub-Graph ---

def create_brain_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("think", think_node)
    workflow.add_node("action", brain_tool_node)
    workflow.add_node("update", update_node)
    workflow.add_node("respond_to_mind", respond_to_mind_node)
    
    workflow.set_entry_point("think")
    
    workflow.add_conditional_edges(
        "think",
        route_brain_action,
        {
            "action": "action",
            "respond_to_mind": "respond_to_mind"
        }
    )
    
    # The crucial loop: After acting, update, then think again.
    workflow.add_edge("action", "update")
    workflow.add_edge("update", "think")

    workflow.add_edge("respond_to_mind", END)
    
    return workflow.compile()
