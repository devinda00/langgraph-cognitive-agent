# langgraph_cognitive_arch/agent/graphs/mind.py
# This file is now the main entry point for the entire agent graph.
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.state import AgentState, MindAction
from agent.tools.brain_tools import web_search_mind
from agent.graphs.brain import create_brain_graph

# --- LLMs and Tools ---
mind_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    convert_system_message_to_human=True
).with_structured_output(MindAction)

mind_tool_node = ToolNode([web_search_mind])
brain_sub_graph = create_brain_graph() # The entire brain is a "tool" for the mind

# --- Nodes for the Mind's Cognitive Loop ---

async def load_node(state: AgentState) -> dict:
    """
    Step 1: LOAD
    The entry point for user interaction. It simply passes the state through.
    The user's message is already in state['messages'].
    """
    print("---MIND LOOP: Load---")
    # In a real "always on" system, this could check for new input.
    # Here, it's the start of the processing chain for a new user message.
    return {}

async def generate_node(state: AgentState) -> dict:
    """
    Step 2: GENERATE
    Generates a clear internal question from the current context.
    """
    print("---MIND LOOP: Generate---")
    # This node is conceptually important but can be combined with 'think' for efficiency.
    # For now, we'll pass its role to the think node.
    return {}

async def think_node(state: AgentState) -> dict:
    """
    Step 3: THINK
    The core of the Mind. It analyzes the situation and decides on an action.
    """
    print("---MIND LOOP: Think---")
    prompt = f"""
You are the "Mind," a fast, efficient AI controller. Your job is to analyze the user's request and decide the next action.
You have three choices:
1.  `respond_to_user`: The request is simple, conversational, or has been fully answered. Provide a direct response.
2.  `use_mind_tool`: The request requires a quick piece of external information. Use the web search tool.
3.  `call_brain`: The request is complex and requires deep reasoning, planning, or access to long-term memory.

Current Conversation:
{state['messages']}

Temp Knowledge:
{state['temp_knowledge']}

Based on this, what is your reasoning and the next action to take?
"""
    mind_action = await mind_llm.ainvoke(prompt)
    print(f"---MIND Thought: {mind_action.reasoning} -> Action: {mind_action.action}---")
    return {"mind_action": mind_action}

async def update_node(state: AgentState, action_result: dict) -> dict:
    """
    Step 5: UPDATE
    Updates the temporary knowledge base with the results of the last action.
    """
    print("---MIND LOOP: Update---")
    last_message = state['messages'][-1]
    # Here we would add logic to distill and save important info to temp_knowledge
    # For now, we'll just log that we passed through the update step.
    return {}

# --- Action-related nodes ---

async def respond_to_user_node(state: AgentState) -> dict:
    """The node that formulates and returns the final response to the user."""
    return {"messages": [AIMessage(content=state['mind_action'].tool_input, name="Mind")]}

# --- Main Conditional Router ---

def route_action(state: AgentState) -> str:
    """The main router for the entire agent, driven by the Mind's decision."""
    action = state['mind_action'].action
    if action == 'call_brain':
        return 'brain'
    elif action == 'use_mind_tool':
        return 'mind_tools'
    else: # respond_to_user
        return 'respond_to_user'

# --- Define and Compile the Unified Agent Graph ---

def create_agent_graph():
    """This function now creates the entire, unified agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("load", load_node)
    workflow.add_node("think", think_node)
    
    # Action nodes
    workflow.add_node("mind_tools", mind_tool_node)
    workflow.add_node("brain", brain_sub_graph) # The brain is a node
    workflow.add_node("respond_to_user", respond_to_user_node)

    workflow.add_node("update", update_node)

    workflow.set_entry_point("load")
    workflow.add_edge("load", "think")
    
    # The Action router
    workflow.add_conditional_edges(
        "think",
        route_action,
        {
            "mind_tools": "mind_tools",
            "brain": "brain",
            "respond_to_user": "respond_to_user"
        }
    )
    
    # After an action, update and loop back to think
    workflow.add_edge("mind_tools", "update")
    workflow.add_edge("brain", "update")
    workflow.add_edge("update", "think") # The main loop!

    workflow.add_edge("respond_to_user", END)

    return workflow.compile()
