# langgraph_cognitive_arch/agent/graphs/brain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools.web_search import web_search

# The BRAIN: Powerful, deliberative, and used for complex reasoning.
brain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0
).bind_tools([web_search]) # Bind the tool for the Brain to use

# The tool node for executing the web search tool
tool_node = ToolNode([web_search])

# --- Nodes for the Brain's Sub-Graph ---

async def brain_planner_node(state: AgentState) -> AgentState:
    """
    This is the core reasoning node for the Brain.
    It receives the task and decides whether to use a tool or to respond directly.
    """
    print("---SUB-GRAPH (BRAIN): Generating plan---")
    response = await brain_llm.ainvoke(state["messages"])
    return {"messages": [response]}

def should_continue_tool_use(state: AgentState) -> str:
    """
    Conditional router for the Brain's ReAct loop.
    Decides if another tool call is needed.
    """
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        # No tool calls, so we're done
        return "end"
    else:
        # There's a tool call, so continue the loop
        return "continue"

async def brain_synthesizer_node(state: AgentState) -> AgentState:
    """
    After all tool calls are done, this node synthesizes the final answer.
    """
    print("---SUB-GRAPH (BRAIN): Synthesizing final answer---")
    synthesizer_prompt = (
        "You are a helpful assistant. All the necessary information from tool calls"
        " is now available. Synthesize this information to provide a final, comprehensive"
        " answer to the user's original query."
    )
    # Add the synthesizer prompt to the message history
    all_messages = state["messages"] + [("user", synthesizer_prompt)]
    
    response = await brain_llm.ainvoke(all_messages)
    # We remove the tool-calling info from the final response for cleanliness
    response.tool_calls = [] 
    return {"messages": [response]}

# --- Define and Compile the Brain's Sub-Graph ---

def create_brain_graph():
    """Factory function to create the brain's sub-graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", brain_planner_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("synthesizer", brain_synthesizer_node)
    
    workflow.set_entry_point("planner")
    
    workflow.add_conditional_edges(
        "planner",
        should_continue_tool_use,
        {
            "continue": "tools",
            "end": "synthesizer",
        },
    )
    workflow.add_edge("tools", "planner") # Loop back to planner after tool use
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()
