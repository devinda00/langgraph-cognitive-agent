# langgraph_cognitive_arch/agent/graphs/brain.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.state import AgentState, Thought
from agent.tools.web_search import web_search

# LLM for the Brain's internal processes
brain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    convert_system_message_to_human=True
)
# Bind the web search tool to the Brain LLM
brain_llm_with_tools = brain_llm.bind_tools([web_search])

# Tool node for executing actions
tool_node = ToolNode([web_search])

# --- Nodes for the Brain's Cognitive Loop ---

async def generate_node(state: AgentState) -> AgentState:
    """
    Step 1: GENERATE
    Generates a plan to answer the user's query. It can choose to use a tool.
    This node is the starting point and the re-evaluation point after each tool use.
    """
    print("---BRAIN LOOP: Generate---")
    
    # The first message is the original user query. Subsequent messages can be tool outputs.
    response = await brain_llm_with_tools.ainvoke(state["messages"])
    
    # The 'generate' node in the brain directly produces the action/response
    return {"messages": [response]}

async def update_node(state: AgentState) -> AgentState:
    """
    Step 2: UPDATE (Implicit after Action/Tool Call)
    The ToolNode implicitly handles this. It takes the tool calls from the last AIMessage,
    executes them, and appends the ToolMessage results to the state.
    This function's logic is therefore handled by the combination of the 'action' (ToolNode)
    and the graph's persistence of state. After this, we loop back to 'generate'.
    """
    # This node is conceptually important but functionally handled by the ToolNode
    # and the graph structure. We don't need to add explicit logic here.
    # The loop back to 'generate' is the key.
    pass

# --- Conditional Edge for the Brain's Graph ---

def should_continue_or_finish(state: AgentState) -> str:
    """
    After the 'generate' node, this decides if we need to call a tool or if we are done.
    If the last message from the LLM has tool calls, we continue to the 'action' step.
    Otherwise, we finish.
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return "finish"
    return "continue"


# --- Define and Compile the Brain's Sub-Graph ---

def create_brain_graph():
    """
    Factory function to create the brain's cognitive loop sub-graph,
    which follows the pattern: Generate -> Action -> Update -> Generate ...
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate", generate_node)
    workflow.add_node("action", tool_node) # Action is executing the tool
    
    workflow.set_entry_point("generate")
    
    workflow.add_conditional_edges(
        "generate",
        should_continue_or_finish,
        {
            "continue": "action",
            "finish": END,
        },
    )
    
    # This is the crucial loop from your diagram: Action -> Update -> Generate
    # The ToolNode ('action') implicitly updates state, and we explicitly loop back to 'generate'
    workflow.add_edge("action", "generate")
    
    return workflow.compile()
