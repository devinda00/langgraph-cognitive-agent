# langgraph_cognitive_arch/agent/router.py
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.graphs.mind import mind_node
from agent.graphs.brain import create_brain_graph

# --- High-Level Nodes for the Main Router Graph ---

# The mind_node is imported directly as it's a single, simple node.
# The brain_graph is a compiled, runnable sub-graph.
brain_sub_graph = create_brain_graph()

async def simple_response_node(state: AgentState) -> AgentState:
    """Generates the final simple response from the Mind's decision."""
    print("---ROUTER: Generating simple response.---")
    mind_decision = state["mind_decision"]
    response = AIMessage(content=mind_decision.simple_response, name="Mind")
    return {"messages": [response]}

async def brain_invoker_node(state: AgentState) -> AgentState:
    """This node invokes the entire Brain sub-graph."""
    print("---ROUTER: Invoking Brain sub-graph.---")
    
    # Add a message to show the Mind is passing the task to the Brain
    escalation_message = AIMessage(content="That's a complex question. Let me think more deeply about it.", name="Mind")
    
    # We create a new, clean state for the brain to work on,
    # but carry over the original user messages.
    brain_initial_state = {"messages": state["messages"] + [escalation_message]}
    
    # Stream the results from the brain sub-graph
    final_brain_state = None
    async for step in brain_sub_graph.astream(brain_initial_state):
        # We only care about the final state of the brain's work
        final_brain_state = step
        
    return {"messages": final_brain_state[list(final_brain_state.keys())[0]]['messages']}

# --- Triage Logic for the Router ---

def triage(state: AgentState) -> str:
    """
    The main conditional router. Based on the Mind's decision, it decides whether
    to invoke the Brain's sub-graph or generate a simple response.
    """
    print("---ROUTER: Routing based on Mind's decision.---")
    if state["mind_decision"].should_escalate:
        return "brain"
    else:
        return "simple_response"

# --- Define and Compile the Top-Level Router Graph ---

def create_agent_graph():
    """Factory function to create the main agent graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("mind", mind_node)
    workflow.add_node("simple_response", simple_response_node)
    workflow.add_node("brain", brain_invoker_node)
    
    workflow.set_entry_point("mind")
    
    workflow.add_conditional_edges(
        "mind",
        triage,
        {
            "brain": "brain",
            "simple_response": "simple_response"
        }
    )
    workflow.add_edge("simple_response", END)
    workflow.add_edge("brain", END)
    
    return workflow.compile()
