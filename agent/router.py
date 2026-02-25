# langgraph_cognitive_arch/agent/router.py
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.graphs.mind import create_mind_graph
from agent.graphs.brain import create_brain_graph

# --- Compile the Sub-Graphs ---
mind_sub_graph = create_mind_graph()
brain_sub_graph = create_brain_graph()

# --- High-Level Nodes for the Main Router Graph ---

async def mind_invoker_node(state: AgentState) -> AgentState:
    """Invokes the entire Mind sub-graph."""
    print("---ROUTER: Invoking Mind sub-graph.---")
    
    # The Mind graph operates on the initial state
    final_mind_state = None
    async for step in mind_sub_graph.astream(state):
        final_mind_state = step[list(step.keys())[0]]
        
    # We return the complete final state of the mind's process
    return final_mind_state

async def brain_invoker_node(state: AgentState) -> AgentState:
    """This node invokes the entire Brain sub-graph."""
    print("---ROUTER: Invoking Brain sub-graph.---")
    
    # The brain works on the current message history.
    brain_initial_state = {"messages": state["messages"]}
    
    final_brain_state = None
    async for step in brain_sub_graph.astream(brain_initial_state):
        final_brain_state = step[list(step.keys())[0]]
        
    return {"messages": final_brain_state['messages']}

# --- Triage Logic for the Router ---

def triage(state: AgentState) -> str:
    """
    The main conditional router. Based on the Mind's 'escalate_to_brain' flag,
    it decides whether to invoke the Brain's sub-graph or end.
    """
    print("---ROUTER: Routing based on Mind's decision.---")
    if state["escalate_to_brain"]:
        return "brain"
    else:
        # If the mind didn't escalate, its response is already in the state's messages
        return END

# --- Define and Compile the Top-Level Router Graph ---

def create_agent_graph():
    """Factory function to create the main agent graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("mind", mind_invoker_node)
    workflow.add_node("brain", brain_invoker_node)
    
    workflow.set_entry_point("mind")
    
    workflow.add_conditional_edges(
        "mind",
        triage,
        {
            "brain": "brain",
            END: END
        }
    )
    workflow.add_edge("brain", END)
    
    return workflow.compile()
