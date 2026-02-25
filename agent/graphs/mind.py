# langgraph_cognitive_arch/agent/graphs/mind.py
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from agent.state import AgentState, Thought
from agent.tools.web_search import web_search

# LLM for the Mind's internal processes
mind_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    convert_system_message_to_human=True
)

# --- Nodes for the Mind's Cognitive Loop ---

async def generate_node(state: AgentState) -> AgentState:
    """
    Step 1: GENERATE
    Generates a clear, internal question from the user's latest message.
    """
    print("---MIND LOOP: Generate---")
    prompt = f"""
You are the 'Generate' component of an AI's Mind. 
Your task is to rephrase the user's last message into a clear, actionable internal question.
This question will be used by the 'Think' component to decide on the next step.

User's last message: "{state['messages'][-1].content}"

Generated Internal Question:
"""
    response = await mind_llm.ainvoke(prompt)
    return {"generated_question": response.content}

async def think_node(state: AgentState) -> AgentState:
    """
    Step 2: THINK
    Thinks about the generated question and decides whether to act or escalate.
    """
    print("---MIND LOOP: Think---")
    prompt = f"""
You are the 'Think' component of an AI's Mind.
Your task is to analyze the internal question and decide on the next step.
You have two choices:
1.  `act`: The question is simple and can be answered with a web search or a direct response.
2.  `escalate`: The question is complex, requires deep reasoning, multi-step planning, or coding.

Internal Question: "{state['generated_question']}"

Your thought process and final decision:
"""
    structured_llm = mind_llm.with_structured_output(Thought)
    thought = await structured_llm.ainvoke(prompt)
    
    escalate = thought.decision == 'escalate'
    print(f"---MIND Thought: {thought.reasoning} -> Decision: {thought.decision}---")
    
    return {"thought": thought, "escalate_to_brain": escalate}

async def action_node(state: AgentState) -> AgentState:
    """
    Step 3: ACTION
    Acts on the thought. For the Mind, this is either a simple response or a quick web search.
    """
    print("---MIND LOOP: Action---")
    # This is a simplified action node for the Mind. It can be expanded with tool calling.
    # For now, it just formulates a simple response based on the generated question.
    prompt = f"""
You are the 'Action' component of an AI's Mind.
Your task is to provide a concise and helpful answer to the internal question.
If the question is conversational, just respond naturally.

Internal Question: "{state['generated_question']}"

Answer:
"""
    response = await mind_llm.ainvoke(prompt)
    return {"messages": [AIMessage(content=response.content, name="Mind")]}

# --- Conditional Edge for the Mind's Graph ---

def should_act_or_end(state: AgentState) -> str:
    """If the thought is to act, go to action, otherwise end the sub-graph."""
    if state["thought"].decision == 'act':
        return "act"
    return END # Escalation is handled by the main router based on 'escalate_to_brain' flag

# --- Define and Compile the Mind's Sub-Graph ---

def create_mind_graph():
    """Factory function to create the mind's cognitive loop sub-graph."""
    workflow = StateGraph(AgentState)
    
    # The user's input is the implicit "Load" step
    workflow.add_node("generate", generate_node)
    workflow.add_node("think", think_node)
    workflow.add_node("action", action_node)
    
    workflow.set_entry_point("generate")
    
    workflow.add_edge("generate", "think")
    workflow.add_conditional_edges(
        "think",
        should_act_or_end,
        {
            "act": "action",
            END: END,
        },
    )
    workflow.add_edge("action", END)
    
    return workflow.compile()
