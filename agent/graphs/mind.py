# langgraph_cognitive_arch/agent/graphs/mind.py
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from agent.state import AgentState, Thought

# LLM for the Mind's internal processes
mind_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    convert_system_message_to_human=True
)

# --- Nodes for the Mind's Cognitive Loop ---

async def generate_node(state: AgentState) -> AgentState:
    """
    Step 1 (and loop entry): GENERATE
    Generates a clear internal question based on the latest messages and knowledge.
    """
    print("---MIND LOOP: Generate---")
    
    # Format the knowledge base for inclusion in the prompt
    kb_string = "\n".join([f"- {key}: {value}" for key, value in state['knowledge_base'].items()])
    if not kb_string:
        kb_string = "is empty."

    prompt = f"""
You are the 'Generate' component of an AI's Mind. 
Your task is to rephrase the user's last message and current knowledge base into a clear, actionable internal question.
The last message might be from the user or context from the knowledge base.

User's last message: "{state['messages'][-1].content}"
The Temp Knowledge base currently {kb_string}

Generated Internal Question:
"""
    response = await mind_llm.ainvoke(prompt)
    return {"generated_question": response.content}

async def think_node(state: AgentState) -> AgentState:
    """
    Step 2: THINK
    Analyzes the question and decides to respond, escalate, or retrieve knowledge.
    """
    print("---MIND LOOP: Think---")
    prompt = f"""
You are the 'Think' component of an AI's Mind. Your task is to analyze the internal question and the state of the temp knowledge base, then decide on the next step.

You have three choices:
1.  `retrieve_knowledge`: The question requires information that might be in the temp knowledge base, but hasn't been accessed yet. Use this if the user asks a follow-up or refers to something that was said before.
2.  `respond`: The question is simple and can be answered directly with the current information.
3.  `escalate`: The question is complex and requires the Brain (e.g., multi-step reasoning, planning, coding).

Internal Question: "{state['generated_question']}"
Temp Knowledge Keys: {list(state['knowledge_base'].keys())}

Your thought process and final decision:
"""
    structured_llm = mind_llm.with_structured_output(Thought)
    thought = await structured_llm.ainvoke(prompt)
    
    escalate = thought.decision == 'escalate'
    print(f"---MIND Thought: {thought.reasoning} -> Decision: {thought.decision}---")
    
    return {"thought": thought, "escalate_to_brain": escalate}

async def retrieve_knowledge_node(state: AgentState) -> AgentState:
    """
    Step 2.5 (Optional): RETRIEVE KNOWLEDGE
    Retrieves information from the temporary knowledge base (dict) and adds it to the message history for context.
    """
    print("---MIND LOOP: Retrieve Knowledge---")
    query = state['thought'].knowledge_query
    if query and query in state['knowledge_base']:
        retrieved_data = state['knowledge_base'][query]
        context_message = HumanMessage(
            content=f"Context from Temp Knowledge for '{query}': {retrieved_data}",
            name="KnowledgeBase"
        )
        print(f"---MIND: Found knowledge for '{query}': {retrieved_data}---")
    else:
        context_message = HumanMessage(
            content=f"Could not find information for '{query}' in Temp Knowledge.",
            name="KnowledgeBase"
        )
        print(f"---MIND: No knowledge found for '{query}'---")
        
    return {"messages": [context_message]}

async def action_node(state: AgentState) -> AgentState:
    """
    Step 3: ACTION / RESPOND
    Formulates a simple, direct response to the user.
    """
    print("---MIND LOOP: Respond---")
    prompt = f"""
You are the 'Action' component of an AI's Mind. Your task is to provide a concise and helpful answer to the internal question. If the question is conversational, just respond naturally.

Internal Question: "{state['generated_question']}"

Answer:
"""
    response = await mind_llm.ainvoke(prompt)
    
    # Let's also update our temp knowledge with our response
    new_knowledge = state['knowledge_base']
    new_knowledge[state['generated_question']] = response.content
    
    return {
        "messages": [AIMessage(content=response.content, name="Mind")],
        "knowledge_base": new_knowledge
    }

# --- Conditional Edge for the Mind's Graph ---

def route_from_think(state: AgentState) -> str:
    """Routes from the 'think' node to the appropriate next step."""
    decision = state["thought"].decision
    if decision == 'retrieve_knowledge':
        return "retrieve_knowledge"
    if decision == 'respond':
        return "respond"
    return END # 'escalate' decision ends the sub-graph

# --- Define and Compile the Mind's Sub-Graph ---

def create_mind_graph():
    """Factory function to create the mind's cognitive loop sub-graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate", generate_node)
    workflow.add_node("think", think_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("respond", action_node)
    
    workflow.set_entry_point("generate")
    
    workflow.add_edge("generate", "think")
    
    # The new 3-way decision point
    workflow.add_conditional_edges(
        "think",
        route_from_think,
        {
            "retrieve_knowledge": "retrieve_knowledge",
            "respond": "respond",
            END: END,
        },
    )
    
    # The new loop: after retrieving knowledge, we go back to generate a new question
    workflow.add_edge("retrieve_knowledge", "generate")
    workflow.add_edge("respond", END)
    
    return workflow.compile()
