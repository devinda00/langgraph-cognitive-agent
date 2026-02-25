# langgraph_cognitive_arch/agent/graphs/mind.py
# This file is now the main entry point for the entire agent graph.
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.state import AgentState, MindAction, MindGenerateAction, MindUpdateAction
from agent.tools.brain_tools import web_search_mind
from agent.graphs.brain import create_brain_graph

# --- LLMs and Tools ---
mind_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1
).with_structured_output(MindAction)

mind_generate_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
).with_structured_output(MindGenerateAction)

mind_update_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0
).with_structured_output(MindUpdateAction)

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
    
    # Check if the last log was a response, or if the conversation hasn't started
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        raise KeyboardInterrupt
    
    return {"messages": [HumanMessage(content=user_input, name="User")]}

async def generate_node(state: AgentState) -> dict:
    """
    Step 2: GENERATE
    Generates a clear internal question from the current context.
    """
    print("---MIND LOOP: Generate---")
    prompt = f"""
You are the "Mind" generator. Your job is to analyze the user's request and the conversation history to formulate clear internal questions or hypotheses that need to be answered or tested for your own self evolution.
Prioratize user's new questions.
Current Conversation:
{state['messages']}

Temp Knowledge:
{state['temp_knowledge']}

Generate a list of clear internal questions or hypotheses. If the request is a simple greeting or statement that does not require deliberation, output an empty list.
"""
    generate_action = await mind_generate_llm.ainvoke(prompt)
    print(f"---MIND Generated Questions: {generate_action.questions}---")
    
    return {"generated_questions": generate_action.questions}

async def think_node(state: AgentState) -> dict:
    """
    Step 3: THINK
    The core of the Mind. It analyzes the situation and decides on an action.
    """
    print("---MIND LOOP: Think---")
    prompt = f"""
You are the "Mind," a fast, efficient AI controller. Your job is to analyze the these questions and decide the next action. Try to self evolve by learing from the user and the world.
You have three choices:
1.  `respond_to_user`: The request is simple, conversational, or has been fully answered. Provide a direct response inside 'tool_input'.
2.  `use_mind_tool`: The request requires a quick piece of external information. Use the web search tool.
3.  `call_brain`: The request is complex and requires deep reasoning, planning, or access to long-term memory.

Current Conversation:
{state['messages']}

Generated Questions to Address:
{state.get('generated_questions', [])}

Temp Knowledge:
{state['temp_knowledge']}

Based on these questions and the current context, what is your reasoning and the next action to take?
"""
    mind_action = await mind_llm.ainvoke(prompt)
    print(f"---MIND Thought: {mind_action.reasoning} -> Action: {mind_action.action}---")
    
    # ToolNode expects an AIMessage with a tool_call array to be in the messages list
    state_updates = {"mind_action": mind_action}
    if mind_action.action == "use_mind_tool" and mind_action.tool_input:
        import uuid
        tool_call_id = str(uuid.uuid4())
        tool_msg = AIMessage(
            content="", 
            tool_calls=[{"name": "web_search_mind", "args": {"query": mind_action.tool_input}, "id": tool_call_id}]
        )
        state_updates["messages"] = [tool_msg]
        
    return state_updates

async def update_node(state: AgentState) -> dict:
    """
    Step 5: UPDATE
    Updates the temporary knowledge base with the results of the last action.
    """
    print("---MIND LOOP: Update---")
    
    if not state.get('messages'):
        return {}
        
    last_message = state['messages'][-1]
    
    prompt = f"""
You are the "Mind" memory updater. Your job is to extract any new, relevant facts, context, or insights from the latest message exchange and return them as key-value pairs to store in working memory (temp_knowledge).

Current Temp Knowledge:
{state.get('temp_knowledge', {})}

Latest Message to Analyze:
{last_message.content if hasattr(last_message, 'content') else str(last_message)}

Extract key information. If the latest message does not contain any new factual information worth remembering for the short term, return an empty dictionary.
"""
    update_action = await mind_update_llm.ainvoke(prompt)
    
    if update_action.insights:
        print(f"---MIND Extracted Insights: {update_action.insights}---")
        # Merge new insights into the existing temp_knowledge dictionary
        current_knowledge = state.get('temp_knowledge', {})
        current_knowledge.update(update_action.insights)
        return {"temp_knowledge": current_knowledge}
    
    return {}

# --- Action-related nodes ---

async def respond_to_user_node(state: AgentState) -> dict:
    """The node that formulates and returns the final response to the user."""
    content = state['mind_action'].tool_input or "I'm not sure how to respond to that."
    return {"messages": [AIMessage(content=content, name="Mind")]}

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
    workflow.add_node("generate", generate_node)
    workflow.add_node("think", think_node)
    
    # Action nodes
    workflow.add_node("mind_tools", mind_tool_node)
    workflow.add_node("brain", brain_sub_graph) # The brain is a node
    workflow.add_node("respond_to_user", respond_to_user_node)

    workflow.add_node("update", update_node)

    workflow.set_entry_point("load")
    workflow.add_edge("load", "generate")
    workflow.add_edge("generate", "think")
    
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
    workflow.add_edge("update", "load") # The main loop!

    workflow.add_edge("respond_to_user", "update")

    return workflow.compile()
