# langgraph_cognitive_arch/agent/graphs/mind.py
# This file is now the main entry point for the entire agent graph.
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.state import AgentState, MindAction, MindGenerateAction, MindUpdateAction
from agent.tools.brain_tools import web_search_mind
from agent.graphs.brain import create_brain_graph
from agent.logging_config import get_logger
from agent.message_bus import bus, MessageType
from agent.prompts import PROMPTS, update_prompt, list_prompt_keys
from agent.config import create_llm

import asyncio

log = get_logger("mind")

# --- Tools ---
mind_tool_node = ToolNode([web_search_mind])
brain_sub_graph = create_brain_graph() # The entire brain is a "tool" for the mind

# --- Nodes for the Mind's Cognitive Loop ---

async def load_node(state: AgentState) -> dict:
    """
    Step 1: LOAD
    Non-blocking: checks for user input. If there is one, flag it.
    If not, the Mind continues its autonomous cognitive loop.
    """
    log.info("MIND LOOP: Load — checking for user input")

    msg = bus.try_get_user_input(timeout=0.3)

    if msg is not None:
        if msg.type == MessageType.SHUTDOWN:
            raise KeyboardInterrupt
        log.info("Received user message: %s", msg.content)
        return {
            "messages": [HumanMessage(content=msg.content, name="User")],
            "has_new_input": True,
        }

    # No user input — continue autonomously
    log.info("No user input, continuing autonomous loop")
    return {"has_new_input": False}

async def generate_node(state: AgentState) -> dict:
    """
    Step 2: GENERATE
    Generates a clear internal question from the current context.
    """
    log.info("MIND LOOP: Generate")
    bus.send_agent_status("Generating questions...")
    prompt = PROMPTS["mind_generate"].format(
        messages=state['messages'],
        temp_knowledge=state['temp_knowledge'],
    )
    llm = create_llm("mind_generate_llm", MindGenerateAction)
    generate_action = await llm.ainvoke(prompt)
    log.info("MIND Generated Questions: %s", generate_action.questions)
    
    return {"generated_questions": generate_action.questions}


async def think_node(state: AgentState) -> dict:
    """
    Step 3: THINK
    Reviews the generated questions and decides on the next single action.
    """
    log.info("MIND LOOP: Think")
    bus.send_agent_status("Thinking...")
    prompt = PROMPTS["mind_think"].format(
        generated_questions=state.get('generated_questions', []),
        temp_knowledge=state['temp_knowledge'],
        recent_messages=state['messages'][-5:],
    )
    llm = create_llm("mind_llm", MindAction)
    mind_thought = await llm.ainvoke(prompt)
    log.info("MIND Thought: %s -> Action: %s", mind_thought.reasoning, mind_thought.action)
    return {"mind_thought": mind_thought}


async def action_node(state: AgentState) -> dict:
    """
    Step 3: ACTION
    The core of the Mind. It analyzes the situation and decides on an action.
    """
    log.info("MIND LOOP: Action")
    bus.send_agent_status("Taking action...")

    has_input = state.get("has_new_input", False)
    respond_allowed = (
        " (ALLOWED — there is a pending user message)" if has_input
        else " (NOT ALLOWED right now — no new user message. Choose call_brain, use_mind_tool, or idle instead.)"
    )
    mind_thought = state.get('mind_thought')
    thought_reasoning = mind_thought.reasoning if mind_thought else "No thoughts generated yet."
    thought_action = mind_thought.action if mind_thought else "none"
    prompt = PROMPTS["mind_action"].format(
        respond_allowed=respond_allowed,
        recent_messages=state['messages'][-5:],
        temp_knowledge=state['temp_knowledge'],
        thought_reasoning=thought_reasoning,
        thought_action=thought_action,
    )
    llm = create_llm("mind_llm", MindAction)
    mind_action = await llm.ainvoke(prompt)
    log.info("MIND Thought: %s -> Action: %s", mind_action.reasoning, mind_action.action)
    
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
    log.info("MIND LOOP: Update")
    
    if not state.get('messages'):
        return {}
        
    last_message = state['messages'][-1]
    prompt = PROMPTS["mind_update"].format(
        temp_knowledge=state.get('temp_knowledge', {}),
        last_message=last_message.content if hasattr(last_message, 'content') else str(last_message),
        editable_prompts=list_prompt_keys(),
    )
    llm = create_llm("mind_update_llm", MindUpdateAction)
    update_action = await llm.ainvoke(prompt)
    
    state_updates = {}

    if update_action.insights:
        log.info("MIND Extracted Insights: %s", update_action.insights)
        current_knowledge = state.get('temp_knowledge', {})
        current_knowledge.update(update_action.insights)
        state_updates["temp_knowledge"] = current_knowledge

    # Apply any self-evolution prompt updates
    if update_action.prompt_updates:
        for pu in update_action.prompt_updates:
            try:
                result = update_prompt(pu.key, pu.new_template, reason=pu.reason)
                log.info("MIND SELF-EVOLVE: %s — Reason: %s", result, pu.reason)
            except ValueError as e:
                log.warning("MIND SELF-EVOLVE rejected: %s", e)

    return state_updates

# --- Action-related nodes ---

async def respond_to_user_node(state: AgentState) -> dict:
    """The node that formulates and returns the final response to the user."""
    content = state['mind_action'].tool_input or "I'm not sure how to respond to that."
    log.info("MIND responding to user: %s", content)
    bus.send_agent_response(content, sender="Mind")
    return {"messages": [AIMessage(content=content, name="Mind")], "has_new_input": False}

async def idle_node(state: AgentState) -> dict:
    """
    The Mind has nothing to do. Sleep briefly before looping back to load
    so we don't burn CPU / API calls in a tight loop.
    """
    log.info("MIND LOOP: Idle — sleeping 2s before next check")
    await asyncio.sleep(2)
    return {}

# --- Main Conditional Router ---

def route_action(state: AgentState) -> str:
    """The main router for the entire agent, driven by the Mind's decision."""
    action = state['mind_action'].action
    has_input = state.get('has_new_input', False)

    if action == 'call_brain':
        return 'brain'
    elif action == 'use_mind_tool':
        return 'mind_tools'
    elif action == 'respond_to_user' and has_input:
        return 'respond_to_user'
    else:
        # idle, or respond_to_user without input → go idle
        return 'idle'

# --- Define and Compile the Unified Agent Graph ---

def create_agent_graph():
    """This function now creates the entire, unified agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("load", load_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("think", think_node)
    workflow.add_node("action", action_node)
    
    # Action nodes
    workflow.add_node("mind_tools", mind_tool_node)
    workflow.add_node("brain", brain_sub_graph) # The brain is a node
    workflow.add_node("respond_to_user", respond_to_user_node)
    workflow.add_node("idle", idle_node)

    workflow.add_node("update", update_node)

    workflow.set_entry_point("load")
    workflow.add_edge("load", "generate")
    workflow.add_edge("generate", "think")
    workflow.add_edge("think", "action")
    # The Action router
    workflow.add_conditional_edges(
        "action",
        route_action,
        {
            "mind_tools": "mind_tools",
            "brain": "brain",
            "respond_to_user": "respond_to_user",
            "idle": "idle",
        }
    )
    
    # After an action, update and loop back to think
    workflow.add_edge("mind_tools", "update")
    workflow.add_edge("brain", "update")
    workflow.add_edge("update", "load") # The main loop!
    workflow.add_edge("idle", "load")   # Idle loops back without update

    workflow.add_edge("respond_to_user", "update")

    return workflow.compile()
