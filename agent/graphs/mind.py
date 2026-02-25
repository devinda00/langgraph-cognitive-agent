# langgraph_cognitive_arch/agent/graphs/mind.py
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState, MindDecision

# The MIND: Fast, reactive, and responsible for triage.
mind_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    convert_system_message_to_human=True
).with_structured_output(MindDecision)

async def mind_node(state: AgentState) -> AgentState:
    """
    The first node that processes the user's input. It uses the fast 'Mind' LLM
    to reason about the query and decide on the next step. This is the entry point
    to the entire cognitive architecture.
    """
    print("---SUB-GRAPH (MIND): Analyzing request---")
    
    prompt = f"""
You are the "Mind," a fast and efficient AI assistant. Your job is to analyze the user's request and decide the best course of action.
You have two choices:
1.  **Simple Response:** If the question is simple, straightforward, or conversational, you can answer it directly.
2.  **Escalate to Brain:** If the question is complex, requires multi-step reasoning, deep analysis, coding, or creativity, you must escalate it to the "Brain."

Based on the latest user message, make your decision.

User's last message:
---
{state['messages'][-1].content}
---
"""
    
    mind_decision = await mind_llm.ainvoke(prompt)
    
    print(f"---MIND Decision:---\nReasoning: {mind_decision.reasoning}")
    if mind_decision.should_escalate:
        print("Decision: ESCALATE to Brain.")
    else:
        print("Decision: SIMPLE response.")
        
    return {"mind_decision": mind_decision}
