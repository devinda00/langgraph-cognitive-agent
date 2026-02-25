
import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- Load API Keys ---
# Create a .env file in this directory and add your keys
# OPENAI_API_KEY="sk-..."
# GROQ_API_KEY="gsk_..."
load_dotenv()

# --- 1. Define the State for our Graph ---
# This is the memory of our agent. It's a dictionary that holds the
# conversation history and the Mind's decision.

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    mind_decision: "MindDecision" 

# --- 2. Define the Pydantic model for the Mind's structured output ---
# This forces the "Mind" LLM to think in a structured way and make a clear decision.

class MindDecision(BaseModel):
    """
    The structured output from the Mind's reasoning process.
    The Mind must decide whether the user's query is simple enough
    for an immediate answer or if it requires deeper thought from the Brain.
    """
    reasoning: str = Field(description="The thought process and reasoning behind the decision.")
    should_escalate: bool = Field(description="True if the query is complex and requires the Brain's full reasoning power. False otherwise.")
    simple_response: str = Field(description="If should_escalate is False, provide a direct, simple answer here. Otherwise, this can be empty.", default="")

# --- 3. Define the Models: Mind and Brain ---

# The MIND: Fast, reactive, and responsible for triage.
# We use Groq's Llama3-8b for its incredible speed.
mind_llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
).with_structured_output(MindDecision)

# The BRAIN: Powerful, deliberative, and used for complex reasoning.
# We use OpenAI's GPT-4o for its advanced capabilities.
brain_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# --- 4. Define the Nodes of the Graph ---

async def mind_node(state: AgentState) -> AgentState:
    """
    The first node that processes the user's input.
    It uses the fast 'Mind' LLM to reason about the query and decide on the next step.
    """
    print("---MIND: Analyzing request---")
    
    # The prompt for the Mind tells it its role: be a fast, intelligent router.
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
    
    # Invoke the Mind LLM to get a structured decision
    mind_decision = await mind_llm.ainvoke(prompt)
    
    print(f"---MIND Decision:---\nReasoning: {mind_decision.reasoning}")
    if mind_decision.should_escalate:
        print("Decision: ESCALATE to Brain.")
        # Add a message to show the Mind is passing the task to the Brain
        escalation_message = AIMessage(content="That's a complex question. Let me think more deeply about it.", name="Mind")
        return {"messages": [escalation_message], "mind_decision": mind_decision}
    else:
        print("Decision: SIMPLE response.")
        # If the Mind can handle it, provide its simple response directly
        simple_message = AIMessage(content=mind_decision.simple_response, name="Mind")
        return {"messages": [simple_message], "mind_decision": mind_decision}

async def brain_node(state: AgentState) -> AgentState:
    """
    The node for complex, deliberative thought.
    It uses the powerful 'Brain' LLM to provide a comprehensive answer.
    """
    print("---BRAIN: Engaging for complex reasoning---")
    
    # The Brain gets the full conversation history to understand the context.
    response = await brain_llm.ainvoke(state["messages"])
    
    # The Brain's response is the final answer.
    return {"messages": [response]}

# --- 5. Define the Conditional Edge for Triage ---

def triage(state: AgentState) -> str:
    """
    The conditional router. Based on the Mind's decision, it decides whether
    to route to the 'brain' node or end the process.
    """
    print("---TRIAGE: Routing based on Mind's decision---")
    if state["mind_decision"].should_escalate:
        return "brain"
    else:
        return END

# --- 6. Assemble the Graph ---

workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("mind", mind_node)
workflow.add_node("brain", brain_node)

# Define the edges
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

# Compile the graph into a runnable app
app = workflow.compile()

# --- 7. Run the Agent ---

async def run_agent():
    print("Cognitive Agent is running. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # We start the conversation with the user's message
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        
        final_state = None
        # The 'stream' method lets us see the output from each step
        async for step in app.astream(initial_state):
            step_name = list(step.keys())[0]
            step_output = step[step_name]
            
            print(f"\n-> Executed node: {step_name}")
            
            if "messages" in step_output:
                for message in step_output["messages"]:
                    if isinstance(message, AIMessage):
                        print(f"   LLM ({message.name}): {message.content}")

            final_state = step_output


        print("\n---FINAL RESPONSE---")
        final_message = final_state["messages"][-1]
        print(f"{final_message.name}: {final_message.content}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_agent())
