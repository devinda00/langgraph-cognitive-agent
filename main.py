# langgraph_cognitive_arch/main.py
import asyncio
from langchain_core.messages import HumanMessage, AIMessage

from agent.graphs.mind import create_agent_graph
from agent.permanent_knowledge import PERMANENT_KNOWLEDGE

async def run_agent():
    """The main entry point for running the cognitive agent."""
    # The new unified graph is created from the mind module
    app = create_agent_graph()
    
    # Pre-populate permanent knowledge with a piece of information
    PERMANENT_KNOWLEDGE.add_memory("Clawed is a helpful AI assistant created by Devinda.")
    
    print("Cognitive Agent is running. Type 'exit' to end the conversation.")
    print("Example simple query: 'Hello, who are you?'")
    print("Example complex query requiring brain: 'Recall your identity.'")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # We start the conversation with the user's message and the knowledge stores
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "temp_knowledge": {},
            "permanent_knowledge": PERMANENT_KNOWLEDGE,
            "generated_questions": [],
            "brain_thought": None,
            "mind_action": None
        }
        
        final_state = None
        # The 'stream' method lets us see the output from each step
        async for step in app.astream(initial_state):
            step_name = list(step.keys())[0]
            final_state = step[step_name]
            
            print(f"\n---> Executing Step: {step_name} <---")
            print("Full State:", final_state)


        print("\n---FINAL RESPONSE---")
        if final_state and final_state.get("messages"):
            final_message = final_state["messages"][-1]
            if isinstance(final_message, AIMessage):
                print(f"{final_message.name}: {final_message.content}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_agent())
