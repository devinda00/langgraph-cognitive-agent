# langgraph_cognitive_arch/main.py
import asyncio
from langchain_core.messages import HumanMessage, AIMessage

from agent.router import create_agent_graph

async def run_agent():
    """The main entry point for running the cognitive agent."""
    app = create_agent_graph()
    
    print("Cognitive Agent is running. Type 'exit' to end the conversation.")
    print("Try a simple question like 'hello' and a complex one like 'what's the weather in Paris?'")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # We start the conversation with the user's message and default values
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "generated_question": "",
            "thought": None,
            "escalate_to_brain": False,
            "knowledge_base": {} # Initialize as an empty dictionary
        }
        
        final_state = None
        # The 'stream' method lets us see the output from each step
        async for step in app.astream(initial_state):
            step_name = list(step.keys())[0]
            print(f"\n---> Executing Router Step: {step_name} <---")
            final_state = step[step_name]

        print("\n---FINAL RESPONSE---")
        final_message = final_state["messages"][-1]
        # Ensure final_message is not a ToolMessage before printing
        if isinstance(final_message, AIMessage):
            print(f"{final_message.name}: {final_message.content}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_agent())
