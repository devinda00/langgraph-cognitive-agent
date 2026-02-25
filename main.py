# langgraph_cognitive_arch/main.py
import asyncio
from langchain_core.messages import HumanMessage, AIMessage

from agent.router import create_agent_graph

async def run_agent():
    """The main entry point for running the cognitive agent."""
    app = create_agent_graph()
    
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
            print(f"\n---> Executing step: {step_name} <---")
            
            if "messages" in step[step_name]:
                 for message in step[step_name]["messages"]:
                    if isinstance(message, AIMessage) and step_name != "mind":
                        print(f"   LLM ({message.name}): {message.content}")

            final_state = step

        print("\n---FINAL RESPONSE---")
        final_message = final_state[list(final_state.keys())[0]]["messages"][-1]
        print(f"{final_message.name}: {final_message.content}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_agent())
