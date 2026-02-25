import os
import sys

# Ensure the parent directory is in the path
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()
print("API KEY:", bool(os.getenv("GOOGLE_API_KEY")))

import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from agent.state import AgentState
from agent.permanent_knowledge import PERMANENT_KNOWLEDGE

async def run_update():
    from agent.graphs.brain import update_node
    state = {
        "messages": [
            HumanMessage(content="What is Devinda's favorite database?", name="User"),
            AIMessage(content="Devinda really likes PostgreSQL for relational data.", name="Brain")
        ],
        "temp_knowledge": {},
        "permanent_knowledge": PERMANENT_KNOWLEDGE,
        "generated_questions": [],
        "brain_thought": None,
        "mind_action": None
    }
    
    try:
        print("Testing brain's update_node...")
        res = await update_node(state)
        print("RESULT:")
        print(res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_update())
