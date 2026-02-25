# langgraph_cognitive_arch/agent/tools/brain_tools.py
from langchain_core.tools import tool
from agent.state import AgentState

@tool
def access_temp_knowledge(state: AgentState, key: str) -> str:
    """Accesses a value from the temporary knowledge base (a dictionary)."""
    print(f"---BRAIN TOOL: Accessing Temp Knowledge for key: '{key}'---")
    return state['temp_knowledge'].get(key, f"No value found for key '{key}' in temporary knowledge.")

@tool
def access_permanent_knowledge(state: AgentState, query: str) -> str:
    """Recalls relevant information from the permanent knowledge vector database."""
    print(f"---BRAIN TOOL: Accessing Permanent Knowledge for query: '{query}'---")
    return state['permanent_knowledge'].recall_memory(query)

@tool
def write_to_permanent_knowledge(state: AgentState, content: str) -> str:
    """Writes a new, distilled piece of information to the permanent knowledge vector database."""
    print(f"---BRAIN TOOL: Writing to Permanent Knowledge: '{content}'---")
    state['permanent_knowledge'].add_memory(content)
    return f"Successfully wrote '{content}' to permanent knowledge."

# A simple tool for the Mind as well
@tool
def web_search_mind(query: str) -> str:
    """A simple web search tool for the Mind to answer factual questions."""
    # Note: This reuses the logic from the old web_search tool but is a distinct tool.
    from duckduckgo_search import DDGS
    print(f"---MIND TOOL: Performing web search for '{query}'---")
    with DDGS() as ddgs:
        results = [r['body'] for r in ddgs.text(query, max_results=3)]
    return "\n".join(results) if results else "No results found."
