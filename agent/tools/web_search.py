# langgraph_cognitive_arch/agent/tools/web_search.py
from langchain_core.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    """
    Run a web search using DuckDuckGo to get information from the internet.
    Use this for simple, factual questions.
    """
    print(f"---TOOL: Performing web search for '{query}'---")
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
    return "\n".join(results) if results else "No results found."
