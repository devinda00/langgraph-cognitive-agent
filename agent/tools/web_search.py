# langgraph_cognitive_arch/agent/tools/web_search.py
from langchain_core.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    """
    Run a web search using DuckDuckGo to get information from the internet.
    """
    print(f"---TOOL: Performing web search for '{query}'---")
    with DDGS() as ddgs:
        # Using generator to get results, then formatting them
        results_gen = ddgs.text(query, max_results=3)
        results = [r for r in results_gen] if results_gen else []
        
    if not results:
        return "No results found."
        
    # Format results for the LLM
    formatted_results = "\n\n".join(
        [f"Title: {res['title']}\nSnippet: {res['body']}" for res in results]
    )
    return f"Search Results for '{query}':\n\n{formatted_results}"
