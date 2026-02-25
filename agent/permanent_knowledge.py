# langgraph_cognitive_arch/agent/permanent_knowledge.py
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import os

class VectorStore:
    """
    A wrapper for a lightweight, in-memory vector store to manage Permanent Knowledge.
    This version uses LangChain's InMemoryVectorStore to avoid heavy dependencies like FAISS.
    """
    def __init__(self):
        print("---INITIALIZING Permanent Knowledge (In-Memory Vector DB with Google Embeddings)---")
        
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
            
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-latest")
        
        # InMemoryVectorStore is initialized with the embedding function and can be empty.
        self.vector_store = InMemoryVectorStore.from_texts(
            texts=[], # Start with an empty store
            embedding=embedding_model
        )

    def add_memory(self, content: str, metadata: dict = None):
        """Adds a new piece of information to the permanent knowledge."""
        print(f"---PK: Adding memory: '{content}'---")
        self.vector_store.add_documents([Document(page_content=content, metadata=metadata or {})])

    def recall_memory(self, query: str, k: int = 2) -> str:
        """Recalls the most relevant memories based on a query."""
        print(f"---PK: Recalling memory for query: '{query}'---")
        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            return "No relevant memories found."
        return "\n".join([f"- {doc.page_content}" for doc in results])

# Global instance to be used by the agent
PERMANENT_KNOWLEDGE = VectorStore()
