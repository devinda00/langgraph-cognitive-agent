# langgraph_cognitive_arch/agent/permanent_knowledge.py
import faiss
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import os

class VectorStore:
    """A wrapper for a FAISS vector store to manage Permanent Knowledge."""
    def __init__(self):
        print("---INITIALIZING Permanent Knowledge (Vector DB with Google Embeddings)---")
        
        # Ensure the API key is available
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
            
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Dimension of the Google embedding model
        embedding_size = 768 
        index = faiss.IndexFlatL2(embedding_size)
        
        # In-memory document store
        docstore = InMemoryDocstore({})
        
        self.vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id={}
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
