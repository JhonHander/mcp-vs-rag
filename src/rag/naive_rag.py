"""
Naive RAG implementation using dense vector similarity search.
"""
from typing import List
import os
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class NaiveRAG:
    """Naive RAG implementation with simple dense vector similarity search."""

    def __init__(self, qdrant_url: str = None, collection_name: str = None):
        self.qdrant_url = qdrant_url or os.getenv(
            "QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION_NAME", "medical_knowledge_base")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        self.vectorstore = Qdrant.from_existing_collection(
            embedding=self.embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name,
            vector_name="dense"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant contexts using dense vector similarity search.

        Args:
            query: The search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved document contents
        """
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]
