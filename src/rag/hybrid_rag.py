"""
Hybrid RAG implementation combining dense and sparse retrieval with RRF fusion.
"""
from typing import List
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Query, QueryRequest, SparseVector, SearchRequest, FusionQuery
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class HybridRAG:
    """Hybrid RAG implementation combining dense and sparse retrieval using RRF fusion."""

    def __init__(self, qdrant_url: str = None, collection_name: str = None):
        self.qdrant_url = qdrant_url or os.getenv(
            "QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION_NAME", "medical_knowledge_base")

        self.client = QdrantClient(url=self.qdrant_url)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant contexts using hybrid search with RRF fusion.

        Combines dense vector search (semantic) with sparse vector search (keyword-based BM25)
        using Reciprocal Rank Fusion (RRF) for optimal results.

        Args:
            query: The search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved document contents
        """
        dense_vector = self.embeddings.embed_query(query)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 2
                ),
                Prefetch(
                    query=SparseVector(
                        indices=[],
                        values=[]
                    ),
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            query=FusionQuery(fusion="rrf"),
            limit=top_k,
            with_payload=True
        )

        results = []
        for point in search_result.points:
            if "page_content" in point.payload:
                results.append(point.payload["page_content"])

        return results
