"""
Script to generate embeddings from chunks_final.json and store them in Qdrant.
Uses OpenAI's text-embedding-3-large model for dense vectors and BM25 for sparse vectors.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, Modifier
import uuid
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CHUNKS_FILE = Path(__file__).parent.parent / "chunks" / "chunks_final.json"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "medical_knowledge_base")
EMBEDDING_MODEL = "text-embedding-3-large"


def load_chunks():
    """Load chunks from JSON file."""
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks


def create_collection(client: QdrantClient, vector_size: int = 3072):
    """Create Qdrant collection with both dense and sparse vector configurations."""
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                modifier=Modifier.IDF
            )
        }
    )


def create_embeddings():
    """Create embeddings from chunks and return points ready for insertion."""
    chunks = load_chunks()

    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    points = []

    for idx, chunk in enumerate(chunks):
        text = chunk["content"]

        dense_vector = embeddings_model.embed_query(text)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vector,
                "sparse": {
                    "text": text,
                    "model": "Qdrant/bm25"
                }
            },
            payload={
                "page_content": text,
                "metadata": {
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "source": chunk["source"],
                    "section_number": chunk["section_number"],
                    "section_title": chunk["section_title"],
                    "char_count": chunk["char_count"],
                    "word_count": chunk["word_count"]
                }
            }
        )
        points.append(point)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(chunks)} chunks")

    print(f"Created embeddings for {len(points)} chunks")
    return points


def populate_database(points):
    """Populate Qdrant database with the provided points."""
    client = QdrantClient(url=QDRANT_URL)

    # Delete existing collection if it exists
    collections = client.get_collections()
    if COLLECTION_NAME in [col.name for col in collections.collections]:
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(collection_name=COLLECTION_NAME)

    # Get vector size from first point
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    sample_embedding = embeddings_model.embed_query("sample")
    vector_size = len(sample_embedding)

    # Create collection
    print(f"Creating collection: {COLLECTION_NAME}")
    create_collection(client, vector_size)

    # Insert points in batches
    batch_size = 100
    total_batches = (len(points) + batch_size - 1) // batch_size

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"Inserted batch {i // batch_size + 1}/{total_batches}")

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Database populated with {collection_info.points_count} points")
    return collection_info


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("Step 1: Creating embeddings...")
    points = create_embeddings()

    print("\nStep 2: Populating database...")
    collection_info = populate_database(points)

    print(f"\nCollection created: {COLLECTION_NAME}")
    print(f"Vectors count: {collection_info.points_count}")
