"""
Script to generate embeddings from chunks_final.json and store them in Qdrant.
Uses OpenAI's text-embedding-3-large model.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configuration
CHUNKS_FILE = Path(__file__).parent.parent / "chunks" / "chunks_final.json"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "medical_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-large"


def load_chunks():
    """Load chunks from JSON file."""
    print(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def create_documents(chunks):
    """Convert chunks to LangChain Document objects with metadata."""
    print("Converting chunks to Document objects...")
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "source": chunk["source"],
                "section_number": chunk["section_number"],
                "section_title": chunk["section_title"],
                "char_count": chunk["char_count"],
                "word_count": chunk["word_count"]
            }
        )
        documents.append(doc)
    print(f"Created {len(documents)} Document objects")
    return documents


def create_embeddings_and_store():
    """Main function to create embeddings and store in Qdrant."""
    # Load chunks
    chunks = load_chunks()

    # Create Document objects
    documents = create_documents(chunks)

    # Initialize OpenAI embeddings
    print(f"Initializing OpenAI embeddings with model: {EMBEDDING_MODEL}...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Create Qdrant vector store from documents
    print(f"Creating Qdrant vector store at {QDRANT_URL}...")
    print(f"Collection name: {COLLECTION_NAME}")
    print("This may take several minutes depending on the number of chunks...")

    qdrant = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True  # Recreate collection if it exists
    )

    print("Successfully created embeddings and stored in Qdrant!")
    print(f"Collection '{COLLECTION_NAME}' is ready for use")

    # Verify by doing a simple search
    print("\nVerifying with a test search...")
    test_results = qdrant.similarity_search("control prenatal", k=3)
    print(f"Found {len(test_results)} results for test query")
    for i, doc in enumerate(test_results, 1):
        print(f"\n  Result {i}:")
        print(f"    Content preview: {doc.page_content[:100]}...")
        print(f"    Metadata: {doc.metadata}")


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Embeddings and Storing in Qdrant")
    print("=" * 60)

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    create_embeddings_and_store()

    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)
