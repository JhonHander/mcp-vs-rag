"""
Test script for NaiveRAG and HybridRAG implementations.
"""
from src.rag.hybrid_rag import HybridRAG
from src.rag.naive_rag import NaiveRAG
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_naive_rag():
    """Test NaiveRAG retrieval."""
    print("=" * 70)
    print("Testing Naive RAG (Dense Vector Search)")
    print("=" * 70)

    rag = NaiveRAG()

    queries = [
        "¿Qué profesional debe llevar a cabo el control prenatal?",
        "¿Cuál es el tratamiento recomendado para las náuseas en el embarazo?",
        "¿Qué vacunas se deben aplicar durante el embarazo?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 70)

        results = rag.retrieve(query, top_k=3)

        for j, result in enumerate(results, 1):
            print(f"\nResult {j}:")
            print(result[:200] + "..." if len(result) > 200 else result)

        print("\n")


def test_hybrid_rag():
    """Test HybridRAG retrieval."""
    print("=" * 70)
    print("Testing Hybrid RAG (Dense + Sparse with RRF Fusion)")
    print("=" * 70)

    rag = HybridRAG()

    queries = [
        "¿Qué profesional debe llevar a cabo el control prenatal?",
        "¿Cuál es el tratamiento recomendado para las náuseas en el embarazo?",
        "¿Qué vacunas se deben aplicar durante el embarazo?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 70)

        results = rag.retrieve(query, top_k=3)

        for j, result in enumerate(results, 1):
            print(f"\nResult {j}:")
            print(result[:200] + "..." if len(result) > 200 else result)

        print("\n")


def compare_rag_methods():
    """Compare both RAG methods side by side."""
    print("=" * 70)
    print("Comparing Naive RAG vs Hybrid RAG")
    print("=" * 70)

    naive_rag = NaiveRAG()
    hybrid_rag = HybridRAG()

    query = "¿Cuáles son las intervenciones recomendadas para el tratamiento del dolor lumbar?"

    print(f"\nQuery: {query}")
    print("=" * 70)

    print("\nNaive RAG Results (Dense Only):")
    print("-" * 70)
    naive_results = naive_rag.retrieve(query, top_k=3)
    for i, result in enumerate(naive_results, 1):
        print(f"\n{i}. {result[:150]}...")

    print("\n" + "=" * 70)
    print("\nHybrid RAG Results (Dense + Sparse + RRF):")
    print("-" * 70)
    hybrid_results = hybrid_rag.retrieve(query, top_k=3)
    for i, result in enumerate(hybrid_results, 1):
        print(f"\n{i}. {result[:150]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "naive":
            test_naive_rag()
        elif mode == "hybrid":
            test_hybrid_rag()
        elif mode == "compare":
            compare_rag_methods()
        else:
            print("Usage: python test_rag.py [naive|hybrid|compare]")
    else:
        print("Running all tests...\n")
        test_naive_rag()
        print("\n" * 2)
        test_hybrid_rag()
        print("\n" * 2)
        compare_rag_methods()
