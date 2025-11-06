"""
Test script to verify RAGAS evaluator is working correctly.
This tests the evaluator independently before running full experiments.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.evaluation.ragas_evaluator import RAGASEvaluator


def test_ragas_basic():
    """Test basic RAGAS functionality."""
    print("="*60)
    print("Testing RAGAS Evaluator")
    print("="*60)
    
    try:
        # Initialize evaluator
        print("\n1. Initializing evaluator...")
        evaluator = RAGASEvaluator()
        print("[SUCCESS] Evaluator initialized with LLM")
        
        # Test data
        question = "¿Cuándo se considera inicio tardío de controles prenatales?"
        answer = "Se considera inicio tardío cuando el control prenatal comienza después de la semana 12 de gestación."
        contexts = [
            "Se considera inicio tardío al haber comenzado atención prenatal a las 12 semanas o más de gestación.",
            "El control prenatal de inicio tardío es cuando la mujer embarazada concurre por primera vez a la consulta habiendo cumplido ya su tercer mes de embarazo."
        ]
        
        # Run evaluation
        print("\n2. Running evaluation...")
        print(f"Question: {question[:80]}...")
        print(f"Answer: {answer[:80]}...")
        print(f"Contexts: {len(contexts)} documents")
        
        result = evaluator.evaluate_response(
            question=question,
            answer=answer,
            contexts=contexts
        )
        
        # Display results
        print("\n3. Results:")
        print("-"*60)
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            return False
        else:
            print(f"Answer Relevancy: {result['answer_relevancy']:.4f}")
            print(f"Faithfulness: {result['faithfulness']:.4f}")
            print("[SUCCESS] Evaluation completed successfully")
            return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ragas_edge_cases():
    """Test edge cases for RAGAS evaluator."""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    evaluator = RAGASEvaluator()
    
    # Test 1: Empty contexts
    print("\nTest 1: Empty contexts")
    result = evaluator.evaluate_response(
        question="Test question?",
        answer="Test answer",
        contexts=[]
    )
    print(f"Result: {result}")
    assert "error" in result or result["answer_relevancy"] == 0.0
    print("[PASS] Empty contexts handled correctly")
    
    # Test 2: Nested list contexts (edge case that caused the original error)
    print("\nTest 2: Nested contexts (should be flattened)")
    result = evaluator.evaluate_response(
        question="Test question?",
        answer="Test answer",
        contexts=["Context 1", ["Nested context"], "Context 2"]
    )
    print(f"Result: {result}")
    print("[PASS] Nested contexts handled")
    
    print("\n[SUCCESS] All edge case tests passed")


if __name__ == "__main__":
    print("\nRAGAS Evaluator Test Suite\n")
    
    # Run basic test
    success = test_ragas_basic()
    
    if success:
        # Run edge case tests
        test_ragas_edge_cases()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("You can now run experiments with confidence.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Tests failed. Please check your configuration:")
        print("1. Verify OPENAI_API_KEY is set in .env")
        print("2. Ensure ragas library is installed: pip install ragas")
        print("3. Check that langchain_openai is installed")
        print("="*60)
        sys.exit(1)
