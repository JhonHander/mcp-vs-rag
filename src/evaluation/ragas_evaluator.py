from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness
from datasets import Dataset
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()


class RAGASEvaluator:
    """Evaluator using RAGAS metrics for RAG system assessment."""

    def __init__(self):
        self.metrics = [AnswerRelevancy(), Faithfulness()]

    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a single response using RAGAS metrics.

        Args:
            question: The input question
            answer: Generated answer from the system  
            contexts: Retrieved contexts used for generation

        Returns:
            Dictionary with metric scores
        """
        try:
            # Create dataset in the format expected by RAGAS
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            })

            # Run evaluation
            result = evaluate(dataset=dataset, metrics=self.metrics)

            return {
                "answer_relevancy": float(result["answer_relevancy"]),
                "faithfulness": float(result["faithfulness"])
            }

        except Exception as e:
            print(f"Error in RAGAS evaluation: {e}")
            return {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "error": str(e)
            }

    def batch_evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]]
    ) -> List[Dict[str, float]]:
        """Evaluate multiple responses in batch."""
        results = []

        for question, answer, contexts in zip(questions, answers, contexts_list):
            result = self.evaluate_response(question, answer, contexts)
            results.append(result)

        return results
