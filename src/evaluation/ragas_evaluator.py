from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
from typing import Dict, List
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class RAGASEvaluator:
    """Evaluator using RAGAS metrics for RAG system assessment."""

    def __init__(self, llm=None):
        """
        Initialize RAGAS evaluator with LLM.

        Args:
            llm: Optional LangChain LLM instance. If None, uses default OpenAI GPT-4.
        """
        if llm is None:
            # Use GPT-4 as default for evaluation
            base_llm = ChatOpenAI(
                model="gpt-4.1",
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            base_llm = llm

        # Wrap LLM for RAGAS
        evaluator_llm = LangchainLLMWrapper(base_llm)

        # Initialize metrics with LLM
        self.metrics = [
            AnswerRelevancy(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm)
        ]

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
            # Validate and clean contexts
            if not contexts:
                print("Warning: Empty contexts provided for evaluation")
                return {
                    "answer_relevancy": 0.0,
                    "faithfulness": 0.0,
                    "error": "Empty contexts"
                }

            # Ensure contexts is a flat list of strings
            cleaned_contexts = []
            for ctx in contexts:
                if isinstance(ctx, str):
                    cleaned_contexts.append(ctx)
                elif isinstance(ctx, list):
                    # Flatten nested lists
                    cleaned_contexts.extend([str(c) for c in ctx if c])
                else:
                    cleaned_contexts.append(str(ctx))

            if not cleaned_contexts:
                print("Warning: No valid contexts after cleaning")
                return {
                    "answer_relevancy": 0.0,
                    "faithfulness": 0.0,
                    "error": "No valid contexts"
                }

            # Create dataset in the format expected by RAGAS
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [cleaned_contexts]
            })

            # Run evaluation
            # evaluate() returns an EvaluationResult object
            result = evaluate(dataset=dataset, metrics=self.metrics)

            # Convert EvaluationResult to pandas DataFrame to extract scores
            # According to RAGAS docs, this is the correct way to access individual scores
            df = result.to_pandas()

            # Extract scalar values from the first (and only) row
            # The DataFrame has one row per evaluation sample
            answer_relevancy = float(df['answer_relevancy'].iloc[0])
            faithfulness = float(df['faithfulness'].iloc[0])

            return {
                "answer_relevancy": answer_relevancy,
                "faithfulness": faithfulness
            }

        except Exception as e:
            print(f"Error in RAGAS evaluation: {e}")
            import traceback
            traceback.print_exc()
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
