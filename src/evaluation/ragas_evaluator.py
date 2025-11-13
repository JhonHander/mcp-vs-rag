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
            llm: Optional LangChain LLM instance. If None, uses default OpenAI GPT-4o-mini.
        """
        if llm is None:
            # Use GPT-4o-mini: fastest and cheapest model for RAGAS evaluations
            # Optimized settings to avoid timeouts and slow evaluations
            base_llm = ChatOpenAI(
                model="gpt-4o-mini",  # Correct model name
                temperature=0.1,
                request_timeout=60,  # 60 second timeout per request (increased from 30s)
                max_retries=3,  # Retry 3 times to handle network issues
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
            # Basic validation
            if not contexts:
                return {"answer_relevancy": 0.0, "faithfulness": 0.0}

            # Clean contexts (handle nested lists)
            cleaned_contexts = []
            for ctx in contexts:
                if isinstance(ctx, list):
                    cleaned_contexts.extend([str(c) for c in ctx if c])
                else:
                    cleaned_contexts.append(str(ctx))

            # Create dataset
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [cleaned_contexts]
            })

            # Run evaluation
            result = evaluate(dataset=dataset, metrics=self.metrics)
            df = result.to_pandas()

            return {
                "answer_relevancy": float(df['answer_relevancy'].iloc[0]),
                "faithfulness": float(df['faithfulness'].iloc[0])
            }

        except Exception as e:
            print(f"❌ Error in RAGAS evaluation: {e}")
            return {"answer_relevancy": 0.0, "faithfulness": 0.0}

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
