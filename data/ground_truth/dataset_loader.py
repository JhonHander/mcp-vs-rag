"""
Ground truth dataset loader.
Provides utilities to load and access the question-answer pairs.
"""

import json
import os
from typing import List, Dict, Optional
import random


class GroundTruthDataset:
    """Loads and manages the ground truth question-answer dataset."""

    def __init__(self, json_path: str = None):
        """
        Initialize the dataset loader.

        Args:
            json_path: Path to questions.json file. If None, uses default path.
        """
        if json_path is None:
            # Default path relative to project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "questions.json")

        self.json_path = json_path
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """Load the JSON data."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Questions dataset not found at {self.json_path}\n"
                "Please run: python data/ground_truth/process_dataset.py"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")

    def get_all_questions(self) -> List[Dict]:
        """Get all questions from the dataset."""
        return self.data.get("questions", [])

    def get_question_by_id(self, question_id: int) -> Optional[Dict]:
        """Get a specific question by ID."""
        questions = self.get_all_questions()
        for q in questions:
            if q.get("id") == question_id:
                return q
        return None

    def get_random_question(self) -> Dict:
        """Get a random question from the dataset."""
        questions = self.get_all_questions()
        if not questions:
            raise ValueError("No questions available in dataset")
        return random.choice(questions)

    def get_random_questions(self, n: int) -> List[Dict]:
        """
        Get n random questions from the dataset.

        Args:
            n: Number of questions to return

        Returns:
            List of question dictionaries
        """
        questions = self.get_all_questions()
        if n > len(questions):
            print(
                f"WARNING: Requested {n} questions but only {len(questions)} available")
            return questions
        return random.sample(questions, n)

    def get_questions_slice(self, start: int, end: int) -> List[Dict]:
        """
        Get a slice of questions by index.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            List of question dictionaries
        """
        questions = self.get_all_questions()
        return questions[start:end]

    def total_questions(self) -> int:
        """Get total number of questions in the dataset."""
        return len(self.get_all_questions())

    def get_metadata(self) -> Dict:
        """Get dataset metadata."""
        return self.data.get("metadata", {})

    def print_summary(self):
        """Print dataset summary."""
        metadata = self.get_metadata()
        questions = self.get_all_questions()

        print("Ground Truth Dataset Summary")
        print("-" * 50)
        print(f"Total questions: {len(questions)}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Description: {metadata.get('description', 'N/A')}")

        if questions:
            print("\nSample questions:")
            for i, q in enumerate(questions[:3], 1):
                print(f"\n{i}. {q['question'][:100]}...")
                print(f"   Answer: {q['ground_truth'][:100]}...")


def load_ground_truth_dataset(json_path: str = None) -> GroundTruthDataset:
    """
    Convenience function to load the ground truth dataset.

    Args:
        json_path: Optional path to questions.json

    Returns:
        GroundTruthDataset instance
    """
    return GroundTruthDataset(json_path)


if __name__ == "__main__":
    # Test the loader
    try:
        dataset = load_ground_truth_dataset()
        dataset.print_summary()

        print("\nRandom question sample:")
        random_q = dataset.get_random_question()
        print(f"Q: {random_q['question']}")
        print(f"A: {random_q['ground_truth']}")

    except Exception as e:
        print(f"ERROR: {e}")
