"""
Ground truth dataset package.
Provides access to the question-answer dataset for evaluation.
"""

from .dataset_loader import (
    GroundTruthDataset,
    load_ground_truth_dataset
)

__all__ = [
    'GroundTruthDataset',
    'load_ground_truth_dataset'
]
