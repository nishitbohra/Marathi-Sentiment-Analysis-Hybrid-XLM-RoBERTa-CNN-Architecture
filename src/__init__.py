"""
Marathi Sentiment Analysis package.
"""

__version__ = "1.0.0"
__author__ = "Sentiment Analysis Research Team"

from .config import Config, get_config
from .data_loader import MarathiSentimentDataset, get_dataloaders_from_config
from .preprocessor import preprocess_marathi_text
from .train import train_model
from .evaluate import evaluate_model

__all__ = [
    'Config',
    'get_config',
    'MarathiSentimentDataset',
    'get_dataloaders_from_config',
    'preprocess_marathi_text',
    'train_model',
    'evaluate_model'
]
