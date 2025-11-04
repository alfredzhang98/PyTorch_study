"""Simple LSTM text classification package.

Exports:
- LSTMClassifier: A PyTorch nn.Module for sequence classification using LSTM.
"""

from .lstm import LSTMClassifier

__all__ = ["LSTMClassifier"]
__version__ = "0.1.0"

