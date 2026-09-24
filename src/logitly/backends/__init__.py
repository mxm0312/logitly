"""Inference backends.

HuggingFaceBackend lives in logitly.backends.huggingface and is imported on
demand, so an API-only install needs no torch.
"""

from logitly.backends.api import OpenAIBackend
from logitly.backends.base import Backend

__all__ = ["Backend", "OpenAIBackend"]
