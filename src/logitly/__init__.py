"""logitly: closed questions for LLMs, answered from the logits and calibrated.

HuggingFaceBackend is not exported here: it is imported on demand from
logitly.backends.huggingface, so an API-only install needs no torch.
"""

from logitly.backends import Backend, OpenAIBackend
from logitly.calibration import Calibration, CalibrationReport, Example, Metrics, fit_calibration
from logitly.config import ApiConfig, CalibrationConfig, FitConfig, PromptConfig
from logitly.core import Answer, Bool, BoolAnswer, Choice, Question, Score, ScoreAnswer
from logitly.errors import (
    BackendError,
    CalibrationError,
    LogitlyError,
    QuestionError,
    TokenizationError,
)
from logitly.model import Logitly

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "ApiConfig",
    "Backend",
    "BackendError",
    "Bool",
    "BoolAnswer",
    "Calibration",
    "CalibrationConfig",
    "CalibrationError",
    "CalibrationReport",
    "Choice",
    "Example",
    "FitConfig",
    "Logitly",
    "LogitlyError",
    "Metrics",
    "OpenAIBackend",
    "PromptConfig",
    "Question",
    "QuestionError",
    "Score",
    "ScoreAnswer",
    "TokenizationError",
    "__version__",
    "fit_calibration",
]
