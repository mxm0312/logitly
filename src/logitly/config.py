"""Every tunable default in one place."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LabelStyle = Literal["letter", "number"]
Scoring = Literal["letter", "text"]

DEFAULT_DIRECTIVE = "Answer with the label of the single best option, and nothing else."
DEFAULT_BATCH_SIZE = 8


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PromptConfig(Frozen):
    """How a question is laid out in text, and how options are scored.

    Defaults suit instruction-tuned chat models; override for a model fine-tuned
    on a different answer format.
    """

    chat: bool | Literal["auto"] = "auto"
    label_style: LabelStyle = "letter"
    answer_prefix: str | None = Field(
        default=None,
        description='Where the answer starts: "" in chat mode, "Answer:" otherwise.',
    )
    directive: str = DEFAULT_DIRECTIVE
    input_label: str = "Input"
    length_normalize: bool = Field(
        default=True,
        description='For scoring="text": divide by token count so long options are not penalised.',
    )

    def prefix_for(self, chat: bool) -> str:
        return self.answer_prefix if self.answer_prefix is not None else ("" if chat else "Answer:")


class FitConfig(Frozen):
    """Adam settings for the softmax((z - b) / T) fit."""

    learning_rate: float = Field(default=0.05, gt=0)
    max_iters: int = Field(default=500, ge=1)
    tol: float = Field(default=1e-8, ge=0, description="Stop when the loss moves less than this.")
    prior: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Strength of a Gaussian prior pulling T towards 1 and the bias towards 0. "
            "Divided by the number of examples, so it only bites on small sets, where "
            "a separable split would otherwise send T to zero."
        ),
    )
    init_temperature: float = Field(default=1.0, gt=0)
    fit_bias: bool = Field(default=True, description="False gives plain temperature scaling.")
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    max_log_temperature: float = Field(default=5.0, gt=0)


class ApiConfig(Frozen):
    """Transport settings for an OpenAI-compatible endpoint."""

    top_logprobs: int = Field(
        default=20,
        ge=1,
        description=(
            "Candidates requested at the answer position. 20 is the OpenAI ceiling; "
            "vLLM allows more if started with a higher --max-logprobs."
        ),
    )
    concurrency: int = Field(default=8, ge=1, description="Requests in flight.")
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    temperature: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Servers apply sampling settings before reporting logprobs, so 1.0 keeps "
            "them the model's own distribution."
        ),
    )


class CalibrationConfig(Frozen):
    """Splitting and reporting around the fit."""

    test_size: float = Field(default=0.25, ge=0, lt=1)
    seed: int = 0
    n_bins: int = Field(default=10, ge=1, description="Confidence bins for ECE.")
    fit: FitConfig = FitConfig()
