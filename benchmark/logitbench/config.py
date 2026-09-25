"""config.yml, validated. Nothing in the benchmark is configured anywhere else."""

import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, model_validator

from logitbench.metrics import METRICS
from logitly import ApiConfig, Choice, FitConfig, Logitly, PromptConfig
from logitly.config import Frozen, Scoring


class HuggingFaceModel(Frozen):
    """Local weights, run through transformers."""

    kind: Literal["huggingface"]
    model: str = Field(description="Hub id or local path.")
    prompt: PromptConfig = PromptConfig()
    load: dict[str, Any] = Field(
        default_factory=dict, description="Forwarded to HuggingFaceBackend.from_pretrained."
    )

    def build(self) -> Logitly:
        return Logitly(self.model, prompt=self.prompt, **self.load)


class ApiModel(Frozen):
    """A model behind an OpenAI-compatible endpoint (vLLM, OpenAI, ...)."""

    kind: Literal["api"]
    model: str
    base_url: str
    api_key_env: str | None = Field(
        default=None, description="Environment variable holding the key."
    )
    prompt: PromptConfig = PromptConfig()
    api: ApiConfig = ApiConfig()
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra request fields, e.g. chat_template_kwargs to turn thinking off.",
    )

    def build(self) -> Logitly:
        key = os.environ[self.api_key_env] if self.api_key_env else None
        return Logitly.from_api(
            self.model,
            self.base_url,
            api_key=key,
            config=self.api,
            prompt=self.prompt,
            extra_body=self.extra_body,
        )


class FakeModel(Frozen):
    """A deterministic stand-in, for checking a config end to end without weights."""

    kind: Literal["fake"]
    model: str = "fake"
    prompt: PromptConfig = PromptConfig()
    backend: dict[str, Any] = Field(default_factory=dict, description="FakeBackend arguments.")

    def build(self) -> Logitly:
        from logitly.testing import FakeBackend

        return Logitly(FakeBackend(**self.backend), prompt=self.prompt)


ModelSpec = Annotated[HuggingFaceModel | ApiModel | FakeModel, Field(discriminator="kind")]


class DatasetSpec(Frozen):
    """One classification task: where its text is, and what its label ids mean."""

    path: str = Field(description="Hub id passed to datasets.load_dataset.")
    config: str | None = None
    text_field: str = "text"
    label_field: str = "label"
    test_split: str = "test"
    calib_split: str = "train"
    test_size: int | None = Field(default=None, gt=0, description="Overrides experiment.test_size.")
    options: list[str] = Field(min_length=2, description="Option text, indexed by label id.")
    instructions: str
    context: str | None = None
    scoring: Scoring = "letter"
    load_kwargs: dict[str, Any] = Field(default_factory=dict)

    def question(self, name: str) -> Choice:
        return Choice(
            name=name,
            instructions=self.instructions,
            options=self.options,
            context=self.context,
            scoring=self.scoring,
        )


class ExperimentSpec(Frozen):
    """The sweep itself."""

    test_size: int = Field(gt=0, description="Examples the metrics are measured on.")
    calib_pool: int = Field(
        gt=0, description="Scored once per model; calibration sets come from it."
    )
    calib_sizes: list[int] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1, description="Each reshuffles the calibration set.")
    data_seed: int = Field(default=0, description="Fixes the test set and the pool.")
    balanced_calibration: bool = Field(
        default=False, description="Equal examples per class instead of the natural mix."
    )
    n_bins: int = Field(default=15, ge=1, description="Confidence bins for ECE and reliability.")
    fit: FitConfig = FitConfig()

    @model_validator(mode="after")
    def _check_sizes(self) -> "ExperimentSpec":
        if min(self.calib_sizes) < 1:
            raise ValueError("calibration sizes must be positive")
        if max(self.calib_sizes) > self.calib_pool:
            raise ValueError(
                f"calib_sizes go up to {max(self.calib_sizes)} > pool {self.calib_pool}"
            )
        return self


class ReportSpec(Frozen):
    """What the generated report and plots contain."""

    curve_metrics: list[str] = Field(default=["accuracy", "nll", "ece", "brier"])
    reliability: bool = True
    dpi: int = Field(default=150, gt=0)

    @model_validator(mode="after")
    def _check_metrics(self) -> "ReportSpec":
        unknown = set(self.curve_metrics) - set(METRICS)
        if unknown:
            raise ValueError(f"unknown metrics {sorted(unknown)}; choose from {list(METRICS)}")
        return self


class BenchConfig(Frozen):
    """A whole benchmark run."""

    output_dir: Path = Field(
        default=Path("results"),
        exclude=True,  # a local path; it says nothing about the run and must not travel with it
        description="Where the run writes; relative to this file.",
    )
    experiment: ExperimentSpec
    models: dict[str, ModelSpec] = Field(min_length=1)
    datasets: dict[str, DatasetSpec] = Field(min_length=1)
    report: ReportSpec = ReportSpec()

    @classmethod
    def load(cls, path: str | Path) -> "BenchConfig":
        """Read a config; relative output paths resolve against the config's own directory."""
        path = Path(path)
        config = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        return config.model_copy(update={"output_dir": (path.parent / config.output_dir).resolve()})

    def select(self, models: list[str], datasets: list[str]) -> "BenchConfig":
        """Narrow the run to the named models and datasets."""
        return self.model_copy(
            update={
                "models": _pick(self.models, models, "model"),
                "datasets": _pick(self.datasets, datasets, "dataset"),
            }
        )


def _pick(items: dict[str, Any], keys: list[str], what: str) -> dict[str, Any]:
    unknown = [key for key in keys if key not in items]
    if unknown:
        raise KeyError(f"unknown {what}(s) {unknown}; config has {sorted(items)}")
    return {key: items[key] for key in keys} if keys else items