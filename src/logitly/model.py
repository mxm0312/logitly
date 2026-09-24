"""Load a model, ask closed questions, calibrate the answers."""

import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from logitly.backends import Backend, OpenAIBackend
from logitly.calibration import (
    Calibration,
    CalibrationReport,
    Metrics,
    fit_calibration,
    normalize_dataset,
    stratified_split,
)
from logitly.config import ApiConfig, CalibrationConfig, PromptConfig
from logitly.core import Answer, PromptRenderer, Question
from logitly.errors import BackendError, CalibrationError

ProgressLike = bool | Callable[[int, int], None]


def _console_progress(done: int, total: int) -> None:
    sys.stderr.write(f"\rscoring {done}/{total}{chr(10) if done >= total else ''}")
    sys.stderr.flush()


class Logitly:
    """A loaded model you can ask closed questions.

    Args:
        model: A Hub id, a local path, or a ready-made Backend.
        prompt: Default formatting for questions that do not carry their own.
        load_kwargs: Forwarded to HuggingFaceBackend.from_pretrained, e.g.
            device, dtype, batch_size, device_map.

    Example:
        llm = Logitly("Qwen/Qwen2.5-1.5B-Instruct")
        tone = Choice(instructions="What is the tone?", options=["calm", "angry"])
        llm.ask(tone, "I was charged twice!").label
    """

    def __init__(
        self,
        model: str | Backend,
        *,
        prompt: PromptConfig | None = None,
        **load_kwargs: Any,
    ):
        if isinstance(model, Backend):
            if load_kwargs:
                raise BackendError(
                    f"a ready-made backend takes no loading arguments: {load_kwargs}"
                )
            self.backend = model
        else:
            # Imported here so that an API-only install needs no torch.
            from logitly.backends.huggingface import HuggingFaceBackend

            self.backend = HuggingFaceBackend.from_pretrained(model, **load_kwargs)

        self.renderer = PromptRenderer(self.backend, prompt or PromptConfig())

    @classmethod
    def from_api(
        cls,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        config: ApiConfig | None = None,
        prompt: PromptConfig | None = None,
        **backend_kwargs: Any,
    ) -> "Logitly":
        """Use a model served behind an OpenAI-compatible API.

        Args:
            model: The model name the server serves.
            base_url: The endpoint, e.g. "http://my-host:8000/v1".
            api_key: Sent as a bearer token; omit for servers that want none.
            config: Transport settings.
            prompt: Default formatting for questions that do not carry their own.

        Example:
            llm = Logitly.from_api("Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1")
        """
        backend = OpenAIBackend(model, base_url, api_key=api_key, config=config, **backend_kwargs)
        return cls(backend, prompt=prompt)

    @property
    def name(self) -> str:
        """Identifier of the underlying model, as it appears in reports."""
        return self.backend.name

    def __repr__(self) -> str:
        return f"Logitly({self.name!r})"

    def render(self, question: Question, state: Any = None) -> str:
        """The prompt as the backend will send it. Print it when results look odd."""
        return self.backend.text_for(self.renderer.render(question, state))

    def scores(
        self,
        question: Question,
        states: Sequence[Any],
        *,
        progress: ProgressLike = False,
    ) -> np.ndarray:
        """Raw, uncalibrated option scores, one row per input.

        Feed the result to calibrate_from_scores to fit a calibration without
        paying for inference twice.
        """
        self._warn_on_foreign_calibration(question)
        config = self.renderer.config_for(question)
        prompts = [self.renderer.render(question, state) for state in states]
        report = _console_progress if progress is True else (progress or None)

        if question.scoring == "letter":
            scores = self.backend.score_labels(
                prompts, question.labels(config.label_style), progress=report
            )
        else:
            scores = self.backend.score_texts(
                prompts,
                question.options,
                length_normalize=config.length_normalize,
                progress=report,
            )

        expected = (len(prompts), question.n_options)
        if scores.shape != expected:
            raise BackendError(
                f"backend returned scores of shape {scores.shape}, expected {expected}"
            )
        return scores

    def ask(self, question: Question, state: Any = None) -> Answer:
        """Answer one question about one input."""
        return question.make_answer(self.scores(question, [state])[0])

    def ask_many(
        self,
        question: Question,
        states: Iterable[Any],
        *,
        progress: ProgressLike = False,
    ) -> list[Answer]:
        """Answer one question about many inputs, batched."""
        states = list(states)
        if not states:
            return []
        scores = self.scores(question, states, progress=progress)
        return [question.make_answer(row) for row in scores]

    def ask_all(
        self,
        questions: Mapping[str, Question] | Sequence[Question],
        state: Any = None,
    ) -> dict[str, Answer]:
        """Answer several questions about the same input.

        Each question is its own forward pass; the answers come back in one dict
        keyed by the mapping key, or by question.name.
        """
        items = (
            questions.items()
            if isinstance(questions, Mapping)
            else ((q.name or f"question_{i}", q) for i, q in enumerate(questions))
        )
        return {key: self.ask(question, state) for key, question in items}

    def _warn_on_foreign_calibration(self, question: Question) -> None:
        fitted_on = question.calibration.model if question.calibration else None
        if fitted_on is not None and fitted_on != self.name:
            warnings.warn(
                f"{question.name or 'this question'} carries a calibration fitted on "
                f"{fitted_on}, but this Logitly runs {self.name}. T and b describe one "
                "model's logits; refit, or clear question.calibration.",
                stacklevel=3,
            )

    def calibrate(
        self,
        question: Question,
        data: Iterable[Any],
        *,
        config: CalibrationConfig | None = None,
        verbose: bool = True,
        attach: bool = True,
    ) -> CalibrationReport:
        """Fit softmax((z - b) / T) for this question on labelled data.

        Runs the model over the data, splits it, fits temperature and per-option
        bias on the train half by gradient descent, and scores both halves so you
        can see whether it helped on examples the fit never saw.

        Args:
            question: The question to calibrate; mutated in place unless attach
                is False.
            data: Example objects, or dicts with input and answer keys. A few
                hundred is plenty - this is two parameters per option.
            config: Split, metric and optimiser settings.
            verbose: Print the report and a progress line.
            attach: Store the fitted calibration on question.calibration.

        Returns:
            A CalibrationReport comparing before and after on the test split.
        """
        inputs, labels = normalize_dataset(question, data)
        scores = self.scores(question, inputs, progress=verbose)
        return self.calibrate_from_scores(
            question, scores, labels, config=config, verbose=verbose, attach=attach
        )

    def calibrate_from_scores(
        self,
        question: Question,
        scores: np.ndarray,
        labels: Sequence[int],
        *,
        config: CalibrationConfig | None = None,
        verbose: bool = True,
        attach: bool = True,
    ) -> CalibrationReport:
        """Calibrate from scores already computed with Logitly.scores."""
        cfg = config or CalibrationConfig()
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray([question.index_of(label) for label in labels], dtype=np.int64)
        if scores.ndim != 2 or scores.shape[1] != question.n_options:
            raise CalibrationError(
                f"scores must have shape (n, {question.n_options}), got {scores.shape}"
            )
        if len(y) != len(scores):
            raise CalibrationError(f"got {len(scores)} score rows but {len(y)} labels")

        train, test = stratified_split(y, cfg.test_size, cfg.seed)
        held_out = len(test) > 0
        evaluate = test if held_out else train

        result = fit_calibration(
            scores[train],
            y[train],
            options=question.options,
            model=self.name,
            config=cfg.fit,
        )
        identity = Calibration.identity(question.options)
        report = CalibrationReport(
            model=self.name,
            question=question.instructions,
            options=question.options,
            n_train=len(train),
            n_test=len(test),
            fit=result,
            before=Metrics.compute(identity.apply(scores[evaluate]), y[evaluate], cfg.n_bins),
            after=Metrics.compute(
                result.calibration.apply(scores[evaluate]), y[evaluate], cfg.n_bins
            ),
            held_out=held_out,
        )
        if attach:
            question.calibration = result.calibration
        if verbose:
            print(report)
        return report
