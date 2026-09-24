"""Building what the model is shown."""

import string
from collections.abc import Mapping, Sequence
from typing import Any

from logitly.config import Frozen, PromptConfig
from logitly.errors import BackendError, QuestionError

LETTERS = string.ascii_uppercase


class Prompt(Frozen):
    """One rendered question, in the neutral form every backend can consume.

    A local backend turns this into a string with the model's chat template; an
    HTTP backend posts the messages and lets the server do it.

    Attributes:
        messages: An optional system turn carrying the question's context, then
            the user turn.
        answer_prefix: Text the answer follows, for backends that can control
            where the model's turn starts.
        chat: Whether this was rendered for a chat model.
    """

    messages: list[dict[str, str]]
    answer_prefix: str = ""
    chat: bool = True

    @property
    def body(self) -> str:
        """The user turn on its own, for backends without a chat template."""
        return self.messages[-1]["content"]


def option_labels(n: int, label_style: str = "letter") -> list[str]:
    """The short labels the model must emit: A, B, C or 1, 2, 3."""
    if label_style == "number":
        return [str(i + 1) for i in range(n)]
    if n > len(LETTERS):
        raise QuestionError(
            f"letter labels support at most {len(LETTERS)} options, got {n}; "
            'use PromptConfig(label_style="number") or scoring="text"'
        )
    return list(LETTERS[:n])


def render_state(state: Any, input_label: str) -> str:
    """Render the per-call input; a mapping becomes one labelled block per key."""
    if state is None:
        return ""
    if isinstance(state, Mapping):
        return "\n\n".join(f"{key}:\n{value}" for key, value in state.items())
    text = str(state).strip()
    return f"{input_label}:\n{text}" if input_label else text


def build_message(
    instructions: str,
    options: Sequence[str],
    state: Any,
    config: PromptConfig,
    background: str | None = None,
) -> str:
    """Assemble the user turn."""
    labels = option_labels(len(options), config.label_style)
    blocks = (
        (background or "").strip(),
        instructions.strip(),
        render_state(state, config.input_label),
        "Options:\n"
        + "\n".join(f"{label}. {option}" for label, option in zip(labels, options, strict=True)),
        config.directive.strip(),
    )
    return "\n\n".join(block for block in blocks if block)


def label_prefix(text: str) -> str:
    """What the label is glued to when the model continues text.

    Tokenizers fold a leading space into the token, so a prompt ending in
    "Answer:" is continued by " A" while one ending in a newline is continued by
    "A". Getting this wrong reads the wrong logit.
    """
    return "" if text[-1:].isspace() else " "


class PromptRenderer:
    """Turns a question and an input into a Prompt for one backend."""

    def __init__(self, backend, config: PromptConfig):
        self._backend = backend
        self._config = config

    def config_for(self, question) -> PromptConfig:
        return question.prompt or self._config

    def uses_chat(self, config: PromptConfig) -> bool:
        if config.chat == "auto":
            return self._backend.has_chat_template
        if config.chat and not self._backend.has_chat_template:
            raise BackendError(
                f"{self._backend.name} has no chat template; use PromptConfig(chat=False)"
            )
        return config.chat

    def render(self, question, state: Any = None) -> Prompt:
        config = self.config_for(question)
        chat = self.uses_chat(config)
        body = build_message(
            question.instructions,
            question.options,
            state,
            config,
            background=None if chat else question.context,
        )
        messages = [{"role": "user", "content": body}]
        if chat and question.context:
            messages.insert(0, {"role": "system", "content": question.context})
        return Prompt(messages=messages, answer_prefix=config.prefix_for(chat), chat=chat)
