"""The contract every inference backend implements."""

import abc
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from logitly.core.prompt import Prompt
from logitly.errors import BackendError

ProgressFn = Callable[[int, int], None]


class Backend(abc.ABC):
    """Scores a batch of rendered prompts against a fixed list of options.

    Two ways to score, and a backend need only offer the first: read the logits
    of the option labels at the answer position, or score each option's own text.
    How a label maps onto something measurable is the backend's business - token
    ids locally, token strings over an API.
    """

    name: str = "backend"

    @property
    def has_chat_template(self) -> bool:
        return False

    def apply_chat_template(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Render messages into a prompt ending at the model's turn."""
        raise NotImplementedError(f"{type(self).__name__} has no chat template")

    def text_for(self, prompt: Prompt) -> str:
        """The prompt as this backend will send it. For inspection and debugging."""
        if prompt.chat and self.has_chat_template:
            return self.apply_chat_template(prompt.messages) + prompt.answer_prefix
        return prompt.body + "\n\n" + prompt.answer_prefix

    @abc.abstractmethod
    def score_labels(
        self,
        prompts: Sequence[Prompt],
        labels: Sequence[str],
        *,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        """Score each option by its label at the answer position.

        Args:
            prompts: One per example.
            labels: The option labels as they appear in the prompt, in order.
            progress: Called with (done, total).

        Returns:
            (len(prompts), len(labels)) array of scores, comparable within a row.
        """

    def score_texts(
        self,
        prompts: Sequence[Prompt],
        options: Sequence[str],
        *,
        length_normalize: bool = True,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        """Score each option by the log-likelihood of its full text."""
        raise BackendError(f'{self.name} does not support scoring="text"')

    @staticmethod
    def batches(items: Sequence, size: int) -> list[Sequence]:
        return [items[i : i + size] for i in range(0, len(items), size)]
