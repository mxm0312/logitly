"""Mapping option labels onto the token ids whose logits we read."""

from collections.abc import Sequence
from typing import Any

from logitly.errors import TokenizationError


def resolve_label_tokens(tokenizer_like: Any, labels: Sequence[str], prefix: str) -> list[int]:
    """The single token id that stands for each label at the answer position.

    Args:
        tokenizer_like: Anything with a first_token_id(text) method.
        labels: The option labels, as they appear in the prompt.
        prefix: What the label follows, from label_prefix.
    """
    ids = [tokenizer_like.first_token_id(prefix + label) for label in labels]

    missing = [label for label, i in zip(labels, ids, strict=True) if i is None]
    if missing:
        raise TokenizationError(f"labels {missing} encode to nothing in this tokenizer")

    clashes = {i for i in ids if ids.count(i) > 1}
    if clashes:
        shared = [label for label, i in zip(labels, ids, strict=True) if i in clashes]
        raise TokenizationError(
            f"labels {shared} share a token id in this tokenizer; "
            'use PromptConfig(label_style="number") or scoring="text"'
        )
    return [int(i) for i in ids]
