"""Shared fixtures data: a task whose correct answer is written into the input."""

import numpy as np

OPTIONS = ["calm", "frustrated", "angry"]


def make_dataset(n: int = 120, options: list[str] = OPTIONS) -> list[dict]:
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(options), size=n)
    return [
        {"input": f"ticket {i} sounds {options[pick]}", "answer": options[pick]}
        for i, pick in enumerate(picks)
    ]


def oracle_from(options: list[str] = OPTIONS):
    """Reads the planted answer back out of the rendered prompt."""

    def oracle(prompt: str) -> int:
        return next(i for i, option in enumerate(options) if f"sounds {option}" in prompt)

    return oracle
