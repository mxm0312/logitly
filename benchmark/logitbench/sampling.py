"""Stratified index sampling: the only source of randomness in the benchmark."""

import numpy as np


def _quota(counts: np.ndarray, n: int, balanced: bool, rng: np.random.Generator) -> np.ndarray:
    """How many examples to draw from each class. Largest remainder, ties broken at random."""
    exact = np.full(len(counts), n / len(counts)) if balanced else n * counts / counts.sum()
    quota = np.floor(exact).astype(np.int64)
    shuffled = rng.permutation(len(counts))
    ranked = shuffled[np.argsort(-(exact - quota)[shuffled], kind="stable")]
    quota[ranked[: n - quota.sum()]] += 1
    return quota


def sample(
    labels: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    balanced: bool = False,
    where: np.ndarray | None = None,
) -> np.ndarray:
    """Draw `n` indices, keeping the class mix intact.

    Args:
        labels: Class index per row.
        n: How many to draw.
        balanced: Spread `n` evenly over the classes instead of following their
            natural frequencies.
        where: Restrict the draw to these row indices.

    Returns:
        Sorted row indices into `labels`.
    """
    rows = np.arange(len(labels)) if where is None else np.asarray(where, dtype=np.int64)
    if n > len(rows):
        raise ValueError(f"asked for {n} examples but only {len(rows)} are available")

    classes, counts = np.unique(labels[rows], return_counts=True)
    quota = _quota(counts, n, balanced, rng)
    short = [(c, q, k) for c, q, k in zip(classes, quota, counts, strict=True) if q > k]
    if short:
        raise ValueError(f"not enough examples for a balanced draw of {n}: need/have {short}")

    picks = [
        rng.choice(rows[labels[rows] == cls], size=size, replace=False)
        for cls, size in zip(classes, quota, strict=True)
    ]
    return np.sort(np.concatenate(picks).astype(np.int64))
