"""Inference is the expensive part, so it happens once and lands on disk."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def fingerprint(signature: dict[str, Any]) -> str:
    """A short digest of everything that would change the scores."""
    blob = json.dumps(signature, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=6).hexdigest()


class ScoreStore:
    """One .npz per (model, dataset): re-runs and report tweaks cost nothing."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        name: str,
        signature: dict[str, Any],
        compute: Callable[[], dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        path = self.directory / f"{name}.{fingerprint(signature)}.npz"
        if path.exists():
            with np.load(path) as data:
                return dict(data)
        arrays = compute()
        np.savez_compressed(path, **arrays)
        return arrays
