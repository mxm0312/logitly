"""Hugging Face transformers backend.

Runs under torch.inference_mode and hands back plain numpy, so no autograd
graph or device tensor escapes into the rest of the library.
"""

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logitly.backends.base import Backend, ProgressFn
from logitly.config import DEFAULT_BATCH_SIZE
from logitly.core.prompt import Prompt, label_prefix
from logitly.core.tokens import resolve_label_tokens
from logitly.errors import BackendError


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _best_dtype(device: str) -> "torch.dtype":
    """Half precision only where its kernels are trustworthy.

    Reading one logit per option needs more precision than generating text does.
    On mps, fp16 moves a padded batch's logits by whole units; on cpu it is also
    slow. Both get fp32, which costs memory but is exact enough to calibrate.
    """
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class HuggingFaceBackend(Backend):
    """Wraps a causal LM and its tokenizer.

    Args:
        model: A loaded AutoModelForCausalLM.
        tokenizer: Its tokenizer.
        batch_size: Prompts per forward pass.
        name: Identifier shown in reports.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        name: str | None = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.name = name or getattr(model.config, "_name_or_path", "") or type(model).__name__

        if batch_size < 1:
            raise BackendError(f"batch_size must be >= 1, got {batch_size}")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise BackendError(f"{self.name}: set tokenizer.pad_token before using this model")
            tokenizer.pad_token = tokenizer.eos_token

        forward_args = inspect.signature(type(model).forward).parameters
        # Recent transformers can skip materialising logits for every position.
        self._keep_kwarg = next(
            (k for k in ("logits_to_keep", "num_logits_to_keep") if k in forward_args), None
        )
        self._takes_position_ids = "position_ids" in forward_args
        self._token_cache: dict[tuple[tuple[str, ...], str], list[int]] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str | None = None,
        dtype: Any = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> "HuggingFaceBackend":
        """Load a model and tokenizer from the Hub or a local directory.

        Args:
            device: Defaults to cuda, then mps, then cpu. Ignored when
                device_map is forwarded to transformers.
            dtype: "auto" picks bf16/fp16 on accelerators and fp32 on CPU.
            model_kwargs: Forwarded to AutoModelForCausalLM.from_pretrained.
        """
        device = device or _best_device()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=_best_dtype(device) if dtype == "auto" else dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        if "device_map" not in model_kwargs:
            model.to(device)
        return cls(model, tokenizer, batch_size=batch_size, name=model_name_or_path)

    @property
    def device(self) -> "torch.device":
        return next(self.model.parameters()).device

    @property
    def has_chat_template(self) -> bool:
        return self.tokenizer.chat_template is not None

    def apply_chat_template(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )

    def first_token_id(self, text: str) -> int | None:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return int(ids[0]) if ids else None

    def score_labels(
        self,
        prompts: Sequence[Prompt],
        labels: Sequence[str],
        *,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        texts = [self.text_for(prompt) for prompt in prompts]
        token_ids = self._label_tokens(tuple(labels), label_prefix(texts[0]))
        columns = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        keep = {self._keep_kwarg: 1} if self._keep_kwarg else {}
        add_special_tokens = not prompts[0].chat
        out = []

        for batch in self.batches(texts, self.batch_size):
            encoded = self._encode_left_padded(batch, add_special_tokens)
            with torch.inference_mode():
                logits = self.model(**encoded, **keep).logits
            # Left padding puts a real token last, so -1 is the answer position.
            out.append(logits[:, -1, :].float().index_select(1, columns).cpu().numpy())
            if progress:
                progress(sum(map(len, out)), len(texts))
        return np.concatenate(out).astype(np.float64)

    def score_texts(
        self,
        prompts: Sequence[Prompt],
        options: Sequence[str],
        *,
        length_normalize: bool = True,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        texts = [self.text_for(prompt) for prompt in prompts]
        add_special_tokens = not prompts[0].chat

        # Prompt and continuation are encoded separately so the boundary is exact;
        # encoding the concatenation lets a merge swallow the first option token.
        option_ids = [self.tokenizer.encode(text, add_special_tokens=False) for text in options]
        empty = [text for text, ids in zip(options, option_ids, strict=True) if not ids]
        if empty:
            raise BackendError(f"options {empty} encode to zero tokens")

        pairs = [
            (self.tokenizer.encode(text, add_special_tokens=add_special_tokens), ids)
            for text in texts
            for ids in option_ids
        ]
        scores: list[float] = []

        for batch in self.batches(pairs, self.batch_size):
            width = max(len(prompt) + len(option) for prompt, option in batch)
            input_ids = torch.full((len(batch), width), self.tokenizer.pad_token_id)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            answer = torch.zeros((len(batch), width - 1), dtype=torch.bool)
            for row, (prompt_ids, option) in enumerate(batch):
                sequence = prompt_ids + option
                input_ids[row, : len(sequence)] = torch.tensor(sequence)
                attention[row, : len(sequence)] = 1
                # The token at position t is predicted by the logits at t - 1.
                answer[row, len(prompt_ids) - 1 : len(sequence) - 1] = True

            input_ids = input_ids.to(self.device)
            attention = attention.to(self.device)
            answer = answer.to(self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention).logits
            logits = logits[:, :-1, :].float()
            chosen = logits.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            token_logprobs = (chosen - torch.logsumexp(logits, dim=-1)) * answer

            total = token_logprobs.sum(dim=-1)
            if length_normalize:
                total = total / answer.sum(dim=-1)
            scores.extend(total.cpu().tolist())
            if progress:
                progress(len(scores) // len(options), len(texts))
        return np.asarray(scores, dtype=np.float64).reshape(len(texts), len(options))

    def _label_tokens(self, labels: tuple[str, ...], prefix: str) -> list[int]:
        key = (labels, prefix)
        if key not in self._token_cache:
            self._token_cache[key] = resolve_label_tokens(self, labels, prefix)
        return self._token_cache[key]

    def _encode_left_padded(self, prompts: Sequence[str], add_special_tokens: bool) -> dict:
        """Tokenise with the padding on the left, so position -1 is a real token."""
        previous = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                add_special_tokens=add_special_tokens,
            )
        finally:
            self.tokenizer.padding_side = previous

        batch = {key: value.to(self.device) for key, value in encoded.items()}
        if self._takes_position_ids:
            # Left padding shifts every real token, and a bare forward pass numbers
            # positions from 0 regardless of the mask. Count real tokens instead.
            mask = batch["attention_mask"]
            batch["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0)
        return batch
