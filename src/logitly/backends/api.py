"""Any server that speaks the OpenAI chat API: vLLM, OpenAI, and the rest.

One request per example, one generated token, and the top logprobs at that
position. No tokenizer and no weights: options are matched by the text of the
tokens the server sends back.
"""

import math
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np

from logitly.backends.base import Backend, ProgressFn
from logitly.config import ApiConfig
from logitly.core.prompt import Prompt
from logitly.errors import BackendError

_STRIP = " \t\n\r*_`\"'()[]{}<>.,:;!?-"
_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _normalize(token: str) -> str:
    """A token's text reduced to what it would mean as an answer.

    Equality after stripping, never a prefix test: " A", "A." and "a" all mean
    option A, while "Answer" means the model started a sentence.
    """
    return token.strip(_STRIP).casefold()


class OpenAIBackend(Backend):
    """Scores options through an OpenAI-compatible chat endpoint.

    Args:
        model: The model name the server serves.
        base_url: The endpoint, including any version path, e.g.
            "http://my-host:8000/v1".
        api_key: Sent as a bearer token. Omit for servers that want none.
        config: Transport settings.
        extra_body: Extra JSON fields, for server-specific options.
        client: A preconfigured httpx client, for custom transports or auth.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        config: ApiConfig | None = None,
        extra_body: dict[str, Any] | None = None,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.config = config or ApiConfig()
        self.extra_body = extra_body or {}
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=self.config.timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )

    @property
    def has_chat_template(self) -> bool:
        return True  # the server applies it

    def text_for(self, prompt: Prompt) -> str:
        return "\n\n".join(f"[{turn['role']}]\n{turn['content']}" for turn in prompt.messages)

    def score_labels(
        self,
        prompts: Sequence[Prompt],
        labels: Sequence[str],
        *,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        distributions = self._request_all(prompts, progress)
        scores = np.empty((len(prompts), len(labels)))
        censored = 0

        for row, distribution in enumerate(distributions):
            # Anything outside the top-k is at most the smallest value we were shown.
            floor = math.exp(min(logprob for _, logprob in distribution))
            for column, label in enumerate(labels):
                wanted = _normalize(label)
                mass = sum(
                    math.exp(logprob)
                    for token, logprob in distribution
                    if _normalize(token) == wanted
                )
                if mass == 0.0:
                    censored += 1
                    mass = floor
                scores[row, column] = math.log(mass)

        if censored:
            warnings.warn(
                f"{censored} option(s) over {len(prompts)} example(s) fell outside the top "
                f"{self.config.top_logprobs} and were floored, so those probabilities are "
                "upper bounds. Tighten the prompt, or raise top_logprobs if the server allows it.",
                stacklevel=2,
            )
        return scores

    def score_texts(self, prompts, options, *, length_normalize=True, progress=None) -> np.ndarray:
        raise BackendError(
            'scoring="text" needs log-probabilities of the prompt, which the chat API does '
            'not expose. Use scoring="letter", or a local backend.'
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAIBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request_all(
        self, prompts: Sequence[Prompt], progress: ProgressFn | None
    ) -> list[list[tuple[str, float]]]:
        distributions = []
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            for done, result in enumerate(pool.map(self._top_logprobs, prompts), start=1):
                distributions.append(result)
                if progress:
                    progress(done, len(prompts))
        return distributions

    def _top_logprobs(self, prompt: Prompt) -> list[tuple[str, float]]:
        data = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": prompt.messages,
                "max_tokens": 1,
                "temperature": self.config.temperature,
                "top_p": 1.0,
                "logprobs": True,
                "top_logprobs": self.config.top_logprobs,
                **self.extra_body,
            },
        )
        try:
            entries = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            return [(entry["token"], float(entry["logprob"])) for entry in entries]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                f"{self.name} returned no token logprobs; the server must support logprobs "
                f"and top_logprobs on /chat/completions. Response: {data}"
            ) from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.config.max_retries + 1):
            cause: Exception | None = None
            try:
                response = self._client.post(path, json=payload)
            except httpx.HTTPError as exc:
                cause = exc
                failure = BackendError(f"could not reach {self.base_url}{path}: {exc}")
                retryable = True
            else:
                if response.status_code < 400:
                    return response.json()
                failure = BackendError(
                    f"{self.base_url}{path} returned {response.status_code}: {response.text}"
                )
                retryable = response.status_code in _RETRY_STATUS

            if not retryable or attempt == self.config.max_retries:
                raise failure from cause
            time.sleep(2**attempt)
