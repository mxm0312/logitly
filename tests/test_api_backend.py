import httpx
import numpy as np
import pytest

from logitly import ApiConfig, BackendError, Choice, Logitly, OpenAIBackend
from logitly.core import Prompt

BASE_URL = "http://test-server:8000/v1"


def response_with(*tokens: tuple[str, float]) -> dict:
    entries = [{"token": token, "logprob": logprob} for token, logprob in tokens]
    return {"choices": [{"logprobs": {"content": [{"top_logprobs": entries}]}}]}


def backend_returning(*tokens: tuple[str, float], **kwargs) -> OpenAIBackend:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_with(*tokens))

    return backend_with(handler, **kwargs)


def backend_with(handler, **kwargs) -> OpenAIBackend:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    return OpenAIBackend("test-model", BASE_URL, client=client, **kwargs)


def prompt(text: str = "hello") -> Prompt:
    return Prompt(messages=[{"role": "user", "content": text}])


def test_the_request_asks_for_one_token_and_its_logprobs():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=response_with((" A", -0.1), (" B", -2.0)))

    backend_with(handler).score_labels([prompt()], ["A", "B"])

    assert seen["max_tokens"] == 1
    assert seen["logprobs"] is True and seen["top_logprobs"] == 20
    assert seen["temperature"] == 1.0 and seen["top_p"] == 1.0
    assert seen["messages"] == [{"role": "user", "content": "hello"}]


def test_scores_are_the_log_probabilities_of_the_matching_tokens():
    scores = backend_returning((" A", -0.1), (" B", -2.0)).score_labels([prompt()], ["A", "B"])
    assert scores == pytest.approx(np.array([[-0.1, -2.0]]))


@pytest.mark.parametrize("token", [" A", "A", "A.", "a", " **A**", "(A)"])
def test_a_label_is_recognised_however_it_is_spelled(token):
    scores = backend_returning((token, -0.1), (" B", -2.0)).score_labels([prompt()], ["A", "B"])
    assert scores[0, 0] == pytest.approx(-0.1)


def test_a_word_starting_with_the_label_is_not_the_label():
    """'Answer' must never count as option A; matching is equality, not prefix."""
    with pytest.warns(UserWarning, match="fell outside"):
        scores = backend_returning(("Answer", -0.1), (" B", -2.0), (" C", -5.0)).score_labels(
            [prompt()], ["A", "B"]
        )
    assert scores[0, 0] == pytest.approx(-5.0)  # floored, not credited with Answer's -0.1
    assert scores[0, 0] < scores[0, 1]


def test_several_spellings_of_one_label_are_added_up():
    scores = backend_returning((" A", np.log(0.3)), ("A", np.log(0.2)), (" B", np.log(0.5)))
    result = scores.score_labels([prompt()], ["A", "B"])
    assert np.exp(result[0, 0]) == pytest.approx(0.5)


def test_a_missing_option_is_floored_and_reported():
    with pytest.warns(UserWarning, match="fell outside the top"):
        scores = backend_returning((" B", -0.1), (" C", -3.0)).score_labels([prompt()], ["A", "B"])
    assert scores[0, 0] == pytest.approx(-3.0)


def test_no_warning_when_every_option_is_present(recwarn):
    backend_returning((" A", -0.1), (" B", -2.0)).score_labels([prompt()], ["A", "B"])
    assert not recwarn.list


def test_a_server_without_logprobs_is_reported_clearly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

    with pytest.raises(BackendError, match="top_logprobs"):
        backend_with(handler).score_labels([prompt()], ["A", "B"])


def test_a_client_error_is_raised_without_retrying():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, text="model not found")

    with pytest.raises(BackendError, match="404"):
        backend_with(handler).score_labels([prompt()], ["A", "B"])
    assert len(calls) == 1


def test_a_server_error_is_retried_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=response_with((" A", -0.1), (" B", -2.0)))

    backend = backend_with(handler, config=ApiConfig(max_retries=3))
    scores = backend.score_labels([prompt()], ["A", "B"])
    assert len(calls) == 3
    assert scores[0, 0] == pytest.approx(-0.1)


def test_text_scoring_is_refused_with_a_reason():
    with pytest.raises(BackendError, match="scoring"):
        backend_returning((" A", -0.1)).score_texts([prompt()], ["yes", "no"])


def test_end_to_end_through_llm():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_with((" B", -0.05), (" A", -3.0)))

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    llm = Logitly.from_api("test-model", BASE_URL, client=client)
    question = Choice(instructions="Is it urgent?", options=["no", "yes"])

    answer = llm.ask(question, "the server is on fire")
    assert answer.label == "yes"
    assert answer.confidence > 0.9
    assert "[user]" in llm.render(question, "x")
