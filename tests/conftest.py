import pytest

from helpers import OPTIONS
from logitly import Choice, Logitly
from logitly.testing import FakeBackend


@pytest.fixture
def question() -> Choice:
    return Choice(instructions="What is the tone?", options=OPTIONS, name="tone")


@pytest.fixture
def llm() -> Logitly:
    return Logitly(FakeBackend())
