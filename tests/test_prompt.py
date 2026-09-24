import pytest

from logitly import BackendError, Bool, Choice, Logitly, PromptConfig, QuestionError
from logitly.core import build_message, label_prefix, option_labels, resolve_label_tokens
from logitly.errors import TokenizationError
from logitly.testing import FakeBackend


class ChatBackend(FakeBackend):
    """A backend whose template ends in a newline, like most instruct models."""

    @property
    def has_chat_template(self) -> bool:
        return True

    def apply_chat_template(self, messages):
        turns = "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages)
        return f"{turns}<|assistant|>\n"


def test_options_are_labelled_in_declaration_order(question):
    message = build_message("q", ["a", "b"], "state", PromptConfig())
    assert "Options:\nA. a\nB. b" in message
    assert message.index("q") < message.index("state") < message.index("Options:")


def test_a_mapping_state_becomes_labelled_blocks():
    message = build_message("q", ["a", "b"], {"subject": "hi", "body": "there"}, PromptConfig())
    assert "subject:\nhi" in message and "body:\nthere" in message


def test_number_labels_and_a_dropped_directive():
    config = PromptConfig(label_style="number", directive="")
    message = build_message("q", ["a", "b"], None, config)
    assert "1. a" in message
    assert "single best option" not in message


def test_letters_run_out_past_the_alphabet():
    with pytest.raises(QuestionError):
        option_labels(27)
    assert option_labels(27, "number")[-1] == "27"


def test_completion_prompts_end_at_the_answer(question):
    llm = Logitly(FakeBackend())
    prompt = llm.render(question, "hello")
    assert prompt.endswith("Answer:")
    assert label_prefix(prompt) == " "


def test_chat_prompts_use_the_template_and_carry_context(question):
    question.context = "You triage support tickets."
    llm = Logitly(ChatBackend())
    prompt = llm.render(question, "hello")
    assert prompt.endswith("<|assistant|>\n")
    assert "<|system|>You triage support tickets." in prompt
    assert label_prefix(prompt) == ""


def test_chat_can_be_forced_off(question):
    question.prompt = PromptConfig(chat=False)
    prompt = Logitly(ChatBackend()).render(question, "hello")
    assert "<|assistant|>" not in prompt


def test_asking_for_chat_without_a_template_is_an_error(question):
    question.prompt = PromptConfig(chat=True)
    with pytest.raises(BackendError):
        Logitly(FakeBackend()).render(question, "hello")


def test_context_is_inlined_when_there_is_no_chat_template(question):
    question.context = "You triage support tickets."
    prompt = Logitly(FakeBackend()).render(question, "hello")
    assert prompt.startswith("You triage support tickets.")


def test_label_tokens_are_one_per_option_and_distinct():
    backend = FakeBackend()
    ids = resolve_label_tokens(backend, ["A", "B", "C"], " ")
    assert len(set(ids)) == 3
    assert ids == [
        backend.first_token_id(" A"),
        backend.first_token_id(" B"),
        backend.first_token_id(" C"),
    ]


def test_the_prefix_changes_which_token_is_read():
    backend = FakeBackend()
    assert resolve_label_tokens(backend, ["A"], " ") != resolve_label_tokens(backend, ["A"], "")


def test_labels_sharing_a_token_are_rejected():
    class Collapsing(FakeBackend):
        def first_token_id(self, text):
            return 7

    with pytest.raises(TokenizationError):
        resolve_label_tokens(Collapsing(), ["A", "B"], " ")


def test_labels_that_encode_to_nothing_are_rejected():
    class Empty(FakeBackend):
        def first_token_id(self, text):
            return None

    with pytest.raises(TokenizationError):
        resolve_label_tokens(Empty(), ["A", "B"], " ")


def test_yes_no_defaults():
    assert Bool(instructions="is it billing?").options == ["yes", "no"]


def test_per_question_prompt_config_overrides_the_llm_default():
    llm = Logitly(FakeBackend(), prompt=PromptConfig(input_label="Ticket"))
    question = Choice(instructions="q", options=["a", "b"])
    assert "Ticket:" in llm.render(question, "hello")

    question.prompt = PromptConfig(input_label="Email")
    assert "Email:" in llm.render(question, "hello")
