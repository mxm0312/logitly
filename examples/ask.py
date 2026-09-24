"""Ask one model several closed questions about the same text.

python examples/ask.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse

from logitly import Bool, Choice, Logitly, Score

TICKET = "I was charged twice for my subscription this morning. Please fix this today."

QUESTIONS = {
    "billing": Bool(instructions="Is this ticket about billing?"),
    "tone": Choice(
        instructions="What is the customer's tone?",
        options=["calm", "frustrated", "angry"],
    ),
    "urgency": Score(
        instructions="How urgent is this ticket?",
        options=["can wait", "this week", "today"],
    ),
}


def connect(args) -> Logitly:
    """Local weights by default; a served model when --base-url is given."""
    if args.base_url:
        return Logitly.from_api(args.model, args.base_url, api_key=args.api_key)
    return Logitly(args.model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://host:8000/v1")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    llm = connect(args)
    answers = llm.ask_all(QUESTIONS, TICKET)

    print(f"ticket: {TICKET}\n")
    for name, answer in answers.items():
        print(f"{name:<9} {answer.label:<12} p={answer.confidence:.2f}  margin={answer.margin:.2f}")

    print(f"\nurgency on a 0-2 scale: {answers['urgency'].expected_score:.2f}")
    print(f"is it billing?          {answers['billing'].value}")
    print(f"full distribution:      {answers['tone'].probs}")


if __name__ == "__main__":
    main()
