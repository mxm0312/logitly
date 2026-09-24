"""Calibrate a question on labelled data, then save it for reuse.

python examples/calibrate.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse

from logitly import CalibrationConfig, Choice, Logitly, Question

REVIEWS = [
    ("Works exactly as described, very happy.", "positive"),
    ("Arrived early and the quality is great.", "positive"),
    ("Best purchase I have made this year.", "positive"),
    ("Sturdy, cheap and does the job.", "positive"),
    ("Sound quality is excellent for the price.", "positive"),
    ("My kids love it, we use it daily.", "positive"),
    ("Setup took two minutes and it just works.", "positive"),
    ("Comfortable and lighter than I expected.", "positive"),
    ("Broke on the first day, total waste.", "negative"),
    ("Cheap plastic, stopped working immediately.", "negative"),
    ("Never arrived and support ignored me.", "negative"),
    ("Terrible battery, dies within an hour.", "negative"),
    ("Does not fit, the photos are misleading.", "negative"),
    ("Returned it, far too noisy to use.", "negative"),
    ("Fell apart in the wash, avoid.", "negative"),
    ("Overpriced for what you actually get.", "negative"),
]


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
    parser.add_argument("--out", default="sentiment.json")
    args = parser.parse_args()

    llm = connect(args)
    question = Choice(
        instructions="Is this review positive or negative?",
        options=["positive", "negative"],
        context="You label product reviews.",
        name="sentiment",
    )

    held_out = "The zip broke within a week, very disappointing."
    print(f"before calibration: {llm.ask(question, held_out)!r}\n")

    # A labelled set is a list of {"input": ..., "answer": ...}. The report at the
    # end is measured on the part of it the fit never saw.
    data = [{"input": text, "answer": label} for text, label in REVIEWS]
    llm.calibrate(question, data, config=CalibrationConfig(test_size=0.35))

    print(f"\nafter calibration:  {llm.ask(question, held_out)!r}")

    question.save(args.out)
    print(f"\nsaved to {args.out}; reload it with Question.load and it keeps its calibration")
    print(Question.load(args.out).calibration)


if __name__ == "__main__":
    main()
