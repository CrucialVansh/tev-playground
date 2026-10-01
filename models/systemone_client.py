import time
from typing import Optional, Sequence

from config import token_cost
from decisions import Label, aliases, choice_question, resolve_label
from models.http import post_json
from models.two_step import two_step_decide
from schema import Answer, DecisionOutput


class SystemOneClient:
    """Hosted Jev call. Each request matches the /v1/systemone schema."""

    def __init__(
        self,
        name: str,
        model: str,
        url: str,
        api_key: str,
        input_price: float,
        output_price: float,
    ):
        if not api_key:
            raise ValueError(f"Missing API key for {name}.")
        self.name = name
        self.model = model
        self.url = url
        self.api_key = api_key
        self.input_price = input_price
        self.output_price = output_price

    def decide(self, text: str) -> DecisionOutput:
        return two_step_decide(
            self.name,
            text,
            self.ask,
            lambda tokens_in, tokens_out: token_cost(tokens_in, tokens_out, self.input_price, self.output_price),
        )

    def ask(self, text: str, question: str, labels: Sequence[Label]) -> Answer:
        started = time.perf_counter()
        payload = post_json(
            self.url,
            self.api_key,
            {"model": self.model, "state": text, "questions": {"answer": choice_question(question, labels)}},
        )
        return answer_from_systemone(payload, labels, (time.perf_counter() - started) * 1000.0)


def answer_from_systemone(payload: object, labels: Sequence[Label], latency_ms: float) -> Answer:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    answer = answers.get("answer") if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        raise ValueError("response did not include an answer")
    choice = answer.get("choice")
    return Answer(
        key=resolve_label(None if choice is None else str(choice), aliases(labels)),
        confidence=_confidence(answer),
        raw=str(choice),
        latency_ms=latency_ms,
        input_tokens=int(usage.get("input_tokens") or 0) if isinstance(usage, dict) else 0,
        output_tokens=int(usage.get("output_tokens") or 0) if isinstance(usage, dict) else 0,
    )


def _confidence(answer: dict) -> Optional[float]:
    value = answer.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
