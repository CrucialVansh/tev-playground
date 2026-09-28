from typing import Optional

from config import token_cost
from decisions import priority_from_score, resolve_label, systemone_questions, QUEUE_ALIASES, TYPE_ALIASES
from models.http import post_json
from schema import DecisionOutput


class SystemOneClient:
    """Hosted Jev call. The request matches the /v1/systemone schema."""

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
        self.questions = systemone_questions()

    def decide(self, text: str) -> DecisionOutput:
        import time

        started = time.perf_counter()
        try:
            payload = post_json(
                self.url,
                self.api_key,
                {"model": self.model, "state": text, "questions": self.questions},
            )
        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return DecisionOutput(
                model=self.name,
                queue=None,
                priority=None,
                ticket_type=None,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                parse_success=False,
                raw_response=f"error: {error}",
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return decision_from_systemone(
            self.name,
            payload,
            latency_ms,
            input_price=self.input_price,
            output_price=self.output_price,
        )


def decision_from_systemone(
    name: str,
    payload: object,
    latency_ms: float,
    *,
    input_price: float = 0.0,
    output_price: float = 0.0,
    cost_usd: Optional[float] = None,
) -> DecisionOutput:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        return DecisionOutput(
            model=name,
            queue=None,
            priority=None,
            ticket_type=None,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            parse_success=False,
            raw_response="error: response did not include answers",
        )

    queue = _choice(answers.get("queue"), QUEUE_ALIASES)
    priority = _priority(answers.get("priority"))
    ticket_type = _choice(answers.get("type"), TYPE_ALIASES)
    input_tokens = int(usage.get("input_tokens") or 0) if isinstance(usage, dict) else 0
    output_tokens = int(usage.get("output_tokens") or 0) if isinstance(usage, dict) else 0
    if cost_usd is None:
        cost_usd = token_cost(input_tokens, output_tokens, input_price, output_price)
    return DecisionOutput(
        model=name,
        queue=queue,
        priority=priority,
        ticket_type=ticket_type,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        parse_success=all((queue, priority, ticket_type)),
        raw_response=_compact(answers),
    )


def _choice(answer: object, aliases: dict) -> Optional[str]:
    if isinstance(answer, str):
        return resolve_label(answer, aliases)
    if not isinstance(answer, dict):
        return None
    return resolve_label(None if answer.get("choice") is None else str(answer.get("choice")), aliases)


def _priority(answer: object) -> Optional[str]:
    if not isinstance(answer, dict):
        return None
    return priority_from_score(answer)


def _compact(answers: dict) -> str:
    parts = []
    for name, answer in answers.items():
        if not isinstance(answer, dict):
            continue
        value = answer.get("choice", answer.get("score", answer.get("noul")))
        parts.append(f"{name}={value}")
    return " | ".join(parts)
