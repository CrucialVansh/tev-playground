import json
import time
from typing import Optional

from openai import OpenAI

from config import (
    GPT4O_INPUT_PRICE_PER_M,
    GPT4O_OUTPUT_PRICE_PER_M,
    OPENAI_MODEL_NAME,
    env_key,
    token_cost,
)
from decisions import PRIORITIES, QUEUES, TYPES, Label, resolve_label, PRIORITY_ALIASES, QUEUE_ALIASES, TYPE_ALIASES
from schema import DecisionOutput


def _catalog(labels: tuple[Label, ...]) -> str:
    return "\n".join(f"- {label.key}: {label.description}" for label in labels)


SYSTEM_PROMPT = (
    "You classify customer support tickets. Reply with JSON only, using exactly these keys: "
    '{"queue":"<key>","priority":"<key>","type":"<key>"}.\n\n'
    "queue, one of:\n"
    f"{_catalog(QUEUES)}\n\n"
    "priority, one of, from least urgent to most urgent:\n"
    f"{_catalog(PRIORITIES)}\n\n"
    "type, one of:\n"
    f"{_catalog(TYPES)}"
)


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or env_key("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY in .env to call gpt-4o.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = OPENAI_MODEL_NAME
        self.name = "openai"

    def decide(self, text: str) -> DecisionOutput:
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            return DecisionOutput(
                model="openai",
                queue=None,
                priority=None,
                ticket_type=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                parse_success=False,
                raw_response=f"error: {error}",
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        raw = (response.choices[0].message.content or "").strip()
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}

        queue = resolve_label(_as_text(parsed.get("queue")), QUEUE_ALIASES) if isinstance(parsed, dict) else None
        priority = resolve_label(_as_text(parsed.get("priority")), PRIORITY_ALIASES) if isinstance(parsed, dict) else None
        ticket_type = resolve_label(_as_text(parsed.get("type")), TYPE_ALIASES) if isinstance(parsed, dict) else None
        return DecisionOutput(
            model="openai",
            queue=queue,
            priority=priority,
            ticket_type=ticket_type,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=token_cost(input_tokens, output_tokens, GPT4O_INPUT_PRICE_PER_M, GPT4O_OUTPUT_PRICE_PER_M),
            parse_success=all((queue, priority, ticket_type)),
            raw_response=raw,
        )


def _as_text(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)
