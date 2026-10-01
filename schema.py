from dataclasses import dataclass
from typing import Optional


@dataclass
class QuerySample:
    sample_id: int
    text: str
    intent: str
    topic: str


@dataclass
class Answer:
    """One choice question answered by one model."""

    key: Optional[str]
    confidence: Optional[float]
    raw: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass
class DecisionOutput:
    model: str
    topic: Optional[str]
    intent: Optional[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    parse_success: bool
    raw_response: str
    topic_confidence: Optional[float] = None
    intent_confidence: Optional[float] = None


@dataclass
class SampleScore:
    sample_id: int
    model: str
    gold_intent: str
    topic: Optional[str]
    intent: Optional[str]
    topic_correct: bool
    intent_correct: bool
    in_scope: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    parse_success: bool
    raw_response: str
    confidence: Optional[float] = None
