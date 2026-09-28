from dataclasses import dataclass
from typing import Optional


@dataclass
class TicketSample:
    sample_id: int
    text: str
    queue: str
    priority: str
    ticket_type: Optional[str]
    language: str


@dataclass
class DecisionOutput:
    model: str
    queue: Optional[str]
    priority: Optional[str]
    ticket_type: Optional[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    parse_success: bool
    raw_response: str


@dataclass
class SampleScore:
    sample_id: int
    model: str
    queue: Optional[str]
    priority: Optional[str]
    ticket_type: Optional[str]
    queue_correct: bool
    priority_correct: bool
    type_correct: Optional[bool]
    priority_abs_error: Optional[int]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    parse_success: bool
    raw_response: str
