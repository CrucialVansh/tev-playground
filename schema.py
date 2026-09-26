from dataclasses import dataclass
from typing import Optional

@dataclass
class DataSample:
    sample_id: int
    text: str
    ground_truth_name: str
    ground_truth_letter: str

@dataclass
class PipelineExecutionResult:
    sample_id: int
    architecture: str
    ground_truth_category: str
    predicted_category: str
    predicted_letter: str
    is_correct: bool
    downstream_executed: bool
    classification_latency_ms: float
    downstream_latency_ms: float
    total_latency_ms: float
    classification_cost_usd: float
    downstream_cost_usd: float
    total_cost_usd: float
    classification_input_tokens: int
    classification_output_tokens: int
    downstream_input_tokens: int
    downstream_output_tokens: int
    parse_success: bool
    downstream_output: Optional[str] = None
