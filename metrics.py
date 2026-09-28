import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

from decisions import priority_distance
from schema import DecisionOutput, SampleScore, TicketSample


@dataclass
class ModelSummary:
    model: str
    sample_count: int
    queue_accuracy_pct: float
    priority_accuracy_pct: float
    priority_mae: Optional[float]
    type_accuracy_pct: Optional[float]
    type_scored_count: int
    parse_success_rate_pct: float
    lat_mean_ms: float
    lat_p50_ms: float
    lat_p95_ms: float
    total_cost_usd: float
    cost_per_1k_usd: float
    queue_agreement_with_openai_pct: Optional[float]


def score_output(sample: TicketSample, output: DecisionOutput) -> SampleScore:
    type_correct = None if sample.ticket_type is None else output.ticket_type == sample.ticket_type
    return SampleScore(
        sample_id=sample.sample_id,
        model=output.model,
        queue=output.queue,
        priority=output.priority,
        ticket_type=output.ticket_type,
        queue_correct=output.queue == sample.queue,
        priority_correct=output.priority == sample.priority,
        type_correct=type_correct,
        priority_abs_error=priority_distance(output.priority, sample.priority),
        latency_ms=output.latency_ms,
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens,
        cost_usd=output.cost_usd,
        parse_success=output.parse_success,
        raw_response=output.raw_response,
    )


def summarize(scores: List[SampleScore], openai_scores: Optional[List[SampleScore]] = None) -> ModelSummary:
    count = len(scores)
    if count == 0:
        raise ValueError("Cannot summarize an empty result list.")

    latencies = [score.latency_ms for score in scores]
    total_cost = sum(score.cost_usd for score in scores)
    type_scores = [score.type_correct for score in scores if score.type_correct is not None]
    priority_errors = [score.priority_abs_error for score in scores if score.priority_abs_error is not None]

    agreement = None
    if openai_scores is not None and scores and scores[0].model != "openai":
        paired = min(len(scores), len(openai_scores))
        matches = sum(
            1
            for index in range(paired)
            if scores[index].queue and scores[index].queue == openai_scores[index].queue
        )
        agreement = (matches / paired) * 100.0 if paired else None

    return ModelSummary(
        model=scores[0].model,
        sample_count=count,
        queue_accuracy_pct=_percent(sum(score.queue_correct for score in scores), count),
        priority_accuracy_pct=_percent(sum(score.priority_correct for score in scores), count),
        priority_mae=(sum(priority_errors) / len(priority_errors)) if priority_errors else None,
        type_accuracy_pct=_percent(sum(type_scores), len(type_scores)) if type_scores else None,
        type_scored_count=len(type_scores),
        parse_success_rate_pct=_percent(sum(score.parse_success for score in scores), count),
        lat_mean_ms=statistics.mean(latencies),
        lat_p50_ms=_percentile(latencies, 0.50),
        lat_p95_ms=_percentile(latencies, 0.95),
        total_cost_usd=total_cost,
        cost_per_1k_usd=(total_cost / count) * 1000.0,
        queue_agreement_with_openai_pct=agreement,
    )


def _percent(hits: int, total: int) -> float:
    return (hits / total) * 100.0 if total else 0.0


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def summaries_by_model(results: Dict[str, List[SampleScore]]) -> List[ModelSummary]:
    openai_scores = results.get("openai")
    return [summarize(results[name], openai_scores) for name in results]
