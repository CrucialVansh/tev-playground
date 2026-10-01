import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

from decisions import OUT_OF_SCOPE
from schema import DecisionOutput, QuerySample, SampleScore

CONFIDENCE_THRESHOLDS = (0.5, 0.7, 0.9)


@dataclass
class HandoffBand:
    """What happens if every answer below `threshold` confidence also goes to a person."""

    threshold: float
    handled_pct: float
    handled_accuracy_pct: Optional[float]
    oos_caught_pct: Optional[float]


@dataclass
class ModelSummary:
    model: str
    sample_count: int
    in_scope_count: int
    oos_count: int
    overall_accuracy_pct: float
    in_scope_accuracy_pct: Optional[float]
    topic_accuracy_pct: Optional[float]
    oos_recall_pct: Optional[float]
    false_handoff_pct: Optional[float]
    parse_success_rate_pct: float
    lat_mean_ms: float
    lat_p50_ms: float
    lat_p95_ms: float
    total_cost_usd: float
    cost_per_1k_usd: float
    agreement_with_openai_pct: Optional[float]
    handoff_bands: Optional[List[HandoffBand]]


def score_output(sample: QuerySample, output: DecisionOutput) -> SampleScore:
    confidences = [value for value in (output.topic_confidence, output.intent_confidence) if value is not None]
    return SampleScore(
        sample_id=sample.sample_id,
        model=output.model,
        gold_intent=sample.intent,
        topic=output.topic,
        intent=output.intent,
        topic_correct=output.topic == sample.topic,
        intent_correct=output.intent == sample.intent,
        in_scope=sample.intent != OUT_OF_SCOPE,
        latency_ms=output.latency_ms,
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens,
        cost_usd=output.cost_usd,
        parse_success=output.parse_success,
        raw_response=output.raw_response,
        confidence=min(confidences) if confidences else None,
    )


def summarize(scores: List[SampleScore], openai_scores: Optional[List[SampleScore]] = None) -> ModelSummary:
    count = len(scores)
    if count == 0:
        raise ValueError("Cannot summarize an empty result list.")

    in_scope = [score for score in scores if score.in_scope]
    oos = [score for score in scores if not score.in_scope]
    latencies = [score.latency_ms for score in scores]
    total_cost = sum(score.cost_usd for score in scores)

    agreement = None
    if openai_scores is not None and scores[0].model != "openai":
        paired = min(len(scores), len(openai_scores))
        matches = sum(
            1 for index in range(paired) if scores[index].intent and scores[index].intent == openai_scores[index].intent
        )
        agreement = _percent(matches, paired) if paired else None

    return ModelSummary(
        model=scores[0].model,
        sample_count=count,
        in_scope_count=len(in_scope),
        oos_count=len(oos),
        overall_accuracy_pct=_percent(sum(score.intent_correct for score in scores), count),
        in_scope_accuracy_pct=_optional_percent(sum(score.intent_correct for score in in_scope), len(in_scope)),
        topic_accuracy_pct=_optional_percent(sum(score.topic_correct for score in in_scope), len(in_scope)),
        oos_recall_pct=_optional_percent(sum(score.intent == OUT_OF_SCOPE for score in oos), len(oos)),
        false_handoff_pct=_optional_percent(sum(score.intent == OUT_OF_SCOPE for score in in_scope), len(in_scope)),
        parse_success_rate_pct=_percent(sum(score.parse_success for score in scores), count),
        lat_mean_ms=statistics.mean(latencies),
        lat_p50_ms=_percentile(latencies, 0.50),
        lat_p95_ms=_percentile(latencies, 0.95),
        total_cost_usd=total_cost,
        cost_per_1k_usd=(total_cost / count) * 1000.0,
        agreement_with_openai_pct=agreement,
        handoff_bands=handoff_bands(scores),
    )


def handoff_bands(scores: List[SampleScore]) -> Optional[List[HandoffBand]]:
    """A request is handed to a person when the model says out of scope or its confidence
    at either step is below the threshold. Handled accuracy counts an out-of-scope
    request the model kept as wrong."""
    if not any(score.confidence is not None for score in scores):
        return None
    oos = [score for score in scores if not score.in_scope]
    bands = []
    for threshold in CONFIDENCE_THRESHOLDS:
        handled = [score for score in scores if not _handed_off(score, threshold)]
        bands.append(
            HandoffBand(
                threshold=threshold,
                handled_pct=_percent(len(handled), len(scores)),
                handled_accuracy_pct=_optional_percent(sum(score.intent_correct for score in handled), len(handled)),
                oos_caught_pct=_optional_percent(sum(_handed_off(score, threshold) for score in oos), len(oos)),
            )
        )
    return bands


def _handed_off(score: SampleScore, threshold: float) -> bool:
    if score.intent is None or score.intent == OUT_OF_SCOPE:
        return True
    return score.confidence is not None and score.confidence < threshold


def _percent(hits: int, total: int) -> float:
    return (hits / total) * 100.0 if total else 0.0


def _optional_percent(hits: int, total: int) -> Optional[float]:
    return (hits / total) * 100.0 if total else None


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
