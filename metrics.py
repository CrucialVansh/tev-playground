from typing import List, Dict, Any
import statistics
from dataclasses import dataclass
from schema import PipelineExecutionResult

@dataclass
class MetricSummary:
    architecture: str
    sample_count: int
    accuracy_pct: float
    
    # Classification Latency (ms)
    class_lat_mean_ms: float
    class_lat_p50_ms: float
    class_lat_p90_ms: float
    class_lat_p95_ms: float
    
    # End-to-End Pipeline Latency (ms)
    e2e_lat_mean_ms: float
    e2e_lat_p50_ms: float
    e2e_lat_p90_ms: float
    e2e_lat_p95_ms: float
    
    # Cost & Economics (USD)
    total_class_cost_usd: float
    total_downstream_cost_usd: float
    total_pipeline_cost_usd: float
    cost_per_1k_classifications_usd: float
    cost_per_1k_pipeline_reqs_usd: float
    
    # Reliability
    parse_success_rate_pct: float
    downstream_triggered_count: int

def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * p
    f = int(k)
    c = min(f + 1, len(data) - 1)
    d = k - f
    sorted_d = sorted(data)
    return sorted_d[f] + d * (sorted_d[c] - sorted_d[f])

def compute_pipeline_metrics(results: List[PipelineExecutionResult], architecture_name: str) -> MetricSummary:
    n = len(results)
    if n == 0:
        raise ValueError("Cannot calculate metrics on empty results list.")

    correct_count = sum(1 for r in results if r.is_correct)
    accuracy_pct = (correct_count / n) * 100.0

    class_latencies = [r.classification_latency_ms for r in results]
    e2e_latencies = [r.total_latency_ms for r in results]

    total_class_cost = sum(r.classification_cost_usd for r in results)
    total_down_cost = sum(r.downstream_cost_usd for r in results)
    total_pipeline_cost = sum(r.total_cost_usd for r in results)

    cost_per_1k_class = (total_class_cost / n) * 1000.0
    cost_per_1k_pipeline = (total_pipeline_cost / n) * 1000.0

    parse_successes = sum(1 for r in results if r.parse_success)
    parse_rate = (parse_successes / n) * 100.0

    downstream_count = sum(1 for r in results if r.downstream_executed)

    return MetricSummary(
        architecture=architecture_name,
        sample_count=n,
        accuracy_pct=accuracy_pct,
        class_lat_mean_ms=statistics.mean(class_latencies),
        class_lat_p50_ms=_percentile(class_latencies, 0.50),
        class_lat_p90_ms=_percentile(class_latencies, 0.90),
        class_lat_p95_ms=_percentile(class_latencies, 0.95),
        e2e_lat_mean_ms=statistics.mean(e2e_latencies),
        e2e_lat_p50_ms=_percentile(e2e_latencies, 0.50),
        e2e_lat_p90_ms=_percentile(e2e_latencies, 0.90),
        e2e_lat_p95_ms=_percentile(e2e_latencies, 0.95),
        total_class_cost_usd=total_class_cost,
        total_downstream_cost_usd=total_down_cost,
        total_pipeline_cost_usd=total_pipeline_cost,
        cost_per_1k_classifications_usd=cost_per_1k_class,
        cost_per_1k_pipeline_reqs_usd=cost_per_1k_pipeline,
        parse_success_rate_pct=parse_rate,
        downstream_triggered_count=downstream_count
    )

def compute_architectural_comparison(
    mono_results: List[PipelineExecutionResult],
    decoupled_results: List[PipelineExecutionResult]
) -> Dict[str, Any]:
    """
    Computes delta and comparative metrics between Monolithic and Decoupled architectures.
    """
    mono_summary = compute_pipeline_metrics(mono_results, "Monolithic (GPT-4o)")
    decoupled_summary = compute_pipeline_metrics(decoupled_results, "Decoupled (Tev1 + GPT-4o)")

    # Agreement rate
    paired_count = min(len(mono_results), len(decoupled_results))
    agreements = sum(
        1 for i in range(paired_count)
        if mono_results[i].predicted_letter == decoupled_results[i].predicted_letter
    )
    agreement_rate_pct = (agreements / paired_count * 100.0) if paired_count > 0 else 0.0

    # Speedups
    class_speedup = (
        mono_summary.class_lat_mean_ms / decoupled_summary.class_lat_mean_ms
        if decoupled_summary.class_lat_mean_ms > 0 else 1.0
    )
    e2e_speedup = (
        mono_summary.e2e_lat_mean_ms / decoupled_summary.e2e_lat_mean_ms
        if decoupled_summary.e2e_lat_mean_ms > 0 else 1.0
    )

    # Cost savings
    class_cost_reduction_pct = (
        (1.0 - (decoupled_summary.cost_per_1k_classifications_usd / mono_summary.cost_per_1k_classifications_usd)) * 100.0
        if mono_summary.cost_per_1k_classifications_usd > 0 else 0.0
    )
    pipeline_cost_reduction_pct = (
        (1.0 - (decoupled_summary.cost_per_1k_pipeline_reqs_usd / mono_summary.cost_per_1k_pipeline_reqs_usd)) * 100.0
        if mono_summary.cost_per_1k_pipeline_reqs_usd > 0 else 0.0
    )

    return {
        "monolithic": mono_summary,
        "decoupled": decoupled_summary,
        "agreement_rate_pct": agreement_rate_pct,
        "classification_latency_speedup": class_speedup,
        "pipeline_latency_speedup": e2e_speedup,
        "classification_cost_reduction_pct": class_cost_reduction_pct,
        "pipeline_cost_reduction_pct": pipeline_cost_reduction_pct,
    }
