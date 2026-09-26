from typing import Optional

from config import ACTIONABLE_CATEGORIES
from schema import DataSample, PipelineExecutionResult
from models.baseline_client import BaselineClient, BaselineClassificationResult, DownstreamResult

class MonolithicPipeline:
    """
    Architecture A: Monolithic GenAI Pipeline
    All decisions & downstream tasks route through the heavy generative LLM (GPT-4o).
    """
    def __init__(self, baseline_client: BaselineClient):
        self.baseline_client = baseline_client

    def run_sample(self, sample: DataSample) -> PipelineExecutionResult:
        # Step 1: Classification via GPT-4o
        class_res: BaselineClassificationResult = self.baseline_client.classify(sample.text)

        is_correct = (class_res.predicted_letter == sample.ground_truth_letter)
        should_execute_downstream = class_res.predicted_category in ACTIONABLE_CATEGORIES

        downstream_latency_ms = 0.0
        downstream_cost_usd = 0.0
        downstream_in_tok = 0
        downstream_out_tok = 0
        downstream_text = None

        # Step 2: Selective downstream generation via GPT-4o
        if should_execute_downstream:
            down_res: DownstreamResult = self.baseline_client.generate_downstream(
                sample.text, class_res.predicted_category
            )
            downstream_latency_ms = down_res.latency_ms
            downstream_cost_usd = down_res.cost_usd
            downstream_in_tok = down_res.input_tokens
            downstream_out_tok = down_res.output_tokens
            downstream_text = down_res.content

        total_latency_ms = class_res.latency_ms + downstream_latency_ms
        total_cost_usd = class_res.cost_usd + downstream_cost_usd

        return PipelineExecutionResult(
            sample_id=sample.sample_id,
            architecture="Monolithic (GPT-4o)",
            ground_truth_category=sample.ground_truth_name,
            predicted_category=class_res.predicted_category,
            predicted_letter=class_res.predicted_letter,
            is_correct=is_correct,
            downstream_executed=should_execute_downstream,
            classification_latency_ms=class_res.latency_ms,
            downstream_latency_ms=downstream_latency_ms,
            total_latency_ms=total_latency_ms,
            classification_cost_usd=class_res.cost_usd,
            downstream_cost_usd=downstream_cost_usd,
            total_cost_usd=total_cost_usd,
            classification_input_tokens=class_res.input_tokens,
            classification_output_tokens=class_res.output_tokens,
            downstream_input_tokens=downstream_in_tok,
            downstream_output_tokens=downstream_out_tok,
            parse_success=class_res.parse_success,
            downstream_output=downstream_text
        )
