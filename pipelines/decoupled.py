from config import ACTIONABLE_CATEGORIES
from schema import DataSample, PipelineExecutionResult
from models.tev_client import TevClient, TevClassificationResult
from models.baseline_client import BaselineClient, DownstreamResult

class DecoupledPipeline:
    """
    Architecture B: Decoupled System-1 Offload Pipeline
    Fast, structured classification is delegated to Tev (together/Tev1-4B-experimental).
    GPT-4o is selectively invoked only when downstream generative synthesis is required.
    """
    def __init__(self, tev_client: TevClient, baseline_client: BaselineClient):
        self.tev_client = tev_client
        self.baseline_client = baseline_client

    def run_sample(self, sample: DataSample) -> PipelineExecutionResult:
        # Step 1: Fast System-1 Classification via Tev
        tev_res: TevClassificationResult = self.tev_client.classify(sample.text)

        is_correct = (tev_res.predicted_letter == sample.ground_truth_letter)
        should_execute_downstream = tev_res.predicted_category in ACTIONABLE_CATEGORIES

        downstream_latency_ms = 0.0
        downstream_cost_usd = 0.0
        downstream_in_tok = 0
        downstream_out_tok = 0
        downstream_text = None

        # Step 2: Downstream generative action triggered only if category is actionable
        if should_execute_downstream:
            down_res: DownstreamResult = self.baseline_client.generate_downstream(
                sample.text, tev_res.predicted_category
            )
            downstream_latency_ms = down_res.latency_ms
            downstream_cost_usd = down_res.cost_usd
            downstream_in_tok = down_res.input_tokens
            downstream_out_tok = down_res.output_tokens
            downstream_text = down_res.content

        total_latency_ms = tev_res.latency_ms + downstream_latency_ms
        total_cost_usd = tev_res.cost_usd + downstream_cost_usd

        return PipelineExecutionResult(
            sample_id=sample.sample_id,
            architecture="Decoupled (Tev1 + GPT-4o)",
            ground_truth_category=sample.ground_truth_name,
            predicted_category=tev_res.predicted_category,
            predicted_letter=tev_res.predicted_letter,
            is_correct=is_correct,
            downstream_executed=should_execute_downstream,
            classification_latency_ms=tev_res.latency_ms,
            downstream_latency_ms=downstream_latency_ms,
            total_latency_ms=total_latency_ms,
            classification_cost_usd=tev_res.cost_usd,
            downstream_cost_usd=downstream_cost_usd,
            total_cost_usd=total_cost_usd,
            classification_input_tokens=tev_res.input_tokens,
            classification_output_tokens=tev_res.output_tokens,
            downstream_input_tokens=downstream_in_tok,
            downstream_output_tokens=downstream_out_tok,
            parse_success=tev_res.parse_success,
            downstream_output=downstream_text
        )
