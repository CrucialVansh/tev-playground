import time

from decisions import systemone_questions
from models.systemone_client import decision_from_systemone
from schema import DecisionOutput


class LayaCoreClient:
    """Local Laya Core ML. Weights download once, then every decision stays on this Mac."""

    def __init__(self, model: str):
        import laya_coreml

        self.name = "laya"
        self.model_id = model
        self.questions = systemone_questions()
        print(f"[*] Loading Laya Core ML from {model}. The first launch downloads the bundle and compiles it.")
        self.agent = laya_coreml.load(model)
        self.agent.predict("ready", {"ready": {"type": "noul", "instructions": "Is this text non-empty?"}})
        print("[✓] Laya Core ML is ready.")

    def decide(self, text: str) -> DecisionOutput:
        started = time.perf_counter()
        try:
            payload = self.agent.predict(text, self.questions)
        except Exception as error:
            return DecisionOutput(
                model=self.name,
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
        return decision_from_systemone(
            self.name,
            payload,
            (time.perf_counter() - started) * 1000.0,
            cost_usd=0.0,
        )
