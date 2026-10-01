import time
from typing import Sequence

from decisions import Label, choice_question
from models.systemone_client import answer_from_systemone
from models.two_step import two_step_decide
from schema import Answer, DecisionOutput


class LayaCoreClient:
    """Local Laya Core ML. Weights download once, then every decision stays on this Mac."""

    def __init__(self, model: str):
        import laya_coreml

        self.name = "laya"
        self.model_id = model
        print(f"[*] Loading Laya Core ML from {model}. The first launch downloads the bundle and compiles it.")
        self.agent = laya_coreml.load(model)
        self.agent.predict("ready", {"ready": {"type": "noul", "instructions": "Is this text non-empty?"}})
        print("[✓] Laya Core ML is ready.")

    def decide(self, text: str) -> DecisionOutput:
        return two_step_decide(self.name, text, self.ask, lambda _in, _out: 0.0)

    def ask(self, text: str, question: str, labels: Sequence[Label]) -> Answer:
        started = time.perf_counter()
        payload = self.agent.predict(text, {"answer": choice_question(question, labels)})
        return answer_from_systemone(payload, labels, (time.perf_counter() - started) * 1000.0)
