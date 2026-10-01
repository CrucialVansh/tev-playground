import unittest
from collections import Counter

from data_loader import normalize_row, select_samples
from decisions import (
    INTENTS,
    OUT_OF_SCOPE,
    TOPIC_BY_INTENT,
    TOPICS,
    intent_options,
    parse_option_letter,
    tev_options,
)
from metrics import score_output, summarize
from models.systemone_client import answer_from_systemone
from models.two_step import two_step_decide
from schema import Answer, DecisionOutput, QuerySample


def _answer(key, confidence=None):
    return Answer(key=key, confidence=confidence, raw=str(key), latency_ms=10.0, input_tokens=5, output_tokens=1)


def _scripted(*answers):
    queue = list(answers)
    asked = []

    def ask(text, question, labels):
        asked.append([label.key for label in labels])
        return queue.pop(0)

    return ask, asked


class LabelTests(unittest.TestCase):
    def test_clinc_topics_cover_all_150_intents(self):
        self.assertEqual(len(TOPIC_BY_INTENT), 150)
        self.assertTrue(all(len(intents) == 15 for intents in INTENTS.values()))

    def test_every_question_fits_tev_and_laya(self):
        self.assertLessEqual(len(tev_options(TOPICS)), 24)
        for topic in INTENTS:
            options = intent_options(topic)
            self.assertEqual(options[-1].key, OUT_OF_SCOPE)
            self.assertLessEqual(len(tev_options(options)), 24)

    def test_tev_letter_maps_back_to_the_option(self):
        self.assertEqual(parse_option_letter("B", TOPICS), TOPICS[1].key)


class TwoStepTests(unittest.TestCase):
    def test_asks_the_intent_within_the_chosen_topic(self):
        ask, asked = _scripted(_answer("banking", 0.9), _answer("balance", 0.6))
        output = two_step_decide("jev", "how much is in my checking", ask, lambda _in, _out: 0.0)
        self.assertEqual((output.topic, output.intent), ("banking", "balance"))
        self.assertEqual(asked[1], [label.key for label in intent_options("banking")])
        self.assertEqual((output.topic_confidence, output.intent_confidence), (0.9, 0.6))
        self.assertEqual(output.latency_ms, 20.0)
        self.assertTrue(output.parse_success)

    def test_out_of_scope_topic_skips_the_second_question(self):
        ask, asked = _scripted(_answer(OUT_OF_SCOPE, 0.8))
        output = two_step_decide("tev", "what is the price of bitcoin", ask, lambda _in, _out: 0.0)
        self.assertEqual(output.intent, OUT_OF_SCOPE)
        self.assertEqual(len(asked), 1)

    def test_an_error_is_an_unparsed_decision(self):
        def ask(text, question, labels):
            raise RuntimeError("down")

        output = two_step_decide("jev", "hi", ask, lambda _in, _out: 0.0)
        self.assertFalse(output.parse_success)
        self.assertTrue(output.raw_response.startswith("error:"))

    def test_systemone_answer_reads_choice_and_confidence(self):
        payload = {"answers": {"answer": {"choice": "banking", "confidence": 0.42}}, "usage": {"input_tokens": 90}}
        answer = answer_from_systemone(payload, TOPICS, 12.0)
        self.assertEqual((answer.key, answer.confidence, answer.input_tokens), ("banking", 0.42, 90))


class DataTests(unittest.TestCase):
    def test_normalize_row_adds_the_topic(self):
        self.assertEqual(normalize_row({"text": "freeze my account", "intent": "freeze_account"}).topic, "banking")
        self.assertEqual(normalize_row({"text": "who won", "intent": OUT_OF_SCOPE}).topic, OUT_OF_SCOPE)
        self.assertIsNone(normalize_row({"text": "x", "intent": "not_an_intent"}))

    def test_select_samples_balances_intents_and_keeps_the_oos_share(self):
        rows = [{"text": f"b{index}", "intent": "balance"} for index in range(10)]
        rows += [{"text": f"t{index}", "intent": "timer"} for index in range(10)]
        rows += [{"text": f"o{index}", "intent": OUT_OF_SCOPE} for index in range(10)]
        chosen = select_samples(rows, num_samples=10, seed=3, oos_fraction=0.2)
        self.assertEqual(Counter(sample.intent for sample in chosen), {"balance": 4, "timer": 4, OUT_OF_SCOPE: 2})
        again = select_samples(rows, num_samples=10, seed=3, oos_fraction=0.2)
        self.assertEqual([sample.text for sample in chosen], [sample.text for sample in again])
        self.assertEqual(sorted(sample.sample_id for sample in chosen), list(range(1, 11)))


class MetricTests(unittest.TestCase):
    def _output(self, intent, topic=None, confidence=None):
        return DecisionOutput("jev", topic, intent, 5, 0, 0, 0.0, intent is not None, "ok", confidence, confidence)

    def test_summary_splits_in_scope_and_out_of_scope(self):
        samples = [
            QuerySample(1, "a", "balance", "banking"),
            QuerySample(2, "b", "timer", "utility"),
            QuerySample(3, "c", OUT_OF_SCOPE, OUT_OF_SCOPE),
            QuerySample(4, "d", OUT_OF_SCOPE, OUT_OF_SCOPE),
        ]
        outputs = [
            self._output("balance", "banking"),
            self._output(OUT_OF_SCOPE, OUT_OF_SCOPE),
            self._output(OUT_OF_SCOPE, OUT_OF_SCOPE),
            self._output("weather", "utility"),
        ]
        summary = summarize([score_output(sample, output) for sample, output in zip(samples, outputs)])
        self.assertEqual(summary.overall_accuracy_pct, 50.0)
        self.assertEqual(summary.in_scope_accuracy_pct, 50.0)
        self.assertEqual(summary.oos_recall_pct, 50.0)
        self.assertEqual(summary.false_handoff_pct, 50.0)
        self.assertIsNone(summary.handoff_bands)

    def test_handoff_bands_send_low_confidence_answers_to_a_person(self):
        samples = [
            QuerySample(1, "a", "balance", "banking"),
            QuerySample(2, "b", "timer", "utility"),
            QuerySample(3, "c", OUT_OF_SCOPE, OUT_OF_SCOPE),
            QuerySample(4, "d", OUT_OF_SCOPE, OUT_OF_SCOPE),
        ]
        outputs = [
            self._output("balance", "banking", 0.95),
            self._output("alarm", "utility", 0.6),
            self._output("weather", "utility", 0.4),
            self._output(OUT_OF_SCOPE, OUT_OF_SCOPE, 0.99),
        ]
        summary = summarize([score_output(sample, output) for sample, output in zip(samples, outputs)])
        bands = {band.threshold: band for band in summary.handoff_bands}
        self.assertEqual((bands[0.5].handled_pct, bands[0.5].handled_accuracy_pct, bands[0.5].oos_caught_pct), (50.0, 50.0, 100.0))
        self.assertEqual((bands[0.9].handled_pct, bands[0.9].handled_accuracy_pct), (25.0, 100.0))


if __name__ == "__main__":
    unittest.main()
