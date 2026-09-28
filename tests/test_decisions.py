import unittest

from data_loader import normalize_row, select_samples
from decisions import PRIORITIES, QUEUES, parse_option_letter, priority_from_score, systemone_questions, tev_options
from metrics import score_output, summarize
from schema import DecisionOutput, TicketSample


def _ticket(**overrides):
    row = {
        "subject": "Charged twice",
        "body": "Please refund invoice 4411.",
        "answer": "SECRET AGENT ANSWER",
        "type": "Request",
        "queue": "Billing and Payments",
        "priority": "high",
        "language": "en",
    }
    row.update(overrides)
    return row


class DecisionTests(unittest.TestCase):
    def test_keeps_department_ticket_and_hides_the_agent_answer(self):
        sample = normalize_row(_ticket(), "en")
        self.assertIsNotNone(sample)
        self.assertEqual(sample.queue, "billing_and_payments")
        self.assertEqual(sample.priority, "high")
        self.assertEqual(sample.ticket_type, "request")
        self.assertNotIn("SECRET AGENT ANSWER", sample.text)

    def test_drops_vertical_queues_and_other_languages(self):
        self.assertIsNone(normalize_row(_ticket(queue="Sports"), "en"))
        self.assertIsNone(normalize_row(_ticket(language="de"), "en"))
        german = normalize_row(_ticket(language="de"), "all")
        self.assertEqual(german.language, "de")

    def test_missing_type_stays_unscored(self):
        sample = normalize_row(_ticket(type=None), "en")
        output = DecisionOutput("tev", sample.queue, sample.priority, "incident", 10, 1, 1, 0.0, True, "ok")
        scored = score_output(sample, output)
        self.assertIsNone(scored.type_correct)
        self.assertTrue(scored.queue_correct)
        self.assertEqual(scored.priority_abs_error, 0)

    def test_priority_score_uses_the_most_likely_level(self):
        label = priority_from_score({"score": 1.2, "probabilities": {"0": 0.1, "1": 0.2, "3": 0.7}})
        self.assertEqual(label, "high")
        self.assertEqual(priority_from_score({"score": 0.4}), "very_low")

    def test_tev_letter_maps_back_to_the_department(self):
        self.assertEqual(parse_option_letter("C", QUEUES), QUEUES[2].key)
        self.assertEqual(len(tev_options(QUEUES)), 10)
        self.assertLessEqual(len(tev_options(PRIORITIES)), 24)

    def test_systemone_asks_for_every_decision_at_once(self):
        questions = systemone_questions()
        self.assertEqual(questions["queue"]["type"], "choice")
        self.assertEqual(questions["priority"]["type"], "score")
        self.assertEqual(len(questions["priority"]["criteria"]), 5)
        self.assertEqual(questions["type"]["type"], "choice")

    def test_select_samples_is_stable_and_bounded(self):
        rows = [_ticket(subject=f"Ticket {index}", priority="low" if index % 2 == 0 else "critical") for index in range(8)]
        rows.append(_ticket(queue="News"))
        first = select_samples(rows, num_samples=3, seed=7, language="en")
        second = select_samples(rows, num_samples=3, seed=7, language="en")
        self.assertEqual([item.text for item in first], [item.text for item in second])
        self.assertEqual([item.sample_id for item in first], [1, 2, 3])

    def test_summary_counts_a_missed_parse_as_incorrect(self):
        sample = TicketSample(1, "Subject: Hi\n\nBody", "it_support", "low", "incident", "en")
        missed = DecisionOutput("jev", None, None, None, 5, 0, 0, 0, False, "error: down")
        summary = summarize([score_output(sample, missed)])
        self.assertEqual(summary.queue_accuracy_pct, 0.0)
        self.assertEqual(summary.parse_success_rate_pct, 0.0)
        self.assertIsNone(summary.priority_mae)


if __name__ == "__main__":
    unittest.main()
