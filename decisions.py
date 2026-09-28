"""Customer-support decisions shared by every model.

Queue is limited to the ten helpdesk departments in the dataset. The other
queue strings are vertical categories (Sports, News, and so on), and Tev
accepts at most 24 options, so those rows are left out of the comparison.
"""

from dataclasses import dataclass
import json
import re
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Label:
    key: str
    name: str
    description: str


QUEUES: tuple[Label, ...] = (
    Label("technical_support", "Technical Support", "Bugs, errors, login failures, and technical troubleshooting."),
    Label("product_support", "Product Support", "How a product works, features, setup, and compatibility."),
    Label("customer_service", "Customer Service", "General customer inquiries and service requests."),
    Label("it_support", "IT Support", "Internal IT, accounts, infrastructure, and system administration."),
    Label("billing_and_payments", "Billing and Payments", "Invoices, charges, subscriptions, refunds, and failed payments."),
    Label("returns_and_exchanges", "Returns and Exchanges", "Returning or exchanging a product."),
    Label("service_outages_and_maintenance", "Service Outages and Maintenance", "Outages, downtime, and planned maintenance."),
    Label("sales_and_pre_sales", "Sales and Pre-Sales", "Pricing, purchases, demos, and questions before buying."),
    Label("human_resources", "Human Resources", "Employees, hiring, payroll, and other HR questions."),
    Label("general_inquiry", "General Inquiry", "A general question that does not belong to another team."),
)

# Ordered from least urgent to most urgent. Index 0 is the low end of a score.
PRIORITIES: tuple[Label, ...] = (
    Label("very_low", "very_low", "Routine. No deadline and no real impact."),
    Label("low", "low", "Non-urgent. A minor inconvenience or a general question."),
    Label("medium", "medium", "Needs a timely reply. Degraded performance or a detailed question."),
    Label("high", "high", "Important. The customer is blocked or a service is impaired."),
    Label("critical", "critical", "Immediate. Outage, security incident, data loss, or safety risk."),
)

TYPES: tuple[Label, ...] = (
    Label("incident", "Incident", "An unexpected break that needs attention."),
    Label("request", "Request", "A routine inquiry or a request for something new."),
    Label("problem", "Problem", "An underlying cause of repeated incidents."),
    Label("change", "Change", "A planned change, update, or modification."),
)

MAX_BODY_CHARS = 2000
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _aliases(labels: Sequence[Label]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for label in labels:
        for raw in (label.key, label.name):
            mapping[_normalize_token(raw)] = label.key
    return mapping


def _normalize_token(value: str) -> str:
    token = value.strip().lower().replace("&", " and ")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


QUEUE_ALIASES = _aliases(QUEUES)
PRIORITY_ALIASES = _aliases(PRIORITIES)
TYPE_ALIASES = _aliases(TYPES)
QUEUE_BY_NAME = {label.name: label.key for label in QUEUES}
PRIORITY_INDEX = {label.key: index for index, label in enumerate(PRIORITIES)}


def resolve_label(value: Optional[str], aliases: Mapping[str, str]) -> Optional[str]:
    if value is None:
        return None
    token = _normalize_token(str(value))
    if not token:
        return None
    return aliases.get(token)


def ticket_text(subject: object, body: object) -> Optional[str]:
    subject_text = "" if subject is None else str(subject).strip()
    body_text = "" if body is None else str(body).strip()
    if not subject_text and not body_text:
        return None
    if len(body_text) > MAX_BODY_CHARS:
        body_text = body_text[:MAX_BODY_CHARS].rstrip() + "..."
    return f"Subject: {subject_text}\n\n{body_text}"


def choice_criteria(labels: Sequence[Label]) -> Dict[str, str]:
    return {label.key: label.description for label in labels}


def score_criteria(labels: Sequence[Label]) -> List[str]:
    return [f"{label.name}: {label.description}" for label in labels]


def systemone_questions() -> Dict[str, dict]:
    return {
        "queue": {
            "type": "choice",
            "instructions": "Which department should handle this customer support ticket?",
            "criteria": choice_criteria(QUEUES),
        },
        "priority": {
            "type": "score",
            "instructions": "How urgent is this customer support ticket?",
            "criteria": score_criteria(PRIORITIES),
        },
        "type": {
            "type": "choice",
            "instructions": "What kind of customer support ticket is this?",
            "criteria": choice_criteria(TYPES),
        },
    }


def tev_options(labels: Sequence[Label]) -> List[dict]:
    if not 2 <= len(labels) <= 24:
        raise ValueError(f"Tev accepts 2 to 24 options, got {len(labels)}.")
    return [
        {"label": OPTION_LETTERS[index], "key": label.key, "description": label.description}
        for index, label in enumerate(labels)
    ]


def tev_task(state: str, question: str, labels: Sequence[Label]) -> str:
    payload = {"state": state, "question": question, "options": tev_options(labels)}
    return json.dumps(payload, ensure_ascii=False)


def parse_option_letter(raw: str, labels: Sequence[Label]) -> Optional[str]:
    if not raw:
        return None
    allowed = {OPTION_LETTERS[index]: label.key for index, label in enumerate(labels)}
    match = re.search(r"\b([A-Z])\b", raw.upper())
    if match and match.group(1) in allowed:
        return allowed[match.group(1)]
    return resolve_label(raw, _aliases(labels))


def priority_from_score(answer: Mapping[str, object]) -> Optional[str]:
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, Mapping) and probabilities:
        best_key, _best_value = max(probabilities.items(), key=lambda item: float(item[1]))
        if str(best_key).isdigit():
            index = int(str(best_key))
            if 0 <= index < len(PRIORITIES):
                return PRIORITIES[index].key
        resolved = resolve_label(str(best_key), PRIORITY_ALIASES)
        if resolved:
            return resolved

    score = answer.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        index = int(round(float(score)))
        index = max(0, min(len(PRIORITIES) - 1, index))
        return PRIORITIES[index].key
    return None


def priority_distance(predicted: Optional[str], gold: str) -> Optional[int]:
    if predicted is None or predicted not in PRIORITY_INDEX or gold not in PRIORITY_INDEX:
        return None
    return abs(PRIORITY_INDEX[predicted] - PRIORITY_INDEX[gold])


def eligible_language(language: object, requested: str) -> bool:
    if requested == "all":
        return language in {"en", "de"}
    return language == requested

