from typing import Callable, List, Optional, Sequence

from decisions import OUT_OF_SCOPE, TOPIC_QUESTION, TOPICS, Label, intent_options, intent_question
from schema import Answer, DecisionOutput

Ask = Callable[[str, str, Sequence[Label]], Answer]
Cost = Callable[[int, int], float]


def two_step_decide(name: str, text: str, ask: Ask, cost: Cost) -> DecisionOutput:
    """Topic first, then the intent within that topic. Every model answers the same two
    questions; out of scope at either step ends the request."""
    answers: List[Answer] = []
    topic: Optional[str] = None
    intent: Optional[str] = None
    try:
        topic_answer = ask(text, TOPIC_QUESTION, TOPICS)
        answers.append(topic_answer)
        topic = topic_answer.key
        intent_answer: Optional[Answer] = None
        if topic == OUT_OF_SCOPE:
            intent = OUT_OF_SCOPE
        elif topic is not None:
            intent_answer = ask(text, intent_question(topic), intent_options(topic))
            answers.append(intent_answer)
            intent = intent_answer.key
    except Exception as error:
        return _output(name, answers, topic, None, cost, raw=f"error: {error}", parsed=False)

    return _output(
        name,
        answers,
        topic,
        intent,
        cost,
        raw=" | ".join(answer.raw for answer in answers),
        parsed=topic is not None and intent is not None,
        topic_confidence=answers[0].confidence if answers else None,
        intent_confidence=intent_answer.confidence if intent_answer else None,
    )


def _output(
    name: str,
    answers: List[Answer],
    topic: Optional[str],
    intent: Optional[str],
    cost: Cost,
    *,
    raw: str,
    parsed: bool,
    topic_confidence: Optional[float] = None,
    intent_confidence: Optional[float] = None,
) -> DecisionOutput:
    input_tokens = sum(answer.input_tokens for answer in answers)
    output_tokens = sum(answer.output_tokens for answer in answers)
    return DecisionOutput(
        model=name,
        topic=topic,
        intent=intent,
        latency_ms=sum(answer.latency_ms for answer in answers),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost(input_tokens, output_tokens),
        parse_success=parsed,
        raw_response=raw,
        topic_confidence=topic_confidence,
        intent_confidence=intent_confidence,
    )
