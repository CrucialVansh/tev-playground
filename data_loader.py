import random
from typing import Iterable, List, Optional

from decisions import PRIORITY_ALIASES, QUEUE_BY_NAME, TYPE_ALIASES, eligible_language, resolve_label, ticket_text
from schema import TicketSample


def load_ticket_samples(num_samples: int = 50, seed: int = 42, language: str = "en") -> List[TicketSample]:
    from datasets import load_dataset

    from config import DATASET_ID

    print(f"[*] Loading {DATASET_ID} ({language} tickets in the ten support departments)...")
    dataset = load_dataset(DATASET_ID, split="train")
    samples = select_samples(dataset, num_samples=num_samples, seed=seed, language=language)
    print(f"[✓] Selected {len(samples)} tickets.")
    return samples


def select_samples(rows: Iterable[dict], num_samples: int, seed: int, language: str) -> List[TicketSample]:
    eligible: List[TicketSample] = []
    for row in rows:
        sample = normalize_row(row, language)
        if sample is not None:
            eligible.append(sample)

    if not eligible:
        raise RuntimeError(f"No tickets matched language={language!r} and the ten support departments.")

    random.Random(seed).shuffle(eligible)
    chosen = eligible[:num_samples]
    for index, sample in enumerate(chosen, start=1):
        sample.sample_id = index
    return chosen


def normalize_row(row: dict, language: str) -> Optional[TicketSample]:
    row_language = row.get("language")
    if not eligible_language(row_language, language):
        return None

    queue = QUEUE_BY_NAME.get(row.get("queue"))
    priority = resolve_label(None if row.get("priority") is None else str(row.get("priority")), PRIORITY_ALIASES)
    text = ticket_text(row.get("subject"), row.get("body"))
    if queue is None or priority is None or text is None:
        return None

    ticket_type = resolve_label(None if row.get("type") is None else str(row.get("type")), TYPE_ALIASES)
    return TicketSample(
        sample_id=0,
        text=text,
        queue=queue,
        priority=priority,
        ticket_type=ticket_type,
        language=str(row_language),
    )
