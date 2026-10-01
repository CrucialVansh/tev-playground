import random
from typing import Dict, Iterable, List, Optional, Sequence

from decisions import OUT_OF_SCOPE, TOPIC_BY_INTENT, query_text
from schema import QuerySample


def load_query_samples(num_samples: int = 400, seed: int = 42, oos_fraction: float = 0.2) -> List[QuerySample]:
    from datasets import load_dataset

    from config import DATASET_CONFIG, DATASET_ID, DATASET_SPLIT

    print(f"[*] Loading {DATASET_ID} ({DATASET_CONFIG}, {DATASET_SPLIT} split)...")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
    names = dataset.features["intent"].names
    rows = ({"text": row["text"], "intent": names[row["intent"]]} for row in dataset)
    samples = select_samples(rows, num_samples=num_samples, seed=seed, oos_fraction=oos_fraction)
    in_scope = sum(sample.intent != OUT_OF_SCOPE for sample in samples)
    print(f"[✓] Selected {len(samples)} requests: {in_scope} in scope, {len(samples) - in_scope} out of scope.")
    return samples


def select_samples(rows: Iterable[dict], num_samples: int, seed: int, oos_fraction: float) -> List[QuerySample]:
    in_scope: List[QuerySample] = []
    out_of_scope: List[QuerySample] = []
    for row in rows:
        sample = normalize_row(row)
        if sample is None:
            continue
        (out_of_scope if sample.intent == OUT_OF_SCOPE else in_scope).append(sample)

    if not in_scope:
        raise RuntimeError("No in-scope requests found.")

    shuffler = random.Random(seed)
    shuffler.shuffle(in_scope)
    shuffler.shuffle(out_of_scope)
    oos_count = min(len(out_of_scope), round(num_samples * oos_fraction))
    chosen = _balanced(in_scope, num_samples - oos_count, key=lambda sample: sample.intent) + out_of_scope[:oos_count]
    shuffler.shuffle(chosen)
    for index, sample in enumerate(chosen, start=1):
        sample.sample_id = index
    return chosen


def _balanced(samples: Sequence[QuerySample], count: int, key) -> List[QuerySample]:
    """Takes one request per intent in turn, so every intent is equally represented
    until a small one runs out."""
    groups: Dict[str, List[QuerySample]] = {}
    for sample in samples:
        groups.setdefault(key(sample), []).append(sample)

    chosen: List[QuerySample] = []
    pools = list(groups.values())
    depth = 0
    while len(chosen) < count and any(depth < len(pool) for pool in pools):
        for pool in pools:
            if depth < len(pool) and len(chosen) < count:
                chosen.append(pool[depth])
        depth += 1
    return chosen


def normalize_row(row: dict) -> Optional[QuerySample]:
    text = query_text(row.get("text"))
    intent = row.get("intent")
    if text is None or intent is None:
        return None
    if intent == OUT_OF_SCOPE:
        topic = OUT_OF_SCOPE
    elif intent in TOPIC_BY_INTENT:
        topic = TOPIC_BY_INTENT[intent]
    else:
        return None
    return QuerySample(sample_id=0, text=text, intent=intent, topic=topic)
