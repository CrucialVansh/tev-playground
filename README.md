# Intent routing: Tev, Jev, Laya Core ML, and GPT-4o on CLINC150

This benchmark scores four models on the same requests from [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) (the `plus` test split): 150 intents across ten topics, plus requests that fit none of them.

Every model answers the same two questions:

1. **Topic.** One of CLINC's ten topics (banking, credit cards, travel, work, and so on), or out of scope.
2. **Intent.** One of that topic's fifteen intents, or none of them.

An out-of-scope answer at either step means the request goes to a person. The topic grouping is CLINC's own [`domains.json`](https://github.com/clinc/oos-eval/blob/master/data/domains.json). No question has more than sixteen options, so Tev (24 at most) and the Laya Core ML export (32 at most) get exactly the same task as Jev and GPT-4o.

- **Tev** answers each question with an option letter, one call per step.
- **Jev** answers each question as a `choice` in one `/v1/systemone` request per step.
- **Laya Core ML** is the local Core ML port of the English Laya checkpoint, [`aac6fef/laya-coreml`](https://huggingface.co/aac6fef/laya-coreml). It downloads once, then runs on this Mac with no API key and no token charge.
- **GPT-4o** returns the chosen option key as JSON.

The sample is balanced across the 150 intents, and 20% of it is out of scope by default.

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

`uv sync` installs Python 3.13 if needed, creates `.venv`, and installs the locked dependencies from `uv.lock`.

`.env` needs `TOGETHER_API_KEY`, `TYPESAFE_API_KEY`, and `OPENAI_API_KEY`. Laya Core ML does not use a key.

`laya-coreml` runs on Python 3.13. This project is pinned to that version (`.python-version` and `requires-python` in `pyproject.toml`) because the Core ML runtime does not ship a working build for 3.14.

Without uv, `pip install -r requirements.txt` in a Python 3.13 virtual environment also works.

## Run

```bash
uv run run_experiment.py
```

```bash
uv run run_experiment.py --samples 1000 --oos-fraction 0.2 --seed 42 --output experiment_results.json
```

The first table reports:

- **Overall accuracy.** The standard CLINC150 score: the exact intent for in-scope requests, and out of scope for the rest.
- **In-scope intent and topic accuracy.**
- **Out of scope caught.** The share of out-of-scope requests the model sent to a person.
- **In scope sent to a person.** In-scope requests the model wrongly gave up on.
- Parse rate, latency (both steps summed), cost per 1,000 requests, and intent agreement with GPT-4o.

The "always out of scope" column shows what giving up on every request would score.

A second table covers Jev and Laya, which return a confidence for each answer. At confidence thresholds of 0.5, 0.7, and 0.9, it also sends answers below the threshold to a person. It shows the share of requests handled automatically, the accuracy on those, and the share of out-of-scope requests caught. Tev and GPT-4o return no confidence.

Set `LAYA_MODEL` to another Core ML bundle, such as `aac6fef/laya-typed-decisions-coreml`, or to a directory created with `hf download`.

## Results

From `uv run run_experiment.py --samples 1000 --oos-fraction 0.2 --seed 42` (800 in scope, 200 out of scope, single run, 2026-10-01). Always answering out of scope would score 20.0%.

| Model | Overall | In-scope intent | Topic | Out of scope caught | In scope sent to a person | Mean / p50 / p95 latency (ms) | Cost per 1,000 | Agreement with GPT-4o |
|---|---|---|---|---|---|---|---|---|
| Jev | **83.8%** | **84.0%** | **86.8%** | 83.0% | **10.6%** | 575 / 523 / 715 | $0.050 | 87.3% |
| GPT-4o | 81.9% | 79.1% | 84.5% | 93.0% | 17.8% | 1,475 / 1,438 / 2,347 | $1.479 | — |
| Tev | 77.4% | 73.4% | 80.5% | 93.5% | 19.5% | **560** / **519** / 745 | $0.038 | 83.2% |
| Laya Core ML | 30.3% | 13.6% | 15.5% | 97.0% | 84.5% | 626 / 522 / 1,084 | $0 (local) | 41.0% |

All four models parsed 100% of responses.

Confidence thresholds (answers below the threshold go to a person):

| Model | Threshold | Handled automatically | Accuracy on handled | Out of scope caught |
|---|---|---|---|---|
| Jev | 0.5 | 71.6% | 91.6% | 86.5% |
| Jev | 0.7 | 64.3% | 94.6% | 94.0% |
| Jev | 0.9 | 53.9% | 97.0% | 97.0% |
| Laya Core ML | 0.5 | 13.0% | 83.8% | 97.0% |
| Laya Core ML | 0.7 | 12.5% | 84.0% | 97.0% |
| Laya Core ML | 0.9 | 9.6% | 86.5% | 98.0% |

Takeaways:

- Jev is the most accurate and gives up on the fewest valid requests, but catches the fewest out-of-scope requests of the API models.
- Tev and Jev run at about 40% of GPT-4o's latency and cost roughly 30–39× less.
- Jev's confidence is useful for routing: at 0.9 it handles 54% of requests alone at 97% accuracy.
- Laya Core ML costs nothing to run and is about as fast, but it answers out of scope for most requests, so its high out-of-scope catch rate comes with 84.5% of valid requests sent to a person.
