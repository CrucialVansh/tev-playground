# Customer support decisions: Tev, Jev, Laya Core ML, and GPT-4o

This benchmark scores four models on the same support tickets from [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets).

Each ticket is judged on three decisions:

- **Queue.** Which of ten helpdesk departments should take it.
- **Priority.** `very_low`, `low`, `medium`, `high`, or `critical`.
- **Type.** `incident`, `request`, `problem`, or `change`.

The ticket text is the subject and body. The agent answer, tags, and gold labels are not shown to the model.

Jev answers all three questions in one `/v1/systemone` call. Queue and type are choices. Priority is a score, and the predicted level is the most probable point on that scale. Tev accepts one choice per call and at most 24 options, so it makes three lettered choices on the same labels. GPT-4o returns the three keys as JSON.

Laya Core ML is the local Core ML port of the English Laya checkpoint, [`aac6fef/laya-coreml`](https://huggingface.co/aac6fef/laya-coreml). It downloads once, then runs on this Mac with no API key and no token charge. That export answers one question per forward pass, and the reported latency is the sum. States longer than 512 tokens are truncated by the model. The default sample is English because this checkpoint is the English model. Pass `--language de` or `--language all` to include German.

Rows whose queue is a vertical category, such as Sports or News, are excluded. Those strings are not support departments, and they would also push the choice past Tev's 24-option limit.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

`.env` needs `TOGETHER_API_KEY`, `TYPESAFE_API_KEY`, and `OPENAI_API_KEY`. Laya Core ML does not use a key.

`laya-coreml` runs on Python 3.13. This project is pinned to that version because the Core ML runtime does not ship a working build for 3.14.

## Run

```bash
python3 run_experiment.py --samples 50
```

```bash
python3 run_experiment.py --samples 100 --language en --seed 42 --output experiment_results.json
```

The report shows accuracy, how many priority levels a miss is off by, parse rate, latency, and cost per 1,000 tickets. A label the model fails to return counts as incorrect. Type accuracy ignores tickets that have no type in the dataset. Tev's latency is the sum of its three calls. Laya's cost stays at zero because inference is local.

Set `LAYA_MODEL` to another Core ML bundle, such as `aac6fef/laya-typed-decisions-coreml`, or to a directory created with `hf download`.
