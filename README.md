# 🧪 Architectural Experiment: System-1 Offloading with Tev1 vs. Monolithic LLM

This repository evaluates a fundamental AI systems architecture problem: **System-1 Decision Offloading**.

Specifically, it benchmarks:
- **Pattern A (Monolithic GenAI Architecture):** All input classification and downstream generative tasks are routed through an expensive, general-purpose frontier LLM (`gpt-4o`).
- **Pattern B (Decoupled System-1 Architecture):** High-frequency structured classification and routing are offloaded to **Tev1** (`together/Tev1-4B-experimental`, Together AI's Jev-like decision classifier), reserving `gpt-4o` solely for selective downstream generative tasks.

---

## 🏛️ Architectural Comparison

```
Pattern A: Monolithic GenAI Architecture
[Input Text] ──────► [ GPT-4o Classifier ] ──────► [ Action Required? ] ─Yes─► [ GPT-4o Downstream Generator ]
                     • High latency (~500-1200ms)        │ No
                     • $2.50 / 1M prompt tokens          └─────────► [ Exit ]
                     • Autoregressive text generation

Pattern B: Decoupled System-1 Architecture
[Input Text] ──────► [ Tev1-4B Classifier ] ─────► [ Action Required? ] ─Yes─► [ GPT-4o Downstream Generator ]
                     • Ultra-low latency (~70-200ms)     │ No
                     • $0.042 / 1M prompt tokens         └─────────► [ Exit ]
                     • Deterministic decision token
```

### Why This Matters Architecturally
1. **Compute Economics**: Classifying a user request with GPT-4o costs **~$2.50 per million input tokens**. With Tev1 on Together AI, the same decision costs **$0.042 per million input tokens**—a **98.3% cost reduction** in the decision layer.
2. **Latency & Throughput**: Non-autoregressive or small bounded decision heads respond significantly faster than heavy frontier models.
3. **Selective Downstream Invocation**: By pairing Tev1 with an action policy (only *Sci/Tech* and *Business* articles trigger executive briefings), systems avoid burning frontier LLM tokens on non-actionable traffic.

---

## 📊 Measured Metrics

- **Classification Latency**: Mean, Median ($p_{50}$), $p_{90}$, and $p_{95}$ in milliseconds.
- **End-to-End Pipeline Latency**: Total request lifecycle including selective downstream generation.
- **Classification Accuracy**: Correct classifications relative to ground truth labels in AG News.
- **Architectural Agreement**: Percentage of decisions where Tev1 and GPT-4o chose the exact same classification.
- **Financial Economics**: Cost per 1,000 requests for the classification layer and the full end-to-end pipeline.
- **Reliability / Parse Success**: Deterministic extraction of single-letter choices (`A`, `B`, `C`, `D`).

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Add your API keys:
```env
TOGETHER_API_KEY=your_together_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

- **Together AI API Key:** Required to access `together/Tev1-4B-experimental`. Obtain from [Together AI](https://api.together.xyz/settings/api-keys).
- **OpenAI API Key:** Required for `gpt-4o`. Obtain from [OpenAI Platform](https://platform.openai.com/api-keys).

---

## 🏃 Running the Experiment

Run the benchmark with the default 200 samples:
```bash
python3 run_experiment.py --samples 200
```

### CLI Options:
```bash
python3 run_experiment.py \
  --samples 250 \
  --seed 42 \
  --delay 0.05 \
  --output experiment_results.json
```

- `--samples`: Number of test items to evaluate (default: `200`).
- `--seed`: Random seed for reproducible shuffling of test items (default: `42`).
- `--delay`: Inter-request delay in seconds to manage rate limits (default: `0.05`).
- `--output`: Filepath to save detailed per-sample execution logs and telemetry (default: `experiment_results.json`).

---

## 📁 Repository Structure

```
tev-playground/
├── config.py                 # Taxonomy, model IDs, pricing models, action policies
├── data_loader.py            # AG News dataset loader (Hugging Face + offline fallback)
├── metrics.py                # Latency percentiles, cost calculation, comparative stats
├── run_experiment.py         # Main CLI runner with rich visual reporting
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── models/
│   ├── tev_client.py         # Together AI Tev1-4B-experimental client
│   └── baseline_client.py    # OpenAI GPT-4o client (classification & downstream)
└── pipelines/
    ├── monolithic.py         # Architecture A implementation
    └── decoupled.py          # Architecture B implementation
```
