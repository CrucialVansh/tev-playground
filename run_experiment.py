#!/usr/bin/env python3
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_SPLIT,
    GPT4O_INPUT_PRICE_PER_M,
    GPT4O_OUTPUT_PRICE_PER_M,
    JEV_API_URL,
    JEV_INPUT_PRICE_PER_M,
    JEV_MODEL_NAME,
    JEV_OUTPUT_PRICE_PER_M,
    LAYA_MODEL_NAME,
    OPENAI_MODEL_NAME,
    TEV_MODEL_NAME,
    env_key,
)
from data_loader import load_query_samples
from metrics import ModelSummary, score_output, summaries_by_model
from schema import SampleScore

console = Console()

DISPLAY_NAMES = {
    "tev": "Tev1-4B",
    "jev": "Jev",
    "laya": "Laya Core ML",
    "openai": "GPT-4o",
}


def validate_environment() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)

    required = {
        "TOGETHER_API_KEY": "Tev1 on Together AI",
        "TYPESAFE_API_KEY": "Jev at api.typesafe.ai",
        "OPENAI_API_KEY": "gpt-4o",
    }
    missing = [f"{name} ({purpose})" for name, purpose in required.items() if not env_key(name)]
    if missing:
        console.print(
            Panel(
                "[bold red]API keys are missing[/bold red]\n\n"
                + "\n".join(f"  • [yellow]{item}[/yellow]" for item in missing)
                + "\n\nAdd them to [bold cyan].env[/bold cyan]. Laya Core ML runs locally and does not need a key.",
                title="Configuration required",
                border_style="red",
            )
        )
        sys.exit(1)


def build_clients():
    from models.laya_client import LayaCoreClient
    from models.openai_client import OpenAIClient
    from models.systemone_client import SystemOneClient
    from models.tev_client import TevClient

    return [
        TevClient(),
        SystemOneClient("jev", JEV_MODEL_NAME, JEV_API_URL, env_key("TYPESAFE_API_KEY"), JEV_INPUT_PRICE_PER_M, JEV_OUTPUT_PRICE_PER_M),
        LayaCoreClient(LAYA_MODEL_NAME),
        OpenAIClient(),
    ]


def print_header(num_samples: int, oos_fraction: float) -> None:
    console.print(
        Panel(
            "[bold cyan]CLINC150 intent routing: Tev, Jev, Laya Core ML, GPT-4o[/bold cyan]\n"
            "Every model answers the same two questions: which of ten topics (or out of scope), "
            "then which of that topic's fifteen intents (or none of them).\n\n"
            f"[bold]Dataset:[/bold] {DATASET_ID} ({DATASET_CONFIG}, {DATASET_SPLIT})\n"
            f"[bold]Requests:[/bold] {num_samples}, balanced across the 150 intents, {oos_fraction:.0%} out of scope\n"
            f"[bold]Tev:[/bold] {TEV_MODEL_NAME}\n"
            f"[bold]Jev:[/bold] {JEV_MODEL_NAME}\n"
            f"[bold]Laya Core ML:[/bold] {LAYA_MODEL_NAME} on this Mac\n"
            f"[bold]OpenAI:[/bold] {OPENAI_MODEL_NAME}",
            border_style="cyan",
        )
    )


def _percent(value) -> str:
    return "—" if value is None else f"{value:.1f}%"


def render_table(summaries: List[ModelSummary]) -> None:
    first = summaries[0]
    table = Table(title="Intent routing", border_style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Always out of scope", justify="right", style="dim")
    for summary in summaries:
        table.add_column(DISPLAY_NAMES.get(summary.model, summary.model), justify="right")

    def row(label: str, base: str, values: List[str]) -> None:
        table.add_row(label, base, *values)

    always_oos = _percent(100.0 * first.oos_count / first.sample_count)
    row("Overall accuracy", always_oos, [_percent(item.overall_accuracy_pct) for item in summaries])
    row(f"In-scope intent accuracy ({first.in_scope_count})", "0.0%", [_percent(item.in_scope_accuracy_pct) for item in summaries])
    row("In-scope topic accuracy", "0.0%", [_percent(item.topic_accuracy_pct) for item in summaries])
    row(f"Out of scope caught ({first.oos_count})", "100.0%", [_percent(item.oos_recall_pct) for item in summaries])
    row("In scope sent to a person", "100.0%", [_percent(item.false_handoff_pct) for item in summaries])
    row("Parse success", "", [_percent(item.parse_success_rate_pct) for item in summaries])
    row("Mean latency", "", [f"{item.lat_mean_ms:.0f} ms" for item in summaries])
    row("Latency p50", "", [f"{item.lat_p50_ms:.0f} ms" for item in summaries])
    row("Latency p95", "", [f"{item.lat_p95_ms:.0f} ms" for item in summaries])
    row("Cost / 1k requests", "", [f"${item.cost_per_1k_usd:.4f}" for item in summaries])
    row("Intent agreement with GPT-4o", "", [_percent(item.agreement_with_openai_pct) for item in summaries])
    console.print(table)
    console.print()
    render_handoff_table(summaries)

    best = max(summaries, key=lambda item: item.overall_accuracy_pct)
    fastest = min(summaries, key=lambda item: item.lat_mean_ms)
    cheapest = min(summaries, key=lambda item: item.cost_per_1k_usd)
    console.print(
        Panel(
            f"[bold]Accuracy:[/bold] {DISPLAY_NAMES[best.model]} at {best.overall_accuracy_pct:.1f}% overall\n"
            f"[bold]Latency:[/bold] {DISPLAY_NAMES[fastest.model]} at {fastest.lat_mean_ms:.0f} ms mean\n"
            f"[bold]Cost:[/bold] {DISPLAY_NAMES[cheapest.model]} at ${cheapest.cost_per_1k_usd:.4f} per 1,000 requests\n\n"
            "Overall accuracy is the standard CLINC150 score: the exact intent for in-scope requests, "
            "and out of scope for the rest. Latency is the sum of both steps; an out-of-scope answer at the "
            "first step skips the second. Laya Core ML runs locally, so its cost is zero. "
            "A missed parse counts as incorrect. "
            f"GPT-4o list price is ${GPT4O_INPUT_PRICE_PER_M:.2f} / ${GPT4O_OUTPUT_PRICE_PER_M:.2f} per million input / output tokens.",
            title="Reading the table",
            border_style="bright_white",
        )
    )


def render_handoff_table(summaries: List[ModelSummary]) -> None:
    scored = [item for item in summaries if item.handoff_bands]
    if not scored:
        return
    table = Table(
        title="Also handing low-confidence answers to a person",
        caption="Confidence is the lower of the two steps. Tev and GPT-4o return no confidence.",
        border_style="magenta",
    )
    table.add_column("Confidence below", style="bold white")
    for summary in scored:
        name = DISPLAY_NAMES.get(summary.model, summary.model)
        table.add_column(f"{name} handled", justify="right")
        table.add_column(f"{name} accuracy", justify="right")
        table.add_column(f"{name} OOS caught", justify="right")
    for index, band in enumerate(scored[0].handoff_bands):
        cells = []
        for summary in scored:
            item = summary.handoff_bands[index]
            cells.extend([_percent(item.handled_pct), _percent(item.handled_accuracy_pct), _percent(item.oos_caught_pct)])
        table.add_row(f"{band.threshold:.1f}", *cells)
    console.print(table)
    console.print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare Tev, Jev, Laya Core ML, and GPT-4o on CLINC150")
    parser.add_argument("--samples", type=int, default=400, help="Number of requests (default: 400)")
    parser.add_argument("--oos-fraction", type=float, default=0.2, help="Share of out-of-scope requests (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between requests")
    parser.add_argument("--output", default="experiment_results.json", help="JSON results path")
    args = parser.parse_args()

    validate_environment()
    print_header(args.samples, args.oos_fraction)
    samples = load_query_samples(num_samples=args.samples, seed=args.seed, oos_fraction=args.oos_fraction)
    clients = build_clients()

    results: Dict[str, List[SampleScore]] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for client in clients:
            model_name = client.name
            label = DISPLAY_NAMES.get(model_name, model_name)
            task = progress.add_task(f"[cyan]{label}", total=len(samples))
            scores: List[SampleScore] = []
            warned = False
            for sample in samples:
                output = client.decide(sample.text)
                scores.append(score_output(sample, output))
                if output.raw_response.startswith("error:") and not warned:
                    progress.console.print(f"[yellow]{label}: {output.raw_response}[/yellow]")
                    warned = True
                progress.advance(task)
                if args.delay > 0:
                    time.sleep(args.delay)
            results[model_name] = scores

    summaries = summaries_by_model(results)
    console.print()
    render_table(summaries)

    payload = {
        "metadata": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "num_samples": len(samples),
            "oos_fraction": args.oos_fraction,
            "seed": args.seed,
            "models": {
                "tev": TEV_MODEL_NAME,
                "jev": JEV_MODEL_NAME,
                "laya": LAYA_MODEL_NAME,
                "openai": OPENAI_MODEL_NAME,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summaries": [asdict(item) for item in summaries],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "intent": sample.intent,
                "topic": sample.topic,
                "predictions": {model_name: _public_score(results[model_name][index]) for model_name in results},
            }
            for index, sample in enumerate(samples)
        ],
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    console.print(f"\n[dim]Wrote {args.output}[/dim]")


def _public_score(score: SampleScore) -> dict:
    raw = score.raw_response if len(score.raw_response) <= 500 else score.raw_response[:500] + "..."
    data = asdict(score)
    data["raw_response"] = raw
    return data


if __name__ == "__main__":
    main()
