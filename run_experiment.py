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
    DATASET_ID,
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
from data_loader import load_ticket_samples
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


def print_header(num_samples: int, language: str) -> None:
    language_note = "English tickets" if language == "en" else f"language={language}"
    console.print(
        Panel(
            "[bold cyan]Customer support decisions: Tev, Jev, Laya Core ML, GPT-4o[/bold cyan]\n"
            "Each model routes the ticket, rates its urgency, and labels its type.\n\n"
            f"[bold]Dataset:[/bold] {DATASET_ID}\n"
            f"[bold]Rows:[/bold] {num_samples} {language_note} in the ten helpdesk departments\n"
            f"[bold]Tev:[/bold] {TEV_MODEL_NAME} (three choices; one call per question)\n"
            f"[bold]Jev:[/bold] {JEV_MODEL_NAME} (queue choice, priority score, type choice)\n"
            f"[bold]Laya Core ML:[/bold] {LAYA_MODEL_NAME} on this Mac\n"
            f"[bold]OpenAI:[/bold] {OPENAI_MODEL_NAME}",
            border_style="cyan",
        )
    )


def render_table(summaries: List[ModelSummary]) -> None:
    table = Table(title="Support ticket decisions", border_style="cyan")
    table.add_column("Metric", style="bold white")
    for summary in summaries:
        table.add_column(DISPLAY_NAMES.get(summary.model, summary.model), justify="right")

    def row(label: str, values: List[str]) -> None:
        table.add_row(label, *values)

    row("Queue accuracy", [f"{item.queue_accuracy_pct:.1f}%" for item in summaries])
    row("Priority accuracy", [f"{item.priority_accuracy_pct:.1f}%" for item in summaries])
    row(
        "Priority error (levels)",
        ["—" if item.priority_mae is None else f"{item.priority_mae:.2f}" for item in summaries],
    )
    row(
        "Type accuracy",
        ["—" if item.type_accuracy_pct is None else f"{item.type_accuracy_pct:.1f}%" for item in summaries],
    )
    row("Parse success", [f"{item.parse_success_rate_pct:.1f}%" for item in summaries])
    row("Mean latency", [f"{item.lat_mean_ms:.0f} ms" for item in summaries])
    row("Latency p50", [f"{item.lat_p50_ms:.0f} ms" for item in summaries])
    row("Latency p95", [f"{item.lat_p95_ms:.0f} ms" for item in summaries])
    row("Cost / 1k tickets", [f"${item.cost_per_1k_usd:.4f}" for item in summaries])
    row(
        "Queue agreement with GPT-4o",
        ["—" if item.queue_agreement_with_openai_pct is None else f"{item.queue_agreement_with_openai_pct:.1f}%" for item in summaries],
    )
    console.print(table)
    console.print()

    best_queue = max(summaries, key=lambda item: item.queue_accuracy_pct)
    fastest = min(summaries, key=lambda item: item.lat_mean_ms)
    cheapest = min(summaries, key=lambda item: item.cost_per_1k_usd)
    console.print(
        Panel(
            f"[bold]Queue:[/bold] {DISPLAY_NAMES[best_queue.model]} at {best_queue.queue_accuracy_pct:.1f}%\n"
            f"[bold]Latency:[/bold] {DISPLAY_NAMES[fastest.model]} at {fastest.lat_mean_ms:.0f} ms mean\n"
            f"[bold]Cost:[/bold] {DISPLAY_NAMES[cheapest.model]} at ${cheapest.cost_per_1k_usd:.4f} per 1,000 tickets\n\n"
            "Tev latency is the sum of its three calls. Jev answers all three questions in one request. "
            "Laya Core ML runs locally, so its cost is zero; this export runs one question per forward pass and the latency is the sum. "
            "A missed parse counts as an incorrect label. Type accuracy skips tickets whose type label is empty. "
            f"GPT-4o list price is ${GPT4O_INPUT_PRICE_PER_M:.2f} / ${GPT4O_OUTPUT_PRICE_PER_M:.2f} per million input / output tokens.",
            title="Reading the table",
            border_style="bright_white",
        )
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare Tev, Jev, Laya Core ML, and GPT-4o on support tickets")
    parser.add_argument("--samples", type=int, default=50, help="Number of tickets (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument("--language", choices=("en", "de", "all"), default="en", help="Ticket language (default: en)")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between calls")
    parser.add_argument("--output", default="experiment_results.json", help="JSON results path")
    args = parser.parse_args()

    validate_environment()
    print_header(args.samples, args.language)
    samples = load_ticket_samples(num_samples=args.samples, seed=args.seed, language=args.language)
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
                scored = score_output(sample, output)
                scores.append(scored)
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
            "num_samples": len(samples),
            "seed": args.seed,
            "language": args.language,
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
                "language": sample.language,
                "text": sample.text,
                "queue": sample.queue,
                "priority": sample.priority,
                "type": sample.ticket_type,
                "predictions": {
                    model_name: _public_score(results[model_name][index])
                    for model_name in results
                },
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
