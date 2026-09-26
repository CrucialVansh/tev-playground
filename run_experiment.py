#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    console = Console()
except ImportError:
    class DummyConsole:
        def print(self, *args, **kwargs):
            clean_args = [str(a) for a in args]
            print(*clean_args)
    console = DummyConsole()
    Table = None
    Panel = None

from config import TEV_MODEL_NAME, BASELINE_MODEL_NAME, ACTIONABLE_CATEGORIES
from data_loader import load_ag_news_samples
from metrics import compute_architectural_comparison, MetricSummary

from pathlib import Path

def validate_environment():
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)

    together_key = os.getenv("TOGETHER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    placeholders = {"your_together_api_key_here", "your_openai_api_key_here", ""}
    missing = []

    if not together_key or together_key.strip() in placeholders or together_key.strip().startswith("your_"):
        missing.append("TOGETHER_API_KEY (Needed for together/Tev1-4B-experimental)")
    if not openai_key or openai_key.strip() in placeholders or openai_key.strip().startswith("your_"):
        missing.append("OPENAI_API_KEY (Needed for gpt-4o)")

    if missing:
        console.print(
            Panel(
                "[bold red]API Keys in .env Not Configured[/bold red]\n\n"
                f"The benchmark loaded '[bold cyan]{env_path}[/bold cyan]', but the following keys are missing or still set to placeholders:\n"
                + "\n".join(f"  • [yellow]{m}[/yellow]" for m in missing)
                + "\n\nPlease open your [bold cyan].env[/bold cyan] file and paste your actual API keys:\n"
                + "  [dim]TOGETHER_API_KEY=tgp_v1_...[/dim]\n"
                + "  [dim]OPENAI_API_KEY=sk-proj-...[/dim]",
                title="Configuration Required",
                border_style="red"
            )
        )
        sys.exit(1)

def print_header(num_samples: int):
    console.print(
        Panel(
            f"[bold cyan]Architectural Benchmark: System-1 Offloading with Tev1[/bold cyan]\n"
            f"[white]Comparing Monolithic LLM (GPT-4o) vs. Decoupled Pipeline (Tev1-4B + GPT-4o)[/white]\n\n"
            f"[bold]Parameters:[/bold]\n"
            f"• [dim]System-1 Classifier:[/dim] [green]{TEV_MODEL_NAME}[/green]\n"
            f"• [dim]Baseline & Downstream:[/dim] [blue]{BASELINE_MODEL_NAME}[/blue]\n"
            f"• [dim]Dataset:[/dim] AG News (4 classes: World, Sports, Business, Sci/Tech)\n"
            f"• [dim]Sample Size:[/dim] {num_samples} samples\n"
            f"• [dim]Action Policy:[/dim] Downstream GPT-4o triggered on {list(ACTIONABLE_CATEGORIES)}",
            border_style="cyan"
        )
    )

def render_comparison_tables(comparison: Dict[str, Any]):
    mono: MetricSummary = comparison["monolithic"]
    dec: MetricSummary = comparison["decoupled"]

    # 1. Latency Table
    latency_table = Table(title="⏱️  Latency Performance (Milliseconds)", border_style="bright_blue")
    latency_table.add_column("Metric / Stage", style="bold white")
    latency_table.add_column("Monolithic (GPT-4o)", justify="right", style="cyan")
    latency_table.add_column("Decoupled (Tev1 + GPT-4o)", justify="right", style="green")
    latency_table.add_column("Delta / Speedup", justify="right", style="bold yellow")

    class_speedup = comparison["classification_latency_speedup"]
    e2e_speedup = comparison["pipeline_latency_speedup"]

    latency_table.add_row(
        "Classification Mean",
        f"{mono.class_lat_mean_ms:.1f} ms",
        f"{dec.class_lat_mean_ms:.1f} ms",
        f"[green]{class_speedup:.2f}x faster[/green]" if class_speedup > 1 else f"{class_speedup:.2f}x"
    )
    latency_table.add_row(
        "Classification Median (p50)",
        f"{mono.class_lat_p50_ms:.1f} ms",
        f"{dec.class_lat_p50_ms:.1f} ms",
        f"{(mono.class_lat_p50_ms / dec.class_lat_p50_ms):.2f}x faster" if dec.class_lat_p50_ms > 0 else "-"
    )
    latency_table.add_row(
        "Classification p95",
        f"{mono.class_lat_p95_ms:.1f} ms",
        f"{dec.class_lat_p95_ms:.1f} ms",
        f"{(mono.class_lat_p95_ms / dec.class_lat_p95_ms):.2f}x faster" if dec.class_lat_p95_ms > 0 else "-"
    )
    latency_table.add_section()
    latency_table.add_row(
        "End-to-End Pipeline Mean",
        f"{mono.e2e_lat_mean_ms:.1f} ms",
        f"{dec.e2e_lat_mean_ms:.1f} ms",
        f"[green]{e2e_speedup:.2f}x overall[/green]" if e2e_speedup > 1 else f"{e2e_speedup:.2f}x"
    )
    latency_table.add_row(
        "End-to-End Pipeline p95",
        f"{mono.e2e_lat_p95_ms:.1f} ms",
        f"{dec.e2e_lat_p95_ms:.1f} ms",
        f"{(mono.e2e_lat_p95_ms / dec.e2e_lat_p95_ms):.2f}x overall" if dec.e2e_lat_p95_ms > 0 else "-"
    )
    console.print(latency_table)
    console.print()

    # 2. Quality & Reliability Table
    quality_table = Table(title="🎯 Quality, Agreement & System Reliability", border_style="magenta")
    quality_table.add_column("Evaluation Metric", style="bold white")
    quality_table.add_column("Monolithic (GPT-4o)", justify="right", style="cyan")
    quality_table.add_column("Decoupled (Tev1 + GPT-4o)", justify="right", style="green")
    quality_table.add_column("Comparison", justify="right", style="bold yellow")

    acc_diff = dec.accuracy_pct - mono.accuracy_pct
    acc_diff_str = f"{acc_diff:+.1f}%" if acc_diff != 0 else "0.0%"

    quality_table.add_row(
        "Classification Accuracy",
        f"{mono.accuracy_pct:.1f}%",
        f"{dec.accuracy_pct:.1f}%",
        f"[green]{acc_diff_str}[/green]" if acc_diff >= 0 else f"[red]{acc_diff_str}[/red]"
    )
    quality_table.add_row(
        "Architectural Decision Agreement",
        "-",
        "-",
        f"[bold]{comparison['agreement_rate_pct']:.1f}%[/bold]"
    )
    quality_table.add_row(
        "Structured Parse Success Rate",
        f"{mono.parse_success_rate_pct:.1f}%",
        f"{dec.parse_success_rate_pct:.1f}%",
        "Deterministic single token"
    )
    quality_table.add_row(
        "Downstream Actions Triggered",
        f"{mono.downstream_triggered_count} / {mono.sample_count}",
        f"{dec.downstream_triggered_count} / {dec.sample_count}",
        f"Δ {dec.downstream_triggered_count - mono.downstream_triggered_count:+d}"
    )
    console.print(quality_table)
    console.print()

    # 3. Cost & Economics Table
    cost_table = Table(title="💰 Economic Analysis (Projected Cost per 1,000 Requests)", border_style="green")
    cost_table.add_column("Cost Component", style="bold white")
    cost_table.add_column("Monolithic (GPT-4o)", justify="right", style="cyan")
    cost_table.add_column("Decoupled (Tev1 + GPT-4o)", justify="right", style="green")
    cost_table.add_column("Savings", justify="right", style="bold yellow")

    class_savings = comparison["classification_cost_reduction_pct"]
    pipeline_savings = comparison["pipeline_cost_reduction_pct"]

    cost_table.add_row(
        "Classification Layer (1k reqs)",
        f"${mono.cost_per_1k_classifications_usd:.4f}",
        f"${dec.cost_per_1k_classifications_usd:.4f}",
        f"[bold green]{class_savings:.1f}% Reduction[/bold green]"
    )
    cost_table.add_row(
        "Full Pipeline E2E (1k reqs)",
        f"${mono.cost_per_1k_pipeline_reqs_usd:.4f}",
        f"${dec.cost_per_1k_pipeline_reqs_usd:.4f}",
        f"[bold green]{pipeline_savings:.1f}% Net Savings[/bold green]"
    )
    console.print(cost_table)
    console.print()

    # 4. Architectural Summary Panel
    console.print(
        Panel(
            f"[bold underline]Architectural Takeaway:[/bold underline]\n\n"
            f"1. [bold cyan]Cost Reduction:[/bold cyan] Delegating high-frequency classification to Tev1 cuts classification compute spend by [bold green]{class_savings:.1f}%[/bold green].\n"
            f"2. [bold cyan]Latency Advantage:[/bold cyan] Tev1 delivers classification decisions [bold green]{class_speedup:.2f}x faster[/bold green] on average compared to prompting GPT-4o directly.\n"
            f"3. [bold cyan]Model Alignment:[/bold cyan] Tev1 achieved a [bold green]{comparison['agreement_rate_pct']:.1f}%[/bold green] decision agreement rate with GPT-4o while running at a fraction of the parameter weight.",
            title="🎯 Architectural Summary",
            border_style="bright_white"
        )
    )

def main():
    parser = argparse.ArgumentParser(description="Architectural benchmark: Tev1 vs. Monolithic LLM")
    parser.add_argument("--samples", type=int, default=200, help="Number of dataset samples (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample shuffling (default: 42)")
    parser.add_argument("--output", type=str, default="experiment_results.json", help="Path to save detailed JSON results")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay in seconds between calls (default: 0.05)")
    args = parser.parse_args()

    validate_environment()
    print_header(args.samples)

    from schema import PipelineExecutionResult
    from models.tev_client import TevClient
    from models.baseline_client import BaselineClient
    from pipelines.monolithic import MonolithicPipeline
    from pipelines.decoupled import DecoupledPipeline

    # 1. Load Data
    samples = load_ag_news_samples(num_samples=args.samples, seed=args.seed)

    # 2. Initialize Clients
    console.print("[*] Initializing clients for [green]Tev1-4B[/green] and [blue]GPT-4o[/blue]...")
    tev_client = TevClient()
    baseline_client = BaselineClient()

    mono_pipeline = MonolithicPipeline(baseline_client)
    dec_pipeline = DecoupledPipeline(tev_client, baseline_client)

    mono_results: List[PipelineExecutionResult] = []
    dec_results: List[PipelineExecutionResult] = []

    # 3. Execute Benchmarks with Rich Progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task_mono = progress.add_task("[cyan]Running Architecture A (Monolithic GPT-4o)...", total=len(samples))
        for sample in samples:
            res = mono_pipeline.run_sample(sample)
            mono_results.append(res)
            progress.advance(task_mono)
            if args.delay > 0:
                time.sleep(args.delay)

        task_dec = progress.add_task("[green]Running Architecture B (Decoupled Tev1 + GPT-4o)...", total=len(samples))
        for sample in samples:
            res = dec_pipeline.run_sample(sample)
            dec_results.append(res)
            progress.advance(task_dec)
            if args.delay > 0:
                time.sleep(args.delay)

    console.print("\n[bold green]✓ Execution completed for all samples.[/bold green]\n")

    # 4. Compute Metrics
    comparison = compute_architectural_comparison(mono_results, dec_results)

    # 5. Render Tables
    render_comparison_tables(comparison)

    # 6. Save Artifact
    serializable_data = {
        "metadata": {
            "num_samples": args.samples,
            "seed": args.seed,
            "tev_model": TEV_MODEL_NAME,
            "baseline_model": BASELINE_MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "comparison_metrics": {
            "agreement_rate_pct": comparison["agreement_rate_pct"],
            "classification_latency_speedup": comparison["classification_latency_speedup"],
            "pipeline_latency_speedup": comparison["pipeline_latency_speedup"],
            "classification_cost_reduction_pct": comparison["classification_cost_reduction_pct"],
            "pipeline_cost_reduction_pct": comparison["pipeline_cost_reduction_pct"],
            "monolithic_summary": vars(comparison["monolithic"]),
            "decoupled_summary": vars(comparison["decoupled"])
        },
        "sample_executions": [
            {
                "sample_id": s.sample_id,
                "text": s.text,
                "ground_truth": s.ground_truth_name,
                "monolithic": vars(mono_results[i]),
                "decoupled": vars(dec_results[i])
            }
            for i, s in enumerate(samples)
        ]
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2)

    console.print(f"[dim]Full experiment log saved to: [underline]{args.output}[/underline][/dim]\n")

if __name__ == "__main__":
    main()
