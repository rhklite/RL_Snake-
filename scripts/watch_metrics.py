#!/usr/bin/env python3
"""Watch a metrics.jsonl file and print a live-updating progress table.

Usage:
    python3 scripts/watch_metrics.py <path/to/metrics.jsonl> [--total N]

If --total is not given, it reads num_updates from the config.yaml in the
same run directory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text


def _load_total(metrics_path: Path, override: int | None) -> int | None:
    if override:
        return override
    config = metrics_path.parent / "config.yaml"
    if config.exists():
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(config)
            batch = cfg.training.num_envs * cfg.training.num_steps
            return cfg.training.total_timesteps // batch
        except Exception:
            pass
    return None


def _read_latest(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def _build_table(rows: list[dict]) -> Table:
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Update", justify="right", style="dim")
    table.add_column("Avg Return", justify="right", style="green")
    table.add_column("Avg Length", justify="right")
    table.add_column("Snake Size", justify="right", style="magenta")
    table.add_column("Policy Loss", justify="right", style="yellow")
    table.add_column("Value Loss", justify="right", style="yellow")
    table.add_column("Entropy", justify="right")
    table.add_column("SPS", justify="right", style="dim")

    display = rows[-30:]
    for r in display:
        table.add_row(
            str(r.get("update", "?")),
            f"{r['avg_return']:.2f}" if "avg_return" in r else "-",
            f"{r['avg_length']:.1f}" if "avg_length" in r else "-",
            f"{r['avg_snake_length']:.1f}" if "avg_snake_length" in r else "-",
            f"{r['policy_loss']:.4f}" if "policy_loss" in r else "-",
            f"{r['value_loss']:.4f}" if "value_loss" in r else "-",
            f"{r['entropy']:.4f}" if "entropy" in r else "-",
            str(r.get("sps", "-")),
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a metrics.jsonl file live")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--total", type=int, default=None, help="Total number of updates")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    total = _load_total(args.metrics, args.total)
    console = Console()

    progress = Progress(
        TextColumn("[bold blue]Training"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("{task.description}"),
        console=console,
    )
    task = progress.add_task("", total=total or 100)

    with Live(console=console, refresh_per_second=4) as live:
        while True:
            rows = _read_latest(args.metrics)
            last = rows[-1] if rows else {}
            update = last.get("update", 0)

            if total:
                pct = update / total
                desc = (
                    f"[dim]Update {update}/{total} | "
                    f"Return {last.get('avg_return', 'N/A')}"
                    if "avg_return" in last
                    else f"[dim]Update {update}/{total}"
                )
                progress.update(task, completed=update, description=desc)

            table = _build_table(rows)
            header = Text(f"  {args.metrics}", style="bold")

            from rich.panel import Panel

            panel = Panel(table, title=str(args.metrics.parent.name), border_style="dim")
            live.update(Columns([progress, "\n", panel]) if total else panel)

            if total and update >= total:
                console.print("[bold green]Training complete.[/bold green]")
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
