#!/usr/bin/env python3
"""Watch a metrics.jsonl file and print one-line-per-update stats.

Usage:
    python3 scripts/watch_metrics.py <path/to/metrics.jsonl> [--total N] [--pid-file PATH]

If --total is not given, it reads num_updates from the config.yaml in the
same run directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


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


def _read_pid(pid_file: Path | None) -> int | None:
    if pid_file is None:
        return None
    try:
        return int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _process_footprint_gb(pid: int) -> float:
    """Return process memory footprint in GB using macOS ``footprint`` tool.

    Includes MPS/GPU unified memory. Falls back to psutil RSS if the
    ``footprint`` command is unavailable or fails.
    """
    try:
        out = subprocess.check_output(
            ["footprint", str(pid)],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if "Footprint:" in line:
                for token in line.split():
                    if token.replace(".", "", 1).isdigit():
                        val = float(token)
                        if "GB" in line:
                            return val
                        return val / 1024
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return psutil.Process(pid).memory_info().rss / 1e9


def _mem_info(train_pid: int | None) -> str:
    vm = psutil.virtual_memory()
    used_gb = vm.used / 1e9
    free_gb = vm.available / 1e9
    used_pct = vm.percent
    free_pct = 100.0 - used_pct

    train_str = "-"
    if train_pid:
        try:
            proc_gb = _process_footprint_gb(train_pid)
            proc_pct = 100.0 * proc_gb / (vm.total / 1e9)
            train_str = f"{proc_gb:.1f}G({proc_pct:.0f}%)"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            train_str = "dead"

    return (
        f"Mem: train={train_str} "
        f"sys={used_gb:.1f}G({used_pct:.0f}%) "
        f"free={free_gb:.1f}G({free_pct:.0f}%)"
    )


def _format_line(r: dict, update: int, total: int | None, train_pid: int | None) -> str:
    pct = f"{100.0 * update / total:.1f}" if total else "?"
    ret = f"{r['avg_return']:.2f}" if "avg_return" in r else "-"
    length = f"{r['avg_length']:.1f}" if "avg_length" in r else "-"
    snake = f"{r['avg_snake_length']:.1f}" if "avg_snake_length" in r else "-"
    sps = str(r.get("sps", "-"))
    ts = datetime.now().strftime("%d %H:%M")
    total_str = str(total) if total else "?"
    mem = _mem_info(train_pid)
    return (
        f"[{pct}%] {update}/{total_str} | "
        f"Return: {ret} | Length: {length} | Snake: {snake} | "
        f"SPS: {sps} | {mem} | {ts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a metrics.jsonl file live")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument(
        "--total", type=int, default=None, help="Total number of updates"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Poll interval in seconds"
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help="Path to PID file for training process",
    )
    args = parser.parse_args()

    total = _load_total(args.metrics, args.total)
    pid_file = args.pid_file
    if pid_file is None:
        pid_file = args.metrics.parent.parent / ".active_training_pid"
    lines_seen = 0

    print(f"Watching {args.metrics}  (Ctrl+C to stop)\n", flush=True)

    while True:
        train_pid = _read_pid(pid_file)

        try:
            with open(args.metrics, encoding="utf-8") as f:
                all_lines = f.readlines()
        except FileNotFoundError:
            time.sleep(args.interval)
            continue

        new_lines = all_lines[lines_seen:]
        for raw in new_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
            except json.JSONDecodeError:
                continue
            lines_seen += 1
            update = lines_seen
            print(_format_line(r, update, total, train_pid), flush=True)

            if total and update >= total:
                print("\nTraining complete.", flush=True)
                sys.exit(0)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
