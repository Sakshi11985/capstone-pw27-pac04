"""Shared helpers: one seed function, config loading, and the append-only run log."""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_CSV = REPO_ROOT / "results" / "runs.csv"
RUNS_HEADER = ["git_commit", "timestamp", "script", "config", "dataset",
               "scenes", "seed", "params", "metrics"]


def set_seed(seed: int) -> np.random.Generator:
    """Seed every RNG the project uses. Returns a numpy Generator to pass around."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def with_channels(cfg: dict, channels) -> dict:
    """Override features.channels from the command line (e.g. ["rgb", "diff"])."""
    if channels:
        cfg.setdefault("features", {})["channels"] = list(channels)
    return cfg


def resolve(path: str | Path) -> Path:
    """Relative paths in configs are relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def log_run(script: str, cfg: dict, params: dict, metrics: dict,
            runs_csv: Path | None = None) -> None:
    """Append one row to results/runs.csv. Old rows are never edited."""
    runs_csv = Path(runs_csv or RUNS_CSV)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not runs_csv.exists() or runs_csv.stat().st_size == 0
    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(RUNS_HEADER)
        w.writerow([
            git_commit(),
            _dt.datetime.now().isoformat(timespec="seconds"),
            script,
            cfg.get("_config_path", ""),
            cfg.get("dataset", ""),
            " ".join(cfg.get("scenes") or ["all"]),
            cfg.get("seed", ""),
            json.dumps(params, sort_keys=True),
            json.dumps(metrics, sort_keys=True),
        ])
