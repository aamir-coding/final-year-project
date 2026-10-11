"""Seeding, device selection, git info and the runs.csv ledger."""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

RUNS_COLUMNS = [
    "timestamp_utc", "run_type", "data_kind", "model", "seed", "config_hash", "git_sha",
    "git_dirty", "run_name", "checkpoint", "split", "wall_clock_s", "metrics_json",
]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we use. Determinism matters more than speed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except Exception:
        return "nogit", False


def append_run(runs_csv: str | Path, **row) -> None:
    """Append one row to runs.csv (invariant 6). Creates the file with a header."""
    runs_csv = Path(runs_csv)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    sha, dirty = git_info()
    row.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    row.setdefault("git_sha", sha)
    row.setdefault("git_dirty", dirty)
    if isinstance(row.get("metrics_json"), dict):
        row["metrics_json"] = json.dumps(row["metrics_json"], sort_keys=True, default=float)
    unknown = set(row) - set(RUNS_COLUMNS)
    if unknown:
        raise KeyError(f"unknown runs.csv columns: {sorted(unknown)}")
    new = not runs_csv.exists()
    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUNS_COLUMNS})


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
