"""Config loading, CLI overrides, hashing and output-directory resolution."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

# Sections that describe *where* things are, not *what* the experiment is. The seed is
# excluded too, so every seed of one experiment shares a hash (run names carry the seed).
HASH_EXCLUDED_SECTIONS = ("paths", "runtime", "seed")
SYNTHETIC_MARKER = "SYNTHETIC_DATA_MARKER"


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Load a YAML config. Supports `_base_: other.yaml` inheritance and
    `key.sub=value` overrides (value parsed as YAML)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _deep_update(load_config(path.parent / base), cfg)
    for ov in overrides or ():
        set_by_dotted(cfg, ov)
    return cfg


def set_by_dotted(cfg: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"override must look like key.sub=value, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def config_hash(cfg: dict, length: int = 12) -> str:
    """Stable hash of the experiment-defining part of the config."""
    core = {k: v for k, v in cfg.items() if k not in HASH_EXCLUDED_SECTIONS}
    blob = json.dumps(core, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def is_synthetic(cfg: dict) -> bool:
    """Synthetic if declared in config OR the metadata folder carries the marker file."""
    if str(cfg.get("data", {}).get("kind", "real")).lower() == "synthetic":
        return True
    meta = Path(cfg["paths"]["metadata_csv"])
    return (meta.parent / SYNTHETIC_MARKER).exists()


def output_dirs(cfg: dict) -> dict:
    """Where results / checkpoints go. Synthetic data NEVER writes to results/ (invariant 9)."""
    p = cfg["paths"]
    if is_synthetic(cfg):
        root = Path(p["synthetic_results_dir"])
        return {
            "results": root / "results",
            "checkpoints": root / "checkpoints",
            "splits": root / "splits",
            "label": "SYNTHETIC",
        }
    return {
        "results": Path(p["results_dir"]),
        "checkpoints": Path(p["checkpoints_dir"]),
        "splits": Path(p["splits_dir"]),
        "label": "REAL",
    }


def metadata_fields(cfg: dict) -> tuple[list[str], list[str]]:
    d = cfg["data"]
    return list(d["categorical_fields"]), list(d["continuous_fields"])
