"""Field-level metadata masking: the ONE code path for both training-time modality
dropout and the inference-time missingness sweep.

A "field" is one metadata column: the categorical fields come first, then the
continuous ones (same order as configs/default.yaml). Masking a field makes it look
exactly like a naturally missing value:
    categorical -> index 0 (reserved "missing")
    continuous  -> 0.0 (the train mean after standardisation) and missing flag = 1
Masks never look at labels, and the inference masks depend only on
(mask_seed, per-image key), so they do not change with batch size or order.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


def n_masked_fields(rate: float, n_fields: int) -> int:
    """Number of fields masked per sample at `rate` (round half up: 0.5*21 -> 11)."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    return int(math.floor(rate * n_fields + 0.5))


def observed_fields(cat: torch.Tensor, cont_missing: torch.Tensor) -> torch.Tensor:
    """(B, F) bool: True where the field currently carries a value."""
    return torch.cat([cat != 0, cont_missing == 0], dim=1)


def missingness_ratio(cat: torch.Tensor, cont_missing: torch.Tensor) -> torch.Tensor:
    """(B,) fraction of fields that are missing (natural or masked)."""
    obs = observed_fields(cat, cont_missing)
    return 1.0 - obs.float().mean(dim=1)


def _choose(rng: np.random.Generator, candidates: np.ndarray, k: int, n_fields: int) -> np.ndarray:
    row = np.zeros(n_fields, dtype=bool)
    k = min(k, len(candidates))
    if k > 0:
        row[rng.choice(candidates, size=k, replace=False)] = True
    return row


def keyed_field_mask(keys: torch.Tensor | Sequence[int], n_fields: int, rate: float, seed: int,
                     observed: torch.Tensor | None = None, pool: str = "all") -> torch.Tensor:
    """Deterministic inference-time mask, (B, F) bool.

    Each sample gets exactly n_masked_fields(rate, F) masked fields, drawn uniformly
    from all fields (pool="all") or only from currently observed fields
    (pool="observed", capped at the number observed). The draw for a sample depends
    only on (seed, key), never on its batch neighbours.
    """
    keys = [int(k) for k in (keys.tolist() if torch.is_tensor(keys) else keys)]
    k = n_masked_fields(rate, n_fields)
    if pool not in ("all", "observed"):
        raise ValueError(pool)
    if pool == "observed" and observed is None:
        raise ValueError("pool='observed' needs the observed-field matrix")
    rows = []
    for i, key in enumerate(keys):
        rng = np.random.default_rng([int(seed), key])
        cand = np.arange(n_fields) if pool == "all" else np.flatnonzero(observed[i].cpu().numpy())
        rows.append(_choose(rng, cand, k, n_fields))
    return torch.from_numpy(np.stack(rows)) if rows else torch.zeros(0, n_fields, dtype=torch.bool)


def modality_dropout_mask(batch_size: int, n_fields: int, md_cfg: dict,
                          rng: np.random.Generator) -> torch.Tensor:
    """Random training-time mask, (B, F) bool. Per sample: with p_full_mask drop every
    field; otherwise with p_apply drop n_masked_fields(r, F) fields, r drawn from
    rate_choices; otherwise keep everything."""
    rows = np.zeros((batch_size, n_fields), dtype=bool)
    if not md_cfg.get("enabled", False):
        return torch.from_numpy(rows)
    for i in range(batch_size):
        u = rng.random()
        if u < md_cfg["p_full_mask"]:
            rows[i] = True
        elif u < md_cfg["p_full_mask"] + md_cfg["p_apply"]:
            r = float(rng.choice(md_cfg["rate_choices"]))
            rows[i] = _choose(rng, np.arange(n_fields), n_masked_fields(r, n_fields), n_fields)
    return torch.from_numpy(rows)


def apply_field_mask(cat: torch.Tensor, cont: torch.Tensor, cont_missing: torch.Tensor,
                     mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked COPIES of the tabular tensors (inputs are never modified)."""
    n_cat = cat.shape[1]
    if mask.shape != (cat.shape[0], n_cat + cont.shape[1]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match the batch")
    mask = mask.to(cat.device)
    m_cat, m_cont = mask[:, :n_cat], mask[:, n_cat:]
    cat = cat.masked_fill(m_cat, 0)
    cont = cont.masked_fill(m_cont, 0.0)
    cont_missing = cont_missing.masked_fill(m_cont, 1.0)
    return cat, cont, cont_missing
