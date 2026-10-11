"""Bootstrap confidence intervals and paired model comparison.

Resampling is clustered by patient by default (images of one patient are not
independent). Metrics are recomputed from scratch on every resample.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def _resample_indices(rng: np.random.Generator, n: int, groups: np.ndarray | None):
    if groups is None:
        return rng.integers(0, n, size=n)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    pick = rng.integers(0, len(uniq), size=len(uniq))
    return np.concatenate([members[g] for g in pick])


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable[..., dict],
                      keys: Sequence[str], n_resamples: int, ci: float, seed: int,
                      groups: np.ndarray | None = None) -> dict[str, tuple[float, float, float]]:
    """Return {key: (point, low, high)} with percentile CIs at level `ci`."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    point = metric_fn(y_true, y_pred)
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    samples = {k: np.empty(n_resamples) for k in keys}
    for b in range(n_resamples):
        if groups is None:
            idx = rng.integers(0, len(y_true), size=len(y_true))
        else:
            idx = np.concatenate([members[g] for g in rng.integers(0, len(members), size=len(members))])
        m = metric_fn(y_true[idx], y_pred[idx])
        for k in keys:
            samples[k][b] = m[k]
    alpha = (1.0 - ci) / 2.0
    return {k: (float(point[k]), float(np.quantile(samples[k], alpha)), float(np.quantile(samples[k], 1 - alpha)))
            for k in keys}


def paired_bootstrap_diff(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                          metric_fn: Callable[..., dict], key: str, n_resamples: int, ci: float,
                          seed: int, groups: np.ndarray | None = None) -> dict[str, float]:
    """Metric(A) - Metric(B) on the same resampled images. `p_le_0` is the fraction of
    resamples where A is not better than B (a one-sided bootstrap p-value)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = _resample_indices(rng, len(y_true), groups)
        diffs[b] = metric_fn(y_true[idx], pred_a[idx])[key] - metric_fn(y_true[idx], pred_b[idx])[key]
    alpha = (1.0 - ci) / 2.0
    point = metric_fn(y_true, pred_a)[key] - metric_fn(y_true, pred_b)[key]
    return {"diff": float(point), "ci_low": float(np.quantile(diffs, alpha)),
            "ci_high": float(np.quantile(diffs, 1 - alpha)), "p_le_0": float((diffs <= 0).mean())}


def summarise_sweep(tidy: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over mask seeds for each (model, split, subset, missingness, metric)."""
    keys = ["model", "run_name", "split", "subset", "missingness", "metric"]
    g = tidy.groupby(keys)["value"]
    out = g.agg(mean="mean", std="std", n_mask_seeds="count").reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out
