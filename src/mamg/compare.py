"""Paired comparison of two evaluated runs on identical images.

    python -m mamg.compare --config configs/default.yaml --a gated_s1_<hash> --b late_s1_<hash>

Reads the two `<run>_<split>_predictions.csv` files written by `mamg.evaluate`, checks that
they cover exactly the same (image, missingness, mask seed) rows with the same labels, and,
for every subset in eval.subsets and every missingness rate, reports metric(A) - metric(B)
with a paired, patient-clustered bootstrap CI (`stats.paired_bootstrap_diff`). The mask
seeds of one rate are pooled: a resampled patient brings all of its rows from every seed.

It also tests degradation directly: `<metric>_drop_reduction` rows give
(B's drop - A's drop) from the clean rate to each masked rate, bootstrapping patients once for
both conditions so each patient is their own control. Positive = A degrades less.

Output: results/compare/<a>_vs_<b>_<split>.csv with one row per (subset, missingness, metric).
A CI that includes 0 means "no clear difference" (docs/CLAIMS_GUARDRAIL.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, output_dirs
from .metrics import compute_metrics
from .stats import paired_bootstrap_diff

KEY_COLS = ["img_id", "missingness", "mask_seed"]
DEFAULT_METRICS = ("macro_f1", "malignant_sensitivity")


def _subset_rows(df: pd.DataFrame, subset: str) -> np.ndarray:
    if subset == "all":
        return np.ones(len(df), dtype=bool)
    if subset == "complete_metadata":
        return df["complete_metadata"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    raise ValueError(f"unknown subset {subset!r}")


def align_predictions(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> pd.DataFrame:
    """Join two prediction tables row by row; refuse anything but an exact match of rows and labels."""
    for name, p in (("a", pred_a), ("b", pred_b)):
        if p.duplicated(KEY_COLS).any():
            raise ValueError(f"predictions {name} has duplicate (img_id, missingness, mask_seed) rows")
    ka = set(map(tuple, pred_a[KEY_COLS].to_numpy()))
    kb = set(map(tuple, pred_b[KEY_COLS].to_numpy()))
    if ka != kb:
        raise ValueError(f"runs were evaluated on different rows ({len(ka ^ kb)} not shared); "
                         "compare runs evaluated on the same split, rates and mask seeds")
    m = pred_a.merge(pred_b[KEY_COLS + ["y_true", "y_pred"]], on=KEY_COLS, suffixes=("", "_b"))
    if not (m["y_true"].to_numpy() == m["y_true_b"].to_numpy()).all():
        raise ValueError("labels differ between the two prediction files")
    return m.rename(columns={"y_pred": "y_pred_a"})


def drop_reduction(clean: pd.DataFrame, masked: pd.DataFrame, fn, key: str, n_resamples: int,
                   ci: float, seed: int, group_col: str | None) -> dict[str, float]:
    """How much less A degrades than B from the clean rows to the masked rows:
    (B_clean - B_masked) - (A_clean - A_masked). Positive = A loses less. Patients are
    resampled once per bootstrap draw and bring their rows from both conditions, so each
    patient is their own control. `p_le_0` = fraction of draws where A is not more robust."""
    cols = ("y_true", "y_pred_a", "y_pred_b")
    c_arr = [clean[c].to_numpy() for c in cols]
    m_arr = [masked[c].to_numpy() for c in cols]

    def stat(ci_: np.ndarray, mi_: np.ndarray) -> float:
        ct, ca, cb = (x[ci_] for x in c_arr)
        mt, ma, mb = (x[mi_] for x in m_arr)
        return (fn(ct, cb)[key] - fn(mt, mb)[key]) - (fn(ct, ca)[key] - fn(mt, ma)[key])

    point = stat(np.arange(len(clean)), np.arange(len(masked)))
    if group_col is None:  # resample images: the image id plays the role of the group
        group_col = "img_id"
    cg, mg = clean[group_col].to_numpy(), masked[group_col].to_numpy()
    groups = np.union1d(cg, mg)
    c_idx = {g: np.flatnonzero(cg == g) for g in groups}
    m_idx = {g: np.flatnonzero(mg == g) for g in groups}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        pick = groups[rng.integers(0, len(groups), size=len(groups))]
        draws[b] = stat(np.concatenate([c_idx[g] for g in pick]),
                        np.concatenate([m_idx[g] for g in pick]))
    alpha = (1.0 - ci) / 2.0
    return {"diff": float(point), "ci_low": float(np.quantile(draws, alpha)),
            "ci_high": float(np.quantile(draws, 1 - alpha)), "p_le_0": float((draws <= 0).mean())}


def compare_predictions(pred_a: pd.DataFrame, pred_b: pd.DataFrame, cfg: dict,
                        metrics=DEFAULT_METRICS) -> pd.DataFrame:
    """Tidy table: subset, missingness, metric, a_value, b_value, diff, ci_low, ci_high, p_le_0.
    Rows with metric `<m>` compare levels at one rate (A - B); rows with metric `<m>_drop_reduction`
    compare degradation from the lowest rate to this rate (positive = A degrades less)."""
    ev, d = cfg["eval"], cfg["data"]
    m = align_predictions(pred_a, pred_b)
    fn = lambda yt, yp: compute_metrics(yt, yp, d["classes"], d["malignant"])  # noqa: E731
    group_col = ev.get("bootstrap_group")
    boot = (int(ev["bootstrap_resamples"]), float(ev["ci_level"]), int(ev["bootstrap_seed"]))
    rows = []
    for subset in ev["subsets"]:
        in_subset = _subset_rows(m, subset)
        rates = sorted(m["missingness"].unique())
        clean = m[in_subset & (m["missingness"] == rates[0]).to_numpy()]
        for rate in rates:
            sel = m[in_subset & (m["missingness"] == rate).to_numpy()]
            if sel.empty:
                continue
            yt, ya, yb = (sel[c].to_numpy() for c in ("y_true", "y_pred_a", "y_pred_b"))
            groups = sel[group_col].to_numpy() if group_col else None
            pa, pb = fn(yt, ya), fn(yt, yb)
            base = {"subset": subset, "missingness": float(rate), "n_rows": len(sel),
                    "n_mask_seeds": int(sel["mask_seed"].nunique())}
            for k in metrics:
                r = paired_bootstrap_diff(yt, ya, yb, fn, k, *boot, groups)
                rows.append({**base, "metric": k, "a_value": pa[k], "b_value": pb[k], **r})
            if rate == rates[0] or clean.empty:
                continue
            pa0 = fn(clean["y_true"].to_numpy(), clean["y_pred_a"].to_numpy())
            pb0 = fn(clean["y_true"].to_numpy(), clean["y_pred_b"].to_numpy())
            for k in metrics:
                r = drop_reduction(clean, sel, fn, k, *boot, group_col)
                rows.append({**base, "metric": f"{k}_drop_reduction",
                             "a_value": pa0[k] - pa[k], "b_value": pb0[k] - pb[k], **r})
    return pd.DataFrame(rows)


def _resolve(arg: str, eval_dir: Path, split: str) -> tuple[Path, str]:
    p = Path(arg)
    if p.suffix == ".csv" and p.exists():
        return p, p.name.replace(f"_{split}_predictions.csv", "")
    p = eval_dir / f"{arg}_{split}_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `python -m mamg.evaluate --ckpt ...` for {arg} first")
    return p, arg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--a", required=True, help="run name (or predictions CSV) of model A")
    ap.add_argument("--b", required=True, help="run name (or predictions CSV) of model B")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    dirs = output_dirs(cfg)
    eval_dir = dirs["results"] / "eval"
    pa, name_a = _resolve(a.a, eval_dir, a.split)
    pb, name_b = _resolve(a.b, eval_dir, a.split)
    table = compare_predictions(pd.read_csv(pa), pd.read_csv(pb), cfg, tuple(a.metrics.split(",")))
    table.insert(0, "a_run", name_a)
    table.insert(1, "b_run", name_b)
    table.insert(2, "split", a.split)
    out_dir = dirs["results"] / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name_a}_vs_{name_b}_{a.split}.csv"
    table.to_csv(out, index=False)
    print(f"[{dirs['label']}] wrote {out}")
    for r in table.itertuples():
        verdict = "no clear difference" if r.ci_low <= 0 <= r.ci_high else ("A better" if r.diff > 0 else "B better")
        print(f"  {r.subset:<18} {r.missingness:>4.0%} {r.metric:<22} diff {r.diff:+.4f} "
              f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
