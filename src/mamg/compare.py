"""Paired comparison of two evaluated runs on identical images.

    python -m mamg.compare --config configs/default.yaml --a gated_s1_<hash> --b late_s1_<hash>

Reads the two `<run>_<split>_predictions.csv` files written by `mamg.evaluate`, checks that
they cover exactly the same (image, missingness, mask seed) rows with the same labels, and,
for every subset in eval.subsets and every missingness rate, reports metric(A) - metric(B)
with a paired, patient-clustered bootstrap CI (`stats.paired_bootstrap_diff`). The mask
seeds of one rate are pooled: a resampled patient brings all of its rows from every seed.

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


def compare_predictions(pred_a: pd.DataFrame, pred_b: pd.DataFrame, cfg: dict,
                        metrics=DEFAULT_METRICS) -> pd.DataFrame:
    """Tidy table: subset, missingness, metric, a_value, b_value, diff, ci_low, ci_high, p_le_0."""
    ev, d = cfg["eval"], cfg["data"]
    m = align_predictions(pred_a, pred_b)
    fn = lambda yt, yp: compute_metrics(yt, yp, d["classes"], d["malignant"])  # noqa: E731
    group_col = ev.get("bootstrap_group")
    rows = []
    for subset in ev["subsets"]:
        in_subset = _subset_rows(m, subset)
        for rate in sorted(m["missingness"].unique()):
            sel = m[in_subset & (m["missingness"] == rate).to_numpy()]
            if sel.empty:
                continue
            yt, ya, yb = (sel[c].to_numpy() for c in ("y_true", "y_pred_a", "y_pred_b"))
            groups = sel[group_col].to_numpy() if group_col else None
            pa, pb = fn(yt, ya), fn(yt, yb)
            for k in metrics:
                r = paired_bootstrap_diff(yt, ya, yb, fn, k, int(ev["bootstrap_resamples"]),
                                          float(ev["ci_level"]), int(ev["bootstrap_seed"]), groups)
                rows.append({"subset": subset, "missingness": float(rate), "metric": k,
                             "n_rows": len(sel), "n_mask_seeds": int(sel["mask_seed"].nunique()),
                             "a_value": pa[k], "b_value": pb[k], **r})
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
