"""Evaluation with inference-time missingness.

    python -m mamg.evaluate --ckpt checkpoints/<run>/best.pt                 # validation sweep (development)
    python -m mamg.evaluate --ckpt checkpoints/<run>/best.pt --final-test    # test split, ONCE per checkpoint

For every missingness rate in eval.missingness_rates and every mask seed, metadata
fields are masked on a copy of each batch (training is never repeated), and metrics
are reported on every subset in eval.subsets (`all` and `complete_metadata`).
Outputs (real data -> results/eval/, synthetic -> tmp/synthetic_runs/results/eval/):
  <run>_<split>_sweep.csv        tidy: model, run_name, split, subset, missingness, mask_seed, metric, value, ci_low, ci_high
  <run>_<split>_summary.csv      mean/std over mask seeds
  <run>_<split>_predictions.csv  per-image probabilities, for paired comparisons in stats.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config, output_dirs
from .engine import build_pipeline, make_loader, predict
from .metrics import HEADLINE, compute_metrics
from .models import build_model
from .stats import bootstrap_metrics, summarise_sweep
from .utils import Timer, append_run, get_device

LOCK_COLUMNS = ["timestamp_utc", "checkpoint_sha256", "checkpoint", "run_name"]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def claim_final_test(lock_csv: Path, ckpt: Path, run_name: str) -> None:
    """Invariant 5: refuse a second final-test evaluation of the same checkpoint.
    The lock is written BEFORE evaluating, so even a crashed attempt counts; a human
    must edit the lock file deliberately to re-run."""
    sha = file_sha256(ckpt)
    if lock_csv.exists():
        lock = pd.read_csv(lock_csv, dtype=str)
        if (lock["checkpoint_sha256"] == sha).any() or (lock["run_name"] == run_name).any():
            raise SystemExit(f"REFUSED: final test already run for {run_name} (see {lock_csv}). "
                             "The test split is evaluated once.")
    lock_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not lock_csv.exists()
    with open(lock_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOCK_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "checkpoint_sha256": sha, "checkpoint": str(ckpt), "run_name": run_name})


def load_checkpoint_cfg(ckpt: dict, current_cfg: dict | None) -> dict:
    """Experiment settings come from the checkpoint; machine-specific paths/runtime
    from the current config (so a Kaggle checkpoint evaluates locally)."""
    cfg = copy.deepcopy(ckpt["config"])
    if current_cfg is not None:
        cfg["paths"] = copy.deepcopy(current_cfg["paths"])
        cfg["runtime"] = copy.deepcopy(current_cfg["runtime"])
        cfg["data"]["kind"] = current_cfg["data"].get("kind", cfg["data"]["kind"])
    return cfg


def evaluate(ckpt_path: str | Path, current_cfg: dict | None = None, final_test: bool = False,
             rates=None, mask_seeds=None) -> dict:
    ckpt_path = Path(ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # own checkpoints only
    cfg = load_checkpoint_cfg(ck, current_cfg)
    dirs = output_dirs(cfg)
    run_name, model_name = ck["run_name"], ck["model_name"]
    split = "test" if final_test else "val"
    if dirs["label"] != ck.get("data_kind", dirs["label"]):
        raise RuntimeError(f"checkpoint trained on {ck.get('data_kind')} data, evaluating on {dirs['label']}")
    if final_test:
        claim_final_test(dirs["results"] / "final_test_lock.csv", ckpt_path, run_name)

    ev = cfg["eval"]
    rates = list(ev["missingness_rates"] if rates is None else rates)
    mask_seeds = list(ev["mask_seeds"] if mask_seeds is None else mask_seeds)
    device = get_device(cfg["runtime"]["device"])
    pipe = build_pipeline(cfg, need=(split,), allow_test=final_test, pre_state=ck["preprocessor"])
    ds = pipe.datasets[split]
    loader = make_loader(ds, cfg, int(ev["batch_size"]))
    model = build_model(model_name, cfg, pipe.pre.cat_cardinalities, len(pipe.pre.continuous),
                        cat_modes=pipe.pre.cat_modes)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    classes, malignant = cfg["data"]["classes"], cfg["data"]["malignant"]
    sub_df = ds.df
    groups_all = sub_df[ev["bootstrap_group"]].to_numpy() if ev.get("bootstrap_group") else None
    subset_rows = {"all": np.arange(len(ds)),
                   "complete_metadata": np.flatnonzero(sub_df["complete_metadata"].to_numpy())}
    metric_fn = lambda yt, yp: compute_metrics(yt, yp, classes, malignant)  # noqa: E731

    tidy, pred_frames = [], []
    with Timer() as t:
        for rate in rates:
            for ms in (mask_seeds if rate > 0 else mask_seeds[:1]):
                out = predict(model, loader, device, rate=rate, mask_seed=ms, pool=ev["mask_pool"],
                              max_steps=int(cfg["runtime"]["max_eval_steps"]),
                              amp=bool(cfg["train"]["amp"]))
                order = out["index"]
                for subset in ev["subsets"]:
                    keep = np.isin(order, subset_rows[subset])
                    if not keep.any():
                        continue
                    yt, yp = out["y_true"][keep], out["y_pred"][keep]
                    groups = groups_all[order[keep]] if groups_all is not None else None
                    point = metric_fn(yt, yp)
                    cis = bootstrap_metrics(yt, yp, metric_fn, HEADLINE, int(ev["bootstrap_resamples"]),
                                            float(ev["ci_level"]), int(ev["bootstrap_seed"]), groups)
                    extra = {"gate_mean": float(np.nanmean(out["gate"][keep])) if model_name == "gated" else np.nan,
                             "effective_missing_ratio": float(out["missing_ratio"][keep].mean())}
                    for k, v in {**point, **extra}.items():
                        lo, hi = cis[k][1:] if k in cis else (np.nan, np.nan)
                        tidy.append({"model": model_name, "run_name": run_name, "data_kind": dirs["label"],
                                     "split": split, "subset": subset, "missingness": rate, "mask_seed": ms,
                                     "metric": k, "value": v, "ci_low": lo, "ci_high": hi})
                pf = pd.DataFrame(out["probs"], columns=[f"p_{c}" for c in classes])
                pf.insert(0, "img_id", [ds.img_ids[i] for i in order])
                pf.insert(1, "patient_id", sub_df[cfg["data"]["patient_col"]].to_numpy()[order])
                pf.insert(2, "y_true", out["y_true"])
                pf.insert(3, "y_pred", out["y_pred"])
                pf["gate"], pf["missing_ratio"] = out["gate"], out["missing_ratio"]
                pf["complete_metadata"] = sub_df["complete_metadata"].to_numpy()[order]
                pf["missingness"], pf["mask_seed"] = rate, ms
                pred_frames.append(pf)

    tidy_df = pd.DataFrame(tidy)
    summary = summarise_sweep(tidy_df)
    out_dir = dirs["results"] / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_name}_{split}"
    tidy_df.to_csv(out_dir / f"{stem}_sweep.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / f"{stem}_predictions.csv", index=False)

    head = summary[(summary.subset == "all") & summary.metric.isin(HEADLINE + ("gate_mean",))]
    metrics = {f"{r.metric}@{r.missingness:g}": round(float(r["mean"]), 4) for _, r in head.iterrows()}
    append_run(dirs["results"] / "runs.csv", run_type="final_test" if final_test else "eval",
               data_kind=dirs["label"], model=model_name, seed=ck["seed"], config_hash=ck["config_hash"],
               run_name=run_name, checkpoint=str(ckpt_path), split=split,
               wall_clock_s=round(t.elapsed, 1), metrics_json=metrics)
    print(f"[{dirs['label']}] {run_name} on {split}: wrote {out_dir / (stem + '_sweep.csv')}")
    for _, r in summary[(summary.subset == "all") & (summary.metric == "macro_f1")].iterrows():
        print(f"  missingness {r.missingness:.0%}: macro-F1 {r['mean']:.4f} +/- {r['std']:.4f}")
    return {"tidy": tidy_df, "summary": summary, "out_dir": out_dir}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml", help="supplies paths/runtime only")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--final-test", action="store_true", help="evaluate the TEST split (once per checkpoint)")
    ap.add_argument("--missingness", default=None, help="comma list of percents, e.g. 0,10,30,50")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    rates = [float(x) / 100 for x in a.missingness.split(",")] if a.missingness else None
    evaluate(a.ckpt, load_config(a.config, a.set), final_test=a.final_test, rates=rates)


if __name__ == "__main__":
    main()
