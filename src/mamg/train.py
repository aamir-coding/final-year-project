"""Training.

    python -m mamg.train --config configs/default.yaml --model {image,late,impute,gated} --seed 1 [--resume]

Class-weighted focal loss, minority-pair oversampling, modality dropout (metadata
models), AMP on CUDA, warmup + cosine LR, early stopping on validation macro-F1,
checkpoint every epoch, exact --resume, per-epoch gate log, and one runs.csv row per
run. This script can never load the test split (invariant 5).
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import config_hash, is_synthetic, load_config, output_dirs
from .data import class_weights, minority_oversample
from .engine import EpochIndexSampler, build_pipeline, make_loader, predict, to_device
from .masking import apply_field_mask, modality_dropout_mask
from .metrics import compute_metrics
from .models import MODEL_NAMES, build_model
from .utils import Timer, append_run, get_device, set_seed


class FocalLoss(nn.Module):
    """Class-weighted focal loss: -w_y (1 - p_y)^gamma log p_y, normalised by sum(w_y)
    like weighted cross-entropy (gamma = 0 reproduces F.cross_entropy(weight=w))."""

    def __init__(self, weight: torch.Tensor | None, gamma: float):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        focal = (1.0 - logp.exp()).clamp_min(0.0).pow(self.gamma) * -logp
        w = self.weight[target] if self.weight is not None else torch.ones_like(focal)
        return (w * focal).sum() / w.sum().clamp_min(1e-12)


def lr_lambda_factory(total_steps: int, warmup_steps: int, min_ratio: float):
    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


def run_name_for(model: str, seed: int, chash: str) -> str:
    return f"{model}_s{seed}_{chash[:8]}"


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "random": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["random"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def train(cfg: dict, model_name: str, seed: int, resume: bool = False) -> dict:
    if model_name not in MODEL_NAMES:
        raise ValueError(model_name)
    cfg = dict(cfg)
    cfg["seed"] = int(seed)
    chash = config_hash(cfg)
    dirs = output_dirs(cfg)
    label = dirs["label"]
    run_name = run_name_for(model_name, seed, chash)
    run_dir = dirs["checkpoints"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tcfg, rt = cfg["train"], cfg["runtime"]
    set_seed(seed)
    device = get_device(rt["device"])
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    print(f"[{label}] run={run_name} device={device} amp={use_amp}")

    pipe = build_pipeline(cfg, need=("train", "val"))  # never the test split
    tr_ds, va_ds = pipe.datasets["train"], pipe.datasets["val"]
    labels = tr_ds.labels.numpy()
    n_classes = len(cfg["data"]["classes"])
    mp = tcfg["minority_pair_aug"]
    epoch_idx = (minority_oversample(labels, mp["target_fraction_of_max"], mp["max_multiplier"], seed)
                 if mp["enabled"] else np.arange(len(labels)))
    weights = (class_weights(labels[epoch_idx], n_classes) if tcfg["class_weighting"] == "inverse_frequency"
               else np.ones(n_classes))
    sampler = EpochIndexSampler(epoch_idx, seed)
    tr_loader = make_loader(tr_ds, cfg, int(tcfg["batch_size"]), sampler=sampler)
    va_loader = make_loader(va_ds, cfg, int(cfg["eval"]["batch_size"]))

    model = build_model(model_name, cfg, pipe.pre.cat_cardinalities, len(pipe.pre.continuous),
                        cat_modes=pipe.pre.cat_modes).to(device)
    crit = FocalLoss(torch.tensor(weights, dtype=torch.float32), tcfg["focal_gamma"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    max_steps = int(rt["max_train_steps"])
    steps_per_epoch = min(len(tr_loader), max_steps) if max_steps else len(tr_loader)
    total = steps_per_epoch * int(tcfg["epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(total, steps_per_epoch * int(tcfg["warmup_epochs"]), float(tcfg["min_lr_ratio"])))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = {"epoch": -1, "best_metric": -1.0, "best_epoch": -1, "no_improve": 0,
             "wall_clock_s": 0.0, "finished": False, "best_val": {}}
    last_ckpt, best_ckpt = run_dir / "last.pt", run_dir / "best.pt"
    if resume and last_ckpt.exists():
        # weights_only=False: the checkpoint also stores RNG states; only load our own files.
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        if ck["config_hash"] != chash:
            raise RuntimeError(f"config changed since checkpoint ({ck['config_hash']} != {chash}); refusing to resume")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        _set_rng_state(ck["rng"])
        print(f"resumed from epoch {state['epoch']} (best macro-F1 {state['best_metric']:.4f})")
        if state["finished"]:
            print("run already finished; nothing to do")
            return {"run_name": run_name, "run_dir": run_dir, "state": state}
    elif resume:
        print("no checkpoint found; starting fresh")

    n_fields = pipe.pre.n_fields
    md = tcfg["modality_dropout"]
    res_dir = dirs["results"]
    history_csv = res_dir / "history" / f"{run_name}.csv"
    patience = int(tcfg["early_stopping_patience"])
    metric = tcfg["early_stopping_metric"]

    for epoch in range(state["epoch"] + 1, int(tcfg["epochs"])):
        with Timer() as t:
            sampler.set_epoch(epoch)
            md_rng = np.random.default_rng([seed, epoch, 7])
            model.train()
            losses, train_gates = [], []
            for step, batch in enumerate(tr_loader):
                if max_steps and step >= max_steps:
                    break
                image, cat, cont, cmiss, y = to_device(batch, device)
                if model.uses_metadata and md["enabled"]:
                    m = modality_dropout_mask(len(y), n_fields, md, md_rng)
                    cat, cont, cmiss = apply_field_mask(cat, cont, cmiss, m)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits, aux = model(image, cat, cont, cmiss)
                loss = crit(logits, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip_norm"]))
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
                if "gate" in aux:
                    train_gates.append(aux["gate"].detach().float().cpu())

            ev_max = int(rt["max_eval_steps"])
            val = predict(model, va_loader, device, max_steps=ev_max, amp=use_amp)
            vm = compute_metrics(val["y_true"], val["y_pred"], cfg["data"]["classes"], cfg["data"]["malignant"])
            row = {"run_name": run_name, "epoch": epoch, "train_loss": float(np.mean(losses)),
                   "lr": sched.get_last_lr()[0], **{f"val_{k}": v for k, v in vm.items()}}
            if model_name == "gated":
                tg = torch.cat(train_gates).numpy() if train_gates else np.array([np.nan])
                vall = predict(model, va_loader, device, rate=1.0, max_steps=ev_max, amp=use_amp)
                grow = {"run_name": run_name, "model": model_name, "seed": seed, "epoch": epoch,
                        "data_kind": label,
                        "train_gate_mean": float(np.mean(tg)), "train_gate_std": float(np.std(tg)),
                        "val_gate_mean": float(np.mean(val["gate"])), "val_gate_std": float(np.std(val["gate"])),
                        "val_gate_mean_all_masked": float(np.mean(vall["gate"])),
                        "val_macro_f1": vm["macro_f1"]}
                _append_csv(res_dir / "gate_log.csv", grow)
                if grow["val_gate_std"] < 1e-3 or not 0.02 < grow["val_gate_mean"] < 0.98:
                    print(f"WARNING epoch {epoch}: possible gate collapse "
                          f"(mean {grow['val_gate_mean']:.3f}, std {grow['val_gate_std']:.4f})")
                row.update({k: grow[k] for k in ("val_gate_mean", "val_gate_std", "val_gate_mean_all_masked")})
            _append_csv(history_csv, row)

        state["wall_clock_s"] += t.elapsed
        state["epoch"] = epoch
        improved = vm[metric] > state["best_metric"]
        if improved:
            state.update(best_metric=vm[metric], best_epoch=epoch, no_improve=0, best_val=vm)
        else:
            state["no_improve"] += 1
        stop = state["no_improve"] >= patience or epoch == int(tcfg["epochs"]) - 1
        state["finished"] = stop
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
              "scaler": scaler.state_dict(), "state": state, "rng": _rng_state(), "config": cfg,
              "config_hash": chash, "model_name": model_name, "seed": seed, "run_name": run_name,
              "preprocessor": pipe.pre.state_dict(), "split_file": str(pipe.split_file),
              "class_weights": weights.tolist(), "data_kind": label}
        torch.save(ck, last_ckpt)
        if improved:
            torch.save(ck, best_ckpt)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} val macro-F1 {vm['macro_f1']:.4f} "
              f"mal-sens {vm['malignant_sensitivity']:.3f} ({t.elapsed:.0f}s){' *' if improved else ''}")
        if stop:
            break
        if epoch == int(rt.get("interrupt_after_epoch", -1)):
            print(f"simulated interruption after epoch {epoch}")
            return {"run_name": run_name, "run_dir": run_dir, "state": state, "interrupted": True}

    metrics = {**state["best_val"], "best_epoch": state["best_epoch"], "epochs_run": state["epoch"] + 1}
    append_run(res_dir / "runs.csv", run_type="train", data_kind=label, model=model_name, seed=seed,
               config_hash=chash, run_name=run_name, checkpoint=str(best_ckpt), split="val",
               wall_clock_s=round(state["wall_clock_s"], 1), metrics_json=metrics)
    return {"run_name": run_name, "run_dir": run_dir, "state": state, "best_ckpt": best_ckpt}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--seed", type=int, default=None, help="defaults to cfg.seed")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=2")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if is_synthetic(cfg):
        print("NOTE: SYNTHETIC data - outputs go to", cfg["paths"]["synthetic_results_dir"])
    train(cfg, a.model, a.seed if a.seed is not None else int(cfg["seed"]), resume=a.resume)


if __name__ == "__main__":
    main()
