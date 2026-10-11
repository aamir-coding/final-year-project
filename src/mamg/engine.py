"""Shared plumbing for train / evaluate / xai: build the data pipeline from a config,
make loaders, and run prediction with (optional) inference-time masking."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from .config import metadata_fields, output_dirs
from .data import (PadUfesDataset, TabularPreprocessor, build_image_index, build_transforms,
                   load_metadata, load_split, patient_level_split, split_path, split_to_frame)
from .masking import apply_field_mask, keyed_field_mask, observed_fields
from .utils import worker_init_fn


@dataclass
class Pipeline:
    df: pd.DataFrame
    split_file: Path
    idx: dict[str, np.ndarray]
    pre: TabularPreprocessor
    datasets: dict[str, PadUfesDataset] = field(default_factory=dict)


def ensure_split(cfg: dict, df: pd.DataFrame) -> Path:
    """Return the split file for cfg.split.seed, creating it if absent (deterministic)."""
    dirs = output_dirs(cfg)
    sp = split_path(dirs["splits"], int(cfg["split"]["seed"]))
    if not sp.exists() and dirs["label"] == "REAL":
        raise FileNotFoundError(f"{sp} not found. Run `python -m mamg.prepare_data` from the repo root "
                                "(real-data splits are committed, never silently regenerated).")
    if not sp.exists():
        splits = patient_level_split(df, cfg["split"]["ratios"], int(cfg["split"]["seed"]),
                                     int(cfg["split"]["n_folds"]))
        sp.parent.mkdir(parents=True, exist_ok=True)
        split_to_frame(df, splits, cfg).to_csv(sp, index=False)
        print(f"created split file {sp}")
    return sp


def subset_mask(df: pd.DataFrame, idx: np.ndarray, subset: str) -> np.ndarray:
    if subset == "all":
        return idx
    if subset == "complete_metadata":
        return idx[df["complete_metadata"].to_numpy()[idx]]
    raise ValueError(f"unknown subset {subset!r}")


def build_pipeline(cfg: dict, need=("train", "val"), allow_test: bool = False,
                   pre_state: dict | None = None, train_aug: bool = True,
                   apply_train_subset: bool = True) -> Pipeline:
    """Load metadata + split, fit (train-only) or restore the preprocessor, build datasets.
    `train_subset` applies to train and val during training (the model-selection data);
    evaluation passes apply_train_subset=False and reports every subset separately."""
    df = load_metadata(cfg)
    sp = ensure_split(cfg, df)
    split_df = pd.read_csv(sp, dtype=str, keep_default_na=False)
    idx = {n: load_split(df, split_df, n, cfg, allow_test=allow_test) for n in need}
    sub = cfg["data"].get("train_subset", "all") if apply_train_subset else "all"
    for n in ("train", "val"):
        if n in idx:
            idx[n] = subset_mask(df, idx[n], sub)
    if pre_state is not None:
        pre = TabularPreprocessor.from_state_dict(pre_state)
    else:
        if "train" not in idx:
            raise ValueError("fitting the preprocessor needs the train split")
        cat, cont = metadata_fields(cfg)
        pre = TabularPreprocessor(cat, cont, cfg["data"]["missing_tokens"]).fit(df.iloc[idx["train"]])
    p = cfg["paths"]
    image_index = build_image_index(p.get("image_cache_dir") or "", p["image_dir"])
    datasets = {n: PadUfesDataset(df, idx[n], pre, image_index,
                                  build_transforms(cfg, train=(n == "train" and train_aug)), cfg)
                for n in need}
    return Pipeline(df=df, split_file=sp, idx=idx, pre=pre, datasets=datasets)


class EpochIndexSampler(Sampler):
    """Yields a seeded permutation of `indices` (which may contain repeats from
    minority oversampling). Deterministic per (seed, epoch), so --resume is exact."""

    def __init__(self, indices: np.ndarray, seed: int):
        self.indices = np.asarray(indices)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        return iter(self.indices[rng.permutation(len(self.indices))].tolist())

    def __len__(self) -> int:
        return len(self.indices)


def make_loader(ds, cfg: dict, batch_size: int, sampler=None, shuffle: bool = False) -> DataLoader:
    nw = int(cfg["runtime"]["num_workers"])
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle and sampler is None,
                      num_workers=nw, worker_init_fn=worker_init_fn, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False, drop_last=False)


def to_device(batch: dict, device) -> tuple:
    return (batch["image"].to(device, non_blocking=True), batch["cat"].to(device),
            batch["cont"].to(device), batch["cont_missing"].to(device), batch["label"].to(device))


def inference_mask(batch: dict, n_fields: int, rate: float, mask_seed: int, pool: str) -> torch.Tensor:
    obs = observed_fields(batch["cat"], batch["cont_missing"]) if pool == "observed" else None
    return keyed_field_mask(batch["key"], n_fields, rate, mask_seed, observed=obs, pool=pool)


@torch.no_grad()
def predict(model, loader, device, rate: float = 0.0, mask_seed: int = 0, pool: str = "all",
            max_steps: int = 0, amp: bool = False) -> dict[str, np.ndarray]:
    """Run the model over a loader. If rate > 0 every batch is masked with the
    deterministic keyed masker (a COPY of the batch; the dataset is untouched)."""
    model.eval()
    ys, probs, gates, ratios, idxs = [], [], [], [], []
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        image, cat, cont, cmiss, y = to_device(batch, device)
        if rate > 0:
            m = inference_mask(batch, cat.shape[1] + cont.shape[1], rate, mask_seed, pool)
            cat, cont, cmiss = apply_field_mask(cat, cont, cmiss, m)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, aux = model(image, cat, cont, cmiss)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ys.append(y.cpu())
        idxs.append(batch["index"])
        ratios.append((1.0 - observed_fields(cat, cmiss).float().mean(1)).cpu())
        if "gate" in aux:
            gates.append(aux["gate"].float().cpu())
    out = {"y_true": torch.cat(ys).numpy(), "probs": torch.cat(probs).numpy(),
           "index": torch.cat(idxs).numpy(), "missing_ratio": torch.cat(ratios).numpy()}
    out["y_pred"] = out["probs"].argmax(1)
    out["gate"] = torch.cat(gates).numpy() if gates else np.full(len(out["y_true"]), np.nan)
    return out
