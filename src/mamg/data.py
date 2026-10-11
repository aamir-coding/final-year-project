"""Data layer: schema handling, patient-grouped split, train-only tabular
preprocessing, image transforms and the PyTorch Dataset.

Invariants enforced here (see CLAUDE.md):
  1. grouped split with zero patient / lesion overlap  -> patient_level_split, assert_no_overlap
  2. missing categorical == index 0, never == "no"     -> TabularPreprocessor
  4. statistics fit on train only                       -> TabularPreprocessor.fit
  5. test split needs an explicit opt-in                 -> load_split(allow_test=...)
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import metadata_fields

SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- schema
def required_columns(cfg: dict) -> list[str]:
    d = cfg["data"]
    cat, cont = metadata_fields(cfg)
    return [d["patient_col"], d["lesion_col"], d["image_col"], d["label_col"], *cat, *cont]


def load_metadata(cfg: dict, path: str | Path | None = None) -> pd.DataFrame:
    """Read metadata.csv as strings (no pandas NA guessing), validate the schema,
    normalise tokens, and attach `group_id` and `n_missing_fields`."""
    d = cfg["data"]
    path = Path(path or cfg["paths"]["metadata_csv"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_cols = [c for c in required_columns(cfg) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"metadata.csv is missing expected columns: {missing_cols}")
    df = df.apply(lambda s: s.str.strip())
    aliases = d.get("value_aliases") or {}
    cat, cont = metadata_fields(cfg)
    if aliases:
        df[cat] = df[cat].replace(aliases)
    bad = set(df[d["label_col"]]) - set(d["classes"])
    if bad:
        raise ValueError(f"unknown diagnostic labels: {sorted(bad)}")
    if df[d["image_col"]].duplicated().any():
        raise ValueError("duplicate img_id rows in metadata")
    df["label"] = df[d["label_col"]].map({c: i for i, c in enumerate(d["classes"])}).astype(int)
    df["group_id"] = assign_groups(df, d["patient_col"], d["lesion_col"])
    tokens = set(d["missing_tokens"])
    df["n_missing_fields"] = sum(df[f].isin(tokens).astype(int) for f in cat + cont)
    df["complete_metadata"] = df["n_missing_fields"] <= int(d["complete_max_missing_fields"])
    return df.reset_index(drop=True)


def assign_groups(df: pd.DataFrame, patient_col: str, lesion_col: str) -> np.ndarray:
    """Connected components over patient_id <-> lesion_id. In PAD-UFES-20 some
    lesion_ids are shared by two patient_ids; merging them guarantees that neither a
    patient nor a lesion can straddle two splits."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, l in zip(df[patient_col], df[lesion_col]):
        rp, rl = find("P:" + p), find("L:" + l)
        if rp != rl:
            parent[max(rp, rl)] = min(rp, rl)
    roots = [find("P:" + p) for p in df[patient_col]]
    codes = {r: i for i, r in enumerate(sorted(set(roots)))}
    return np.array([codes[r] for r in roots], dtype=np.int64)


def missingness_profile(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-field missingness on the RAW metadata (empty string vs literal UNK reported apart)."""
    cat, cont = metadata_fields(cfg)
    tokens = set(cfg["data"]["missing_tokens"])
    rows = []
    for f in cat + cont:
        s = df[f]
        n_empty = int((s == "").sum())
        n_unk = int((s.isin(tokens) & (s != "")).sum())
        rows.append({
            "field": f, "type": "categorical" if f in cat else "continuous", "n_rows": len(s),
            "n_empty": n_empty, "n_unk": n_unk, "n_missing": n_empty + n_unk,
            "missing_rate": round((n_empty + n_unk) / len(s), 4),
            "n_observed_categories": int(s[~s.isin(tokens)].nunique()) if f in cat else "",
        })
    return pd.DataFrame(rows)


def missingness_by_class(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Missing rate per field per diagnosis: exposes missingness that predicts the label."""
    cat, cont = metadata_fields(cfg)
    tokens = set(cfg["data"]["missing_tokens"])
    lab = cfg["data"]["label_col"]
    out = []
    for c in cfg["data"]["classes"]:
        sub = df[df[lab] == c]
        row = {"class": c, "n_images": len(sub),
               "frac_complete_metadata": round(float(sub["complete_metadata"].mean()), 4)}
        for f in cat + cont:
            row[f] = round(float(sub[f].isin(tokens).mean()), 4)
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- split
def patient_level_split(df: pd.DataFrame, ratios: Sequence[float], seed: int,
                        n_folds: int = 20, label_col: str = "label", group_col: str = "group_id",
                        patient_col: str = "patient_id", lesion_col: str = "lesion_id"
                        ) -> dict[str, np.ndarray]:
    """Grouped, class-stratified train/val/test split.

    Groups (connected patient/lesion components) are never divided. The data are cut
    into `n_folds` equal folds with sklearn's StratifiedGroupKFold (greedy: keeps the
    class distribution of every fold close to the global one while keeping groups
    whole), then whole folds are given to train/val/test in proportion to `ratios`
    (20 folds of 5% -> 14/3/3 for 70/15/15).
    Returns positional row indices per split; asserts zero overlap before returning.
    """
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"ratios must be 3 numbers summing to 1, got {ratios}")
    fold_counts = ratios * n_folds
    if not np.allclose(fold_counts, np.round(fold_counts)):
        raise ValueError(f"ratios {ratios} are not multiples of 1/{n_folds}")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=np.int64)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(df)), df[label_col], df[group_col])):
        fold[idx] = k
    edges = np.cumsum(np.round(fold_counts).astype(int))[:2]
    split_idx = np.searchsorted(edges, fold, side="right")
    out = {name: np.flatnonzero(split_idx == i) for i, name in enumerate(SPLIT_NAMES)}
    assert_no_overlap(df, out, patient_col=patient_col, lesion_col=lesion_col)
    return out


def assert_no_overlap(df: pd.DataFrame, splits: dict[str, np.ndarray],
                      patient_col: str = "patient_id", lesion_col: str = "lesion_id",
                      image_col: str = "img_id") -> None:
    """Invariant 1: no patient, lesion or image in more than one split."""
    names = list(splits)
    for col in (patient_col, lesion_col, image_col):
        if col not in df.columns:
            continue
        sets = {n: set(df.iloc[splits[n]][col]) for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = sets[a] & sets[b]
                assert not inter, f"LEAK: {len(inter)} {col} values in both {a} and {b}, e.g. {sorted(inter)[:3]}"
    all_idx = np.concatenate([splits[n] for n in names])
    assert len(all_idx) == len(set(all_idx.tolist())), "LEAK: a row is assigned to two splits"


def split_to_frame(df: pd.DataFrame, splits: dict[str, np.ndarray], cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    rows = []
    for name, idx in splits.items():
        sub = df.iloc[idx]
        rows.append(pd.DataFrame({d["image_col"]: sub[d["image_col"]].to_numpy(),
                                  d["patient_col"]: sub[d["patient_col"]].to_numpy(),
                                  d["lesion_col"]: sub[d["lesion_col"]].to_numpy(),
                                  d["label_col"]: sub[d["label_col"]].to_numpy(),
                                  "split": name}))
    return pd.concat(rows, ignore_index=True).sort_values(["split", d["image_col"]]).reset_index(drop=True)


def split_summary(df: pd.DataFrame, splits: dict[str, np.ndarray], cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    rows = []
    for name, idx in splits.items():
        sub = df.iloc[idx]
        row = {"split": name, "n_images": len(sub), "n_patients": sub[d["patient_col"]].nunique(),
               "n_lesions": sub[d["lesion_col"]].nunique(), "frac_images": round(len(sub) / len(df), 4),
               "frac_complete_metadata": round(float(sub["complete_metadata"].mean()), 4)}
        for c in d["classes"]:
            row[c] = int((sub[d["label_col"]] == c).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def split_path(splits_dir: str | Path, seed: int) -> Path:
    return Path(splits_dir) / f"split_seed{seed}.csv"


def load_split(df: pd.DataFrame, split_df: pd.DataFrame, name: str, cfg: dict,
               allow_test: bool = False) -> np.ndarray:
    """Row indices of `df` belonging to split `name`. The test split requires
    allow_test=True, which only the final-evaluation command passes (invariant 5)."""
    if name not in SPLIT_NAMES:
        raise ValueError(name)
    if name == "test" and not allow_test:
        raise PermissionError("The test split is reserved for `mamg.evaluate --final-test`.")
    d = cfg["data"]
    # Re-check overlap on every load, from the file itself.
    by_split = {n: df.index[df[d["image_col"]].isin(split_df.loc[split_df["split"] == n, d["image_col"]])].to_numpy()
                for n in SPLIT_NAMES}
    assert_no_overlap(df, by_split, d["patient_col"], d["lesion_col"], d["image_col"])
    n_assigned = sum(len(v) for v in by_split.values())
    if n_assigned != len(df):
        raise ValueError(f"split file covers {n_assigned} of {len(df)} metadata rows - regenerate it")
    return by_split[name]


# --------------------------------------------------------------------------- tabular
class TabularPreprocessor:
    """Encodes metadata fields. Fit on TRAIN rows only (invariant 4).

    categorical: observed values -> 1..K (sorted, deterministic); missing/UNK/unseen -> 0.
    continuous : z-score with train mean/std; missing -> train mean (= 0 after scaling)
                 plus an explicit missing flag (1 = missing).
    """

    def __init__(self, categorical: Sequence[str], continuous: Sequence[str],
                 missing_tokens: Sequence[str]):
        self.categorical = list(categorical)
        self.continuous = list(continuous)
        self.missing_tokens = set(missing_tokens)
        self.fitted = False

    # -- helpers
    def _is_missing(self, s: pd.Series) -> np.ndarray:
        return s.isin(self.missing_tokens).to_numpy()

    def _to_float(self, s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        vals = pd.to_numeric(s.where(~s.isin(self.missing_tokens)), errors="coerce").to_numpy(dtype=float)
        miss = ~np.isfinite(vals)
        return vals, miss

    def fit(self, train_df: pd.DataFrame) -> "TabularPreprocessor":
        self.vocab: dict[str, dict[str, int]] = {}
        self.cat_modes: list[int] = []
        for f in self.categorical:
            s = train_df[f]
            obs = s[~self._is_missing(s)]
            values = sorted(obs.unique().tolist())
            self.vocab[f] = {v: i + 1 for i, v in enumerate(values)}  # 0 reserved for missing
            self.cat_modes.append(self.vocab[f][obs.value_counts().sort_index().idxmax()] if len(obs) else 0)
        self.cont_mean, self.cont_std = [], []
        for f in self.continuous:
            vals, miss = self._to_float(train_df[f])
            obs = vals[~miss]
            mean = float(obs.mean()) if len(obs) else 0.0
            std = float(obs.std()) if len(obs) > 1 else 1.0
            self.cont_mean.append(mean)
            self.cont_std.append(std if std > 0 else 1.0)
        self.n_fit_rows = len(train_df)
        self.fitted = True
        return self

    @property
    def cat_cardinalities(self) -> list[int]:
        return [len(self.vocab[f]) + 1 for f in self.categorical]

    @property
    def n_fields(self) -> int:
        return len(self.categorical) + len(self.continuous)

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("fit() the preprocessor on the training split first")
        n = len(df)
        cat = np.zeros((n, len(self.categorical)), dtype=np.int64)
        for j, f in enumerate(self.categorical):
            cat[:, j] = df[f].map(self.vocab[f]).fillna(0).astype(np.int64).to_numpy()
        cont = np.zeros((n, len(self.continuous)), dtype=np.float32)
        cont_missing = np.zeros_like(cont)
        for j, f in enumerate(self.continuous):
            vals, miss = self._to_float(df[f])
            z = (vals - self.cont_mean[j]) / self.cont_std[j]
            z[miss] = 0.0  # mean imputation in standardised units
            cont[:, j] = z
            cont_missing[:, j] = miss
        return {"cat": cat, "cont": cont, "cont_missing": cont_missing}

    def state_dict(self) -> dict:
        return {"categorical": self.categorical, "continuous": self.continuous,
                "missing_tokens": sorted(self.missing_tokens), "vocab": self.vocab,
                "cat_modes": self.cat_modes, "cont_mean": self.cont_mean,
                "cont_std": self.cont_std, "n_fit_rows": self.n_fit_rows}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "TabularPreprocessor":
        p = cls(sd["categorical"], sd["continuous"], sd["missing_tokens"])
        p.vocab, p.cat_modes = sd["vocab"], sd["cat_modes"]
        p.cont_mean, p.cont_std, p.n_fit_rows = sd["cont_mean"], sd["cont_std"], sd["n_fit_rows"]
        p.fitted = True
        return p


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1 (fit on the training labels)."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / n_classes, 0.0)
    return w / w[counts > 0].mean()


def minority_oversample(labels: np.ndarray, target_fraction: float, max_multiplier: float,
                        seed: int) -> np.ndarray:
    """Indices for one training 'epoch' in which rare classes are repeated (whole
    image+metadata pairs, never synthesised) up to target_fraction * largest class,
    capped at max_multiplier x. Random augmentation makes each copy different."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels)
    target = target_fraction * counts.max()
    out = [np.arange(len(labels))]
    for c, n in enumerate(counts):
        if n == 0 or n >= target:
            continue
        want = int(round(min(target, max_multiplier * n))) - n
        members = np.flatnonzero(labels == c)
        full, rem = divmod(want, n)
        out += [members] * full
        if rem:
            out.append(rng.choice(members, size=rem, replace=False))
    return np.sort(np.concatenate(out))


# --------------------------------------------------------------------------- images
def build_image_index(*dirs: str | Path) -> dict[str, Path]:
    """Map image file name -> path, searching each directory recursively. Earlier
    directories win (so a resized cache can shadow the raw images)."""
    index: dict[str, Path] = {}
    for d in reversed([Path(x) for x in dirs if x]):
        if d.exists():
            for p in d.rglob("*.png"):
                index[p.name] = p
    return index


def build_transforms(cfg: dict, train: bool):
    im = cfg["image"]
    size = int(im["size"])
    norm = T.Normalize(im["mean"], im["std"])
    if train:
        a = im["augment"]
        return T.Compose([
            T.RandomResizedCrop(size, scale=(float(a["scale_min"]), 1.0)),
            T.RandomHorizontalFlip(a["hflip_p"]),
            T.RandomVerticalFlip(a["vflip_p"]),
            T.RandomRotation(a["rotation_deg"]),
            T.ColorJitter(a["color_jitter"], a["color_jitter"], a["color_jitter"]),
            T.ToTensor(), norm,
        ])
    # deterministic: no randomness in val/test transforms
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), norm])


def stable_key(img_id: str) -> int:
    """Batch-order-independent integer key per image (used to seed inference masks)."""
    return zlib.crc32(img_id.encode("utf-8"))


class PadUfesDataset(Dataset):
    """Returns a dict per sample. Metadata masking is NOT done here; it is applied to
    whole batches by mamg.masking so training dropout and evaluation share one path."""

    def __init__(self, df: pd.DataFrame, indices: np.ndarray, pre: TabularPreprocessor,
                 image_index: dict[str, Path], transform, cfg: dict):
        d = cfg["data"]
        self.df = df.iloc[np.asarray(indices)].reset_index(drop=True)
        tab = pre.transform(self.df)
        self.cat = torch.from_numpy(tab["cat"])
        self.cont = torch.from_numpy(tab["cont"])
        self.cont_missing = torch.from_numpy(tab["cont_missing"])
        self.labels = torch.as_tensor(self.df["label"].to_numpy(), dtype=torch.long)
        self.img_ids = self.df[d["image_col"]].tolist()
        self.keys = torch.as_tensor([stable_key(i) for i in self.img_ids], dtype=torch.long)
        missing = [i for i in self.img_ids if i not in image_index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:3]}")
        self.paths = [image_index[i] for i in self.img_ids]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, i: int) -> dict:
        with Image.open(self.paths[i]) as im:
            img = self.transform(im.convert("RGB"))
        return {"image": img, "cat": self.cat[i], "cont": self.cont[i],
                "cont_missing": self.cont_missing[i], "label": self.labels[i],
                "key": self.keys[i], "index": i}
