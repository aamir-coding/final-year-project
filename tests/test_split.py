"""Invariant 1: patient/lesion-grouped split with zero overlap."""
import numpy as np
import pandas as pd
import pytest
from conftest import real_data

from mamg.data import (assert_no_overlap, load_metadata, load_split, patient_level_split,
                       split_to_frame)

SPLITS = ("train", "val", "test")


def _split(cfg, seed=None):
    df = load_metadata(cfg)
    seed = cfg["split"]["seed"] if seed is None else seed
    return df, patient_level_split(df, cfg["split"]["ratios"], seed, cfg["split"]["n_folds"])


def _check(df, splits, cfg, tol):
    # zero overlap on every identity column (also asserted inside patient_level_split)
    for col in ("patient_id", "lesion_id", "img_id", "group_id"):
        a, b, c = (set(df.iloc[splits[n]][col]) for n in SPLITS)
        assert not (a & b) and not (a & c) and not (b & c), col
    assert sum(len(v) for v in splits.values()) == len(df)
    classes = set(cfg["data"]["classes"])
    for name, r in zip(SPLITS, cfg["split"]["ratios"]):
        frac = len(splits[name]) / len(df)
        assert abs(frac - r) <= tol, (name, frac)
        present = set(df.iloc[splits[name]]["diagnostic"])
        assert present == classes, (name, classes - present)


def test_synthetic_split_no_overlap_and_ratios(synth_cfg):
    df, splits = _split(synth_cfg)
    _check(df, splits, synth_cfg, synth_cfg["split"]["ratio_tolerance_synthetic"])


def test_split_is_deterministic(synth_cfg):
    _, a = _split(synth_cfg, seed=7)
    _, b = _split(synth_cfg, seed=7)
    _, c = _split(synth_cfg, seed=8)
    for n in a:
        np.testing.assert_array_equal(a[n], b[n])
    assert any(not np.array_equal(a[n], c[n]) for n in a), "different seeds should give different splits"


def test_shared_lesion_ids_stay_together(synth_cfg):
    df, splits = _split(synth_cfg)
    split_of = pd.Series("", index=df.index)
    for n, idx in splits.items():
        split_of.iloc[idx] = n
    shared = df.groupby("lesion_id")["patient_id"].nunique()
    shared = shared[shared > 1].index
    assert len(shared) > 0, "fixture should contain lesion ids shared across patients"
    for lid in shared:
        assert split_of[df["lesion_id"] == lid].nunique() == 1


def test_assert_no_overlap_catches_a_leak(synth_cfg):
    df, splits = _split(synth_cfg)
    leaked = dict(splits)
    leaked["val"] = np.append(splits["val"], splits["train"][0])
    with pytest.raises(AssertionError, match="LEAK"):
        assert_no_overlap(df, leaked)


def test_test_split_requires_explicit_opt_in(synth_cfg):
    df, splits = _split(synth_cfg)
    sdf = split_to_frame(df, splits, synth_cfg)
    assert len(load_split(df, sdf, "train", synth_cfg)) == len(splits["train"])
    with pytest.raises(PermissionError):
        load_split(df, sdf, "test", synth_cfg)
    assert len(load_split(df, sdf, "test", synth_cfg, allow_test=True)) == len(splits["test"])


def test_load_split_detects_tampered_file(synth_cfg):
    df, splits = _split(synth_cfg)
    sdf = split_to_frame(df, splits, synth_cfg)
    # move one image of a multi-image train patient into val -> overlap detected on load
    counts = sdf[sdf.split == "train"].groupby("patient_id").size()
    pid = counts[counts > 1].index[0]
    victim = sdf.index[(sdf.patient_id == pid)][0]
    sdf.loc[victim, "split"] = "val"
    with pytest.raises(AssertionError, match="LEAK"):
        load_split(df, sdf, "train", synth_cfg)


@real_data
def test_real_split(base_cfg):
    df, splits = _split(base_cfg)
    assert len(df) == 2298
    assert df["patient_id"].nunique() == 1373
    assert df["lesion_id"].nunique() == 1641
    _check(df, splits, base_cfg, base_cfg["split"]["ratio_tolerance_real"])
