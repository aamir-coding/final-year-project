"""Paired comparison of two runs' predictions (mamg.compare)."""
import numpy as np
import pandas as pd
import pytest

from mamg.compare import align_predictions, compare_predictions


def _preds(base_cfg, seed=0, n_patients=60, per_patient=3, rates=(0.0, 0.3), mask_seeds=(0, 1)):
    """A predictions table shaped like evaluate's output: every image at every (rate, mask seed)."""
    rng = np.random.default_rng(seed)
    n_img = n_patients * per_patient
    imgs = pd.DataFrame({"img_id": [f"img_{i}.png" for i in range(n_img)],
                         "patient_id": [f"PAT_{i // per_patient}" for i in range(n_img)],
                         "y_true": rng.integers(0, len(base_cfg["data"]["classes"]), n_img),
                         "complete_metadata": np.arange(n_img) % 2 == 0})
    frames = []
    for r in rates:
        for ms in (mask_seeds if r > 0 else mask_seeds[:1]):
            f = imgs.copy()
            f["missingness"], f["mask_seed"] = r, ms
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _cfg(base_cfg, resamples=200):
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base_cfg.items()}
    cfg["eval"]["bootstrap_resamples"] = resamples
    return cfg


def test_identical_runs_have_zero_difference(base_cfg):
    p = _preds(base_cfg)
    p["y_pred"] = np.random.default_rng(1).integers(0, 6, len(p))
    t = compare_predictions(p, p.copy(), _cfg(base_cfg))
    assert set(t["subset"]) == {"all", "complete_metadata"}
    assert set(t["missingness"]) == {0.0, 0.3}
    assert (t["diff"] == 0).all() and (t["ci_low"] == 0).all() and (t["ci_high"] == 0).all()


def test_better_run_wins_with_ci_above_zero(base_cfg):
    p = _preds(base_cfg)
    good, bad = p.copy(), p.copy()
    rng = np.random.default_rng(2)
    good["y_pred"] = np.where(rng.random(len(p)) < 0.9, p["y_true"], rng.integers(0, 6, len(p)))
    bad["y_pred"] = rng.integers(0, 6, len(p))
    t = compare_predictions(good, bad, _cfg(base_cfg))
    f1 = t[t["metric"] == "macro_f1"]
    assert (f1["diff"] > 0.3).all()
    assert (f1["ci_low"] > 0).all()
    assert (f1["p_le_0"] == 0).all()
    # A vs B is the mirror image of B vs A
    rev = compare_predictions(bad, good, _cfg(base_cfg))
    assert np.allclose(rev["diff"].to_numpy(), -t["diff"].to_numpy())


def test_complete_subset_ignores_incomplete_rows(base_cfg):
    p = _preds(base_cfg)
    a, b = p.copy(), p.copy()
    a["y_pred"] = p["y_true"]
    b["y_pred"] = np.where(p["complete_metadata"], p["y_true"], (p["y_true"] + 1) % 6)
    t = compare_predictions(a, b, _cfg(base_cfg))
    f1 = t[t["metric"] == "macro_f1"].set_index("subset")
    assert (f1.loc["complete_metadata", "diff"] == 0).all()
    assert (f1.loc["all", "diff"] > 0).all()


def test_mismatched_inputs_are_refused(base_cfg):
    p = _preds(base_cfg)
    p["y_pred"] = 0
    with pytest.raises(ValueError, match="different rows"):
        align_predictions(p, p.iloc[1:].copy())
    q = p.copy()
    q.loc[0, "y_true"] = (q.loc[0, "y_true"] + 1) % 6
    with pytest.raises(ValueError, match="labels differ"):
        align_predictions(p, q)
    with pytest.raises(ValueError, match="duplicate"):
        align_predictions(pd.concat([p, p.iloc[:1]]), p)
