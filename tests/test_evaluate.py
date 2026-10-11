"""WP6: metrics agree with sklearn, bootstrap behaves, the sweep is tidy, and the final
test refuses to run twice."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from mamg.evaluate import evaluate
from mamg.metrics import HEADLINE, compute_metrics
from mamg.stats import bootstrap_metrics, paired_bootstrap_diff, summarise_sweep
from mamg.train import train

CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
MAL = ["BCC", "MEL", "SCC"]
LABELS = list(range(6))


def _fn(yt, yp):
    return compute_metrics(yt, yp, CLASSES, MAL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_match_sklearn_on_random_predictions(seed):
    rng = np.random.default_rng(seed)
    yt = rng.integers(0, 6, 500)
    yp = rng.integers(0, 6, 500)
    m = _fn(yt, yp)
    assert m["macro_f1"] == pytest.approx(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0))
    rec = recall_score(yt, yp, labels=LABELS, average=None, zero_division=0)
    f1 = f1_score(yt, yp, labels=LABELS, average=None, zero_division=0)
    for i, c in enumerate(CLASSES):
        assert m[f"recall_{c}"] == pytest.approx(rec[i])
        assert m[f"f1_{c}"] == pytest.approx(f1[i])
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(yt, yp))
    assert m["malignant_recall_macro"] == pytest.approx(rec[[1, 2, 4]].mean())
    mal_true = np.isin(yt, [1, 2, 4])
    assert m["malignant_sensitivity"] == pytest.approx(np.isin(yp[mal_true], [1, 2, 4]).mean())
    assert 0.1 < m["macro_f1"] < 0.25, "random guessing must score near chance (1/6), not near 1"


def test_metrics_with_absent_class_match_sklearn():
    yt = np.array([0, 0, 1, 1, 3, 3])
    yp = np.array([0, 1, 1, 1, 3, 0])
    m = _fn(yt, yp)
    assert m["macro_f1"] == pytest.approx(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0))
    assert m["recall_MEL"] == 0.0


def test_perfect_predictions():
    yt = np.repeat(np.arange(6), 10)
    m = _fn(yt, yt)
    assert m["macro_f1"] == 1.0 and m["malignant_sensitivity"] == 1.0
    ci = bootstrap_metrics(yt, yt, _fn, HEADLINE, 100, 0.95, 0)
    assert ci["macro_f1"] == (1.0, 1.0, 1.0)


def test_bootstrap_ci_contains_point_and_is_deterministic():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 6, 300)
    yp = np.where(rng.random(300) < 0.6, yt, rng.integers(0, 6, 300))
    groups = np.repeat(np.arange(100), 3)
    a = bootstrap_metrics(yt, yp, _fn, HEADLINE, 200, 0.95, 7, groups)
    b = bootstrap_metrics(yt, yp, _fn, HEADLINE, 200, 0.95, 7, groups)
    assert a == b
    for k, (p, lo, hi) in a.items():
        assert lo <= p <= hi and hi - lo > 0, k


def test_paired_diff_of_identical_models_is_zero():
    rng = np.random.default_rng(1)
    yt, yp = rng.integers(0, 6, 200), rng.integers(0, 6, 200)
    d = paired_bootstrap_diff(yt, yp, yp, _fn, "macro_f1", 100, 0.95, 0)
    assert d["diff"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0 and d["p_le_0"] == 1.0
    better = np.where(rng.random(200) < 0.9, yt, yp)
    d2 = paired_bootstrap_diff(yt, better, yp, _fn, "macro_f1", 200, 0.95, 0)
    assert d2["diff"] > 0 and d2["ci_low"] > 0 and d2["p_le_0"] < 0.05


def test_summarise_sweep():
    tidy = pd.DataFrame({"model": "gated", "run_name": "r", "split": "val", "subset": "all",
                         "missingness": [0.3, 0.3], "mask_seed": [0, 1], "metric": "macro_f1",
                         "value": [0.5, 0.7]})
    s = summarise_sweep(tidy)
    assert s.loc[0, "mean"] == pytest.approx(0.6) and s.loc[0, "n_mask_seeds"] == 2


def test_sweep_and_final_test_lock(synth_cfg):
    cfg = copy.deepcopy(synth_cfg)
    cfg["runtime"].update({"max_train_steps": 2, "max_eval_steps": 0})
    cfg["train"]["epochs"] = 1
    out = train(cfg, "gated", seed=0)
    res = evaluate(out["best_ckpt"], cfg)
    tidy = res["tidy"]
    assert {"model", "split", "subset", "missingness", "mask_seed", "metric", "value", "ci_low", "ci_high"} <= set(tidy.columns)
    assert set(tidy["split"]) == {"val"} and set(tidy["data_kind"]) == {"SYNTHETIC"}
    assert sorted(tidy["missingness"].unique()) == [0.0, 0.1, 0.3, 0.5]
    n_seeds = len(cfg["eval"]["mask_seeds"])
    mf = tidy[(tidy.metric == "macro_f1") & (tidy.subset == "all")]
    assert len(mf) == 1 + 3 * n_seeds
    eff = tidy[(tidy.metric == "effective_missing_ratio") & (tidy.subset == "all")].groupby("missingness")["value"].mean()
    assert eff.is_monotonic_increasing, "more masking must mean more missing fields"
    assert "results" not in str(res["out_dir"]).split("synthetic_runs")[0]

    final = evaluate(out["best_ckpt"], cfg, final_test=True)
    assert set(final["tidy"]["split"]) == {"test"}
    with pytest.raises(SystemExit, match="REFUSED"):
        evaluate(out["best_ckpt"], cfg, final_test=True)
