"""Invariants 2 and 4: missing != "no"; statistics fit on train only."""
import numpy as np
import pandas as pd
import pytest
from conftest import real_data

from mamg.data import (TabularPreprocessor, class_weights, load_metadata, minority_oversample,
                       patient_level_split)

TOKENS = ["", "UNK", "unk", "NaN", "nan"]


def _pre():
    return TabularPreprocessor(["smoke", "region"], ["age"], TOKENS)


def test_missing_is_never_no():
    train = pd.DataFrame({"smoke": ["False", "True", "", "UNK"], "region": ["FACE"] * 4,
                          "age": ["40", "60", "", "50"]})
    p = _pre().fit(train)
    enc = p.transform(train)["cat"][:, 0]
    assert enc[2] == 0 and enc[3] == 0, "empty and UNK must map to reserved index 0"
    assert enc[0] >= 1 and enc[1] >= 1 and enc[0] != enc[1], "False/True are real categories"
    assert 0 not in p.vocab["smoke"].values()
    assert "UNK" not in p.vocab["smoke"] and "" not in p.vocab["smoke"]


def test_fit_uses_train_only_and_unseen_maps_to_zero():
    train = pd.DataFrame({"smoke": ["False", "True"], "region": ["FACE", "ARM"], "age": ["40", "60"]})
    val = pd.DataFrame({"smoke": ["False", "True"], "region": ["NOSE", "FACE"], "age": ["1000", ""]})
    p = _pre().fit(train)
    assert set(p.vocab["region"]) == {"FACE", "ARM"}
    assert p.cont_mean == [50.0]
    out = p.transform(val)
    assert out["cat"][0, 1] == 0, "a category never seen in train is treated as missing"
    assert out["cat"][1, 1] == p.vocab["region"]["FACE"]
    # val statistics do not move the scaler
    assert np.isclose(out["cont"][0, 0], (1000 - 50) / p.cont_std[0])
    assert out["cont"][1, 0] == 0.0 and out["cont_missing"][1, 0] == 1.0
    assert out["cont_missing"][0, 0] == 0.0


def test_mode_and_state_roundtrip():
    train = pd.DataFrame({"smoke": ["False", "False", "True", ""], "region": ["FACE"] * 4,
                          "age": ["1", "2", "3", "4"]})
    p = _pre().fit(train)
    assert p.cat_modes[0] == p.vocab["smoke"]["False"]
    q = TabularPreprocessor.from_state_dict(p.state_dict())
    for k, v in p.transform(train).items():
        np.testing.assert_array_equal(v, q.transform(train)[k])


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        _pre().transform(pd.DataFrame({"smoke": ["x"], "region": ["y"], "age": ["1"]}))


def test_class_weights_and_oversampling():
    labels = np.array([0] * 80 + [1] * 15 + [2] * 5)
    w = class_weights(labels, 4)
    assert w[3] == 0 and w[2] > w[1] > w[0] and np.isclose(w[:3].mean(), 1.0)
    idx = minority_oversample(labels, target_fraction=0.5, max_multiplier=4, seed=0)
    counts = np.bincount(labels[idx], minlength=3)
    assert counts[0] == 80 and counts[1] == 40 and counts[2] == 20  # target 40, 4x cap on class 2
    assert set(idx.tolist()) == set(range(len(labels))), "every original sample kept"
    np.testing.assert_array_equal(idx, minority_oversample(labels, 0.5, 4, seed=0))


def _fit_on_train(cfg):
    df = load_metadata(cfg)
    s = patient_level_split(df, cfg["split"]["ratios"], cfg["split"]["seed"], cfg["split"]["n_folds"])
    cat, cont = cfg["data"]["categorical_fields"], cfg["data"]["continuous_fields"]
    p = TabularPreprocessor(cat, cont, cfg["data"]["missing_tokens"]).fit(df.iloc[s["train"]])
    return df, s, p


def test_synthetic_fit_on_train_rows_only(synth_cfg):
    df, s, p = _fit_on_train(synth_cfg)
    assert p.n_fit_rows == len(s["train"])
    train_age = pd.to_numeric(df.iloc[s["train"]]["age"]).mean()
    all_age = pd.to_numeric(df["age"]).mean()
    assert np.isclose(p.cont_mean[0], train_age)
    assert not np.isclose(p.cont_mean[0], all_age)


@real_data
def test_real_booleans_never_zero(base_cfg):
    df, s, p = _fit_on_train(base_cfg)
    enc = p.transform(df)
    for j, f in enumerate(p.categorical):
        observed = (~df[f].isin(base_cfg["data"]["missing_tokens"])).to_numpy()
        in_vocab = df[f].isin(list(p.vocab[f])).to_numpy()
        assert (enc["cat"][observed & in_vocab, j] > 0).all(), f
        assert (enc["cat"][~observed, j] == 0).all(), f
    assert "BRASIL" not in p.vocab["background_father"], "alias BRASIL->BRAZIL applied"
    assert "biopsed" not in p.categorical + p.continuous
