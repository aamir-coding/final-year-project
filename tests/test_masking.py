"""WP3: exact rate, seed determinism, masked fields look exactly like missing values."""
import numpy as np
import pytest
import torch

from mamg.masking import (apply_field_mask, keyed_field_mask, missingness_ratio,
                          modality_dropout_mask, n_masked_fields, observed_fields)

F_CAT, F_CONT = 18, 3
F = F_CAT + F_CONT


def _batch(b=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    cat = torch.randint(1, 5, (b, F_CAT), generator=g)
    cat[: b // 4, :10] = 0  # some naturally missing fields
    cont = torch.randn(b, F_CONT, generator=g)
    cont_missing = torch.zeros(b, F_CONT)
    cont_missing[: b // 4, 1:] = 1.0
    cont[cont_missing.bool()] = 0.0
    return cat, cont, cont_missing


@pytest.mark.parametrize("rate,k", [(0.0, 0), (0.1, 2), (0.3, 6), (0.5, 11), (1.0, 21)])
def test_exact_number_of_fields(rate, k):
    assert n_masked_fields(rate, F) == k
    m = keyed_field_mask(torch.arange(100), F, rate, seed=3)
    assert m.shape == (100, F) and (m.sum(1) == k).all()


def test_seed_determinism_and_batch_independence():
    keys = torch.tensor([11, 22, 33, 44, 55, 66])
    a = keyed_field_mask(keys, F, 0.3, seed=1)
    b = keyed_field_mask(keys, F, 0.3, seed=1)
    c = keyed_field_mask(keys, F, 0.3, seed=2)
    assert torch.equal(a, b) and not torch.equal(a, c)
    perm = torch.tensor([5, 2, 0, 4, 1, 3])
    assert torch.equal(keyed_field_mask(keys[perm], F, 0.3, seed=1), a[perm]), "mask must follow the sample"
    assert torch.equal(keyed_field_mask(keys[:2], F, 0.3, seed=1), a[:2]), "batch size must not matter"


def test_masks_spread_over_fields():
    m = keyed_field_mask(torch.arange(5000), F, 0.3, seed=0).float().mean(0)
    assert torch.allclose(m, torch.full((F,), 6 / F), atol=0.03)


def test_apply_hits_missing_encoding_and_copies():
    cat, cont, cmiss = _batch()
    orig = (cat.clone(), cont.clone(), cmiss.clone())
    m = keyed_field_mask(torch.arange(len(cat)), F, 0.5, seed=0)
    c2, x2, f2 = apply_field_mask(cat, cont, cmiss, m)
    assert torch.equal(cat, orig[0]) and torch.equal(cont, orig[1]) and torch.equal(cmiss, orig[2])
    assert (c2[m[:, :F_CAT]] == 0).all()
    assert (x2[m[:, F_CAT:]] == 0).all() and (f2[m[:, F_CAT:]] == 1).all()
    keep = ~m
    assert torch.equal(c2[keep[:, :F_CAT]], cat[keep[:, :F_CAT]])
    assert torch.equal(x2[keep[:, F_CAT:]], cont[keep[:, F_CAT:]])
    # missingness can only go up, and full masking means ratio 1
    assert (missingness_ratio(c2, f2) >= missingness_ratio(cat, cmiss)).all()
    full = torch.ones(len(cat), F, dtype=torch.bool)
    c3, _, f3 = apply_field_mask(cat, cont, cmiss, full)
    assert torch.allclose(missingness_ratio(c3, f3), torch.ones(len(cat)))


def test_observed_pool_only_masks_present_fields():
    cat, cont, cmiss = _batch()
    obs = observed_fields(cat, cmiss)
    m = keyed_field_mask(torch.arange(len(cat)), F, 0.5, seed=0, observed=obs, pool="observed")
    assert not (m & ~obs).any()
    expected = torch.floor(0.5 * obs.sum(1).float() + 0.5).long()
    assert (m.sum(1) == expected).all(), "observed pool hides `rate` of each sample's observed fields"
    # the hidden fraction no longer depends on natural missingness
    frac = m.sum(1).float() / obs.sum(1).float()
    assert (frac - 0.5).abs().max() <= 0.5 / obs.sum(1).min().item() + 1e-6


def test_modality_dropout_modes():
    rng = np.random.default_rng(0)
    base = {"enabled": True, "rate_choices": [0.1, 0.3, 0.5, 0.7]}
    assert modality_dropout_mask(32, F, {**base, "p_apply": 0.0, "p_full_mask": 1.0}, rng).all()
    assert not modality_dropout_mask(32, F, {**base, "p_apply": 0.0, "p_full_mask": 0.0}, rng).any()
    assert not modality_dropout_mask(32, F, {**base, "enabled": False, "p_apply": 1, "p_full_mask": 1}, rng).any()
    m = modality_dropout_mask(20000, F, {**base, "p_apply": 0.5, "p_full_mask": 0.1}, rng)
    full = m.all(1).float().mean().item()
    none = (~m.any(1)).float().mean().item()
    assert abs(full - 0.1) < 0.01 and abs(none - 0.4) < 0.015
    partial = m[m.any(1) & ~m.all(1)].sum(1)
    assert set(partial.unique().tolist()) <= {2, 6, 11, 15}


def test_rate_out_of_range():
    with pytest.raises(ValueError):
        n_masked_fields(1.5, F)
