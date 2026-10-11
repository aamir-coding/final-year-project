"""WP4: model behaviour under missing metadata."""
import copy

import pytest
import torch

from mamg.masking import apply_field_mask
from mamg.models import build_model

CARDS = [3] * 11 + [7, 15] + [3] * 5
MODES = [2] * len(CARDS)
N_CONT = 3
F = len(CARDS) + N_CONT


@pytest.fixture()
def cfg(base_cfg):
    c = copy.deepcopy(base_cfg)
    c["model"]["pretrained"] = False
    return c


def _inputs(batch=4, size=64):
    torch.manual_seed(0)
    cat = torch.stack([torch.randint(1, c, (batch,)) for c in CARDS], 1)
    return torch.randn(batch, 3, size, size), cat, torch.randn(batch, N_CONT), torch.zeros(batch, N_CONT)


@torch.no_grad()
def test_gated_all_metadata_masked_is_finite_and_gate_in_unit_interval(cfg):
    model = build_model("gated", cfg, CARDS, N_CONT).eval()
    image, cat, cont, cmiss = _inputs()
    cat, cont, cmiss = apply_field_mask(cat, cont, cmiss, torch.ones(4, F, dtype=torch.bool))
    assert (cat == 0).all() and (cmiss == 1).all()
    logits, aux = model(image, cat, cont, cmiss)
    assert torch.isfinite(logits).all()
    assert ((aux["gate"] >= 0) & (aux["gate"] <= 1)).all()
    assert torch.allclose(aux["missing_ratio"], torch.ones(4))


@torch.no_grad()
def test_gate_sees_missingness_ratio(cfg):
    """The gate input includes the missingness ratio, so masking changes the gate."""
    model = build_model("gated", cfg, CARDS, N_CONT).eval()
    torch.nn.init.constant_(model.gate.weight, 0.0)
    model.gate.weight[0, -1] = -5.0  # weight on the ratio input only
    image, cat, cont, cmiss = _inputs()
    _, clean = model(image, cat, cont, cmiss)
    m = apply_field_mask(cat, cont, cmiss, torch.ones(4, F, dtype=torch.bool))
    _, masked = model(image, *m)
    assert (masked["gate"] < clean["gate"]).all()


@torch.no_grad()
def test_impute_baseline_cannot_tell_missing_from_mode(cfg):
    """ImputeFusion: a masked field is indistinguishable from the train-mode value."""
    model = build_model("impute", cfg, CARDS, N_CONT, cat_modes=MODES).eval()
    image, cat, cont, cmiss = _inputs()
    mask = torch.zeros(4, F, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mcat, mcont, mmiss = apply_field_mask(cat, cont, cmiss, mask)
    as_mode_cat = cat.clone()
    as_mode_cat[:, 0] = MODES[0]
    as_mean_cont = cont.clone()
    as_mean_cont[:, -1] = 0.0
    a, _ = model(image, mcat, mcont, mmiss)
    b, _ = model(image, as_mode_cat, as_mean_cont, torch.zeros_like(cmiss))
    assert torch.allclose(a, b, atol=1e-6)


@torch.no_grad()
def test_late_fusion_distinguishes_missing_from_no(cfg):
    """LateFusion (indicator encoding): missing (index 0) differs from any real value."""
    model = build_model("late", cfg, CARDS, N_CONT).eval()
    image, cat, cont, cmiss = _inputs()
    missing = cat.clone()
    missing[:, 0] = 0
    no = cat.clone()
    no[:, 0] = 1
    a, _ = model(image, missing, cont, cmiss)
    b, _ = model(image, no, cont, cmiss)
    assert (a - b).abs().max() > 1e-6
