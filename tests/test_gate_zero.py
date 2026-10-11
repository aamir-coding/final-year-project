"""Invariant 3 - the paper's central architectural property.

    fused = image_global + gate * attention_out

so with the gate forced to 0 the gated model IS the image-only model. We build a
real ImageOnly model, copy its image encoder and classifier head into a
GatedCrossAttention model, force gate = 0, and require identical logits (1e-6).

To make the 1e-6 tolerance meaningful, BatchNorm statistics are first calibrated on
random images (an untrained, uncalibrated EfficientNet gives nearly image-independent
logits), and we check that the logits really do vary with the image. A g = 0.5 check
against the formula computed by hand catches wrong fusion rules (e.g. (1-g)*glob + g*attn).
"""
import copy

import pytest
import torch

from mamg.models import GatedCrossAttention, ImageOnly

CARDS = [3] * 11 + [7, 15] + [3] * 5   # 18 categorical fields (vocab size + 1 for "missing")
N_CONT = 3
TOL = 1e-6


@pytest.fixture()
def mcfg(base_cfg):
    m = copy.deepcopy(base_cfg["model"])
    m["pretrained"] = False
    return m


def _inputs(batch=4, size=224, seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(batch, 3, size, size, generator=g)
    cat = torch.stack([torch.randint(0, c, (batch,), generator=g) for c in CARDS], dim=1)
    cont = torch.randn(batch, N_CONT, generator=g)
    cont_missing = (torch.rand(batch, N_CONT, generator=g) < 0.3).float()
    return image, cat, cont, cont_missing


@torch.no_grad()
def _calibrate_batchnorm(model, size=224):
    """Replace the default running stats (mean 0, var 1) with real activation statistics."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None  # cumulative average
    model.train()
    g = torch.Generator().manual_seed(123)
    for _ in range(3):
        model(torch.randn(8, 3, size, size, generator=g))
    model.eval()


@pytest.fixture()
def pair(mcfg):
    torch.manual_seed(0)
    image_only = ImageOnly(mcfg, n_classes=6)
    _calibrate_batchnorm(image_only)
    torch.manual_seed(1)  # different init on purpose: equality must come from the copied weights
    gated = GatedCrossAttention(mcfg, 6, cat_cardinalities=CARDS, n_cont=N_CONT).eval()
    gated.encoder.load_state_dict(image_only.encoder.state_dict())
    gated.head.load_state_dict(image_only.head.state_dict())
    return image_only, gated


@torch.no_grad()
def test_logits_depend_on_the_image(pair):
    """Sanity check that makes the 1e-6 comparison below non-trivial."""
    image_only, _ = pair
    image, *_ = _inputs()
    logits, _ = image_only(image)
    assert logits.std(0).max() > 1e-3
    blank, _ = image_only(torch.zeros_like(image))
    assert (blank - logits).abs().max() > 1e-3


@torch.no_grad()
def test_gate_zero_reproduces_image_only_logits(pair):
    image_only, gated = pair
    image, cat, cont, cont_missing = _inputs()

    expected, _ = image_only(image)
    got, aux = gated(image, cat, cont, cont_missing, gate_override=0.0)

    assert torch.all(aux["gate"] == 0)
    assert torch.allclose(got, expected, atol=TOL, rtol=0), (got - expected).abs().max()


@torch.no_grad()
def test_gate_zero_ignores_metadata_entirely(pair):
    """With g = 0, changing or deleting every metadata field cannot change the output."""
    _, gated = pair
    image, cat, cont, cont_missing = _inputs()
    a, _ = gated(image, cat, cont, cont_missing, gate_override=0.0)
    b, _ = gated(image, torch.zeros_like(cat), torch.zeros_like(cont), torch.ones_like(cont_missing),
                 gate_override=0.0)
    assert torch.allclose(a, b, atol=TOL, rtol=0)


@torch.no_grad()
def test_fusion_formula_at_intermediate_gate(pair):
    """At g = 0.5 the logits must equal head(image_global + 0.5 * attn_out), computed by hand."""
    _, gated = pair
    image, cat, cont, cont_missing = _inputs()
    tokens, glob = gated.encoder(image)
    meta = gated.tab(cat, cont, cont_missing)
    attn_out = gated.attn(meta.unsqueeze(1), tokens, tokens)[0].squeeze(1)
    expected = gated.head(glob + 0.5 * attn_out)
    got, _ = gated(image, cat, cont, cont_missing, gate_override=0.5)
    assert torch.allclose(got, expected, atol=TOL, rtol=0)
    assert (got - gated.head(glob)).abs().max() > 1e-4, "the attention path must contribute at g = 0.5"


@torch.no_grad()
def test_learned_gate_is_a_proper_fraction(pair):
    _, gated = pair
    image, cat, cont, cont_missing = _inputs()
    _, aux = gated(image, cat, cont, cont_missing)
    assert ((aux["gate"] > 0) & (aux["gate"] < 1)).all()
