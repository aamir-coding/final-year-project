"""Invariant 3 - the paper's central architectural claim.

    fused = image_global + gate * attention_out

so with the gate forced to 0 the gated model IS the image-only model. We build a
real ImageOnly model, copy its image encoder and classifier head into a
GatedCrossAttention model, force gate = 0, and require identical logits (1e-6).
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


def _pair(mcfg):
    torch.manual_seed(0)
    image_only = ImageOnly(mcfg, n_classes=6).eval()
    torch.manual_seed(1)  # different init on purpose: equality must come from the copied weights
    gated = GatedCrossAttention(mcfg, 6, cat_cardinalities=CARDS, n_cont=N_CONT).eval()
    gated.encoder.load_state_dict(image_only.encoder.state_dict())
    gated.head.load_state_dict(image_only.head.state_dict())
    return image_only, gated


@torch.no_grad()
def test_gate_zero_reproduces_image_only_logits(mcfg):
    image_only, gated = _pair(mcfg)
    image, cat, cont, cont_missing = _inputs()

    expected, _ = image_only(image)
    got, aux = gated(image, cat, cont, cont_missing, gate_override=0.0)

    assert torch.all(aux["gate"] == 0)
    assert torch.allclose(got, expected, atol=TOL, rtol=0), (got - expected).abs().max()


@torch.no_grad()
def test_gate_zero_ignores_metadata_entirely(mcfg):
    """With g = 0, changing or deleting every metadata field cannot change the output."""
    _, gated = _pair(mcfg)
    image, cat, cont, cont_missing = _inputs()
    a, _ = gated(image, cat, cont, cont_missing, gate_override=0.0)
    b, _ = gated(image, torch.zeros_like(cat), torch.zeros_like(cont), torch.ones_like(cont_missing),
                 gate_override=0.0)
    assert torch.allclose(a, b, atol=TOL, rtol=0)


@torch.no_grad()
def test_nonzero_gate_does_use_metadata(mcfg):
    """Guards against a vacuous pass: with g = 1 the metadata path must matter."""
    image_only, gated = _pair(mcfg)
    image, cat, cont, cont_missing = _inputs()
    expected, _ = image_only(image)
    got, _ = gated(image, cat, cont, cont_missing, gate_override=1.0)
    assert (got - expected).abs().max() > 1e-4
    learned, aux = gated(image, cat, cont, cont_missing)
    assert ((aux["gate"] > 0) & (aux["gate"] < 1)).all()
