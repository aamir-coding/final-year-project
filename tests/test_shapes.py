"""WP4: every model runs a CPU forward pass on a batch of 4 and returns (logits, aux)."""
import copy

import pytest
import torch

from mamg.models import MODEL_NAMES, ImageEncoder, build_model

CARDS = [3] * 11 + [7, 15] + [3] * 5
MODES = [1] * len(CARDS)
N_CONT = 3


@pytest.fixture()
def cfg(base_cfg):
    c = copy.deepcopy(base_cfg)
    c["model"]["pretrained"] = False
    return c


def _inputs(batch=4, size=224):
    torch.manual_seed(0)
    return (torch.randn(batch, 3, size, size),
            torch.stack([torch.randint(0, c, (batch,)) for c in CARDS], 1),
            torch.randn(batch, N_CONT), torch.zeros(batch, N_CONT))


def test_encoder_gives_49_tokens_of_256(cfg):
    enc = ImageEncoder(cfg["model"]).eval()
    fmap = enc.backbone.forward_features(torch.randn(2, 3, 224, 224))
    assert fmap.shape == (2, 1280, 7, 7)
    tokens, glob = enc(torch.randn(2, 3, 224, 224))
    assert tokens.shape == (2, 49, 256) and glob.shape == (2, 256)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shapes(cfg, name):
    model = build_model(name, cfg, CARDS, N_CONT, cat_modes=MODES).eval()
    with torch.no_grad():
        logits, aux = model(*_inputs())
    assert logits.shape == (4, len(cfg["data"]["classes"]))
    assert torch.isfinite(logits).all()
    assert isinstance(aux, dict)
    if name == "gated":
        assert aux["gate"].shape == (4,)
        assert aux["attn_weights"].shape == (4, 49)
        assert torch.allclose(aux["attn_weights"].sum(1), torch.ones(4), atol=1e-5)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_backward_reaches_all_branches(cfg, name):
    model = build_model(name, cfg, CARDS, N_CONT, cat_modes=MODES).train()
    logits, _ = model(*_inputs(batch=2, size=64))
    logits.sum().backward()
    assert model.encoder.proj.weight.grad is not None
    if name != "image":
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.tab.parameters())
    if name == "gated":
        assert model.gate.weight.grad is not None and model.gate.weight.grad.abs().sum() > 0


def test_unknown_model_rejected(cfg):
    with pytest.raises(ValueError):
        build_model("transformer", cfg, CARDS, N_CONT)
