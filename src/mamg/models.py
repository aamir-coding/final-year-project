"""The four models. Every model returns (logits, aux).

    ImageOnly            image_global -> head
    LateFusion           [image_global, meta_emb] -> head          (missingness-aware tabular encoder)
    ImputeFusion         [image_global, meta_emb] -> head          (mean/mode imputation, NO indicators)
    GatedCrossAttention  image_global + g * CrossAttn(q=meta_emb, kv=visual tokens) -> head   (ours)

All four share one ImageEncoder, so the image path is identical code everywhere:
EfficientNet-B0 feature map (B, 1280, 7, 7) -> 1x1 conv to 256 -> 49 tokens; the
image-global vector is the mean of those tokens. With g = 0 the gated model is
exactly ImageOnly (tests/test_gate_zero.py).
"""
from __future__ import annotations

from typing import Sequence

import timm
import torch
import torch.nn as nn

from .masking import missingness_ratio

MODEL_NAMES = ("image", "late", "impute", "gated")


class ImageEncoder(nn.Module):
    def __init__(self, mcfg: dict):
        super().__init__()
        self.backbone = timm.create_model(mcfg["backbone"], pretrained=bool(mcfg["pretrained"]),
                                          num_classes=0, global_pool="")
        if self.backbone.num_features != mcfg["feature_channels"]:
            raise ValueError(f"backbone gives {self.backbone.num_features} channels, "
                             f"config says {mcfg['feature_channels']}")
        self.proj = nn.Conv2d(mcfg["feature_channels"], mcfg["token_dim"], kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.backbone.forward_features(x)          # (B, 1280, 7, 7) at 224 px
        tokens = self.proj(fmap).flatten(2).transpose(1, 2)  # (B, 49, 256)
        return tokens, tokens.mean(dim=1)                  # visual tokens, image_global (B, 256)


class TabularEncoder(nn.Module):
    """Per-field categorical embeddings + continuous values -> 3-layer GELU MLP.

    indicators=True : embedding index 0 is a learned "missing" vector and the
                      continuous missing flags are inputs (missingness-aware).
    indicators=False: plain imputation - missing categoricals are replaced by the
                      train mode, continuous values are already mean-imputed and the
                      flags are dropped, so the encoder cannot tell missing from mode.
    """

    def __init__(self, cat_cardinalities: Sequence[int], n_cont: int, mcfg: dict,
                 indicators: bool = True, cat_modes: Sequence[int] | None = None):
        super().__init__()
        self.indicators = indicators
        emb = int(mcfg["cat_embed_dim"])
        self.embeddings = nn.ModuleList(nn.Embedding(int(c), emb) for c in cat_cardinalities)
        if not indicators:
            if cat_modes is None or len(cat_modes) != len(cat_cardinalities):
                raise ValueError("imputation encoder needs one train-mode index per categorical field")
            self.register_buffer("cat_modes", torch.as_tensor(list(cat_modes), dtype=torch.long))
        in_dim = len(cat_cardinalities) * emb + n_cont * (2 if indicators else 1)
        layers, d = [], in_dim
        for _ in range(int(mcfg["meta_layers"]) - 1):
            layers += [nn.Linear(d, mcfg["meta_hidden"]), nn.GELU()]
            d = mcfg["meta_hidden"]
        layers.append(nn.Linear(d, mcfg["meta_dim"]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat: torch.Tensor, cont: torch.Tensor, cont_missing: torch.Tensor) -> torch.Tensor:
        if not self.indicators:
            cat = torch.where(cat == 0, self.cat_modes.unsqueeze(0).expand_as(cat), cat)
        parts = [e(cat[:, j]) for j, e in enumerate(self.embeddings)]
        parts.append(cont)
        if self.indicators:
            parts.append(cont_missing)
        return self.mlp(torch.cat(parts, dim=1))


def _head(in_dim: int, n_classes: int, mcfg: dict) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(mcfg["head_dropout"]), nn.Linear(in_dim, n_classes))


class ImageOnly(nn.Module):
    uses_metadata = False

    def __init__(self, mcfg: dict, n_classes: int, **_):
        super().__init__()
        self.encoder = ImageEncoder(mcfg)
        self.head = _head(mcfg["token_dim"], n_classes, mcfg)

    def forward(self, image, cat=None, cont=None, cont_missing=None):
        _, glob = self.encoder(image)
        return self.head(glob), {}


class LateFusion(nn.Module):
    uses_metadata = True
    indicators = True

    def __init__(self, mcfg: dict, n_classes: int, cat_cardinalities, n_cont, cat_modes=None):
        super().__init__()
        self.encoder = ImageEncoder(mcfg)
        self.tab = TabularEncoder(cat_cardinalities, n_cont, mcfg, indicators=self.indicators,
                                  cat_modes=cat_modes)
        self.head = _head(mcfg["token_dim"] + mcfg["meta_dim"], n_classes, mcfg)

    def forward(self, image, cat, cont, cont_missing):
        _, glob = self.encoder(image)
        meta = self.tab(cat, cont, cont_missing)
        return self.head(torch.cat([glob, meta], dim=1)), {}


class ImputeFusion(LateFusion):
    """Imputation baseline promised by synopsis objective OB3: identical to LateFusion
    except that missing values are imputed (mode / mean) with no missingness indicators."""
    indicators = False


class GatedCrossAttention(nn.Module):
    uses_metadata = True

    def __init__(self, mcfg: dict, n_classes: int, cat_cardinalities, n_cont, cat_modes=None):
        super().__init__()
        if mcfg["meta_dim"] != mcfg["token_dim"]:
            raise ValueError("meta_dim must equal token_dim (metadata embedding is the attention query)")
        self.encoder = ImageEncoder(mcfg)
        self.tab = TabularEncoder(cat_cardinalities, n_cont, mcfg, indicators=True)
        self.attn = nn.MultiheadAttention(mcfg["token_dim"], mcfg["attn_heads"],
                                          dropout=mcfg["attn_dropout"], batch_first=True)
        self.gate = nn.Linear(mcfg["meta_dim"] + 1, 1)
        nn.init.constant_(self.gate.bias, float(mcfg["gate_bias_init"]))
        self.head = _head(mcfg["token_dim"], n_classes, mcfg)

    def forward(self, image, cat, cont, cont_missing, gate_override: float | None = None):
        tokens, glob = self.encoder(image)                       # (B, N, D), (B, D)
        meta = self.tab(cat, cont, cont_missing)                 # (B, D)
        attn_out, attn_w = self.attn(meta.unsqueeze(1), tokens, tokens, need_weights=True)
        attn_out = attn_out.squeeze(1)                           # (B, D)
        ratio = missingness_ratio(cat, cont_missing).to(meta.dtype).unsqueeze(1)
        g = torch.sigmoid(self.gate(torch.cat([meta, ratio], dim=1)))  # (B, 1)
        if gate_override is not None:
            g = torch.full_like(g, float(gate_override))
        fused = glob + g * attn_out                              # invariant 3
        aux = {"gate": g.squeeze(1), "attn_weights": attn_w.squeeze(1), "missing_ratio": ratio.squeeze(1)}
        return self.head(fused), aux


_REGISTRY = {"image": ImageOnly, "late": LateFusion, "impute": ImputeFusion, "gated": GatedCrossAttention}


def build_model(name: str, cfg: dict, cat_cardinalities: Sequence[int], n_cont: int,
                cat_modes: Sequence[int] | None = None) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return _REGISTRY[name](cfg["model"], len(cfg["data"]["classes"]),
                           cat_cardinalities=list(cat_cardinalities), n_cont=n_cont, cat_modes=cat_modes)


def gradcam_target_layer(model: nn.Module, cfg: dict) -> nn.Module:
    return getattr(model.encoder.backbone, cfg["xai"]["target_layer"])
