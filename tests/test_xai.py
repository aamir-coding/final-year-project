"""WP7: deletion/insertion curves are sane, AUCs lie in [0, 1], a correct saliency map
beats a random pixel order, Grad-CAM++ runs on the real model classes, and the
efficiency table is filled."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from mamg.models import build_model
from mamg.train import train
from mamg.xai_eval import (FixedMetadata, curve_auc, efficiency_table, faithfulness,
                           perturbation_curve, saliency_order, xai_sweep)

H = W = 16
STEPS = 20


class RegionModel(nn.Module):
    """Toy classifier: the class-0 logit grows with the mean brightness of the top-left
    quadrant, so the 'true' saliency is exactly that quadrant."""

    def forward(self, x):
        region = x[:, :, : H // 2, : W // 2].mean(dim=(1, 2, 3))
        return torch.stack([4 * region, torch.zeros_like(region)], dim=1)


def _prob_fn(model):
    return lambda x: torch.softmax(model(x), 1)[:, 0]


def _setup():
    image = torch.ones(3, H, W)
    true_sal = np.zeros((H, W))
    true_sal[: H // 2, : W // 2] = 1.0
    true_sal += np.linspace(0, 1e-3, H * W).reshape(H, W)  # break ties deterministically
    return RegionModel(), image, true_sal


def test_deletion_curve_monotone_and_endpoints():
    model, image, sal = _setup()
    base = torch.zeros_like(image)
    curve = perturbation_curve(_prob_fn(model), image, saliency_order(sal), STEPS, "deletion", base)
    assert len(curve) == STEPS + 1
    p_full = float(_prob_fn(model)(image[None]))
    p_base = float(_prob_fn(model)(base[None]))
    assert curve[0] == pytest.approx(p_full) and curve[-1] == pytest.approx(p_base)
    assert np.all(np.diff(curve) <= 1e-7), "deleting important pixels can only lower the probability here"
    ins = perturbation_curve(_prob_fn(model), image, saliency_order(sal), STEPS, "insertion", base)
    assert ins[0] == pytest.approx(p_base) and ins[-1] == pytest.approx(p_full)
    assert np.all(np.diff(ins) >= -1e-7)


def test_true_saliency_beats_random_order():
    model, image, sal = _setup()
    base = torch.zeros_like(image)
    f = faithfulness(_prob_fn(model), image, sal, STEPS, base, base, np.random.default_rng(0))
    for k in ("deletion_auc", "insertion_auc", "deletion_auc_random", "insertion_auc_random"):
        assert 0.0 <= f[k] <= 1.0, k
    assert f["deletion_auc"] < f["deletion_auc_random"]
    assert f["insertion_auc"] > f["insertion_auc_random"]
    # an anti-saliency map (least important first) must do worse than random
    g = faithfulness(_prob_fn(model), image, -sal, STEPS, base, base, np.random.default_rng(0))
    assert g["deletion_auc"] > g["deletion_auc_random"]


def test_curve_auc():
    assert curve_auc(np.ones(21)) == pytest.approx(1.0)
    assert curve_auc(np.zeros(21)) == 0.0
    assert curve_auc(np.linspace(1, 0, 21)) == pytest.approx(0.5)


@pytest.fixture()
def cfg(base_cfg):
    c = copy.deepcopy(base_cfg)
    c["model"]["pretrained"] = False
    c["image"]["size"] = 64
    c["efficiency"].update({"latency_warmup": 1, "latency_runs": 2})
    return c


CARDS = [3] * 11 + [7, 15] + [3] * 5
N_CONT = 3


def test_gradcam_pp_on_untrained_gated_model(cfg):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from mamg.models import gradcam_target_layer

    torch.manual_seed(0)
    model = build_model("gated", cfg, CARDS, N_CONT).eval()
    wrapper = FixedMetadata(model).eval()
    image = torch.randn(1, 3, 64, 64)
    wrapper.set_meta(torch.ones(1, len(CARDS), dtype=torch.long), torch.zeros(1, N_CONT), torch.zeros(1, N_CONT))
    with GradCAMPlusPlus(model=wrapper, target_layers=[gradcam_target_layer(model, cfg)]) as cam:
        sal = cam(input_tensor=image, targets=[ClassifierOutputTarget(0)])
    assert sal.shape == (1, 64, 64) and np.isfinite(sal).all()
    assert sal.min() >= 0 and sal.max() <= 1 + 1e-6
    prob_fn = lambda x: torch.softmax(wrapper(x), 1)[:, 0]  # noqa: E731
    f = faithfulness(prob_fn, image[0], sal[0], STEPS, torch.zeros(3, 64, 64), torch.zeros(3, 64, 64),
                     np.random.default_rng(0))
    for k in ("deletion_auc", "insertion_auc", "deletion_auc_random", "insertion_auc_random"):
        assert 0.0 <= f[k] <= 1.0
    assert len(f["deletion_curve"]) == STEPS + 1


def test_efficiency_table(cfg):
    tab = efficiency_table(cfg, CARDS, N_CONT, [1] * len(CARDS), models=("image", "gated"))
    assert list(tab["model"]) == ["image", "gated"]
    assert (tab["params_total"] > 0).all() and (tab["gmacs"] > 0).all() and (tab["latency_ms_median"] > 0).all()
    img, gated = tab.set_index("model").loc["image"], tab.set_index("model").loc["gated"]
    assert gated["params_total"] > img["params_total"], "gated adds tabular + attention + gate parameters"


def test_xai_sweep_end_to_end(synth_cfg):
    c = copy.deepcopy(synth_cfg)
    c["runtime"]["max_train_steps"] = 2
    c["train"]["epochs"] = 1
    out = train(c, "gated", seed=0)
    df = xai_sweep(out["best_ckpt"], c, rates=[0.0, 0.5])
    assert len(df) == 2 * c["xai"]["n_images"]
    assert set(df["split"]) == {"val"} and set(df["data_kind"]) == {"SYNTHETIC"}
    for k in ("deletion_auc", "insertion_auc", "deletion_auc_random", "insertion_auc_random"):
        assert df[k].between(0, 1).all()
    assert df["gate"].between(0, 1).all()
