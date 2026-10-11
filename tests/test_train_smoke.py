"""WP5 smoke test on the SYNTHETIC fixture, CPU, pretrained=False.

2 epochs must finish, write checkpoints, resume exactly after a simulated crash, and
append to a runs.csv under tmp (never results/)."""
import copy

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from mamg.config import output_dirs
from mamg.train import FocalLoss, lr_lambda_factory, train

ROOT_RESULTS = "results"


def test_focal_gamma0_equals_weighted_ce():
    torch.manual_seed(0)
    logits, y = torch.randn(32, 6), torch.randint(0, 6, (32,))
    w = torch.rand(6) + 0.5
    assert torch.allclose(FocalLoss(w, 0.0)(logits, y), F.cross_entropy(logits, y, weight=w), atol=1e-6)
    # focusing: confident correct examples are down-weighted
    easy = torch.tensor([[10.0, 0, 0, 0, 0, 0]])
    assert FocalLoss(None, 2.0)(easy, torch.tensor([0])) < FocalLoss(None, 0.0)(easy, torch.tensor([0]))


def test_lr_schedule_warmup_then_cosine():
    f = lr_lambda_factory(total_steps=100, warmup_steps=10, min_ratio=0.01)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(0.01)
    assert all(f(s) >= f(s + 1) for s in range(10, 100))


def _smoke_cfg(synth_cfg):
    cfg = copy.deepcopy(synth_cfg)
    cfg["runtime"]["max_train_steps"] = 4
    cfg["train"]["epochs"] = 2
    return cfg


@pytest.mark.parametrize("model_name", ["gated"])
def test_train_resume_is_exact(synth_cfg, model_name):
    cfg = _smoke_cfg(synth_cfg)
    dirs = output_dirs(cfg)
    assert dirs["label"] == "SYNTHETIC" and "results" not in str(dirs["results"]).split("synthetic_runs")[0]

    # run A: interrupted after epoch 0, then resumed
    cfg_a = copy.deepcopy(cfg)
    cfg_a["runtime"]["interrupt_after_epoch"] = 0
    out = train(cfg_a, model_name, seed=3)
    assert out.get("interrupted") and (out["run_dir"] / "last.pt").exists()
    assert not (dirs["results"] / "runs.csv").exists(), "interrupted run must not log a final row"
    resumed = train(cfg, model_name, seed=3, resume=True)
    assert resumed["state"]["epoch"] == 1 and resumed["state"]["finished"]
    a = torch.load(resumed["run_dir"] / "last.pt", map_location="cpu", weights_only=False)["model"]

    # run B: same thing uninterrupted, in a fresh output dir
    cfg_b = copy.deepcopy(cfg)
    cfg_b["paths"]["synthetic_results_dir"] = str(dirs["results"].parent / "b")
    out_b = train(cfg_b, model_name, seed=3)
    b = torch.load(out_b["run_dir"] / "last.pt", map_location="cpu", weights_only=False)["model"]
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-5), f"resume diverged at {k}"

    runs = pd.read_csv(dirs["results"] / "runs.csv")
    assert len(runs) == 1 and runs.loc[0, "data_kind"] == "SYNTHETIC"
    assert {"seed", "config_hash", "git_sha", "metrics_json", "wall_clock_s"} <= set(runs.columns)
    gate_log = pd.read_csv(dirs["results"] / "gate_log.csv")
    assert list(gate_log["epoch"]) == [0, 1]
    assert gate_log["val_gate_mean"].between(0, 1).all()
    # resuming a finished run is a no-op
    again = train(cfg, model_name, seed=3, resume=True)
    assert again["state"]["finished"]


@pytest.mark.parametrize("model_name", ["image", "late", "impute"])
def test_other_models_train(synth_cfg, model_name):
    cfg = _smoke_cfg(synth_cfg)
    cfg["runtime"]["max_train_steps"] = 2
    cfg["runtime"]["max_eval_steps"] = 2
    out = train(cfg, model_name, seed=1)
    assert (out["run_dir"] / "best.pt").exists()
    ck = torch.load(out["run_dir"] / "best.pt", map_location="cpu", weights_only=False)
    assert ck["data_kind"] == "SYNTHETIC" and ck["preprocessor"]["n_fit_rows"] > 0
