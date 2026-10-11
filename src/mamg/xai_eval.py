"""Explainability and efficiency.

    python -m mamg.xai_eval --ckpt checkpoints/<run>/best.pt          # Grad-CAM++ faithfulness sweep (val split)
    python -m mamg.xai_eval --efficiency                              # params / FLOPs / CPU latency, all 4 models

Faithfulness follows RISE (Petsiuk et al., 2018): pixels are removed (deletion) or
revealed (insertion) in descending saliency order in `xai.deletion_steps` equal steps,
the softmax probability of the originally predicted class is recorded at each step,
and the curve is integrated over [0, 1]. Lower deletion AUC / higher insertion AUC =
more faithful saliency. A random pixel order gives the reference baseline. These are
empirical measurements of explanation faithfulness, not proofs.
The sweep runs on the VALIDATION split; the test split stays reserved (invariant 5).
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .config import load_config, metadata_fields, output_dirs
from .data import TabularPreprocessor, load_metadata, load_split
from .engine import build_pipeline, ensure_split
from .evaluate import load_checkpoint_cfg
from .masking import apply_field_mask, keyed_field_mask, observed_fields
from .models import MODEL_NAMES, build_model, gradcam_target_layer
from .utils import append_run, get_device, set_seed


# ------------------------------------------------------------------ faithfulness core
def perturbation_curve(prob_fn: Callable[[torch.Tensor], torch.Tensor], image: torch.Tensor,
                       order: np.ndarray, steps: int, mode: str, baseline: torch.Tensor) -> np.ndarray:
    """Probability of the target class after perturbing the first k pixels of `order`,
    for k = round(i * HW / steps), i = 0..steps. Returns an array of length steps + 1.

    deletion : start from the image, overwrite pixels with `baseline`
    insertion: start from `baseline`, reveal pixels of the image
    All steps+1 perturbed images are scored in one batch.
    """
    if mode not in ("deletion", "insertion"):
        raise ValueError(mode)
    C, H, W = image.shape
    n = H * W
    rank = np.empty(n, dtype=np.int64)
    rank[np.asarray(order)] = np.arange(n)       # position of each pixel in the order
    rank = torch.from_numpy(rank).view(1, H, W)
    ks = torch.tensor([int(round(i * n / steps)) for i in range(steps + 1)]).view(-1, 1, 1)
    touched = (rank < ks).unsqueeze(1)           # (S, 1, H, W): pixel already perturbed at step s
    src, dst = (image, baseline) if mode == "deletion" else (baseline, image)
    batch = torch.where(touched, dst.unsqueeze(0), src.unsqueeze(0))
    with torch.no_grad():
        p = prob_fn(batch)
    return p.detach().float().cpu().numpy()


def curve_auc(curve: np.ndarray) -> float:
    """Trapezoidal area under a curve sampled on an even grid over [0, 1]."""
    x = np.linspace(0.0, 1.0, len(curve))
    return float(np.trapezoid(curve, x)) if hasattr(np, "trapezoid") else float(np.trapz(curve, x))


def saliency_order(saliency: np.ndarray) -> np.ndarray:
    """Pixel indices, most important first (stable for ties)."""
    return np.argsort(-np.asarray(saliency, dtype=np.float64).ravel(), kind="stable")


def faithfulness(prob_fn, image: torch.Tensor, saliency: np.ndarray, steps: int,
                 deletion_baseline: torch.Tensor, insertion_baseline: torch.Tensor,
                 rng: np.random.Generator) -> dict:
    n = image.shape[1] * image.shape[2]
    sal, rnd = saliency_order(saliency), rng.permutation(n)
    out = {}
    for name, order in (("", sal), ("_random", rnd)):
        d = perturbation_curve(prob_fn, image, order, steps, "deletion", deletion_baseline)
        i = perturbation_curve(prob_fn, image, order, steps, "insertion", insertion_baseline)
        out[f"deletion_auc{name}"], out[f"insertion_auc{name}"] = curve_auc(d), curve_auc(i)
        out[f"deletion_curve{name}"], out[f"insertion_curve{name}"] = d, i
    return out


# ------------------------------------------------------------------ Grad-CAM++
class FixedMetadata(nn.Module):
    """Adapts a (image, cat, cont, cont_missing) model to the image-only call that
    pytorch-grad-cam expects. Metadata of batch size 1 is broadcast to the image batch."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.meta = None

    def set_meta(self, cat, cont, cont_missing):
        self.meta = (cat, cont, cont_missing)

    def forward(self, image):
        b = image.shape[0]
        cat, cont, cmiss = (t.expand(b, -1) if t.shape[0] == 1 else t for t in self.meta)
        return self.model(image, cat, cont, cmiss)[0]


def make_baselines(image: torch.Tensor, xcfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if xcfg["deletion_baseline"] != "zero":
        raise ValueError("only the 'zero' (channel-mean after normalisation) deletion baseline is implemented")
    blur = TF.gaussian_blur(image, kernel_size=int(xcfg["blur_kernel"]), sigma=float(xcfg["blur_sigma"]))
    ins = blur if xcfg["insertion_baseline"] == "blur" else torch.zeros_like(image)
    return torch.zeros_like(image), ins


def xai_sweep(ckpt_path: str | Path, current_cfg: dict | None = None, rates=None) -> pd.DataFrame:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # own checkpoints only
    cfg = load_checkpoint_cfg(ck, current_cfg)
    dirs, xcfg, ev = output_dirs(cfg), cfg["xai"], cfg["eval"]
    set_seed(int(ck["seed"]))
    device = get_device(cfg["runtime"]["device"])
    pipe = build_pipeline(cfg, need=("val",), pre_state=ck["preprocessor"])
    ds = pipe.datasets["val"]
    model = build_model(ck["model_name"], cfg, pipe.pre.cat_cardinalities, len(pipe.pre.continuous),
                        cat_modes=pipe.pre.cat_modes)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    wrapper = FixedMetadata(model).eval()
    rates = list(ev["missingness_rates"] if rates is None else rates)
    n_img = min(int(xcfg["n_images"]), len(ds))
    mask_seed = int(ev["mask_seeds"][0])
    rows, curves = [], []
    with GradCAMPlusPlus(model=wrapper, target_layers=[gradcam_target_layer(model, cfg)]) as cam:
        for rate in rates:
            for i in range(n_img):
                s = ds[i]
                image = s["image"].unsqueeze(0).to(device)
                cat, cont, cmiss = (s[k].unsqueeze(0).to(device) for k in ("cat", "cont", "cont_missing"))
                if rate > 0:
                    m = keyed_field_mask(s["key"].view(1), cat.shape[1] + cont.shape[1], rate, mask_seed,
                                         observed=observed_fields(cat, cmiss), pool=ev["mask_pool"])
                    cat, cont, cmiss = apply_field_mask(cat, cont, cmiss, m)
                wrapper.set_meta(cat, cont, cmiss)
                with torch.no_grad():
                    logits, aux = model(image, cat, cont, cmiss)
                pred = int(logits.argmax(1))
                sal = cam(input_tensor=image, targets=[ClassifierOutputTarget(pred)])[0]
                prob_fn = lambda x: torch.softmax(wrapper(x.to(device)), 1)[:, pred]  # noqa: E731
                d_base, i_base = make_baselines(image[0], xcfg)
                rng = np.random.default_rng([int(xcfg["random_baseline_seed"]), i])
                f = faithfulness(prob_fn, image[0], sal, int(xcfg["deletion_steps"]), d_base, i_base, rng)
                rows.append({"model": ck["model_name"], "run_name": ck["run_name"], "data_kind": dirs["label"],
                             "split": "val", "missingness": rate, "img_id": ds.img_ids[i],
                             "y_true": int(s["label"]), "y_pred": pred,
                             "p_pred": float(torch.softmax(logits, 1)[0, pred]),
                             "gate": float(aux["gate"][0]) if "gate" in aux else np.nan,
                             **{k: v for k, v in f.items() if "_auc" in k}})
                for k, v in f.items():
                    if "curve" in k:
                        curves.append({"missingness": rate, "img_id": ds.img_ids[i], "curve": k,
                                       **{f"step_{j}": float(x) for j, x in enumerate(v)}})
    df = pd.DataFrame(rows)
    out_dir = dirs["results"] / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{ck['run_name']}_faithfulness.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / f"{ck['run_name']}_curves.csv", index=False)
    summ = df.groupby("missingness")[[c for c in df.columns if "auc" in c]].mean().round(4)
    append_run(dirs["results"] / "runs.csv", run_type="xai", data_kind=dirs["label"], model=ck["model_name"],
               seed=ck["seed"], config_hash=ck["config_hash"], run_name=ck["run_name"],
               checkpoint=str(ckpt_path), split="val", wall_clock_s="",
               metrics_json={f"{c}@{r:g}": float(summ.loc[r, c]) for r in summ.index for c in summ.columns})
    print(summ.to_string())
    return df


# ------------------------------------------------------------------ efficiency
def count_flops(model: nn.Module, inputs: tuple) -> tuple[float, str]:
    """Multiply-accumulate count for one forward pass (fvcore, falling back to thop)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, inputs)
        fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        return float(fca.total()), "fvcore"
    except Exception:
        from thop import profile
        macs, _ = profile(model, inputs=inputs, verbose=False)
        return float(macs), "thop"


def efficiency_table(cfg: dict, cat_cardinalities, n_cont: int, cat_modes, models=MODEL_NAMES) -> pd.DataFrame:
    e = cfg["efficiency"]
    torch.set_num_threads(int(e["threads"]))
    size, bs = int(cfg["image"]["size"]), int(e["latency_batch_size"])
    gen = torch.Generator().manual_seed(0)
    image = torch.randn(bs, 3, size, size, generator=gen)
    cat = torch.stack([torch.randint(0, c, (bs,), generator=gen) for c in cat_cardinalities], 1)
    cont, cmiss = torch.randn(bs, n_cont, generator=gen), torch.zeros(bs, n_cont)
    rows = []
    for name in models:
        m = build_model(name, cfg, cat_cardinalities, n_cont, cat_modes=cat_modes).eval()
        inputs = (image, cat, cont, cmiss)
        macs, tool = count_flops(m, inputs)
        times = []
        with torch.no_grad():
            for _ in range(int(e["latency_warmup"])):
                m(*inputs)
            for _ in range(int(e["latency_runs"])):
                t0 = time.perf_counter()
                m(*inputs)
                times.append((time.perf_counter() - t0) * 1000)
        rows.append({"model": name, "params_total": sum(p.numel() for p in m.parameters()),
                     "params_trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
                     "gmacs": round(macs / 1e9, 4), "flop_counter": tool,
                     "latency_ms_median": round(float(np.median(times)), 2),
                     "latency_ms_p90": round(float(np.percentile(times, 90)), 2),
                     "batch_size": bs, "image_size": size, "threads": int(e["threads"]),
                     "cpu": platform.processor() or platform.machine(), "torch": torch.__version__})
    return pd.DataFrame(rows)


def _fit_preprocessor(cfg: dict) -> TabularPreprocessor:
    df = load_metadata(cfg)
    split_df = pd.read_csv(ensure_split(cfg, df), dtype=str, keep_default_na=False)
    cat, cont = metadata_fields(cfg)
    return TabularPreprocessor(cat, cont, cfg["data"]["missing_tokens"]).fit(
        df.iloc[load_split(df, split_df, "train", cfg)])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None, help="checkpoint for the Grad-CAM++ faithfulness sweep")
    ap.add_argument("--efficiency", action="store_true", help="write the params/FLOPs/latency table")
    ap.add_argument("--missingness", default=None, help="comma list of percents, e.g. 0,10,30,50")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.efficiency:
        pre = _fit_preprocessor(cfg)
        cfg["model"]["pretrained"] = False  # architecture cost only; weights do not change FLOPs/latency
        tab = efficiency_table(cfg, pre.cat_cardinalities, len(pre.continuous), pre.cat_modes)
        out = output_dirs(cfg)["results"] / "efficiency.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(out, index=False)
        print(tab.to_string(index=False))
    if a.ckpt:
        rates = [float(x) / 100 for x in a.missingness.split(",")] if a.missingness else None
        xai_sweep(a.ckpt, cfg, rates=rates)
    if not (a.efficiency or a.ckpt):
        ap.error("give --ckpt and/or --efficiency")


if __name__ == "__main__":
    main()
