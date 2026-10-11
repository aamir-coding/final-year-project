"""Verify the downloaded dataset, measure raw missingness, and write the split.

    python -m mamg.prepare_data --config configs/default.yaml [--cache-images] [--force-split]

Outputs (real data -> results/ and splits/; synthetic -> tmp/synthetic_runs/):
  missingness_profile.csv   per-field missing rate on the RAW metadata
  missingness_by_class.csv  per-class missing rate (exposes label-correlated missingness)
  split_seed{N}.csv         img_id -> train/val/test (committed; regenerate only deliberately)
  split_summary_seed{N}.csv counts per class per split
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .config import load_config, output_dirs
from .data import (build_image_index, load_metadata, missingness_by_class, missingness_profile,
                   patient_level_split, split_path, split_summary, split_to_frame)

EXPECTED_REAL = {"rows": 2298, "patients": 1373, "lesions": 1641}


def cache_images(cfg: dict, df, index) -> int:
    """Resize every image so its short side is `image.cache_short_side` (faster loading)."""
    out = Path(cfg["paths"]["image_cache_dir"])
    out.mkdir(parents=True, exist_ok=True)
    short = int(cfg["image"]["cache_short_side"])
    n = 0
    for img_id in df[cfg["data"]["image_col"]]:
        dst = out / img_id
        if dst.exists():
            continue
        with Image.open(index[img_id]) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = short / min(w, h)
            if s < 1:
                im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
            im.save(dst)
        n += 1
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. split.seed=1")
    ap.add_argument("--cache-images", action="store_true")
    ap.add_argument("--force-split", action="store_true", help="overwrite an existing split file")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    dirs = output_dirs(cfg)
    d = cfg["data"]

    df = load_metadata(cfg)
    n_pat, n_les = df[d["patient_col"]].nunique(), df[d["lesion_col"]].nunique()
    print(f"[{dirs['label']}] rows={len(df)} patients={n_pat} lesion_ids={n_les} "
          f"groups={df['group_id'].nunique()} complete_metadata={int(df['complete_metadata'].sum())}")
    if dirs["label"] == "REAL":
        got = {"rows": len(df), "patients": n_pat, "lesions": n_les}
        if got != EXPECTED_REAL:
            print(f"WARNING: counts {got} differ from the published {EXPECTED_REAL}")
    index = build_image_index(cfg["paths"]["image_dir"])
    found = int(df[d["image_col"]].isin(index).sum())
    print(f"images found: {found}/{len(df)} under {cfg['paths']['image_dir']}")

    res = dirs["results"]
    res.mkdir(parents=True, exist_ok=True)
    missingness_profile(df, cfg).to_csv(res / "missingness_profile.csv", index=False)
    missingness_by_class(df, cfg).to_csv(res / "missingness_by_class.csv", index=False)
    print(f"wrote {res / 'missingness_profile.csv'} and missingness_by_class.csv")

    seed = int(cfg["split"]["seed"])
    sp = split_path(dirs["splits"], seed)
    splits = patient_level_split(df, cfg["split"]["ratios"], seed, cfg["split"]["n_folds"])  # asserts zero overlap
    if sp.exists() and not a.force_split:
        print(f"split file exists, not overwritten: {sp} (use --force-split)")
    else:
        sp.parent.mkdir(parents=True, exist_ok=True)
        split_to_frame(df, splits, cfg).to_csv(sp, index=False)
        print(f"wrote {sp}")
    summ = split_summary(df, splits, cfg)
    summ.to_csv(sp.parent / f"split_summary_seed{seed}.csv", index=False)
    print(summ.to_string(index=False))

    if a.cache_images:
        if found < len(df):
            raise SystemExit("cannot cache: some images are missing")
        print(f"cached {cache_images(cfg, df, index)} images to {cfg['paths']['image_cache_dir']}")


if __name__ == "__main__":
    main()
