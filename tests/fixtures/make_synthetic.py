"""SYNTHETIC PAD-UFES-20 look-alike for tests and smoke runs. NOT DATA. NOT RESULTS.

Same column schema as the real metadata.csv, ~120 patients, tiny random PNGs, and a
realistic missingness pattern (a questionnaire block missing for ~35% of benign rows,
UNK in grew/changed, a few lesion_ids shared across two patients as in the real file).
Writes a SYNTHETIC_DATA_MARKER file so every consumer routes outputs to tmp/.

    python tests/fixtures/make_synthetic.py --out tmp/synthetic_data --patients 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CLASS_P = [0.28, 0.30, 0.08, 0.12, 0.12, 0.10]
BENIGN = {"ACK", "NEV", "SEK"}
BLOCK = ["smoke", "drink", "background_father", "background_mother", "pesticide", "gender",
         "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system",
         "fitspatrick", "diameter_1", "diameter_2"]
BOOL_ALWAYS = ["itch", "hurt", "bleed", "elevation"]
BOOL_UNK = ["grew", "changed"]
COUNTRIES = ["POMERANIA", "GERMANY", "ITALY", "BRAZIL", "UNK", "NETHERLANDS"]
REGIONS = ["FACE", "FOREARM", "CHEST", "BACK", "ARM", "NOSE", "HAND", "NECK"]
COLUMNS = ["patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother",
           "age", "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
           "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic",
           "itch", "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"]
TINT = {c: np.array(t) for c, t in zip(CLASSES, [(200, 120, 110), (150, 90, 80), (60, 40, 40),
                                                  (120, 80, 60), (170, 110, 100), (110, 90, 70)])}


def tf(rng, p=0.5) -> str:
    return "True" if rng.random() < p else "False"


def make(out: Path, n_patients: int = 120, seed: int = 0, image_size: int = 32) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, lesion_counter, img_counter = [], 1, 1
    # guarantee every class has enough patients for a stratified 3-way split
    patient_classes = [CLASSES[i % len(CLASSES)] for i in range(len(CLASSES) * 6)]
    patient_classes += list(rng.choice(CLASSES, size=max(0, n_patients - len(patient_classes)), p=CLASS_P))
    shared_lesions = []
    for p in range(n_patients):
        pid = f"PAT_{p + 1}"
        dx = patient_classes[p]
        block_missing = dx in BENIGN and rng.random() < 0.35
        age = int(rng.integers(18, 90))
        for _ in range(int(rng.integers(1, 3))):
            lid = str(lesion_counter)
            lesion_counter += 1
            if rng.random() < 0.05 and shared_lesions:   # mimic shared lesion_ids
                lid = shared_lesions[int(rng.integers(len(shared_lesions)))]
            shared_lesions.append(lid)
            base = {
                "patient_id": pid, "lesion_id": lid, "age": str(age), "diagnostic": dx,
                "smoke": tf(rng, .15), "drink": tf(rng, .25),
                "background_father": str(rng.choice(COUNTRIES)), "background_mother": str(rng.choice(COUNTRIES)),
                "pesticide": tf(rng, .4), "gender": str(rng.choice(["FEMALE", "MALE"])),
                "skin_cancer_history": tf(rng, .45), "cancer_history": tf(rng, .5),
                "has_piped_water": tf(rng, .6), "has_sewage_system": tf(rng, .55),
                "fitspatrick": str(float(rng.choice([1, 2, 3, 4], p=[.1, .6, .25, .05]))),
                "region": str(rng.choice(REGIONS)),
                "diameter_1": str(float(rng.integers(2, 30))), "diameter_2": str(float(rng.integers(2, 25))),
                "biopsed": "True" if dx not in BENIGN else tf(rng, .3),
            }
            for f in BOOL_ALWAYS:
                base[f] = tf(rng, .5)
            for f in BOOL_UNK:
                base[f] = "UNK" if rng.random() < 0.17 else tf(rng, .5)
            if block_missing:
                for f in BLOCK:
                    base[f] = ""
            for _ in range(int(rng.integers(1, 3))):
                img_id = f"{pid}_{lid}_{img_counter}.png"
                img_counter += 1
                arr = rng.normal(TINT[dx], 40, size=(image_size, image_size, 3)).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(out / "images" / img_id)
                rows.append({**base, "img_id": img_id})
    df = pd.DataFrame(rows)[COLUMNS]
    df.to_csv(out / "metadata.csv", index=False)
    (out / "SYNTHETIC_DATA_MARKER").write_text("This folder is SYNTHETIC test data. Never report it.\n")
    return out / "metadata.csv"


def synthetic_cfg(base_cfg: dict, data_dir: Path, work_dir: Path) -> dict:
    """Config pointing at a synthetic fixture with tiny, CPU-friendly settings."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["kind"] = "synthetic"
    cfg["paths"].update({"metadata_csv": str(data_dir / "metadata.csv"), "image_dir": str(data_dir / "images"),
                         "image_cache_dir": "", "synthetic_results_dir": str(work_dir)})
    # ~120 patients cannot fill 20 stratified folds; 5 folds of 20% keep every class in every split
    cfg["split"].update({"ratios": [0.6, 0.2, 0.2], "n_folds": 5})
    cfg["image"]["size"] = 64
    cfg["model"]["pretrained"] = False
    cfg["runtime"].update({"device": "cpu", "num_workers": 0})
    cfg["train"].update({"epochs": 2, "batch_size": 8, "warmup_epochs": 0})
    cfg["eval"].update({"batch_size": 16, "bootstrap_resamples": 50, "mask_seeds": [0, 1]})
    cfg["xai"].update({"n_images": 2})
    cfg["efficiency"].update({"latency_warmup": 1, "latency_runs": 2})
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/synthetic_data")
    ap.add_argument("--patients", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("SYNTHETIC fixture written:", make(Path(a.out), a.patients, a.seed))
