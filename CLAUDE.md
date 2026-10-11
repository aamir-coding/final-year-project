# Missingness-Aware Multimodal Gating (Team 26PB67)

## What this is
Skin-lesion classification on PAD-UFES-20 (6 classes: ACK, BCC, MEL, NEV, SCC, SEK;
malignant = BCC, MEL, SCC) combining a smartphone clinical image with a partially
filled patient questionnaire. A learned per-sample scalar gate decides how much the
metadata branch (cross-attention over visual tokens) influences the prediction, so the
model degrades gracefully to image-only when metadata is missing.

Team: Aamir Mohammed Khan (data/tabular), Chirag G K (model/training), Giridhar Prabhu
(evaluation/XAI). Guide: Dr. Sahana P Shankar. Co-guide: Mr. Deepak Varadam. MSRUAS.

## Hard invariants: never violate these
1. Splits are grouped by patient_id (and lesion_id, via connected components). Every
   split asserts zero patient AND zero lesion overlap (`assert_no_overlap`), at split
   time, at load time, and in tests. A random image-level split invalidates every result.
2. A missing categorical value maps to reserved index 0, never to the same index as a
   genuine "no"/"False"/"0". Literal "UNK" is treated as missing, not as a category.
3. fused = image_global + gate * attention_out. gate=0 must reproduce the image-only
   logits exactly. tests/test_gate_zero.py proves it (tolerance 1e-6); keep it passing.
4. Preprocessing statistics (means, stds, vocabularies, modes, class weights) are fit
   on TRAIN only. Unseen categories at val/test map to 0 (missing).
5. The test split is touched only by `python -m mamg.evaluate --final-test`, which
   refuses to run twice for the same checkpoint (results/final_test_lock.csv).
   train.py can never load the test split (`load_split(..., allow_test=False)`).
6. Every training/eval run appends a row to runs.csv: seed, config hash, git SHA,
   metrics, wall-clock.
7. Headline metrics are macro-F1 and per-class recall (+ malignant recall). Never plain
   accuracy as a headline.
8. No magic numbers in code: everything in configs/default.yaml. Seeds via set_seed().
9. NEVER present synthetic-data output as a result. Synthetic outputs go only to
   tmp/synthetic_runs/ (gitignored) and are labelled SYNTHETIC. results/ and figures/
   contain only real-data outputs. `output_dirs(cfg)` enforces this automatically.

## Commands
- pytest -q
- python -m mamg.prepare_data --config configs/default.yaml        # verify data, profile missingness, write split
- python -m mamg.train --config configs/default.yaml --model {image,late,impute,gated} --seed 1 [--resume]
- python -m mamg.evaluate --config configs/default.yaml --ckpt <path>                # val sweep (dev)
- python -m mamg.evaluate --config configs/default.yaml --ckpt <path> --final-test   # ONCE, at the end
- python -m mamg.xai_eval --config configs/default.yaml --ckpt <path> [--efficiency]

## Conventions
- Package lives in src/mamg; `pip install -e .` (or PYTHONPATH=src).
- Every model returns (logits, aux); aux["gate"] holds the gate for the gated model.
- Masking (src/mamg/masking.py) is the single code path for both training-time modality
  dropout and the inference-time missingness sweep.
- Record every non-obvious choice in DECISIONS.md with a date.

## Current state
See STATE.md (single source of truth for progress) and MORNING_REPORT.md.
