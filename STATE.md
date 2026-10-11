# STATE.md — single source of truth for the overnight run

## Environment (detected 2026-10-12 00:28 IST)
- OS: Windows 11 Home 10.0.26300 (Git Bash / MINGW64 + PowerShell)
- Python: 3.10.0 (system), project venv at `.venv/` (`.venv/Scripts/python`)
- GPU: NVIDIA GeForce GTX 1650, 4 GB, driver 616.56 — local torch installed CPU-only (disk space); training is planned for Kaggle/HPC
- CPU cores: 12; Disk free at start: 17 GB
- git 2.54, user configured (Aamir M K)
- Network: data.mendeley.com reachable (HTTP 200), github.com reachable, pypi reachable
- REAL DATA: OBTAINED. metadata.csv + imgs_part_1..3.zip downloaded from Mendeley, SHA-256 verified, 2,298 PNGs extracted to data/raw/images (gitignored). Counts match: 2,298 rows, 1,373 patients, 1,641 lesion ids.
- Missingness measured: results/missingness_profile.csv, results/missingness_by_class.csv (questionnaire block missing in 804 rows = 35%, ALL benign -> see DECISIONS.md)
- Split written: splits/split_seed42.csv (1589/364/345)
- venv: torch 2.14.1+cpu, torchvision 0.29.1, timm 1.0.30, pandas 2.3.3, sklearn 1.7.2, grad-cam 1.5.7, fvcore, thop

## Work packages
- [x] WP1 Scaffold
- [x] WP2 Data layer (data.py, prepare_data.py, synthetic fixture, split + preprocessing tests)
- [x] WP3 Masking utility (masking.py + tests/test_masking.py)
- [x] WP4 Models (models.py; test_gate_zero, test_shapes, test_masking_behaviour)
- [x] WP5 Training (train.py, engine.py, metrics.py; test_train_smoke incl. exact resume)
- [x] WP6 Evaluation (evaluate.py, stats.py; tests/test_evaluate.py incl. sklearn agreement + final-test lock)
- [x] WP7 XAI (xai_eval.py: Grad-CAM++, deletion/insertion AUC + random baseline, efficiency table; tests/test_xai.py)
- [x] WP8 Kaggle runner (kaggle/kernel-metadata.json, runner.ipynb, README.md, configs/kaggle.yaml)
- [x] WP9 Team learning docs (docs/WALKTHROUGH.md, docs/CLAIMS_GUARDRAIL.md)
- [ ] WP10 Hardening + leak hunt + MORNING_REPORT.md

## In progress
WP8 Kaggle runner + WP9 docs (subagent), then WP10.

## Next action
WP8/WP9 files (kaggle/, docs/, configs/kaggle.yaml) then WP10 hardening + leak hunt.

## Blockers
None.

## Resume protocol
Read STATE.md, CLAUDE.md, DECISIONS.md; `git log --oneline -20`; `git status`;
`.venv/Scripts/python -m pytest -q`; continue from the first unchecked WP.
