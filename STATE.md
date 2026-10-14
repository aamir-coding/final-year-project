# STATE.md — single source of truth for the overnight run

## Environment (Aamir's laptop, detected 2026-10-12 00:28 IST)
- OS: Windows 11 Home 10.0.26300 (Git Bash / MINGW64 + PowerShell)
- Python: 3.10.0 (system), project venv at `.venv/` (`.venv/Scripts/python`)
- GPU: NVIDIA GeForce GTX 1650, 4 GB, driver 616.56 — local torch installed CPU-only (disk space); training is planned for Kaggle/HPC
- CPU cores: 12; Disk free at start: 17 GB
- git 2.54, user configured (Aamir M K)
- Network: data.mendeley.com reachable (HTTP 200), github.com reachable, pypi reachable
- REAL DATA: OBTAINED. metadata.csv + imgs_part_1..3.zip downloaded from Mendeley, SHA-256 verified, 2,298 PNGs extracted to data/raw/images (gitignored). Counts match: 2,298 rows, 1,373 patients, 1,641 lesion ids.
- Missingness measured: results/missingness_profile.csv, results/missingness_by_class.csv (questionnaire block missing in 804 rows = 35%, ALL benign -> see DECISIONS.md)
- Split written: splits/split_seed42.csv (1589/364/345)
- Tests: 72 passed (`.venv/Scripts/python -m pytest -q`, ~1 min CPU)
- Git remote: https://github.com/aamir-coding/final-year-project (main). Team syncs via git pull/push (rules in CLAUDE.md)
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
- [x] WP10 Hardening + independent leak hunt (no leak found; 6 weaknesses fixed) + MORNING_REPORT.md

## In progress
Nothing. Overnight run complete.

## Next action
Team: read MORNING_REPORT.md (three decisions at the top), then follow 'Commands for tomorrow'.

## Blockers
None.

## Setting up a new laptop (teammates)
1. `git clone` / `git pull` the repo; create a venv: `python -m venv .venv`
   (Windows: `.venv/Scripts/python`, macOS/Linux: `.venv/bin/python`).
2. Install torch for that machine (CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`;
   NVIDIA GPU: the matching CUDA wheel from pytorch.org), then `pip install -r requirements.txt -e .`.
3. `python -m pytest -q` must show all tests passing (real-data tests skip until data exists).
4. Data (gitignored, ~3.6 GB raw): download metadata.csv + imgs_part_1..3.zip from
   https://data.mendeley.com/datasets/zr7vgbcyr2/1 into data/raw/, extract all PNGs into
   data/raw/images/, then `python -m mamg.prepare_data --cache-images` (must report 2298/2298 images
   and "split file exists, not overwritten"). Alternatively copy Aamir's data/raw/metadata.csv and
   data/processed/ (223 MB) via a drive. Never pass --force-split.
5. Re-run `python -m pytest -q` (real-data tests now run too).

## Resume protocol
`git pull`; read STATE.md, CLAUDE.md, DECISIONS.md; `git log --oneline -20`; `git status`;
`.venv/Scripts/python -m pytest -q`; continue from the first unchecked WP.

ALL_DONE (overnight build, 2026-10-12). Phase 2 (real training on Kaggle/HPC, sweeps, XAI, paper) is ongoing: track it in the checklist below.

## Phase 2 checklist (update as you go)
- [ ] Team decision on the missingness/benign confound protocol (MORNING_REPORT item 1) recorded in DECISIONS.md
- [ ] Review all NEEDS HUMAN REVIEW items in DECISIONS.md
- [ ] Private Kaggle dataset uploaded (metadata.csv + resized images) and kaggle/ placeholders filled
- [ ] First gated + image-only Kaggle runs; gate_log.csv checked for collapse
- [ ] 4 models x 3 seeds trained; results CSVs committed
- [ ] Validation missingness sweeps (both subsets) + paired comparisons
- [ ] XAI sweep + efficiency table
- [ ] Final test (once per final checkpoint), lock file pushed
- [ ] Figures, report, paper draft (every number cites a committed CSV)
