# STATE.md — single source of truth for progress

## Progress: M3 done, ~70% (updated 2026-10-17, Chirag)

| Milestone | % | Done when |
|---|---|---|
| M0 overnight build | 35 | pipeline, tests, real data, split (done 2026-10-12) |
| M1 decisions + GPU smoke run | 45 | review items settled, local GPU smoke run succeeds (both done 2026-10-16) |
| M2 first real runs | 55 | gated + image-only trained, gate_log checked for collapse, results plausible (done 2026-10-17) |
| M3 full grid | 70 | 4 models x 3 seeds, val sweeps on both subsets, paired bootstrap (gated vs late, gated vs impute), complete-only ablation (done 2026-10-17) |
| M4 XAI + figures | 80 | deletion/insertion sweep, efficiency table, figures from committed CSVs |
| M5 final test | 88 | `--final-test` once per chosen checkpoint, lock pushed, final tables |
| M6 write-up + demo | 100 | report, paper draft, demo app, viva notes |

Done this turn: all NEEDS HUMAN REVIEW items decided (DECISIONS.md, 2026-10-16); second laptop set up (RTX 4050, venv outside OneDrive, `configs/local.yaml`), real data in place, 72/72 tests pass; page file fixed; GPU smoke run (gated, 20 steps + val, output in tmp/) succeeded on CUDA with AMP.

2026-10-17: M2 and M3 done. 14 training runs (4 models x seeds 1-3 with config hash `1bd64f00`, plus the complete-metadata ablation for gated and late, seed 1, hash `feff02f0`), validation sweeps for all 14 (`results/eval/`), and 9 paired comparisons (`results/compare/`: gated vs late and gated vs impute per seed, plus 3 ablation pairs). New command `python -m mamg.compare` with level and degradation (`*_drop_reduction`) tests. Results and the claim wording they support: DECISIONS.md, "2026-10-17 — M3 results". Headline: gated is ~4 macro-F1 points below late fusion on complete inputs but degrades less under masking in 2 of 3 seeds; the gate responds the wrong way to masking because of the missingness shortcut, and the complete-metadata ablation reverses that (1 seed).

Checkpoints (`best.pt`) live only on Chirag's laptop (`C:\Users\Chirag G K\mamg-local\checkpoints\`, ~55 MB each, not in git). M4 (XAI sweep) and M5 (final test) need them: either run those steps on that laptop, or copy the checkpoints over a drive.

**Next action (M4):** XAI and efficiency on validation: `python -m mamg.xai_eval --config <cfg> --ckpt "<checkpoints_dir>/gated_s1_1bd64f00/best.pt"` (and late fusion for the same seeds), then `python -m mamg.xai_eval --config <cfg> --efficiency`; then figures in `figures/` built only from committed CSVs (macro-F1 vs masking per subset with seed spread; drop-reduction forest plot from `results/compare/`; gate mean vs masking).

## Environment (Chirag's laptop, 2026-10-16)
- Windows 11 Home 10.0.26200, 16 GB RAM, NVIDIA RTX 4050 Laptop GPU 6 GB (driver 581.29)
- venv: `C:\Users\Chirag G K\mamg-local\.venv` (Python 3.13.3, torch 2.14.1+cu132, torchvision 0.29.1+cu132; rest from requirements.txt)
- Data: metadata.csv in repo `data/raw/` (gitignored); 256-px image cache and checkpoints in `C:\Users\Chirag G K\mamg-local\` (outside OneDrive), wired by gitignored `configs/local.yaml`
- Tests: 78 passed with real data (2026-10-17)
- Training throughput: ~30-70 s per epoch with two runs sharing the GPU (`num_workers: 0`); a 40-epoch-max run with early stopping takes 10-25 min

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

## Blockers
None. (Resolved 2026-10-17: Chirag's laptop GPU dropped off the bus at ~07:04 with three processes on it; a reboot fixed it and every run resumed from `last.pt`. Since then at most two GPU processes run at once. Earlier: page file too small -> system-managed.)

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
6. If the repo is inside OneDrive: keep the venv, images and checkpoints outside it. Copy
   `configs/local.example.yaml` to `configs/local.yaml` (gitignored), set your paths, and pass
   `--config configs/local.yaml`. Windows + CUDA needs a large page file (system-managed).

## Resume protocol
`git pull`; read STATE.md, CLAUDE.md, DECISIONS.md; `git log --oneline -20`; `git status`;
`.venv/Scripts/python -m pytest -q`; continue from the first unchecked WP.

ALL_DONE (overnight build, 2026-10-12). Phase 2 (real training on Kaggle/HPC, sweeps, XAI, paper) is ongoing: track it in the checklist below.

## Phase 2 checklist (update as you go)
- [x] Team decision on the missingness/benign confound protocol (MORNING_REPORT item 1) recorded in DECISIONS.md
- [x] Review all NEEDS HUMAN REVIEW items in DECISIONS.md
- [x] GPU smoke run (local RTX 4050; Kaggle dataset upload + kaggle/ placeholders only if local GPU is not enough)
- [x] First gated + image-only runs; gate_log.csv checked for collapse
- [x] 4 models x 3 seeds trained; results CSVs committed
- [x] Validation missingness sweeps (both subsets) + paired comparisons
- [ ] XAI sweep + efficiency table
- [ ] Final test (once per final checkpoint), lock file pushed
- [ ] Figures, report, paper draft (every number cites a committed CSV)
