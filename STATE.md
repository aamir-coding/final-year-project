# STATE.md — single source of truth for the overnight run

## Environment (detected 2026-10-12 00:28 IST)
- OS: Windows 11 Home 10.0.26300 (Git Bash / MINGW64 + PowerShell)
- Python: 3.10.0 (system), project venv at `.venv/` (`.venv/Scripts/python`)
- GPU: NVIDIA GeForce GTX 1650, 4 GB, driver 616.56 — local torch installed CPU-only (disk space); training is planned for Kaggle/HPC
- CPU cores: 12; Disk free at start: 17 GB
- git 2.54, user configured (Aamir M K)
- Network: data.mendeley.com reachable (HTTP 200), github.com reachable, pypi reachable
- REAL DATA: metadata.csv downloaded + SHA-256 verified; image zips downloading (see below)

## Work packages
- [ ] WP1 Scaffold
- [ ] WP2 Data layer
- [ ] WP3 Masking utility
- [ ] WP4 Models
- [ ] WP5 Training
- [ ] WP6 Evaluation
- [ ] WP7 XAI
- [ ] WP8 Kaggle runner
- [ ] WP9 Team learning docs
- [ ] WP10 Hardening + leak hunt + MORNING_REPORT.md

## In progress
WP1 scaffold.

## Next action
Commit WP1, then WP2 data layer.

## Blockers
None.

## Resume protocol
Read STATE.md, CLAUDE.md, DECISIONS.md; `git log --oneline -20`; `git status`;
`.venv/Scripts/python -m pytest -q`; continue from the first unchecked WP.
