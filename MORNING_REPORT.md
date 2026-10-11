# MORNING REPORT — overnight run, 12 Oct 2026

**No blockers.** The real dataset was downloaded and verified tonight, and all work packages are done. Tonight produced **no scientific results**: nothing was trained to convergence and no numbers have been reported.

## Read this first: three decisions only the team can make

1. **In PAD-UFES-20, missing metadata gives away the label.** 13 questionnaire fields (smoke, drink, background_father/mother, pesticide, gender, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, diameter_1, diameter_2) are missing *together* in 804 of 2,298 rows (35.0%). **All 804 rows are benign** (ACK 447, NEV 169, SEK 188). Every BCC, MEL and SCC row has the questionnaire. A missingness-aware model can learn "questionnaire missing => benign". That would inflate its clean score, and it could misread a masked malignant case as benign. The code now reports every metric on `all` rows and on a `complete_metadata` subset (n = 1,494, at most 2 unrecorded fields; it contains every malignant image), where inference-time masking is a clean, controlled experiment. A switch, `--set data.train_subset=complete_metadata`, trains on complete rows only. **Recommendation:** make the complete-metadata subset the primary setting for the missingness sweep, report both in the paper, and discuss this openly; it strengthens the paper rather than weakening it. See `results/missingness_by_class.csv`.
2. **Some lesion ids belong to two patients.** There are 1,641 lesion ids but 1,891 (patient, lesion) pairs; 250 lesion ids appear under two patient ids (for example lesion 1867 under PAT_995 and PAT_1545). The split merges them into one group (union-find), so no patient or lesion crosses splits. Check with the dataset authors whether these are duplicate registrations.
3. **The other conservative calls marked NEEDS HUMAN REVIEW are listed below.**

## What exists (one line per file)

| File | What it is |
|---|---|
| `CLAUDE.md` | Project brief and the 9 hard invariants, read by every Claude Code session |
| `DECISIONS.md` | Every design choice with its reason, dated. The viva depends on it |
| `STATE.md` | Progress checklist and resume protocol |
| `configs/default.yaml` | All hyperparameters, paths, missingness rates, seeds (no magic numbers in code) |
| `configs/kaggle.yaml` | Kaggle path/runtime overrides (inherits default) |
| `src/mamg/config.py` | YAML loading, `--set` overrides, config hash, routes synthetic output to `tmp/` |
| `src/mamg/utils.py` | `set_seed`, device, git SHA, the `runs.csv` ledger |
| `src/mamg/data.py` | Schema checks, union-find patient/lesion groups, `patient_level_split` (StratifiedGroupKFold), `assert_no_overlap`, train-only `TabularPreprocessor` (missing = index 0), Dataset and transforms |
| `src/mamg/prepare_data.py` | Verifies the download, writes the missingness profile and the split, optional resized-image cache |
| `src/mamg/masking.py` | One field-masking code path for training dropout and the inference sweep (deterministic, keyed per image) |
| `src/mamg/models.py` | ImageOnly, LateFusion, ImputeFusion (OB3 imputation baseline), GatedCrossAttention (ours); all return `(logits, aux)` |
| `src/mamg/engine.py` | Shared pipeline building, seeded sampler, masked prediction |
| `src/mamg/train.py` | Focal loss, minority-pair oversampling, modality dropout, AMP, cosine LR, early stopping, per-epoch checkpoint, exact `--resume`, gate log, runs.csv |
| `src/mamg/metrics.py` | Macro-F1, per-class recall/F1, balanced accuracy, malignant recall (2 definitions); no plain accuracy |
| `src/mamg/stats.py` | Patient-clustered bootstrap CIs, paired bootstrap model comparison, sweep summary |
| `src/mamg/evaluate.py` | Missingness sweep 0/10/30/50% x 5 mask seeds on val; `--final-test` once per checkpoint (lock file) |
| `src/mamg/xai_eval.py` | Grad-CAM++, deletion/insertion AUC with random-order baseline (val split), params/MACs/CPU-latency table |
| `tests/` (10 test files, 72 tests) | Split leakage, preprocessing, masking, gate-zero, shapes, model behaviour, training smoke + exact resume, metrics vs sklearn, bootstrap, final-test lock, XAI, repository guards |
| `tests/fixtures/make_synthetic.py` | SYNTHETIC look-alike dataset (about 120 patients) for tests; outputs auto-routed to `tmp/` |
| `kaggle/` | `kernel-metadata.json` (placeholders), 4-cell `runner.ipynb`, `README.md` with the push/status/output loop |
| `docs/WALKTHROUGH.md` | Plain-English tour of every module, the reason behind each choice, commands, five failure modes, explain-it-back checklists |
| `docs/CLAIMS_GUARDRAIL.md` | Wording to avoid and safer alternatives |
| `results/missingness_profile.csv`, `results/missingness_by_class.csv` | Measured on the REAL raw metadata (descriptive, not model results) |
| `splits/split_seed42.csv`, `splits/split_summary_seed42.csv` | The committed patient-level split |
| `requirements.txt` / `requirements-lock.txt` | Pinned dependencies (torch left to Kaggle's build) / exact local environment |

## Test results

`.venv/Scripts/python -m pytest -q`: **72 passed** (about 1 minute on CPU), on the synthetic fixture plus the real metadata. The real-data tests check the published counts (2,298 / 1,373 / 1,641), split ratios within 2%, every class in every split, zero patient/lesion/image overlap, and that no observed boolean encodes to 0. The gate-zero test passes: with g = 0 the gated model's logits equal a real ImageOnly model's to within 1e-6 (the difference is exactly 0). The test calibrates BatchNorm first and was mutation-checked: three wrong fusion formulas are all caught. A real-data pipeline check (3 training steps with pretrained weights, plus a 1-batch evaluation sweep) also ran end to end on CPU; its output stayed in the gitignored `tmp/pipeline_check/`.

## Real data: obtained

The files came from Mendeley (CC BY 4.0) and all SHA-256 hashes match the API: `metadata.csv` and `imgs_part_1..3.zip`, extracted to 2,298 PNGs. Both the raw data in `data/raw/` and a 256-px resized cache in `data/processed/` (223 MB) are gitignored.

Measured per-field missingness (`results/missingness_profile.csv`). "Missing" means not recorded. The answer "UNK" (patient did not know) is reported separately, because it is a real category:

| Field(s) | Not recorded | Answered "UNK" |
|---|---|---|
| smoke, drink, pesticide, gender, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, diameter_1, diameter_2 | 35.0% each (the same 804 rows) | 0 |
| background_father / background_mother | 35.6% / 35.8% | 4.1% / 3.7% |
| grew / changed | 0% | 17.5% / 17.2% |
| hurt, itch, bleed, elevation | 0% | 0.4% / 0.3% / 0.3% / 0.1% |
| age, region | 0% | 0 |

Complete-metadata fraction by class: BCC, MEL and SCC 100%; ACK 39%, NEV 31%, SEK 20%. In the test split, 102 of 345 images (29.6%) have the block missing, and all of them are benign.

Split (seed 42), train / val / test: 1,589 / 364 / 345 images (69.1 / 15.8 / 15.0%), 960 / 214 / 199 patients. **MEL is only 39 / 6 / 7 images**, so MEL recall will have very wide CIs.

## NEEDS HUMAN REVIEW (full reasoning in DECISIONS.md)

1. The missingness = benign confound and which subset is primary (above).
2. The 250 lesion ids shared across patients are merged into one group.
3. `biopsed` is excluded from the features (it is a label proxy: all cancers are biopsied).
4. The answer `UNK` is a real category with its own embedding, not missing. This reverses the evening's first choice after the leak hunt: UNK leans malignant, while unrecorded leans benign.
5. Modality dropout is applied identically to all three metadata models (fair comparison). This differs from the playbook line "never during training of the evaluated model"; inference masks are separate, keyed and fixed-rate.
6. Inference masking hides `floor(r * n_observed + 0.5)` of each image's own observed fields (`mask_pool: observed`, the default after the leak hunt). The alternative, `mask_pool: all` (exactly `floor(r*21 + 0.5)` of all fields), hid about twice as much real information on malignant rows. Effective missingness is reported either way.
7. "Minority-pair augmentation" is implemented as oversampling intact (image, metadata) pairs up to 50% of the largest class (max 4x), with focal-loss class weights computed on the oversampled distribution.
8. Bootstrap CIs resample patients, not images.
9. The XAI sweep runs on the validation split only (invariant 5).

## Leak hunt (WP10)

An Opus subagent at max effort reviewed `data.py`, `masking.py`, `models.py` and the tests without seeing any of my reasoning, and tried to break invariants 1-5. It found **no actual train/val/test leak** in the shipped pipeline. It instrumented a run and confirmed that the preprocessor is fit on train rows only, class weights come from train labels, the vocabulary is train-only, there is zero patient/lesion/image/group overlap for split seeds 42, 0, 1, 2 and 3, and no near-duplicate images exist across patients. Everything it did find was fixed, tested and committed:

| Severity | Finding | Fix |
|---|---|---|
| Major | `UNK` and "not recorded" shared index 0 but point in opposite label directions | UNK is now a real category |
| Major | `allow_test="False"` passed the test-split guard (truthiness check) | strict `allow_test is True` |
| Major | `mask_pool: all` hid ~2x more real fields on malignant rows, so the sweep axis was confounded with class | default `mask_pool: observed` (hide a fraction of each image's observed fields) |
| Minor | gate-zero test was weak (uncalibrated BatchNorm made logits nearly image-independent; wrong fusion formulas passed) | BN calibration, image-sensitivity check, g=0.5 formula check; mutation-checked |
| Minor | overlap check used bare `assert` (disabled by `python -O`), silently skipped missing id columns, mixed index labels and positions | explicit `LeakError`, error on missing columns, positional indices |
| Minor | '2' vs '2.0', case variants and overflowing numbers (age '1e40' gave inf, then NaN at g=0) | canonical spellings; non-finite numbers count as missing |

It also found **label noise**: 15 byte-identical image pairs (same patient, different lesion ids, 12 with conflicting diagnoses), and 226 lesion ids with more than one diagnosis. None of them cross splits. Before this, a docs-writing subagent's review led to: per-class recall CIs, no subset filtering at evaluation, strict split loading, a config hash that ignores eval-only settings, and glob checkpoints. Details are in DECISIONS.md.

## Known risks and limitations

- **Gate shortcut:** the gate sees the missingness ratio, which separates the block-missing (all benign) rows perfectly (ratio >= 0.3 exactly for those rows). The gate value is therefore not a pure "trust in metadata" signal. Training-time modality dropout reproduces the 13-field pattern with probability ~6e-5 per sample, so it does not break the shortcut. Read `gate_log.csv` together with the complete-metadata results. Consider an ablation that trains on `complete_metadata` only.
- **Label noise** in PAD-UFES-20 (see the leak-hunt section): cite it as a dataset limitation.
- **Tiny MEL counts** (6-7 images in val/test): do not draw conclusions about MEL from point estimates.
- **The torch version is not pinned** for Kaggle (deliberate). Exact local versions are in `requirements-lock.txt`.
- **Kaggle cell 4's checkpoint glob** has not been run on Kaggle yet; check it on the first run.
- **Local machine:** torch is CPU-only, because disk was tight (8 GB free after the data). The GTX 1650 (4 GB) is usable for debugging after installing the CUDA wheel (see DECISIONS.md).
- **Checkpoints use pickle loading** (`weights_only=False`). Only load checkpoints this project produced.

## Commands for tomorrow

```bash
# 0. environment (Windows, from the repo root)
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt -e .
.venv/Scripts/python -m pytest -q

# 1. data (already done on Aamir's laptop; repeat on any new machine)
#    download metadata.csv + imgs_part_1..3.zip from https://data.mendeley.com/datasets/zr7vgbcyr2/1
#    into data/raw/, extract all PNGs to data/raw/images/, then:
.venv/Scripts/python -m mamg.prepare_data --config configs/default.yaml --cache-images

# 2. the split is committed (splits/split_seed42.csv); prepare_data only re-checks it.
#    Never regenerate it with --force-split once results exist.

# 3. first real training on Kaggle: follow kaggle/README.md
#    (push repo to GitHub, upload metadata.csv + data/processed as private dataset pad-ufes-20,
#     fill placeholders, then)
kaggle kernels push -p kaggle/
kaggle kernels status <kaggle-username>/mamg-train
kaggle kernels output <kaggle-username>/mamg-train -p kaggle_output/

# 4. evaluation sweep on validation (development), e.g. locally with a downloaded checkpoint
.venv/Scripts/python -m mamg.evaluate --config configs/default.yaml --ckpt "checkpoints/gated_s1_*/best.pt"

# 5. ONLY at the very end, once per final checkpoint
.venv/Scripts/python -m mamg.evaluate --config configs/default.yaml --ckpt "checkpoints/gated_s1_*/best.pt" --final-test

# 6. XAI and efficiency
.venv/Scripts/python -m mamg.xai_eval --config configs/default.yaml --ckpt "checkpoints/gated_s1_*/best.pt"
.venv/Scripts/python -m mamg.xai_eval --config configs/default.yaml --efficiency
```

The first thing to check after the first Kaggle run is `results/gate_log.csv`. Look for a gate that is neither stuck at 0 nor at 1, and for `val_gate_mean_all_masked` lower than `val_gate_mean`. Also check `wall_clock_s` in `results/runs.csv` against the 20-minute-per-run estimate.

## Suggested split of work

- **Aamir (data / tabular):** decide the missingness-confound protocol with the guide (item 1) and document it in DECISIONS.md. Check the shared-lesion-id question. Upload the Kaggle dataset (resized cache) and push the repo to GitHub. Own `data.py`, `prepare_data.py`, `masking.py`, and `test_split.py` / `test_preprocessing.py` / `test_masking.py`. Make the field-completeness and class-balance figures from the two committed CSVs.
- **Chirag (model / training):** run the first gated and image-only Kaggle runs, then all 4 models x 3 seeds. Watch `gate_log.csv` for collapse. Run the ablations "gated without modality dropout" and "train_subset=complete_metadata". Own `models.py`, `train.py`, `engine.py`, `kaggle/`, and `test_gate_zero.py` / `test_train_smoke.py`. Rehearse the micrograd gate derivation from the playbook.
- **Giridhar (evaluation / XAI):** run the validation sweeps as checkpoints arrive. Build the macro-F1-versus-missingness figure (both subsets, with CIs) and the paired gated-vs-late and gated-vs-impute comparisons (`stats.paired_bootstrap_diff` on the predictions CSVs). Run the XAI sweep and the efficiency table. Own `evaluate.py`, `stats.py`, `xai_eval.py`, `metrics.py` and their tests. Keep `docs/CLAIMS_GUARDRAIL.md` enforced in every slide.
