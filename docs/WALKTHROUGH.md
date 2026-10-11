# Team walkthrough: how the code works and why

Audience: the three of us (Aamir, Chirag, Giridhar), third-year B.Tech level. Goal: each person can explain every module out loud, without the screen, because the panel will examine us and not the repo.

This page describes the code as it is today. Where it disagrees with an earlier brief or with the playbook, there is a note marked **Difference**. "DECISIONS.md" items are cited by their short name, for example (DECISIONS: biopsed excluded).

Contents

1. The project in one minute
2. The most important data finding
3. The gated model as a picture
4. Module by module (`src/mamg`) and the tests
5. How to run each command
6. The five most likely ways this project could go wrong
7. Explain-it-back checklists

---

## 1. The project in one minute

- Task: classify a smartphone photo of a skin lesion into 6 classes (ACK, BCC, MEL, NEV, SCC, SEK) using PAD-UFES-20: 2,298 images, 1,373 patients. BCC, MEL and SCC are malignant; ACK, NEV and SEK are benign.
- Each image comes with a patient questionnaire (age, smoking, region of the body, "did it grow?" and so on). Real questionnaires are often incomplete, so the model must cope when metadata is missing.
- Idea: a small learned number, the **gate** `g` between 0 and 1, decides per patient how much the metadata branch may change the image-based decision. If metadata is missing, we hope `g` goes down and the model behaves like an image-only model.
- The evidence we need to produce: a sweep that masks 0%, 10%, 30% and 50% of the metadata fields at inference time and reports macro-F1 and per-class recall with confidence intervals, for four models: image-only, late fusion, imputation baseline and gated.

Fixed rules (from `CLAUDE.md`, "hard invariants"): grouped patient-level split; missing is never the same as "no"; `gate = 0` must give the image-only logits; preprocessing statistics fit on train only; test split used once; every run logged in `runs.csv`; headline metrics are macro-F1 and per-class recall; no magic numbers outside `configs/default.yaml`; nothing synthetic ever counts as a result.

---

## 2. The most important data finding: missingness predicts the label

**What we found** (`results/missingness_by_class.csv`, `results/missingness_profile.csv`, DECISIONS: "Missingness is a label shortcut"):

- A block of 13 fields is missing together in **804 of 2,298 rows (35%)**: smoke, drink, background_father, background_mother, pesticide, gender, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, diameter_1, diameter_2.
- **All 804 of those rows are benign**: ACK 447, NEV 169, SEK 188.
- Every BCC, MEL and SCC row has the full questionnaire. In the class table, `frac_complete_metadata` is 0.97 for BCC, 0.98 for MEL and 0.96 for SCC, but only 0.39 for ACK, 0.25 for NEV and 0.19 for SEK.

**Why this matters for a missingness-aware model.** Our whole point is that the model sees and uses the missingness pattern (embedding index 0, missing flags, the missing ratio fed to the gate). In this dataset, "the questionnaire block is missing" already tells you "benign". A model can therefore:

1. get a good clean score partly by reading the missingness pattern instead of the skin;
2. then, when we mask fields at inference to simulate missing data, see a malignant case suddenly look like the benign pattern, which can push malignant cases towards benign predictions (the most dangerous error);
3. make the gate look clever when it is simply reacting to the shortcut. Note that the gate's own input includes the missing ratio, which is exactly the shortcut feature.

So "the missingness-aware model does better" can be true for the wrong reason. This is also a fair thing for the panel to ask about, and we should raise it first.

**What the repo does about it (no choice has been made for us; team decision pending):**

- `results/missingness_by_class.csv` documents the pattern.
- `data.py` marks each row `complete_metadata` (at most `complete_max_missing_fields` = 2 missing fields). `evaluate.py` reports every metric on two subsets, `all` and `complete_metadata` (`eval.subsets`). On the complete subset the masking sweep is a controlled experiment: the only missingness is the one we put in.
- `data.train_subset: complete_metadata` (config) trains and validates only on those rows. Default is `all`.
- Recommendation in DECISIONS: use the complete-metadata subset as the primary setting for the missingness sweep, and say so in the paper.

**Things to be careful about when we decide:**

- The complete subset has 1,445 images and a very different class mix: about 71% malignant (about BCC 822, SCC 184, MEL 51) and few benign (about ACK 282, NEV 62, SEK 44, over all splits; computed from the `frac_complete_metadata` column). Per-class results for NEV, SEK and MEL will be noisy.
- **Note:** `data.train_subset` filters train and val only while training (model selection). Evaluation and XAI always load the full split and report the `all` and `complete_metadata` subsets separately.
- Whichever setting we pick, report both subsets, say which is primary, and describe the confound in the same paragraph as the headline result.

---

## 3. The gated model as a picture

```
 IMAGE BRANCH (shared by all four models: class ImageEncoder)

   image 224x224x3
        |
   EfficientNet-B0 (timm, pretrained)
        |
   feature map 7 x 7 x 1280
        |
   1x1 conv  (1280 -> 256)
        |
   49 tokens, each 256-d   -----------------------------+----> keys and values
        |                                               |
   mean over the 49 tokens                              |
        |                                               |
   image_global (256-d) ------------------+             |
                                          |             |
 METADATA BRANCH (class TabularEncoder)   |             |
                                          |             |
   18 categorical fields -> 8-d embedding each          |
   (index 0 = "missing", learned vector)                |
   3 continuous fields   -> z-score value + missing flag|
        |                                               |
   concatenate -> 3-layer MLP with GELU                 |
        |                                               |
   meta_emb (256-d) --> query (1 token) --> 4-head cross-attention
        |                                         |
        |                                    attn_out (256-d)
        |                                         |
        +--> [meta_emb, missing_ratio] -> Linear(257 -> 1) -> sigmoid -> gate g in (0,1)
                                                  |
                                      fused = image_global + g * attn_out
                                                  |
                                      Dropout(0.2) -> Linear(256 -> 6) -> logits
```

Key points to say out loud:

- The metadata does not get its own classifier. It only **asks questions of the image tokens** (cross-attention: "given what I know about this patient, which image regions matter?") and the answer `attn_out` is added to the image vector, scaled by `g`.
- Because of the residual form `image_global + g * attn_out`, setting `g = 0` gives exactly the image-only model. `tests/test_gate_zero.py` checks this on real model classes, within 1e-6.
- `missing_ratio` is the fraction of the 21 fields (18 categorical + 3 continuous) that are missing, natural or masked. It is an extra input to the gate so the gate can react directly to how much of the questionnaire is empty.
- Parameter cost: about 0.43 M more parameters and about 0.007 GMACs on top of EfficientNet-B0's 0.414 GMACs (DECISIONS: "FLOPs are reported as multiply-accumulates").

---

## 4. Module by module

### `src/mamg/__init__.py`
Only the version string.

### `config.py` (owner: Aamir for data settings; Chirag for model settings)

**What it does.** Loads YAML configs, applies command-line overrides, hashes the experiment settings, and decides where outputs go.

**Key functions**
- `load_config(path, overrides)`: reads a YAML file. If it contains `_base_: other.yaml`, that file is loaded first and our file is **deep-merged** on top, so `configs/kaggle.yaml` only needs the keys that change. `--set train.lr=1e-4` style overrides are applied last (`set_by_dotted`).
- `config_hash(cfg)`: 12-hex-digit SHA-256 of the whole config **except** `paths`, `runtime`, `seed` and the evaluation-only sections `eval`, `xai`, `efficiency` (`HASH_EXCLUDED_SECTIONS`).
- `is_synthetic(cfg)` and `output_dirs(cfg)`: if the data is synthetic (declared in config or a `SYNTHETIC_DATA_MARKER` file next to the metadata), results, checkpoints and splits go to `tmp/synthetic_runs/` and the label is `SYNTHETIC`; otherwise they go to `results/`, `checkpoints/`, `splits/` and the label is `REAL`.

**Why**
- *Every tunable number lives in YAML* (invariant 8), so a result can be tied to a config.
- *The hash ignores paths, runtime, seed and evaluation-only settings* (DECISIONS: "Config hash excludes..." and "Review fixes"), so changing an evaluation setting never renames a training run or blocks `--resume`: the same experiment gives the same hash on Kaggle, a laptop or the HPC lab, and the three seeds of one experiment share a hash. The run name `{model}_s{seed}_{hash8}` carries the seed.
- *Synthetic output cannot reach `results/`* (invariant 9) because every script asks `output_dirs` where to write; nobody has to remember.

### `utils.py`

**What it does.** Small shared helpers: seeding, device choice, git info, and the `runs.csv` ledger.

**Key functions**
- `set_seed(seed)`: seeds Python, NumPy and torch (CPU and CUDA) and turns on deterministic cuDNN. Determinism is worth more to us than speed.
- `worker_init_fn`: gives every data-loader worker its own seed.
- `get_device(pref)`: `auto`, `cpu` or `cuda`.
- `git_info()`: short SHA and whether tracked files are modified.
- `append_run(runs_csv, **row)`: appends one row with a fixed column list (`RUNS_COLUMNS`: timestamp, run type, data kind, model, seed, config hash, git SHA, git dirty flag, run name, checkpoint, split, wall clock, metrics as JSON). It refuses unknown columns.
- `Timer`: context manager for wall-clock time.

**Why.** Invariant 6 says every run must be reproducible: seed + config hash + git SHA identify exactly what produced a number. `git_dirty = True` is a warning that the code differed from the commit.

### `data.py` (owner: Aamir)

The largest module. Four jobs: read and clean the metadata, split it without leakage, encode it as numbers, and serve images.

**Reading and grouping**
- `load_metadata(cfg)`: reads the CSV **as strings** with no automatic NA guessing (so we decide what "missing" means), checks required columns, applies value aliases (BRASIL to BRAZIL), checks labels and duplicate image ids, adds the integer `label`, a `group_id`, `n_missing_fields` and the `complete_metadata` flag.
- `assign_groups(...)`: union-find (connected components) over patient ids and lesion ids. If a lesion id appears under two patient ids, they are merged into one group. PAD-UFES-20 has 250 such lesion ids (DECISIONS: "lesion_id is not globally unique"), giving 1,132 groups.

**Why.** Images of the same patient (and of the same lesion) look alike. If they land on both sides of the split the model memorises people, not diagnoses, and every number is inflated (invariant 1).

**Splitting**
- `patient_level_split(...)`: cuts the groups into 20 folds with sklearn's `StratifiedGroupKFold` (keeps groups whole and class mix close to global), then gives whole folds to train/val/test as 14/3/3, which is 70/15/15 by image count. It calls `assert_no_overlap` before returning.
- `assert_no_overlap(...)`: asserts no patient, lesion or image id is in two splits, and no row is assigned twice. The message starts with `LEAK`.
- `split_to_frame`, `split_summary`, `split_path`: write and describe the split file `splits/split_seed42.csv` (committed, so everyone and every machine uses the same split).
- `load_split(..., allow_test=False)`: reads rows of one split **from the file**, re-runs `assert_no_overlap` every time, checks the file covers every metadata row, and raises `PermissionError` for the test split unless `allow_test=True`.

**Why.** The split algorithm went through three versions (DECISIONS: "Split algorithm (revised)"); simple greedy versions put only 4 MEL images in val or skewed the class mix. Seed-42 result: 1589 / 364 / 345 images, MEL 39 / 6 / 7. Re-checking overlap on every load means a hand-edited split file cannot silently leak.

**Encoding tabular data: `TabularPreprocessor`**
- Categorical fields: observed values become indices 1..K (sorted, so deterministic). Missing, "UNK", or a value never seen in train becomes **0**.
- Continuous fields (age, diameter_1, diameter_2): z-score with train mean/std. Missing becomes 0 (the mean, after scaling) **plus** a separate missing flag.
- `fit()` uses train rows only. It also stores the train mode of each categorical field (for the imputation baseline).
- `state_dict()` / `from_state_dict()`: saved inside checkpoints so evaluation uses the same encoding.

**Why**
- *Index 0 is reserved* (invariant 2): if "missing" shared an index with "False", the model could not tell "no smoking" from "did not answer".
- *"UNK" counts as missing, not as a category* (DECISIONS: "Literal UNK treated as missing", marked NEEDS HUMAN REVIEW).
- *Fit on train only* (invariant 4): means, vocabularies and modes computed on val/test would leak information about them into training.
- *`biopsed` is not a feature* (DECISIONS: "biopsed excluded"): every BCC/MEL/SCC is biopsy-proven, so it is almost the label, and a biopsy result is not known at triage.
- *`fitspatrick` is categorical* (only 6 levels).

**Other helpers**
- `class_weights(labels, n)`: inverse-frequency weights, normalised to mean 1, fit on train labels.
- `minority_oversample(...)`: repeats whole (image, metadata) pairs of rare classes up to 50% of the largest class, at most 4x. No synthetic rows (no SMOTE). Each copy gets different random image augmentation (DECISIONS: "Minority-pair augmentation").
- `missingness_profile`, `missingness_by_class`: produce the two CSVs behind section 2.
- `build_image_index`, `build_transforms`: find PNGs recursively (a resized cache folder can shadow the raw images); train transforms are random (crop, flips, rotation, colour jitter); val/test transforms are deterministic (resize, centre crop, normalise).
- `stable_key(img_id)`: CRC32 of the file name, used to seed masks so they do not depend on batch order.
- `PadUfesDataset`: returns image, cat, cont, cont_missing, label, key and index per sample. **It does not mask.** Masking is done on whole batches in `masking.py`, so training and evaluation use the same code.

### `prepare_data.py` (owner: Aamir)

**What it does.** Checks the download (expected 2,298 rows, 1,373 patients, 1,641 lesion ids), counts how many images are found, writes `missingness_profile.csv` and `missingness_by_class.csv`, writes the split file (it will not overwrite an existing one without `--force-split`), writes the split summary, and optionally builds a resized image cache (`--cache-images`).

**Why.** A one-command sanity check that anyone can run first on a new machine. If the counts differ from the published ones it prints a warning.

### `masking.py` (owner: Chirag; shared with Giridhar for the sweep)

**What it does.** The **only** place where metadata fields are hidden. A masked field is made to look exactly like a naturally missing one: categorical to index 0; continuous to 0.0 with missing flag 1.

**Key functions**
- `n_masked_fields(rate, n_fields)`: round-half-up, so 50% of 21 fields is 11, 10% is 2, 30% is 6.
- `keyed_field_mask(keys, n_fields, rate, seed, pool)`: **inference-time** mask. Each sample gets exactly that many masked fields, drawn from a random generator seeded with `(mask_seed, key)`. So the mask belongs to the image, not to its position in a batch.
- `modality_dropout_mask(...)`: **training-time** random mask. Per sample: with probability `p_full_mask` (0.1) hide everything; otherwise with probability `p_apply` (0.5) hide a random share (rate drawn from 0.1, 0.3, 0.5, 0.7); otherwise keep all.
- `apply_field_mask(...)`: returns masked **copies**; inputs are never modified.
- `observed_fields`, `missingness_ratio`: which fields have values now, and the fraction missing (the gate's extra input).

**Why**
- *One code path* (CLAUDE.md conventions): if training dropout and the evaluation sweep were written separately, "masked" could mean two different things.
- *Keyed masks*: the same image gets the same mask no matter how the batch is built; results do not change when the batch size changes.
- *Pool `all`* (DECISIONS: "Inference masking definition"): fields are drawn from all 21, so an already-missing field can be drawn. Therefore evaluation also records `effective_missing_ratio`. The alternative `mask_pool: observed` exists and is tested (NEEDS HUMAN REVIEW).
- *Modality dropout in training* teaches the model to cope with missing metadata; it is applied identically to late, impute and gated so the comparison is fair (DECISIONS: "Modality dropout is applied to all three metadata models").

### `models.py` (owner: Chirag)

**What it does.** Defines the four models; every `forward` returns `(logits, aux)`.

**Classes**
- `ImageEncoder`: timm EfficientNet-B0 without its classifier, then a 1x1 conv to 256 channels, giving 49 tokens (`tokens`) and their mean (`image_global`). Shared by all models.
- `TabularEncoder`: per-field embeddings (8-d) + continuous values (+ flags when `indicators=True`) into a 3-layer GELU MLP. With `indicators=False` it replaces missing categoricals by the train mode and drops the flags, so it cannot tell "missing" from "typical".
- `ImageOnly`: `image_global -> head`.
- `LateFusion`: `[image_global, meta_emb] -> head`, missingness-aware.
- `ImputeFusion(LateFusion)`: same, but with imputation and no indicators. This is the baseline promised by synopsis objective OB3 (DECISIONS: "ImputeFusion = LateFusion architecture with mean/mode imputation").
- `GatedCrossAttention`: our model, see section 3. It also returns `gate`, `attn_weights` and `missing_ratio` in `aux`, and accepts `gate_override` for the unit test.
- `build_model(name, cfg, ...)`, `gradcam_target_layer(...)`: factory, and the layer (`bn2`) Grad-CAM++ hooks into.

**Why**
- *One shared `ImageEncoder`* (DECISIONS: "Image-global vector = mean of the 49 projected tokens"): the image path is identical code in every model, so differences between models come from the metadata path only. It also makes `fused = image_global + g * attn_out` dimensionally exact.
- *Index 0 has a learnable embedding, not `padding_idx`* (DECISIONS): the model can learn what "missing" means for each field.
- *Scalar gate, 4 heads* (playbook "keeping the viva winnable"): simple enough to explain and to plot.
- *Gate bias starts at 0* so `g` starts at 0.5, away from both extremes.
- *Same head and width for late/impute/gated image path*: fair comparison.

### `engine.py` (owner: Chirag; Giridhar reuses it)

**What it does.** Plumbing shared by train, evaluate and XAI.

**Key pieces**
- `build_pipeline(cfg, need, allow_test, pre_state, train_aug, apply_train_subset)`: loads metadata and split, applies `train_subset` to train and val during training (evaluation and XAI pass `apply_train_subset=False`), fits the preprocessor on train (or restores it from a checkpoint's state), builds datasets and the image index.
- `ensure_split(cfg, df)`: returns the split file. For real data it refuses to continue if the committed file is missing (run `prepare_data`); only synthetic data gets a split created on the fly.
- `EpochIndexSampler`: a seeded shuffle that depends on `(seed, epoch)`. This is what makes `--resume` give exactly the same result as an uninterrupted run.
- `make_loader`, `to_device`: loader and tensor movement.
- `inference_mask(...)`: calls `keyed_field_mask` for a batch.
- `predict(model, loader, device, rate, mask_seed, pool, ...)`: runs the model; if `rate > 0` it masks a **copy** of each batch. Returns true labels, probabilities, predictions, gate values, missing ratios and sample indices.

**Why.** Training, evaluation and XAI must preprocess data identically. Copying batches before masking means the dataset itself is never altered, so one sweep point cannot contaminate the next.

**Gotcha.** Paths are relative to the repo root. If you run from another directory the split file is not found and real-data runs stop with a `FileNotFoundError` (deliberately, instead of silently making a new split). The Kaggle runner does `%cd` into the clone for this reason.

### `train.py` (owner: Chirag)

**What it does.** Trains one model for one seed.

**Key pieces**
- `FocalLoss`: class-weighted focal loss, normalised by the sum of the weights, so `gamma = 0` equals weighted cross-entropy exactly (tested).
- `lr_lambda_factory`: 1 epoch of warm-up, then cosine decay to 1% of the learning rate.
- `train(cfg, model_name, seed, resume)`: builds the pipeline (never the test split), oversamples minority pairs, computes class weights from the *effective* training distribution, trains with AdamW, mixed precision on CUDA only, gradient clipping, modality dropout for metadata models, validates every epoch, early-stops on validation macro-F1 (patience 8), saves `last.pt` every epoch and `best.pt` on improvement, writes `results/history/<run>.csv`, and appends one row to `runs.csv` at the end.
- For the gated model, also writes a row per epoch to `results/gate_log.csv` and prints `WARNING ... possible gate collapse` when the validation gate std is below 1e-3 or the mean is outside (0.02, 0.98).
- `--resume` restores model, optimizer, scheduler, scaler and RNG states and refuses to continue if the config hash changed.

**Why**
- *Focal loss + inverse-frequency weights + oversampling* all fight class imbalance; weights come from the post-oversampling distribution so the two corrections do not double-count (DECISIONS: "Minority-pair augmentation"; NEEDS HUMAN REVIEW).
- *Early stopping on macro-F1, not loss or accuracy*: macro-F1 treats rare classes as equally important.
- *`train.py` cannot reach the test split*: it only asks for `("train", "val")`.
- *Checkpoint every epoch*: Kaggle sessions can die; an exact resume is tested in `test_train_smoke.py`.
- *`weights_only=False` when loading* (DECISIONS: "Checkpoints are loaded with weights_only=False"): checkpoints contain RNG states and the preprocessor; only load files this project produced.

### `metrics.py` (owner: Giridhar)

**What it does.** `compute_metrics(y_true, y_pred, classes, malignant)` returns per-class recall and F1, macro-F1, balanced accuracy and two malignant summaries: `malignant_recall_macro` (average recall over BCC, MEL, SCC) and `malignant_sensitivity` (share of malignant images predicted as *any* malignant class).

**Why.** Plain accuracy is deliberately not computed (invariant 7): with 845 BCC images and 52 MEL images, ignoring MEL still scores well. A class with no true images scores recall 0 and F1 0 (same as sklearn with `zero_division=0`), which is tested against sklearn. Two malignant numbers are kept because "BCC mistaken for SCC" is a wrong subtype but not a missed cancer (DECISIONS: "Malignant recall reported two ways").

### `stats.py` (owner: Giridhar)

**What it does.** Bootstrap confidence intervals and paired comparisons.

**Key functions**
- `bootstrap_metrics(...)`: resamples *patients* (not images) with replacement 1,000 times, recomputes metrics from scratch each time, returns `(point, low, high)` percentile intervals.
- `paired_bootstrap_diff(...)`: metric(A) minus metric(B) on the same resampled patients, with CI and a one-sided `p_le_0`.
- `summarise_sweep(tidy)`: mean and std over mask seeds per (model, split, subset, missingness, metric).

**Why.** Images of one patient are correlated, so resampling images would give intervals that are too narrow (DECISIONS: "Bootstrap CIs are patient-clustered"; NEEDS HUMAN REVIEW; `eval.bootstrap_group: null` switches to per-image). Paired differences are more powerful because both models see the same resamples.

### `evaluate.py` (owner: Giridhar)

**What it does.** Loads a checkpoint and runs the missingness sweep: for each rate in 0, 10, 30, 50% and each of 5 mask seeds (rate 0 only once), mask a copy of the batches, predict, and compute metrics on each subset (`all`, `complete_metadata`) with bootstrap CIs.

**Outputs** (in `results/eval/`): `<run>_<split>_sweep.csv` (tidy, with CIs), `_summary.csv` (mean/std over mask seeds), `_predictions.csv` (per-image probabilities for paired comparisons). It also appends to `runs.csv`.

**Key pieces**
- `load_checkpoint_cfg`: experiment settings come **from the checkpoint**, only paths and runtime from the current config. A Kaggle-trained checkpoint can therefore be evaluated on a laptop without config drift (DECISIONS: "Evaluate takes experiment settings from the checkpoint").
- `claim_final_test` and `--final-test`: writes a row to `results/final_test_lock.csv` (checkpoint SHA-256 and run name) **before** evaluating, and refuses a second run for the same checkpoint. A crashed attempt still counts. Re-running needs a deliberate, visible edit of the lock file.
- Refuses to evaluate a checkpoint trained on synthetic data as real data, and the other way round.

**Why.** Invariant 5. The test set decides the numbers in the paper; each time we peek and adjust, the test number stops being an honest estimate.

**Note.** Confidence intervals are computed for the four headline keys (macro-F1, balanced accuracy, malignant recall macro, malignant sensitivity) and for every per-class recall. F1 per class, gate mean and effective missing ratio have empty `ci_low`/`ci_high`. Expect a very wide CI for MEL recall (6-7 images).

### `xai_eval.py` (owner: Giridhar)

**What it does.** Two separate jobs.

1. **Faithfulness sweep** (`--ckpt`): for the first `xai.n_images` (100) validation images, at each missingness rate, compute a Grad-CAM++ saliency map for the predicted class, then measure two curves. *Deletion*: remove pixels from most to least salient, step by step, and watch the predicted class probability fall; lower area under the curve (AUC) is better. *Insertion*: start from a blurred image and reveal pixels in saliency order; higher AUC is better. A random pixel order gives a baseline for both. Writes `results/xai/<run>_faithfulness.csv` and `_curves.csv`.
2. **Efficiency table** (`--efficiency`): parameters, GMACs (fvcore, thop as fallback) and CPU latency (batch 1, 1 thread, median and p90) for all four models into `results/efficiency.csv`. Architecture only, no pretrained weights needed.

**Key functions.** `perturbation_curve`, `curve_auc`, `saliency_order`, `faithfulness`, `FixedMetadata` (lets Grad-CAM, which expects image-only input, work with a model that also takes metadata: the metadata is held fixed while the image changes), `make_baselines`, `xai_sweep`, `count_flops`, `efficiency_table`.

**Why**
- Deletion/insertion follows RISE (Petsiuk et al., 2018). These are **empirical measurements**, not proofs, and the docstring says so; see `docs/CLAIMS_GUARDRAIL.md`.
- *Deletion baseline 0 in normalised space (the ImageNet channel mean); insertion starts from a Gaussian blur* (DECISIONS: "Deletion baseline"). *Target class = the model's prediction at that missingness rate.*
- *Validation split only* (DECISIONS: "XAI sweep runs on the validation split only"; NEEDS HUMAN REVIEW) because invariant 5 reserves the test split.
- Efficiency is reported as parameters, MACs and latency instead of claiming "edge-deployable".

**Note.** The 100 images are a seeded random sample of the validation split (`xai.sample_seed`), the same images for every model. It is not class-balanced, so MEL may be represented by only a handful of images.

### Tests (`tests/`)

Run with `pytest -q`. They use a small **synthetic** fixture (random tiny images, made-up tables) and the real metadata where present (tests marked `real_data` are skipped if `data/raw/metadata.csv` is absent).

| File | What it checks | Invariant / decision |
| --- | --- | --- |
| `conftest.py` | Shared fixtures: base config, synthetic data folder, synthetic config (outputs forced into a temp folder) | 9 |
| `fixtures/make_synthetic.py` | Builds the SYNTHETIC look-alike, with a 35% benign-only missing block, UNK values and shared lesion ids; writes `SYNTHETIC_DATA_MARKER` | 9 |
| `test_split.py` | Zero overlap on patient, lesion, image and group ids; ratios within tolerance; every class present in every split; determinism; shared lesion ids stay together; `assert_no_overlap` really catches a deliberate leak; test split needs `allow_test`; a tampered split file is detected on load; real-data split has 2,298 rows, 1,373 patients, 1,641 lesions | 1, 5 |
| `test_preprocessing.py` | Missing/UNK map to 0 and never equal "False"; vocabularies and scaler fit on train only; unseen category to 0; state round-trips; class weights and oversampling counts; on real data no observed value maps to 0 and `biopsed` is not a feature | 2, 4 |
| `test_masking.py` | Exact number of masked fields per rate; same seed same mask, independent of batch order and size; masks spread evenly over fields; masked values look like natural missing; inputs not modified; `observed` pool only masks present fields; modality dropout proportions | single masking path |
| `test_masking_behaviour.py` | With all metadata masked the gated model still gives finite logits and a gate in (0,1); the gate reacts to the missing ratio; the imputation baseline cannot tell missing from mode; late fusion can tell missing from "no" | model design |
| `test_gate_zero.py` | With `g = 0` the gated logits equal a real `ImageOnly` model's logits (weights copied, 1e-6); with `g = 0` changing all metadata changes nothing; with `g = 1` metadata does matter (guards against a vacuous pass) | 3 |
| `test_shapes.py` | 49 tokens of 256-d from a 7x7x1280 map; all four models give `(4, 6)` logits; attention weights sum to 1; gradients reach every branch | architecture |
| `test_train_smoke.py` | Focal loss at gamma 0 equals weighted cross-entropy; LR schedule shape; a run interrupted after epoch 0 and resumed ends with the same weights as an uninterrupted run; synthetic output lands in the temp folder; `runs.csv` and `gate_log.csv` written; other models train | 6, 9 |
| `test_evaluate.py` | Metrics agree with sklearn; random predictions score near chance; bootstrap deterministic and contains the point estimate; paired difference of identical models is 0; sweep CSV is tidy; effective missingness rises with rate; a second `--final-test` is refused | 5, 7 |
| `test_xai.py` | Deletion curve decreases and insertion curve increases on a toy model; a correct saliency beats random order and a wrong one loses to it; AUC maths; Grad-CAM++ works on the real gated model; efficiency table filled; end-to-end XAI sweep on synthetic data | faithfulness maths |
| `test_invariants.py` | Repository-level guards: `train.py` and `xai_eval.py` never ask for the test split; synthetic outputs are routed away from `results/`; committed CSVs under `results/` contain no synthetic rows | 5, 8, 9 |

**Difference.** `CLAUDE.md` says the gate-zero test "proves" invariant 3. Safer wording (see `docs/CLAIMS_GUARDRAIL.md`): the test verifies that the *implementation* has the residual property. It says nothing about whether the trained gate learns something useful.

---

## 5. How to run each command

Run from the repo root. On Windows use `.venv/Scripts/python`; `python` below means that interpreter. On Kaggle or the HPC lab use the matching config (see `kaggle/README.md`).

| Goal | Command | Notes |
| --- | --- | --- |
| Run the tests | `pytest -q` | Must pass before every merge. 63 tests passed in about 76 s on the overnight laptop (it trains tiny models on CPU). |
| Verify data, profile missingness, write the split | `python -m mamg.prepare_data --config configs/default.yaml` | Writes `results/missingness_*.csv`. Does not overwrite an existing split file; `--force-split` does, which invalidates old checkpoints' comparisons, so avoid it. `--cache-images` builds a resized cache. |
| Train image-only | `python -m mamg.train --config configs/default.yaml --model image --seed 1` | Outputs in `checkpoints/image_s1_<hash>/` and `results/history/`. |
| Train late fusion | `... --model late --seed 1` | |
| Train imputation baseline | `... --model impute --seed 1` | |
| Train gated model | `... --model gated --seed 1` | Also writes `results/gate_log.csv`. Watch for the collapse warning. |
| Resume a stopped run | add `--resume` | Needs `checkpoints/<run>/last.pt` and the same config. If the run already finished, it does nothing. |
| Validation sweep (development) | `python -m mamg.evaluate --config configs/default.yaml --ckpt checkpoints/<run>/best.pt` | Val split, rates 0/10/30/50%, 5 mask seeds, both subsets. Optional `--missingness 0,10,30,50`. |
| Final test, once | `python -m mamg.evaluate --config configs/default.yaml --ckpt checkpoints/<run>/best.pt --final-test` | Only after all decisions are frozen. Refuses a second run for the same checkpoint. Commit `results/final_test_lock.csv` afterwards. |
| Grad-CAM++ faithfulness | `python -m mamg.xai_eval --config configs/default.yaml --ckpt checkpoints/<run>/best.pt` | Val split. Slow (1-2 GPU-hours for the full programme). |
| Efficiency table | `python -m mamg.xai_eval --config configs/default.yaml --efficiency` | Writes `results/efficiency.csv`. |
| Override any setting | add `--set train.epochs=2 --set model.pretrained=false` | Values are parsed as YAML. Changing anything outside `paths`, `runtime`, `seed`, `eval`, `xai`, `efficiency` changes the config hash. |
| Train on complete rows only | `--set data.train_subset=complete_metadata` | See section 2. Changes the hash, so the run name differs. |
| Fast pipeline check | `--set runtime.max_train_steps=2 --set train.epochs=1 --set model.pretrained=false` | Quick smoke run; only for checking the plumbing. On real data it still writes into `results/`, so do not leave such rows around. |

Where things end up: `results/runs.csv` (one row per run), `results/gate_log.csv`, `results/history/`, `results/eval/`, `results/xai/`, `results/efficiency.csv`, `results/final_test_lock.csv`; `checkpoints/` is not in git.

---

## 6. The five most likely ways this project could go wrong

### 6.1 Gate collapse

- **Symptom.** The gate is the same number for every patient (std near 0), or sits at about 0 or 1. It does not drop when metadata is removed. The central claim ("the gate adapts") then has no evidence.
- **Cause.** The metadata branch is ignored (gate near 0) or always trusted (near 1), or the gate learns a constant bias because that is the easiest way to lower the loss. Also possible: the gate only reads `missing_ratio`, which is correlated with the label (section 2), not the content.
- **Check in this repo.** Open `results/gate_log.csv` after the **first** gated run, not in week four. Columns: `train_gate_mean/std`, `val_gate_mean/std` and **`val_gate_mean_all_masked`** (the validation gate with every metadata field masked). A healthy pattern is `val_gate_mean_all_masked` clearly *lower* than `val_gate_mean`, and a gate std well above 0. `train.py` prints `WARNING ... possible gate collapse` when std < 1e-3 or the mean is outside (0.02, 0.98). The eval sweep also stores `gate_mean` per missingness rate. `test_masking_behaviour.py` shows the gate *can* respond to the missing ratio; it does not show that it *does* after training.

### 6.2 Patient leakage

- **Symptom.** Macro-F1 above about 0.90. Nobody in this literature reports that on PAD-UFES-20.
- **Cause.** The same patient, or the same lesion (including lesions registered under two patient ids), has images in both train and val/test. A random image-level split causes exactly this.
- **Check in this repo.** `assert_no_overlap` in `data.py` runs at split time, on every `load_split`, and in `tests/test_split.py` (including a test that plants a leak and expects it to be caught, and a test with a tampered split file). Groups are built with union-find over patient and lesion ids, so the 250 shared lesion ids cannot straddle splits. The split file is committed; never regenerate it casually. Also check that preprocessing is fit on train only (`test_preprocessing.py`).

### 6.3 Masking applied inconsistently between training and test

- **Symptom.** The missingness curve is suspiciously flat, or it falls off a cliff at the first masking rate; two models show gaps that move when you change batch size.
- **Cause.** Training-time masking and evaluation-time masking implemented in different places or with different meanings of "masked" (for example one writes 0 and the other writes index 0 plus a flag), masks that depend on batch order, or evaluation masks that change the stored dataset.
- **Check in this repo.** Everything goes through `masking.py`. Training uses random `modality_dropout_mask` (random fields, random rate, 10% fully masked); evaluation uses `keyed_field_mask` (a fixed number of fields per rate, seeded by `(mask_seed, crc32(img_id))`, applied to a copy of each batch). `test_masking.py` checks exact counts, determinism and batch independence. Report the *effective* missingness too, since masking can hit fields that were already missing. **Difference from the playbook:** the playbook guardrail says masking happens "never during training of the evaluated model". Our code deliberately uses random modality dropout during training (DECISIONS: modality dropout for all three metadata models) and keeps *evaluation* masking on a copy; the clean-data score is evaluated with no masking. A "gated without modality dropout" ablation is recommended.

### 6.4 Class imbalance hiding malignant misses

- **Symptom.** A good-looking headline number, but melanomas or carcinomas are predicted as benign.
- **Cause.** Rare classes (MEL 52 images in total; 39 / 6 / 7 in train / val / test) contribute little to accuracy. Missingness masking can also push malignant cases towards the benign-looking pattern (section 2).
- **Check in this repo.** Use macro-F1, per-class recall (`recall_<class>` columns), `malignant_recall_macro` and **`malignant_sensitivity`**; plain accuracy is not computed. Class weights, focal loss and minority oversampling in training. Read the MEL numbers with their counts: with 6-7 images, one image moves recall by 14-17 points, so the confidence intervals will be very wide. Look at confusion between malignant and benign classes, not only the average. Compare `all` with `complete_metadata`.

### 6.5 Over-claiming

- **Symptom.** A slide or paper sentence says more than a table supports ("robust", "guarantee", "state-of-the-art", "deployable").
- **Cause.** Writing from memory or from the synopsis wording; presenting a synthetic-run number; ignoring the missingness/benign confound.
- **Check in this repo.** `docs/CLAIMS_GUARDRAIL.md` lists wording to avoid and safer alternatives. Every number must cite a committed CSV in `results/` with a run name, and nothing from `tmp/synthetic_runs/` is a result.

---

## 7. Explain-it-back checklists

Rule from the playbook: whoever's module Claude wrote that day explains it to the other two, out loud, without the screen. If you cannot, you read it until you can, before merging.

### Aamir (split, preprocessing, missingness profile)

- [ ] Why a patient-grouped split? What goes wrong with a random image-level split?
- [ ] What is the union-find step for, and what is odd about `lesion_id` in this dataset (250 shared ids)?
- [ ] How does `StratifiedGroupKFold` give a 70/15/15 split, and why 20 folds? What are the final counts, and why does MEL have only 6 and 7 images in val and test?
- [ ] Where is the leakage assertion called (three places) and what does the test with a planted leak show?
- [ ] Why is "missing" index 0 and not a normal category? What would break if "UNK" were a category?
- [ ] Why is every statistic (vocabulary, mean, std, mode, class weights) fit on train only?
- [ ] Why is `biopsed` not a feature?
- [ ] Explain the 804-row finding with the numbers from `missingness_by_class.csv`, why it matters for a missingness-aware model, and what `complete_metadata` and `data.train_subset` do.
- [ ] What does `prepare_data.py` write, and what happens if the split file already exists?

### Chirag (models, gate, training)

- [ ] Walk through the gated forward pass from pixels to logits using the diagram, including every tensor shape.
- [ ] Why `image_global + g * attn_out`, and what does `test_gate_zero.py` check and not check?
- [ ] Why a scalar gate, why is its input `[meta_emb, missing_ratio]`, and what is the risk of feeding it the missing ratio (section 2)?
- [ ] How does the imputation baseline differ from late fusion, and why is that the right comparison?
- [ ] Why does one `ImageEncoder` serve all four models?
- [ ] What are modality dropout's three cases (full, partial, none) and their probabilities? Why is it applied to all metadata models?
- [ ] Focal loss, class weights and oversampling: what does each do, and why are the weights computed after oversampling?
- [ ] How does `--resume` give an identical run, and when does it refuse?
- [ ] What does `gate_log.csv` contain and which column would show collapse?
- [ ] Why does `train.py` never touch the test split?

### Giridhar (evaluation, bootstrap, XAI)

- [ ] Why macro-F1 and per-class recall instead of accuracy? What is the difference between `malignant_recall_macro` and `malignant_sensitivity`?
- [ ] How is an inference-time mask built (number of fields, which fields, which seed), and why are rate-0 results computed only once?
- [ ] What does "effective missingness" mean and why do we report it?
- [ ] Why `all` and `complete_metadata` subsets? What does each answer?
- [ ] Why is the bootstrap clustered by patient? How would a per-image bootstrap differ?
- [ ] What does the paired bootstrap difference tell you that two separate CIs do not?
- [ ] How does the final-test lock work, and what exactly happens if someone runs `--final-test` twice?
- [ ] Explain deletion and insertion AUC in one minute: what is plotted, what is "good", why is a random-order baseline needed, why are these measurements and not proofs?
- [ ] What does `FixedMetadata` do and why is it needed for Grad-CAM++?
- [ ] What do the efficiency numbers measure and what do they not measure (not a phone, architecture only, batch 1, one thread)?
