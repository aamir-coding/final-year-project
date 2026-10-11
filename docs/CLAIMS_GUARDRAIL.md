# Claims guardrail: what we may and may not write

Team 26PB67. Use this page when writing the report, paper, slides, abstract, poster or viva answers. A panel will read our exact words. Wording that sounds stronger than our evidence is the easiest thing for them to attack, and it is also the easiest thing for us to fix before they see it.

The rule behind every row is the same: **say what we measured, on what data, with what uncertainty.**

Some of these phrases are already in our submitted synopsis or slides: the "guarantee" row, the "edge-deployable" row and the "RG-MF / RG-DermNet" row come from the playbook's "Three claims in your submitted documents that need softening". Fix them before the interim review.

## Wording to avoid and safer alternatives

| Wording to avoid | Why it is a problem | Say this instead |
| --- | --- | --- |
| "guarantee", "guarantees transparency", "guarantees the model remains truthful" | Deletion and insertion AUC are measurements on a finite set of images, not a certificate. The word invites the question "guaranteed how?" | "empirically measures faithfulness (deletion/insertion AUC)" and always give the random-order baseline next to it (`results/xai/`) |
| "edge-deployable", "deployable on edge clinic hardware", "edge-deployable T4" | A T4 is a datacentre GPU. Fitting in 16 GB says nothing about phones or clinic PCs | "trains within a single 16 GB T4; deployability is reported as parameters, MACs and CPU latency" (`results/efficiency.csv`, architecture only, batch 1, 1 thread) |
| "mathematically proves", "proves that the gate...", "proven" | We have tests and experiments, not a proof of generalisation. Only the residual form of the fusion is exact by construction | "the gate-zero unit test verifies the residual property of the implementation" (`tests/test_gate_zero.py`: with g = 0 the logits equal the image-only model within 1e-6) or "results suggest" |
| "robust to missing data" | Vague, and it hides how much performance is lost | "degrades by X macro-F1 points from 0% to 50% inference-time masking (95% CI a to b)", with X, a and b read from `results/eval/*_summary.csv` and the sweep CSV |
| "clinically validated", "ready for clinical use", "diagnostic tool", "can replace a dermatologist" | One public dataset, retrospective, one clinic population, no prospective study, no regulatory work | "evaluated retrospectively on one public dataset (PAD-UFES-20)" |
| "state-of-the-art", "outperforms the literature" | Published numbers use different splits, different subsets and sometimes image-level splits with leakage. They are not comparable to ours | "compared with our own baselines (image-only, late fusion, imputation) under an identical patient-level split". Cite other papers only as context, and say the protocols differ |
| "accuracy of X%" as a headline | Classes are very imbalanced (ACK 730, BCC 845, MEL 52 images). A model that ignores MEL still scores a high accuracy | "macro-F1 and per-class recall", plus malignant recall/sensitivity. `mamg.evaluate` gives patient-clustered 95% CIs for macro-F1, balanced accuracy, malignant recall and malignant sensitivity; per-class recall rows currently have no CI, so compute one with `mamg.stats.bootstrap_metrics` before quoting it. (The code does not compute plain accuracy on purpose.) |
| "the gate learns when to trust metadata", "the gate adapts to missing data" | This is our central claim, so it needs evidence, not a sentence | Write it only if `results/gate_log.csv` shows `val_gate_mean_all_masked` clearly lower than `val_gate_mean`, no collapse warning, and the same on the gate columns of the eval sweep. Otherwise write "the gate value changed by X when all metadata was masked" or drop the claim |
| "our missingness-aware model is better" (with no further comment) | In PAD-UFES-20 the questionnaire is missing in 804 rows and all 804 are benign. Missingness itself predicts the label, so a model can gain by reading missingness rather than skin | Also report the `complete_metadata` subset next to `all`, state the confound in the same paragraph, and say which setting is the primary one. See "Missingness is a label shortcut" in `docs/WALKTHROUGH.md` |
| "significantly better" with no test or CI | "Significant" has a statistical meaning | "the paired bootstrap difference in macro-F1 is d (95% CI a to b)" (`mamg.stats.paired_bootstrap_diff`). If the CI includes 0, say "no clear difference" |
| "detects melanoma", "melanoma sensitivity of X%" | MEL has only 6 validation and 7 test images. One image changes recall by 14 points | Report MEL recall as counts (for example "k of 7") and say the interval is very wide; if you want a CI, compute it with `mamg.stats.bootstrap_metrics` on `recall_MEL`. Pool BCC/MEL/SCC as "malignant" for the headline |
| "RG-MF" in one place and "RG-DermNet" in another for the same reference [10] | One paper with two names looks careless | Pick one name, write it in `DECISIONS.md`, and use it everywhere (report, slides, paper, tables) |
| "tuned on the test set" or any test number from before the final run | Breaks the rule that the test split is used once | Model selection uses the validation split. Test numbers come only from `mamg.evaluate --final-test`, once per checkpoint, and are logged in `results/final_test_lock.csv` |

## Every number must cite a committed CSV in `results/`

- A number in the report, slides or paper must come from a CSV that is **committed in `results/`** (or `figures/` built from one). If you cannot point to the file and the row, the number does not go in.
- Cite the file and the run identity next to the number, for example: "macro-F1 0.xx (95% CI a to b); `results/eval/gated_s1_<hash>_val_summary.csv`; git SHA from `results/runs.csv`".
- Every row in `results/runs.csv` carries the seed, config hash and git SHA, so any number can be regenerated. If the `git_dirty` column is True, treat the row as unreproducible and re-run.
- Report seeds honestly: mean and spread over the seeds you ran (3), not the best seed.
- Numbers typed from memory, from a terminal that scrolled away, or from a notebook output that was never saved to `results/` are not citable.

## Never present anything from `tmp/synthetic_runs/` as a result

- `tmp/synthetic_runs/` holds output from the SYNTHETIC test fixture (random tiny images plus made-up tables). It exists only to check that the code runs. It is gitignored.
- Those files are labelled `SYNTHETIC` in the `data_kind` column and by the output path. They say nothing about skin lesions, models or missingness.
- `results/` and `figures/` must contain real-data outputs only. The code routes synthetic runs away from `results/` automatically (`output_dirs` in `src/mamg/config.py`); do not copy files across by hand.
- If a slide or figure contains numbers from a synthetic run, even as a placeholder, remove it. Placeholders get forgotten.

## Quick checklist before anything leaves the team

1. Does each number have a CSV path and a run name?
2. Does each comparison between models have a confidence interval, and does the text match it?
3. Is the `complete_metadata` subset reported next to `all` wherever metadata helps?
4. Is MEL discussed with its counts (6 val, 7 test)?
5. Have the avoided phrases above been searched for in the document (search for "guarant", "prove", "edge", "state-of-the-art", "robust", "clinical", "accuracy")?
6. Is the dataset and its limits stated: one public dataset, 2,298 smartphone images, 1,373 patients, retrospective?
