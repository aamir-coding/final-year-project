# Building This Project With Claude: Team 26PB67 Execution Playbook

Oct 11, 2026 · @Chiru

## The decision in one page

Your whole experimental programme needs roughly 20–30 GPU-hours. You requested 120 from the HPC lab and Kaggle gives about 30 a week free. Compute is not your constraint — correct code and disciplined evaluation are. That changes the whole plan: the fastest path is one well-structured repo that Claude Code writes and tests, with GPU runs fired off as short batch jobs.

| Question you asked | Answer |
| --- | --- |
| Claude Chat or Claude Code? | **Claude Code**, in one Git repo, for everything that produces code, figures, the report or the paper. Chat for learning concepts, reading papers, and rehearsing the viva |
| Which model where? | **Opus** for anything whose error costs a GPU re-run or a wrong number in the paper. **Sonnet** for the other 70% of the work. **Haiku** rarely — the savings at this codebase size are small |
| What runs on Claude? | All code, all tests, all figures, all writing, and — with the Kaggle CLI wired in — launching and collecting GPU runs |
| What needs a GPU elsewhere? | Only the training, the XAI sweep and multi-seed repeats. Kaggle first, HPC lab as overflow and as the story you already told the college |
| How fast can this finish? | **About 7 focused working days.** Finishing before the 22–24 Oct interim review is realistic |
| Biggest risk | Not compute. It is patient-level leakage, a gate that collapses to a constant, and three people who cannot explain code Claude wrote |

### Why 20–30 GPU-hours is enough

About 1,600 training images after a patient-level split, EfficientNet-B0 at 224×224, batch 32 with mixed precision: roughly 20–30 seconds an epoch on a T4, so about 20 minutes for a 40-epoch run. Four architectures × 3 seeds is around 4 GPU-hours. The missingness ablation is inference only and costs minutes. The Grad-CAM++ deletion and insertion sweep is the expensive part at 1–2 hours. Double it all for failed runs and tuning and you are still well under 30.

Say this number out loud at the interim review. "Our full experiment suite is about 25 GPU-hours" is the sentence that tells a panel you understand your own project's scale.

### The one change to make today

Your four-week AI/ML roadmap spends about 51 hours building SaansCast, an air-quality forecaster. That is more hours than this entire project needs, spent on a different project. Keep the roadmap's **Learn** column exactly as it is and re-point its **Build** column at PAD-UFES-20. The next section does that remapping day by day.

## Resolve first: your roadmap is building the wrong capstone

The roadmap is well built, and its teaching order is right. The problem is only its capstone. SaansCast is tabular time-series forecasting with API ingestion and a Streamlit deployment; your project is multimodal image-plus-tabular classification with attention fusion and XAI. They share about four skills and 51 hours of your time.

| Roadmap skill | Transfers to your project? |
| --- | --- |
| pandas, numpy, EDA, missing-value handling | Yes — PAD-UFES-20's metadata CSV is exactly this, and the missing-value decisions *are* your project |
| Leakage, time splits, cross-validation discipline | Yes, with one change: your leakage axis is patient, not time |
| Metrics: precision, recall, thresholds, class imbalance | Yes — macro-F1 and malignant recall are the same machinery |
| micrograd, backprop by hand, PyTorch training loops | Yes, completely. This is the single most valuable week |
| Transfer learning with a pretrained CNN | Yes — that tutorial *is* your EfficientNet-B0 branch |
| Attention and transformers | Yes — your cross-attention is query, key, value |
| API ingestion, rate limits, caching | No. Your data is one Mendeley download |
| Time-series lags, rolling windows, persistence baselines | No |
| GRU and sequence models | No |
| Streamlit deployment | Partly — useful as the final demo artefact, not as an air-quality app |

### The remapping

Same videos, same readings, same daily rhythm. The build block points at your repo instead.

| Roadmap day | Keep the Learn block | Replace the Build block with |
| --- | --- | --- |
| Week 1 (pandas, EDA, charts) | As written | Load PAD-UFES-20's metadata CSV. Measure the real missingness rate per field — that number goes in your paper. Chart class balance and field completeness |
| Week 2 (classical ML, metrics, leakage) | As written | `patient_level_split()` with the zero-overlap assertion. A leakage test. Macro-F1 and per-class recall implemented and sanity-checked on random predictions |
| Week 3 (micrograd, PyTorch) | As written, all of it | Rebuild your own gate in micrograd on toy data: `out = base + sigmoid(w·x) * extra`. Then the PyTorch training loop your project actually uses |
| Week 4 Mon (CNN, transfer learning) | As written | Run the transfer-learning tutorial, then swap ResNet for EfficientNet-B0 and print the 7×7×1280 feature map shape |
| Week 4 Tue (attention) | As written | Push a (1, 256) query and a (49, 256) key/value through `nn.MultiheadAttention` and print every shape. Verify the attention weights sum to 1 across 49 tokens |
| Week 4 Wed (GRU) | Skip | Grad-CAM++ on one image, by hand, before using a library |
| Week 4 Thu–Sun (deploy, ship) | As written | A small demo app for the December final demo: upload a photo, tick whichever history boxes you like, see the prediction and the gate value |

### Why the gate exercise in Week 3 matters

Writing `out = base + sigmoid(w·x) * extra` in micrograd and watching the gradient flow through it is twenty minutes of work and it is the thing a panel will ask you to explain. You will have derived your own architecture by hand. Nobody who has done that can be rattled by "how does your gate learn?"

### The honest trade-off

Re-pointing the capstone costs you a portfolio project for job applications. If that matters, build SaansCast in November or December *after* this project is finished and submitted. Running both in parallel through October is how both end up mediocre.

## Where each kind of work happens

Claude Code is the default venue. The rule: anything that produces a file belongs in the repo, and anything that needs a GPU belongs in a batch job the repo launches.

| Work | Venue | Why |
| --- | --- | --- |
| Every module: data prep, model, training loop, evaluation, XAI | Claude Code | It writes files, runs tests and commits. Code pasted out of a chat window loses its history |
| Unit tests, including the leakage assertion and the gate-zero proof | Claude Code | These run on CPU in seconds and catch the errors that would otherwise cost you a GPU day |
| CPU smoke runs on random tensors | Claude Code | Verifies shapes and the residual property before a GPU ever starts |
| Training, ablation sweep, XAI sweep, multi-seed repeats | Kaggle, then HPC lab | Needs a GPU. Claude's container has none |
| Downloading PAD-UFES-20 | Kaggle or HPC lab | Claude's container reaches package registries and GitHub, not Mendeley |
| Reading results, making figures, statistics, bootstrap CIs | Claude Code | Results come back as CSV; everything after that is CPU work |
| The report, the paper draft, the final deck | Claude Code | Same repo, versioned alongside the numbers they cite |
| Learning concepts, "explain this paper to me", viva rehearsal | Claude Chat | Conversational, no files produced, and you want it on your phone |
| Literature lookups | Claude Chat with the PubMed connector | Metadata, related-article search and full text without leaving the conversation |

### What Claude's container cannot do

Say these out loud once so nobody waits on them:

- **No GPU.** Not slow — absent. Training must run elsewhere.
- **No dataset download.** Outbound network is allow-listed to package registries and GitHub, so the Mendeley download happens on Kaggle or the HPC machine.
- **No persistent machine.** The cloud container is reclaimed between sessions. Everything that matters lives in Git.

None of this limits the project, because the GPU work is a handful of batch jobs, not an interactive activity.

### Kaggle versus the HPC lab

|  | Kaggle | HPC lab |
| --- | --- | --- |
| Availability | Immediate, about 30 GPU-hours a week free (check your current quota) | Booked 01 Oct – 15 Nov, 120 hours |
| Session length | Around 12 hours, runs unattended | Depends on your lab's policy |
| Scriptable from Claude Code | Yes, via the Kaggle API — see the bridge section | Only if you have SSH; otherwise manual |
| Best for | Everything, day to day | Final multi-seed runs, and the line in your report that says institutional compute was used |

Use Kaggle as the daily driver and the HPC lab for the final confirmation runs. You already told the college you would use the lab, so use it for the runs whose numbers go in the report — that keeps the facility request honest.

Colab free tier is the third option. It disconnects more and gives no scripted job submission, so treat it as a backup when Kaggle's quota runs out mid-week.

## Model strategy

Pick by the cost of being wrong, not by how hard the task feels. Your whole codebase is maybe 1,500 lines; an elaborate three-tier routing scheme costs more in coordination than it saves. Two tiers is the honest answer, with a third used opportunistically.

| Use | Model and effort | Tasks |
| --- | --- | --- |
| Errors cost a GPU re-run or a wrong number in the paper | **Opus, high effort** | The fusion and gate implementation. The patient-level split. The missingness masking protocol. Deletion and insertion AUC. Debugging a training run that looks wrong. Reading a results table and deciding what it means. The methods section |
| Ordinary building, roughly 70% of the work | **Sonnet** | Data loaders, transforms, plotting scripts, the config module, tests, refactors, docstrings, README, converting results CSVs into report tables, the Kaggle kernel wrapper |
| Mechanical, no judgement | **Haiku** | Bulk renames, formatting, bibliography into BibTeX, boilerplate file skeletons. Genuinely marginal at this size — use it if it is in front of you, do not build a workflow around it |

**Max effort** earns its cost in exactly three places: the first implementation of the gated fusion module, the first design of the evaluation protocol, and any session where a result looks too good and you need to find the leak. Everywhere else it is slower without being better.

### What actually saves you time

Model choice is a second-order lever. These are first-order:

- **A good `CLAUDE.md`.** Every session starting with a re-explanation of your architecture is the real tax. Write the invariants once, and every session inherits them.
- **Plan mode before anything structural.** Ask for the plan, read it, then let it run. Catching a wrong approach in a plan costs a minute; catching it in a diff costs an hour.
- **One thread per concern.** A long session that drifts from data prep to XAI to the report carries dead context. Start fresh for a new concern.
- **Subagents for genuinely parallel work.** `evaluate.py` and `xai_eval.py` do not touch each other, so two agents can write them at once. Do not fan out work that shares files.
- **Let it run the tests.** Claude Code fixing its own failing test is faster than you reading the diff and reporting the failure back.

### Three people, one repo

Do not all point Claude Code at the same files. Split by module, matching the ownership already in your plan.

| Person | Owns | Files |
| --- | --- | --- |
| A | Data and tabular branch | `config.py`, `data/`, `features.py`, the split and its leakage test |
| B | Model and training | `models.py`, `train.py`, the Kaggle runner |
| C | Evaluation and XAI | `evaluate.py`, `xai_eval.py`, figures, statistics |

Short-lived branches, merged daily. When two of you need the same file, one of you waits — merge conflicts in Claude-generated code are a bad way to spend an evening. For a change that touches several modules at once, use an isolated worktree so it does not disturb anyone's working copy.

## Day-1 setup

Two hours of setup together, on one call, buys you the whole sprint.

### Repo layout

```text
missingness-gating/
  CLAUDE.md               project invariants, read by every session
  DECISIONS.md            every choice and why; the viva depends on this
  configs/default.yaml    seeds, split ratios, hyperparameters, missingness rates
  src/mamg/
    config.py
    data.py               loading, patient_level_split, TabularPreprocessor
    models.py             ImageOnly, LateFusion, Impute, GatedCrossAttention
    train.py
    evaluate.py           macro-F1, per-class recall, missingness sweep
    xai_eval.py           Grad-CAM++, deletion and insertion AUC
    stats.py              bootstrap CIs
  kaggle/                 kernel-metadata.json and the thin runner
  results/                committed CSVs: the project's memory
  figures/
  tests/
  paper/
```

Committing `results/` is deliberate. Every number in your report then traces to a CSV at a known commit. When the panel asks where a figure came from, you show them.

### CLAUDE.md

The highest-leverage file in the repo. Paste this in and extend it as you learn things.

```markdown
# Missingness-Aware Multimodal Gating (Team 26PB67)

## What this is
Skin-lesion classification on PAD-UFES-20 combining a clinical image
with a partially-filled patient questionnaire. A learned per-patient
gate decides how much the metadata branch influences the prediction.

## Invariants: never violate these
- Splits are grouped by patient_id. Every split asserts zero patient
  overlap. A random image-level split invalidates every result.
- A missing categorical field maps to reserved index 0, never to the
  same value as a genuine "no".
- fused = image_global + gate * attention_out. The residual path means
  gate=0 must recover the image-only representation exactly. There is
  a test for this; it must keep passing.
- The test split is evaluated once, at the end. Tune on validation.
- Every run writes a row to results/runs.csv: seed, config hash,
  git SHA, and all metrics.

## Commands
- pytest -q
- python -m mamg.train --config configs/default.yaml --model gated
- python -m mamg.evaluate --ckpt <path> --missingness 0,10,30,50

## Conventions
- No magic numbers in code; everything comes from the config.
- Seeds fixed through set_seed(). Determinism matters more than speed.
- Metrics are macro-F1 and per-class recall. Never plain accuracy.

## Current state
<update this line every session: what works, what is next>
```

### Connectors worth wiring up

| Connector | What it buys you |
| --- | --- |
| GitHub | The repo itself: Claude Code commits, opens PRs, reads CI |
| Kaggle API (CLI, not a connector) | Claude launches GPU runs and pulls results back without you copy-pasting notebooks |
| PubMed | Paper metadata, related-article search and full text while you write the related-work section |
| Google Drive | Your college submission templates and sharing drafts with your guide |

Skip anything else for now. A connector you check once is a connector you did not need.

### Two project skills worth writing

Skills are folders of instructions Claude loads when relevant. Two pay for themselves here:

- **`run-experiment`** — the full loop: validate the config, commit, push the Kaggle kernel, poll until it finishes, pull `results/` back, append to `runs.csv`, commit. After this exists, "run the gated model at three seeds" is one sentence.
- **`report-section`** — how your college wants the report formatted, plus the rule that every number must cite a committed CSV. Stops the report drifting from the results.

Ask Claude Code to write both on day 1; it can propose them as saved skills.

## The compressed sprint

Starting Monday 12 October, a focused team reaches a complete result set in seven working days, leaving the interim review as a progress report on finished work rather than a promise.

&#91;embedded content: build sprint · 8 workstreams, 1 gate\]

The first three days run in parallel because the modules do not touch each other; everything after day 4 is sequential because it consumes the previous stage's output.

### What must be true at the end of each day

- [ ] **Day 1–2.** `pytest` passes. The split asserts zero patient overlap. The real per-field missingness rate is measured and written down.
- [ ] **Day 2–3.** All four model classes instantiate and run a forward pass on random tensors. The gate-zero test passes exactly.
- [ ] **Day 3.** The evaluation harness scores random predictions correctly — macro-F1 near chance, not near 1.
- [ ] **Day 4–5.** Four models × 3 seeds trained on Kaggle. `gate_log.csv` shows gate values that are neither always 0 nor always 1.
- [ ] **Day 5–6.** The missingness sweep produces the macro-F1-versus-missingness table at 0, 10, 30 and 50 percent.
- [ ] **Day 6–7.** Deletion and insertion AUC computed for gated and late-fusion at all four rates, with a random-order baseline.
- [ ] **Day 7–8.** Bootstrap CIs, all figures, and the parameter/FLOPs/latency table that backs your efficiency claim.
- [ ] **Day 8–10.** Report draft and paper draft, every number citing a committed CSV.

### If a day slips

Cut in this order: the paper draft first (it has until December), then the imputation baseline, then seeds 2 and 3. Never cut the missingness sweep or the leakage check — the sweep is the project, and the check is what makes it true.

## The Kaggle bridge

The Kaggle API turns GPU training into something Claude Code can launch and collect on its own. This is the difference between Claude writing code you then babysit through a notebook, and Claude running the experiment.

&#91;embedded content: the build loop · 3 stages, results return through Git\]

The notebook on Kaggle stays deliberately thin: clone, install, run a module, save outputs. All logic lives in the repo, so nothing important exists only inside a notebook cell.

### Setting it up, once

```bash
pip install kaggle
# put kaggle.json (Account > API > Create New Token) at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# upload PAD-UFES-20 once as a private dataset, then reference it from every kernel
kaggle datasets create -p data/pad-ufes-20
```

`kaggle/kernel-metadata.json` declares the GPU and the dataset:

```json
{
  "id": "<username>/mamg-train",
  "title": "mamg-train",
  "code_file": "runner.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<username>/pad-ufes-20"]
}
```

### The loop, every time

```bash
kaggle kernels push -p kaggle/                 # fire the run
kaggle kernels status <username>/mamg-train    # poll until complete
kaggle kernels output <username>/mamg-train -p results/
git add results/ && git commit -m "gated model, seeds 1-3"
```

Four commands. Wrap them in the `run-experiment` skill and it becomes one sentence to Claude Code. Kernels run unattended for hours, so you can fire a sweep and close the laptop.

### The thin runner

The entire notebook is four cells:

```python
!git clone -q https://github.com/<org>/missingness-gating.git
!pip install -q -r missingness-gating/requirements.txt
%cd missingness-gating
!python -m mamg.train --config configs/default.yaml --model gated --seed 1
```

If you find yourself editing logic inside the notebook, stop and move it into the repo. Notebooks that accumulate logic are how teams lose track of which code produced which number — and that is the question your panel will ask.

## Prompt library

The pattern that works: state the invariant, name the file, name the test that proves it. Vague prompts produce plausible code that fails silently, which is the worst outcome in a project judged on its numbers.

### Setting up

> Read CLAUDE.md. Scaffold the repo exactly as the layout there describes: package under `src/mamg`, a `configs/default.yaml` holding seeds, split ratios, the four missingness rates and all hyperparameters, and a pytest suite. No model code yet — just the skeleton, the config loader, `set_seed()`, and a passing `pytest -q`.

### Data and the split

> Implement `patient_level_split(df, ratios, seed)` in `src/mamg/data.py`, grouping by `patient_id`. It must return train/val/test index arrays and assert zero patient overlap between every pair. Write `tests/test_split.py` covering: no patient appears in two splits, the ratios hold within two percent, and the same seed gives the same split. Then report the real per-field missingness rate in the PAD-UFES-20 metadata CSV as a table — that number goes in the paper.

### The model

> Implement four models in `src/mamg/models.py` with one shared interface returning `(logits, aux)` where `aux` carries the gate value: ImageOnly, LateFusion (concatenation), ImputeFusion (mean/mode imputation then concatenation), and GatedCrossAttention.
>
> For the gated model: EfficientNet-B0 truncated before pooling, 1x1 conv projecting 1280 to 256 channels, giving 49 visual tokens. Per-field categorical embeddings with index 0 reserved for missing; continuous features mean-imputed with an explicit missingness flag. `nn.MultiheadAttention` with 4 heads, metadata embedding as query, visual tokens as key and value. Scalar gate = sigmoid(Linear(\[metadata\_embedding, missingness\_ratio\])). Then `fused = image_global + gate * attn_out`.
>
> Write `tests/test_gate_zero.py` proving that with the gate forced to 0, the fused representation equals the image-only representation to within floating-point tolerance. This test is the paper's central claim — make it exact and make it readable.

### Training

> Write `src/mamg/train.py`: class-weighted focal loss (gamma 2), AMP, cosine schedule, early stopping on validation macro-F1, checkpoint every epoch, and a `--resume` flag. Log the mean and standard deviation of the gate value every epoch to `results/gate_log.csv`. Add training-time modality dropout at the rate in the config. Every run appends one row to `results/runs.csv` with seed, config hash, git SHA and final metrics.

### Evaluation

> Write `src/mamg/evaluate.py`. It loads a checkpoint and evaluates at missingness rates 0, 10, 30 and 50 percent, masking only at inference and never re-training. Report macro-F1 and per-class recall, with malignant classes (BCC, MEL, SCC) also aggregated. Repeat the masking over five mask seeds and report mean and standard deviation. Output one tidy CSV: model, missingness, mask\_seed, metric, value.

### XAI

> Write `src/mamg/xai_eval.py` using `pytorch-grad-cam` for Grad-CAM++ on the last convolutional block. Implement deletion and insertion AUC following the RISE paper: delete pixels in descending importance order in twenty steps, record the softmax probability of the originally predicted class at each step, and integrate. Include a random-order deletion baseline. Run across all four missingness rates for the gated and late-fusion models.

### When a number looks wrong

> Our gated model reports 0.94 macro-F1 at 0% missingness. MetaBlock-SE reports about 0.70 balanced accuracy on this dataset. Before I believe this, find the leak. Check the split for patient overlap, check whether any normalisation statistic is computed over the full dataset, check whether augmentation touches validation, and check whether the test set has been evaluated more than once. Report what you find, do not fix anything yet.

This is the single most valuable prompt in this document. Run it the first time a result pleases you.

### Writing

> Draft the methods section from the actual code in `src/mamg/`, not from the synopsis. If the code and the synopsis disagree, say so explicitly rather than describing what we intended. Every number cites the CSV in `results/` it came from.

### Chat prompts for the learning track

> I just ran `nn.MultiheadAttention` with a (1, 256) query and (49, 256) key/value and got a (1, 256) output. Walk me through what happened to produce that, and ask me three questions afterwards to check I actually followed it.

> Here is my `models.py`. Ask me to explain it function by function, like a panel would. Tell me where my explanation is hand-waving.

## Guardrails

Projects like this fail quietly. Nothing crashes; the numbers just stop meaning anything. Each row below has a check that catches it on the day it happens.

| Failure | What you would see | The check |
| --- | --- | --- |
| Patient leakage across splits | Macro-F1 above 0.90, which nobody in this literature reports | The zero-overlap assertion, run on every split, in CI |
| Gate collapses to a constant | Your central claim has no evidence | `results/gate_log.csv` every epoch from the first training run, not week four |
| Test set used more than once | Scores creep up over the week | Test evaluation lives in one script that refuses to run twice without a flag |
| Normalisation fitted on all data | Small, consistent, unexplained gain | Scalers and class weights fit on training folds only; a test asserts it |
| Masking that leaks | Ablation curve is suspiciously flat | Masking happens at inference on a copy of the batch; never during training of the evaluated model |
| Augmentation on validation | Validation tracks training too closely | Separate transform pipelines; assert validation transforms are deterministic |
| Results that cannot be reproduced | A number in the report nobody can regenerate | Every row in `runs.csv` carries seed, config hash and git SHA |

### Three claims in your submitted documents that need softening

These are already written down and your panel has them. Fix the wording before the interim review rather than defending it live.

| As submitted | The problem | Say instead |
| --- | --- | --- |
| "mathematically stress-tests spatial explanations to guarantee the model remains transparent and truthful" | Deletion and insertion AUC are empirical measurements, not proofs. "Guarantee" invites a hostile question | "empirically stress-tests spatial explanations and measures how faithful they remain" |
| "Fully contained within a 16GB VRAM budget on a single T4 GPU, making it deployable on edge clinic hardware" | A T4 is a datacentre card. Fitting in 16 GB says nothing about running on a phone | "Trains within a single 16 GB T4. Deployability is measured and reported as parameter count, FLOPs and CPU latency" |
| "RG-MF" on the synthesis slide versus "RG-DermNet" in the table (reference \[10\]) | Same paper, two names, on adjacent slides | Pick one name and use it everywhere |

### One commitment in the synopsis you may have missed

Objective OB3 says you will compare "against imputation and late-fusion baselines." That is four models, not three: image-only, late-fusion, **an imputation baseline you run yourself**, and gated. Citing MetaBlock-SE's published number does not satisfy what you wrote.

The good news is that it is cheap. A mean/mode imputation baseline sharing your late-fusion architecture is about thirty lines and one extra training run, and it makes your central comparison far stronger: impute versus gate, same backbone, same split, same folds. Add it on day 3.

## Keeping the viva winnable

The risk of moving this fast is arriving in December with a working project you cannot explain. Your college will examine you, not the repo, and a panel can tell within two questions whether you understand your own gate.

Three habits prevent it, and they cost about fifteen minutes a day.

**Explain it back, same day.** Whoever's module Claude wrote that day explains it to the other two, out loud, without the screen. If you cannot, you do not merge it yet — you read it until you can. This is the whole safeguard; the other two are supporting.

**`DECISIONS.md`, one line per choice.** Focal loss over weighted cross-entropy, 4 heads not 8, a scalar gate not per-channel, 0/10/30/50 and not 0/25/50/75. Date each line. By December you will not remember why, and "why did you choose" is the most common panel question in existence.

**Own your module.** Person A can be asked about the split, B about the gate, C about deletion AUC, and each can answer without looking. Three people who each half-know everything is the failure mode; three people who each fully own a third is a team.

### Where to draw the line with Claude

Your own roadmap already has the right rule: use assistants to explain errors, not to write capstone modules you cannot read. For a project at this pace that is too strict, so here is the workable version.

| Fine | Not fine |
| --- | --- |
| Claude writes the data loader, the training loop, plotting, tests, the Kaggle runner | Anyone merging code they could not walk through line by line |
| Claude drafts the report, you rewrite it in your own voice | Submitting a methods section describing code you have not read |
| Claude finds your leak and explains the mechanism | Pasting generated code to fix other generated code, twice in a row, without understanding either |

**Check your college's policy on AI assistance this week, before you are deep in.** Most Indian universities now have one and most permit assisted coding with disclosure. Ask your guide directly and write the answer in `DECISIONS.md`. Finding out in December is the only version of this that hurts.

### The honest framing, if you are asked

"We used Claude as a pair programmer. The architecture, the evaluation protocol and every design decision are ours and documented in DECISIONS.md. We each own a module and can walk through it." That is a normal answer in 2026, and it is true if you keep the explain-it-back habit.

## Open questions

Answer these in the comments and I will tighten the plan around them.

- [ ] **HPC lab access shape.** Is it SSH to a shared machine, a Slurm queue, or a lab PC you physically sit at? SSH or Slurm means Claude Code can drive it directly; a lab PC means Kaggle does the daily work and the lab does final runs.
- [ ] **Does your college have a stated policy on AI-assisted coding?** Ask your guide this week. It changes nothing about the plan, only about what you disclose.
- [ ] **Has anyone downloaded PAD-UFES-20 yet, and did you measure the real missingness rate?** That single number anchors your motivation section, and the whole project is easier once someone has looked at the CSV.
- [ ] **Did your guide ask for the vaguer objectives on slide 9?** The synopsis version ("develop a flexible model framework") is much weaker than your earlier draft. If the format was imposed, fine — keep the specific versions in the report. If not, the specific ones are better.
- [ ] **Who has a machine that can run training locally?** Any GPU with 6 GB or more makes debugging far faster than waiting on a Kaggle queue, even if final runs happen elsewhere.
- [ ] **Are you still targeting EMBC 2027 or JBHI?** The paper's format differs, and it is cheaper to decide before writing than after.

### What I would do next, if you want to start now

Say the word and I will write the scaffolding prompts into a ready-to-run sequence: repo skeleton, config, split with its leakage test, and the gate-zero test — in the order you would run them on day 1. That is about two hours of your time and it puts a working, tested skeleton in GitHub before the weekend ends.
