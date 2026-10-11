# Running training on Kaggle

This folder lets us train on a free Kaggle GPU (a T4 with 16 GB) from the command line and bring the results back through Git. It follows "The Kaggle bridge" in the execution playbook.

The idea in one paragraph: the Kaggle notebook (`runner.ipynb`) is deliberately thin. It clones the GitHub repo, installs it, and runs `python -m mamg.train` and `python -m mamg.evaluate`. All logic lives in the repo, so you can always answer "which code produced this number?" from a git SHA. Every row of `results/runs.csv` records that SHA.

| File | Purpose |
| --- | --- |
| `kaggle/kernel-metadata.json` | Tells Kaggle the kernel id, the GPU, internet, and the dataset to mount |
| `kaggle/runner.ipynb` | The thin notebook: 1 markdown cell + 4 code cells (5 cells in total) |
| `configs/kaggle.yaml` | Inherits `default.yaml`; only changes `paths` and `runtime` (see below) |

`configs/kaggle.yaml` changes only where files live and which device is used. `paths` and `runtime` are excluded from the config hash, so a run on Kaggle has the same hash as the same experiment on a laptop or the HPC lab. The committed `splits/split_seed42.csv` is reused, so Kaggle runs use exactly the same patient-level split as everyone else.

---

## One-time setup

### 1. The repo must be on GitHub first

The runner does `git clone` of your repo, so the code must be pushed before the first run. Push your latest commit before every kernel run too, otherwise Kaggle trains on old code and `git_sha` in `runs.csv` will not match what you think you ran.

- Public repo: nothing more to do.
- Private repo: make a fine-grained GitHub token with read-only access to this one repository, store it as a Kaggle secret named `GITHUB_TOKEN` (Kaggle notebook page, Add-ons, Secrets; attach it to the notebook after the first push and check it is still attached after later pushes), and change cell 1 of `runner.ipynb` to:

  ```python
  from kaggle_secrets import UserSecretsClient
  tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
  !git clone -q https://{tok}@github.com/<github-org>/<repo-name>.git /kaggle/working/repo
  ```

  Never write the token into the repo or the notebook file. If you do not want the extra setup, making the repo public is simpler.

### 2. Install the Kaggle CLI and add your API token

```bash
pip install kaggle
# On kaggle.com: Settings > API > Create New Token. This downloads kaggle.json.
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle --version
```

`kaggle.json` is a password for your Kaggle account. **Never commit it, paste it in chat, or put it inside the repo.** It is listed in `.gitignore` as a safety net, but keep it only in `~/.kaggle/`. (On Windows the folder is `C:\Users\<you>\.kaggle\`; `chmod` is not needed there.)

### 3. Upload PAD-UFES-20 once as a PRIVATE Kaggle dataset

The dataset is called `pad-ufes-20`. Its folder layout must be `metadata.csv` plus an `images/` folder of PNGs. Upload the RESIZED image cache, not the raw images: run `python -m mamg.prepare_data --cache-images` first (writes `data/processed/`, short side 256 px, about 223 MB instead of 3.6 GB). Use the same cache for every run, local or Kaggle, so preprocessing is identical. Build the folder outside the repo:

```bash
mkdir -p ~/pad-ufes-20-upload
cp data/raw/metadata.csv ~/pad-ufes-20-upload/
cp -r data/processed ~/pad-ufes-20-upload/images

kaggle datasets init -p ~/pad-ufes-20-upload
# Edit ~/pad-ufes-20-upload/dataset-metadata.json:
#   "title": "pad-ufes-20"
#   "id":    "<kaggle-username>/pad-ufes-20"
#   keep the licence that the Mendeley Data page of the dataset states

kaggle datasets create -p ~/pad-ufes-20-upload --dir-mode zip
kaggle datasets files <kaggle-username>/pad-ufes-20      # check metadata.csv and the images are listed
```

Notes:

- `--dir-mode zip` is needed because by default the CLI skips sub-folders. Kaggle unpacks the archive for you. Datasets made through the CLI are private by default; leave them private.
- The code searches `image_dir` recursively for the PNG file names, so it does not matter whether the images end up in `images/` or in a deeper folder.
- Kaggle mounts datasets under `/kaggle/input/`. On the first run, add `!ls /kaggle/input` to a scratch notebook to confirm the exact path. If it differs from `/kaggle/input/pad-ufes-20`, fix `metadata_csv` and `image_dir` in `configs/kaggle.yaml`.

### 4. Fill in the placeholders

There are exactly three places. Replace the values with your own:

```bash
sed -i 's#<kaggle-username>#YOUR_KAGGLE_USERNAME#g' kaggle/kernel-metadata.json
sed -i 's#<github-org>/<repo-name>#YOUR_GITHUB_ORG/YOUR_REPO#' kaggle/runner.ipynb
grep -n "<" kaggle/kernel-metadata.json kaggle/runner.ipynb      # should print nothing relevant now
```

Commit these edits and push to GitHub. (Only your Kaggle username goes in the files, never your token.)

---

## The loop (every run)

```bash
git push                                                  # the kernel clones GitHub, not your laptop
kaggle kernels push -p kaggle/                            # starts the run
kaggle kernels status <kaggle-username>/mamg-train        # repeat until it says "complete"
kaggle kernels output <kaggle-username>/mamg-train -p kaggle_output/
```

The output lands in `kaggle_output/`. Because the notebook works in `/kaggle/working/repo`, the files you want are under `kaggle_output/repo/`:

```
kaggle_output/repo/results/runs.csv          one row per train / eval run (seed, config hash, git SHA, metrics)
kaggle_output/repo/results/gate_log.csv      per-epoch gate statistics (gated model only)
kaggle_output/repo/results/history/          per-epoch training curves, one CSV per run
kaggle_output/repo/results/eval/             sweep, summary and prediction CSVs from mamg.evaluate
kaggle_output/repo/checkpoints/              best.pt and last.pt per run (large; NOT for git)
```

### Bring the results back into the repo

`runs.csv` and `gate_log.csv` are append-only logs shared by everyone, so **append the new rows, never overwrite the file**. The per-run files in `history/` and `eval/` have unique names, so they can simply be copied.

```bash
mkdir -p results/history results/eval
cp -n kaggle_output/repo/results/history/* results/history/     # -n: never overwrite an existing file
cp -n kaggle_output/repo/results/eval/* results/eval/

# Merge the two append-only logs (keeps old rows, adds new ones, drops exact duplicates).
python - <<'EOF'
import pandas as pd, os
for name in ("runs.csv", "gate_log.csv"):
    new = pd.read_csv(f"kaggle_output/repo/results/{name}")
    old = pd.read_csv(f"results/{name}") if os.path.exists(f"results/{name}") else new.iloc[0:0]
    pd.concat([old, new], ignore_index=True).drop_duplicates().to_csv(f"results/{name}", index=False)
    print(name, len(old), "->", len(pd.read_csv(f"results/{name}")), "rows")
EOF

git add results/ && git commit -m "gated model, seed 1 (Kaggle run)"
```

Things to keep out of git:

- Checkpoints (`*.pt` is already in `.gitignore`). Keep the ones you need (for example in your own storage) because the XAI and final-test steps need them.
- `kaggle_output/` as a whole. Add `kaggle_output/` to `.gitignore` if it is not listed yet.
- `kaggle.json`, any GitHub token, and anything from `tmp/synthetic_runs/`.

If you ran `mamg.evaluate --final-test` on Kaggle, also copy back `results/final_test_lock.csv` and commit it. The lock file is what stops the test split being evaluated twice. A lock that exists only inside a Kaggle session protects nothing. The safest habit is to run `--final-test` once, locally or on the HPC lab, from a repo that already has the committed lock file.

---

## Changing the model or seed

Edit cell 4 of `kaggle/runner.ipynb`:

```
!python -m mamg.train --config configs/kaggle.yaml --model late --seed 2
!python -m mamg.evaluate --config configs/kaggle.yaml --ckpt "checkpoints/late_s2_*/best.pt"
```

- `--model` is one of `image`, `late`, `impute`, `gated`.
- The checkpoint folder is named `{model}_s{seed}_{config-hash-8}`, which is why cell 4 uses a `*` wildcard for the hash.
- Other settings can be changed without editing YAML, for example `--set train.epochs=20` or `--set data.train_subset=complete_metadata`. Anything under `train`, `model`, `data` and so on changes the config hash and therefore the run name. Changing `paths`, `runtime` or the seed does not.
- `mamg.evaluate` takes the experiment settings from the checkpoint and only paths and device from the config you pass, so `--set` on the evaluate line cannot change the experiment.
- Do not put `--final-test` in the runner by habit. It is a once-only command.

### All 4 models x 3 seeds in one kernel

Replace cell 4 with a bash cell. Keep it as a loop and nothing else:

```bash
%%bash
set -e
for model in image late impute gated; do
  for seed in 1 2 3; do
    python -m mamg.train --config configs/kaggle.yaml --model $model --seed $seed
    python -m mamg.evaluate --config configs/kaggle.yaml \
      --ckpt "checkpoints/${model}_s${seed}_*/best.pt"
  done
done
```

`set -e` stops at the first failing command, so a crash is not buried under later successes. Twelve runs at about 20 minutes each is roughly 4 hours, which fits inside one Kaggle session. To make the download smaller you can delete the optimizer-carrying `last.pt` files at the end of the loop (`rm checkpoints/*/last.pt`), but then you can no longer `--resume` those runs.

---

## When a Kaggle session times out

- A Kaggle GPU session lasts about 12 hours at most, and `/kaggle/working` is saved as kernel output only when the session completes. If the session is killed part-way, assume everything written in that session is lost, including checkpoints.
- So for long sweeps, **split the work into several kernels** (for example one model x 3 seeds per kernel, about 1 hour each) instead of one 10-hour kernel. Duplicate the kernel folder or edit cell 4 and push again with a different `id`/`title` if you want them running side by side.
- `--resume` exists in `mamg.train` (it restores weights, optimizer, scheduler and random states, and it refuses to resume if the config hash changed), but it needs `checkpoints/<run_name>/last.pt` to be present. A fresh Kaggle session starts from a fresh `git clone`, which has no checkpoints. To resume across sessions you would have to add the previous kernel as an input (`"kernel_sources": ["<kaggle-username>/mamg-train"]`), copy `last.pt` from `/kaggle/input/` into `checkpoints/<run_name>/`, and then run `train --resume`. In practice, short kernels are simpler than resuming.

---

## GPU budget

- The playbook estimates 20-30 seconds per epoch on a T4, so about 20 minutes for a 40-epoch run (early stopping with patience 8 can end a run sooner). **Check this on the first real run** by reading `wall_clock_s` in `results/runs.csv` and the per-epoch seconds in the log. Our training set uses minority oversampling, so an epoch is somewhat longer than 1,589 images.
- 4 models x 3 seeds is about 4 GPU-hours. The missingness sweep in `mamg.evaluate` is inference only and takes minutes. The Grad-CAM++ faithfulness sweep (`mamg.xai_eval`) is the expensive part at roughly 1-2 hours. With failed runs and tuning, the whole programme is about 20-30 GPU-hours. Kaggle gives roughly 30 GPU-hours per week (check your current quota).
- `mamg.xai_eval --ckpt ...` needs a checkpoint. Checkpoints are not in git, so run the XAI sweep in the same kernel that trained the model (add a line after the evaluate line), or download the checkpoint and run it somewhere else.
- Use Kaggle as the daily driver. Use the HPC lab (booked 01 Oct - 15 Nov, 120 hours) for the final confirmation runs whose numbers go in the report, so the report can state that institutional compute was used. On the HPC machine the same commands work with `configs/default.yaml` and path overrides, for example `--set paths.metadata_csv=/path/to/metadata.csv --set paths.image_dir=/path/to/images`.

## First-run checklist

1. `.venv/Scripts/python -m pytest -q` passes locally and the latest commit is pushed to GitHub.
2. Placeholders replaced (username, GitHub org and repo).
3. In the first kernel log, confirm `device=cuda`, no `FileNotFoundError` for images, and one `epoch N:` line per epoch.
4. If `kaggle kernels push` rejects the metadata file, run `kaggle kernels init -p /tmp/probe` and compare the generated `kernel-metadata.json` with ours (the CLI template writes true/false as strings).
5. After the run: merge results as described above, then open `results/gate_log.csv` and check that the gate has not collapsed (see `docs/WALKTHROUGH.md`).
