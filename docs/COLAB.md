# Working on Google Colab

The project no longer depends on one laptop. Three places hold everything:

| What | Where | Who can change it |
|---|---|---|
| Code, configs, results CSVs, STATE/DECISIONS | GitHub `aamir-coding/final-year-project` (`main`) | anyone with push access, one driver at a time |
| Dataset (metadata + 2,298 resized images) and the 14 trained models | Google Drive folder **MAMG** (owned by Chirag, shared with the team as Editor) | the notebook only; never edit by hand |
| Your Colab session | a temporary VM, wiped when the runtime ends | nothing is kept there |

`colab/MAMG_colab.ipynb` connects the three. `configs/colab.yaml` only changes paths and the device, so Colab runs have the same config hash, run names and checkpoints as runs anywhere else (`tests/test_invariants.py` checks this). **Nothing has to be retrained:** the M3 models are in the Drive folder and all results are in git.

## One-time setup

### Chirag (owner of the Drive folder), once
1. On drive.google.com create a folder named exactly **MAMG** in My Drive.
2. Upload `MAMG_snapshot_2026-11-04.zip` (962 MB, from the Desktop) into it. Do not unzip it on Drive.
3. Share the folder with Aamir and Giridhar as **Editor** (new training runs write checkpoints into it). Never use "Anyone with the link": checkpoints are loaded with full pickle loading.
4. Open the notebook in Colab (see "Opening the notebook") and run cells 1-5 once. Cell 5 unpacks the 14 models into `MAMG/checkpoints/` and verifies them against `results/checkpoints_sha256.txt`.

### Every team member, once
1. Accept the Drive share, then in **Shared with me** right-click **MAMG** → **Organize** → **Add shortcut** → **My Drive**. The notebook expects `/content/drive/MyDrive/MAMG`.
2. To push from Colab you need (a) push access to the repo: it belongs to Aamir (`aamir-coding`), so **only Aamir** can add a missing teammate under the repo's Settings → Collaborators (anyone who has already pushed has access); and (b) a **classic** token. Fine-grained tokens cannot select a repository owned by another person's account, so they do not work here.
   GitHub → profile picture → Settings → Developer settings → Personal access tokens → **Tokens (classic)** → Generate new token (classic). Note: `MAMG Colab`. Expiration: 90 days. Scopes: tick only **`repo`**. This scope can write to all your repositories, so the token goes only into Colab Secrets; revoke it on GitHub if it ever leaks. (No token? Do GPU work on Colab and commit the small result files from a laptop.)
3. In Colab, open the key icon (Secrets) on the left and add `GITHUB_TOKEN` (the token), `GIT_NAME` and `GIT_EMAIL` (as used on GitHub). Turn on "Notebook access" for each. Never paste a token into a cell, a file or a chat.

### Opening the notebook
Colab → File → Open notebook → GitHub tab → `aamir-coding/final-year-project` → `colab/MAMG_colab.ipynb`. Then File → **Save a copy in Drive** so your edits (like the commit message) don't need a push. When the repo's notebook changes, open it again from GitHub.

## Every session
1. Pick the runtime first (Runtime → Change runtime type): **T4 GPU** only for training, evaluation sweeps and XAI; **CPU** for tests, `mamg.compare`, figures and writing.
2. Run cells 1-5 (about 3-5 minutes: mount, fresh clone, install, data to local disk, checkpoint check).
3. Run what your milestone needs from section 6. Uncomment one block at a time.
4. Update `STATE.md` (and `DECISIONS.md` if you decided something) in the Colab file browser, set `COMMIT_MESSAGE`, run cell 7 to pull, commit and push.
5. **Runtime → Disconnect and delete runtime.**

## Staying inside the free limits
Google does not publish exact Colab limits and they change; heavy GPU use on one account can block GPUs for that account for a day or so. The project's whole remaining GPU need is small (rough estimates on a T4: 15-30 minutes per training run, 1-2 hours for the XAI sweep, about 2 minutes per evaluation sweep), so careful use is enough:

**Do**
- Use a GPU runtime only while a GPU command is actually running. Do setup, tests, comparisons and figures on CPU.
- Disconnect and delete the runtime as soon as you finish. An idle connected GPU runtime still counts.
- Batch GPU work: queue the runs you need in one session rather than many short ones.
- Keep checkpoints on Drive (the config already does). On a disconnect, re-run cells 1-5 and the same train command with `--resume`; at most one epoch is lost.
- Spread GPU work across team members' accounts when there is a lot (each account has its own quota), still one driver at a time for pushing.
- Keep the browser tab open during long runs; Colab disconnects idle tabs (often after about 90 minutes without interaction) and caps a session at about 12 hours.
- Pull before you start and push right after you finish.

**Don't**
- Don't leave a T4 runtime connected while reading, writing or idle.
- Don't run the same training twice: check `results/runs.csv` for the run name first.
- Don't run several GPU jobs in parallel on one runtime (a T4 has 16 GB, but the VM has only 2 CPUs; it only slows both).
- Don't read images from Drive during training (slow); cell 4 copies them to the VM disk.
- Don't commit `data/`, checkpoints, `*.pt`, tokens or `kaggle.json`. Cell 7 only stages `results/`, `figures/`, `STATE.md`, `DECISIONS.md`, `docs/`, `paper/`; stage code changes yourself, and run `python -m pytest -q` before pushing code.
- Don't run `--final-test` without the team's agreement, and never twice for one checkpoint (cell 6f).
- Don't delete or overwrite anything in the Drive `MAMG/checkpoints` folder by hand; the final-test lock is keyed by checkpoint SHA-256.
- Don't use paid compute units or Colab Pro without the team agreeing.

## Backups (three independent copies)
1. GitHub: all code, results and docs.
2. Drive `MAMG/`: the snapshot zip (data, models, a copy of the repo at `b857598`) and the unpacked checkpoints.
3. Chirag's Desktop (OneDrive-synced): `MAMG_snapshot_2026-11-04.zip`, verified (all 14 models match `results/checkpoints_sha256.txt`).

When a new important run finishes (e.g. a final model), its checkpoint is already on Drive; add its SHA-256 to `results/checkpoints_sha256.txt` so everyone can verify it.

## If something goes wrong
- **"No /content/drive/MyDrive/MAMG folder"**: add the shortcut (setup step 1), then re-run cell 1.
- **Cell 5 reports a mismatch**: don't use that checkpoint; tell the team. Re-unpack it from the snapshot by deleting only that run's folder on Drive and re-running cell 5.
- **`CUDA` not available**: you are on a CPU runtime; switch to T4 (or do CPU-only work).
- **GPU unavailable / usage limit**: do CPU work, try later or from another team member's account, or use the Kaggle runner (`kaggle/README.md`; Kaggle has its own weekly GPU quota per account).
- **Push rejected**: someone pushed first. Re-run cell 7 (it pulls with `--rebase`). For conflicts in append-only CSVs (`runs.csv`, `gate_log.csv`, `final_test_lock.csv`) keep both sides' rows.
- **pip warnings about versions**: expected on Colab; only an install error that stops the cell matters. Run the tests (6a) if unsure.
