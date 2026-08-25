# Running the tactile forecaster on ND CRC — full walkthrough

Scheduler: **Univa Grid Engine (UGE)**, `qsub`/`qrsh`/`qstat`. GPU: `-q gpu -l gpu_card=1`
(4-day limit; nodes ~32 cores / 4 GPUs). Docs:
[connecting](https://docs.crc.nd.edu/new_user/connecting_to_crc.html) ·
[GPU](https://docs.crc.nd.edu/resources/gpu.html) ·
[conda](https://docs.crc.nd.edu/popular_modules/conda.html).

> **Virtual environment?** Yes — use a **conda env** (step 2), not a bare `python -m venv`.
> CRC serves Python via modules; conda isolates Python + CUDA PyTorch cleanly.

## 0. Connect (SSH)
You need a CRC account and your ND **NetID**. From **off campus**, first connect to the campus
VPN, *or* hop through the bastion host.
```bash
# on campus / VPN:
ssh -Y netid@crcfe01.crc.nd.edu          # front-end (crcfe02 also works)
# off campus without VPN:
ssh -Y netid@bastion.crc.nd.edu          # then: ssh crcfe01
```
(Optional) passwordless login from your machine:
```bash
ssh-keygen -t ed25519                      # if you don't have a key
ssh-copy-id netid@crcfe01.crc.nd.edu
```
Conda requires bash — check with `echo $0`; if not bash, run `bash`.

## 1. Get the code + data onto CRC
**Code** — preferred: clone your fork (after the work is committed/pushed):
```bash
cd ~
git clone https://github.com/Jiayi459/TouchAnything.git
cd TouchAnything
```
*Or* push from local, *or* rsync the working tree from your machine:
```bash
# (run on your LOCAL machine, repo root)
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'datasets' \
      ./ netid@crcfe01.crc.nd.edu:~/TouchAnything/
```
**Data** — the tactile subset is gitignored, so transfer it separately (small, no MP4s).

> ⚠️ **`/scratch365` no longer exists on CRC** (verified 2026-08-19: `df` on it falls back to
> `/`, and `quota`'s own usage text no longer lists it). The tiers now are `/users/$USER`
> (home, **100 GB**), `/groups`, `/temp180`, `/bluefs`, `/goldfs`, and AFS. As of 2026-08-19
> netid `jhao3` has **no per-user directory on `/temp180`, `/bluefs`, or `/goldfs`** — those
> are allocations you request from crcsupport@nd.edu, not directories you can `mkdir`. Until
> one exists, everything must live under `$HOME` and fit its 100 GB quota, so check
> `quota` (no `-s` — it's a CRC wrapper, not the standard tool) before staging anything big.

```bash
# on CRC — with an allocation (preferred for anything large):
mkdir -p /temp180/$USER/egotouch
ln -sfn /temp180/$USER/egotouch ~/TouchAnything/datasets    # datasets -> allocation
# without one, keep it in home and watch the quota:
mkdir -p ~/data/egotouch && ln -sfn ~/data/egotouch ~/TouchAnything/datasets
# on your LOCAL machine (point the target at whichever you chose):
rsync -avz datasets/grasp_hold_lift_tactile/ \
      netid@crcfe01.crc.nd.edu:~/data/egotouch/grasp_hold_lift_tactile/
# (for pretraining, also rsync the full EgoTouch npz tree)
```

## 2. One-time conda environment (the "virtual environment")
```bash
cd ~/TouchAnything
bash scripts/crc/setup_crc_env.sh
```
This runs `module load conda; conda init; source ~/.bashrc; module unload conda`, creates env
**`tactile`** (Python 3.10 + data deps), and installs **CUDA PyTorch 2.5.1 (cu124)**.
`torch.cuda.is_available()` is `False` on the front-end — verify on a GPU node (step 3).

## 3. Smoke test on a GPU
Interactive GPU shell (quickest):
```bash
qrsh -q gpu -l gpu_card=1                 # waits for a GPU node, drops you into a shell
conda activate tactile
cd ~/TouchAnything
python scripts/crc/smoke_test.py          # prints "SMOKE TEST PASSED"
exit
```

## 4. Train (the experiment)
Edit your **netid** in `scripts/crc/train_gpu.job`, then `mkdir -p logs`.
```bash
# (a) Pretrain on ALL trajectories (needs full EgoTouch npz), once per model:
qsub -v CONFIG=configs/tactile_pixel/convgru.yaml,SCOPE=full scripts/crc/train_gpu.job
#   (or run interactively with --pretrain --out runs/convgru_pretrain)

# (b) Fine-tune + Leave-Trajectory-Out CV (5 folds) on the 82 grasp clips:
for f in 0 1 2 3 4; do
  qsub -v CONFIG=configs/tactile_pixel/convgru.yaml,FOLD=$f,SCOPE=grasp,\
PRETRAINED=runs/convgru_pretrain/best.pt scripts/crc/train_gpu.job
done

# (c) Leave-One-Task-Out: add PROTOCOL=loto, FOLD=0..7
```
Monitor: `qstat -u $USER` (state `qw`=queued, `r`=running); kill with `qdel JOBID`. Logs land in
`logs/`. Models: `convgru` (primary), `convlstm` (baseline), `simvp` (headline CNN).

## 5. Evaluate / collect results
```bash
python -m src.tactile_pixel.eval --ckpt runs/convgru_grasp_lto_f0/best.pt --fold 0
# pull results back to your machine (LOCAL):
rsync -avz netid@crcfe01.crc.nd.edu:~/TouchAnything/runs/ ./runs/
```
Each run writes `best.pt`, `train_log.csv`, `test_metrics.csv`, `summary.json`. Headline =
**mean skill vs persistence** (must be > 0).

## 6. Per-category predictability study (the current experiment)

Goal: measure the real **skill-over-persistence per action category** and confirm/break the
training-free ranking in `docs/ACTION_CATEGORIES.md` (probe says: Cut/Spray/Wash easiest,
Press/Click & holds hardest). Needs the **full** EgoTouch npz tree (`--scope full`).

```bash
# --- (A) stage code (fork not set up yet -> rsync the working tree), from LOCAL repo root ---
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'datasets' --exclude 'runs' \
      ./ NETID@crcfe01.crc.nd.edu:~/TouchAnything/

# --- (B) stage ONLY the pressure grids of full EgoTouch to scratch (forecaster reads just
#         pressure_grids.npz; dir names carry the task label for --category) ---
# on CRC (see the storage note in step 1 -- /scratch365 is gone; use an allocation if you
# have one, else home):
mkdir -p ~/data/egotouch && ln -sfn ~/data/egotouch ~/TouchAnything/datasets
# from LOCAL (only *.npz, preserving scene/task/traj structure):
rsync -avz --prune-empty-dirs --include='*/' --include='pressure_grids.npz' --exclude='*' \
      datasets/EgoTouch/ NETID@crcfe01.crc.nd.edu:~/data/egotouch/EgoTouch/

# --- (C) env + smoke test (once) --- (see step 2-3 above for details)
cd ~/TouchAnything && bash scripts/crc/setup_crc_env.sh
qrsh -q gpu -l gpu_card=1     # then in the shell: conda activate tactile; python scripts/crc/smoke_test.py; exit

# --- (D) submit the sweep (9 categories x 5 folds = 45 SimVP jobs) ---
mkdir -p logs
bash scripts/crc/run_percategory.sh
#   swap model:   CONFIG=configs/tactile_pixel/convgru.yaml bash scripts/crc/run_percategory.sh
#   subset:       CATS="Cut Grasp/Hold/Lift Press/Click" FOLDS="0 1 2" bash scripts/crc/run_percategory.sh
qstat -u $USER            # qw=queued, r=running ;  qdel JOBID to cancel

# --- (E) collect: pull runs back + ranked per-category table ---
# from LOCAL:
rsync -avz NETID@crcfe01.crc.nd.edu:~/TouchAnything/runs/ ./runs/
python scripts/shared/aggregate_results.py        # prints "PER-CATEGORY RANKING" (headline)
```

Run dirs: `runs/simvp_full_<slug>_lto_f<fold>/summary.json`. `aggregate_results.py` parses the
`<slug>` and prints a per-category ranking by mean test skill — compare its order to the probe PI.

## Notes
- The repo's top-level `environment.yaml` (full DINOv2/xformers stack) is **not** needed here —
  `environment_tactile_cuda.yaml` is the lean env.
- Bump the torch/cu124 pin in `setup_crc_env.sh` if the cluster driver needs a newer CUDA.
- `--category` filters by the verb taxonomy in `src/tactile_pixel/categories.py`; LTO (default)
  is the right protocol for cross-category comparison. LOTO needs ≥2 tasks in a category
  (Spray/Cut/Pinch have 1–2 → LTO only).
- Support: crcsupport@nd.edu.
