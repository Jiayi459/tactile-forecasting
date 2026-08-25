# SESSION_LOG.md

Source-of-truth log of plans, modifications, analyses, questions/answers, and decisions.
Newest session at the bottom.

---

## Session 1 — 2026-06-17 — Environment setup, dataset download, working agreement, GitHub fork

### Context / platform
- Repo: TouchAnything (local at `c:\Users\haoji\TouchAnything`), Windows 11, PowerShell + Git Bash.
- `origin` remote = `https://github.com/Jianyi2004/TouchAnything` (the **original/upstream** repo).

### Work completed earlier this session
1. **Environment (no conda originally installed; `environment.yaml` is Linux-only).**
   - Created `.venv\` from system Python 3.10.11.
   - Later installed **Miniconda** at `C:\Users\haoji\miniconda3` (conda 26.3.2), initialized for PowerShell.
   - Built Windows-friendly conda env **`touchanything`** (Python 3.10.20, from `conda-forge` to avoid Anaconda commercial ToS).
   - Both envs hold identical **data-only** deps: huggingface_hub 1.5.0, hf_xet, numpy 1.24.3, h5py 3.15.1, opencv-python 4.8.1, pandas, scipy, tqdm, pillow, decord. **Not** the training stack (torch/lightning/triton/xformers/nvidia-* are Linux/GPU-only and not installable here).
2. **Dataset download** — `zhouzhoujy/EgoTouch` (HF), **metadata-only (no mp4)** via `scripts/download_egotouch.py` (`ignore_patterns=["*.mp4"]`).
   - Result: `datasets/EgoTouch/` = 14.91 GB, 11,258 json + 3,286 npz + `split.json` (full annotation coverage; videos skipped per user choice). Integrity spot-check passed (split.json dict with train/val/test_seen/test_unseen; pressure_grids.npz shapes (1652,21,21)).
   - Note: `run_convert_to_hdf5.sh` → `scripts/core/convert_to_hdf5.py` REQUIRES `chest/left/right.mp4` + `pressure_grids.npz` + `wilor_hands.json`, so HDF5 conversion needs a `--videos` re-download to run end-to-end. `wilor_hands.json` is JSON-Lines.

### This task: working agreement + GitHub fork

**Decisions/actions:**
- Created `CLAUDE.md` (Working Agreement) **verbatim** from user-provided text. ⚠️ See OPEN QUESTION 4 (project name reads "IntelligentCarpet"; references `compute_com.py` / "Session 2" from a different project).
- Created `.gitignore` excluding `datasets/`, `.venv/`, `.claude/`, logs, `__pycache__` — verified: only `CLAUDE.md`, `.gitignore`, `SESSION_LOG.md`, `scripts/download_egotouch.py` remain untracked. **Critical** so the 15 GB dataset / venv are never pushed.
- Created this `SESSION_LOG.md`.

**Goal:** "upload this repo to github as a fork from the original repo" = push current local state to a fork of `Jianyi2004/TouchAnything` under the **user's** GitHub account.

**Blockers found:** no `gh` CLI (either shell); `git user.name`/`user.email` unset; credential helper = `manager`; user's GitHub username unknown.

**Planned steps (PENDING user resolution of open questions — not yet executed):**
1. Set `git user.name` / `user.email`.
2. Create the fork of `Jianyi2004/TouchAnything` under the user's account (method TBD — see OQ2).
3. Re-point remotes: `origin` → user's fork, `upstream` → `Jianyi2004/TouchAnything`.
4. Commit the new local files (CLAUDE.md, .gitignore, SESSION_LOG.md, download script) — scope TBD (OQ3).
5. Push to the fork. (Outward-facing/publishing action — requires explicit go-ahead.)

### OPEN QUESTIONS — RESOLVED (2026-06-17)
1. **GitHub account/username** → to be obtained automatically from `gh auth status` after the user authenticates (see OQ2 answer).
2. **Fork creation method** → **Install `gh` + authenticate.** I install GitHub CLI; user runs `gh auth login` (interactive browser/device flow — I cannot do this for them); then `gh repo fork` + push.
3. **Commit scope** → **All 4 new files**: `CLAUDE.md`, `.gitignore`, `SESSION_LOG.md`, `scripts/download_egotouch.py`. Dataset + `.venv` stay excluded via `.gitignore`.
4. **CLAUDE.md content** → **Adapt to TouchAnything.** Done: "IntelligentCarpet" → "TouchAnything"; removed the `compute_com.py`/"Session 2" parenthetical in directive 5.

### Execution progress
- [x] CLAUDE.md adapted to TouchAnything.
- [x] Installed `gh` 2.94.0 at `C:\Program Files\GitHub CLI\gh.exe` (not on PATH in pre-existing shells; available in new terminals).
- [x] User authenticated `gh` as **Jiayi459** (scopes: repo, workflow, read:org, gist).
- [x] Set repo-local git identity: `Jiayi459 <jh9141@nyu.edu>`.
- [x] Created fork **`Jiayi459/TouchAnything`** (isFork=true, parent=Jianyi2004/TouchAnything).
- [x] Re-pointed remotes: `origin` → `Jiayi459/TouchAnything`, `upstream` → `Jianyi2004/TouchAnything`.
- [x] Committed 4 files (`1509fe9`) and pushed `main` to the fork. Remote HEAD verified = `1509fe9`. Dataset/.venv/.claude excluded (confirmed not staged).

### COMPLETED 2026-06-17. Fork live at https://github.com/Jiayi459/TouchAnything

### OPEN ITEM (not part of commit)
- `README.md` has an **accidental working-tree edit**: the string `& "C:\Program Files\GitHub CLI\gh.exe" auth login` was pasted into line 39 mid-sentence (likely a stray paste in the IDE). It was **not** staged/committed/pushed. Pending user decision: revert via `git restore README.md`, or keep/fix manually.

---

## Session 2 — 2026-06-18 — Dataset action categorization + grasp-success forecast (PLANNING)

### Request (verbatim intent)
Go through the EgoTouch dataset; (1) divide all data into categories based on **hand action**; (2) determine which category is **suitable for grasping an item**; (3) **forecast grasp success possibility**.

### Exploration completed (facts)
- Local dataset = metadata-only (no mp4): `datasets/EgoTouch/`. Structure: `Scene/task_name/trajectory_id/{pressure_grids.npz, wilor_hands.json, hamer_hands.json, rokoko_hands.json, vive_poses.json, manual_contact_annotation.json, masks.npz, jq_pressure.json}`.
- **212 real tasks** (213 folders minus `Home/metadata`), **1933 trajectories**. Scenes: Home 124, Office 13, Outdoor 25, Retail 19, Workbench 32 (task counts).
- Task names encode action verbs: grasp/grip/hold/lift, pick_up (largest), pull/push/drag, open/close, fold/spread/wring, plug_unplug, squeeze/pinch, twist/turn/rotate, play (games/sports), swing/throw/bounce/hit/toss, spray/press/click/slide, use/wash/buy/shop/take/put/move/organize/cut/assemble/write, etc.
- **Signals available** (metadata only): `pressure_grids.npz` = left/right tactile grids (T,21,21) normalized to [0,1] (attr `tactile_max`); hand pose (wilor/hamer); camera/wrist poses (vive/rokoko); masks.
- **No grasp success/failure labels exist.** `manual_contact_annotation.json` only has coarse per-traj `left_contact`/`right_contact` booleans, True in only ~5-6% of a 120-sample → NOT a usable success label.
- Pressure data quality: 120-traj sample had 0 all-NaN grids; some trajectories have partial-NaN frames (e.g., `Home/grasp_cola/20260320_090636_772` left grid). Must handle NaNs.

### Env feasibility
- Current `touchanything` conda env / `.venv`: numpy/scipy/h5py/opencv/pandas present. **No scikit-learn** (would need install) and **no deep-learning stack** (Linux/GPU-only, not installable on Windows). So: heuristic/statistical forecast = feasible now; classical ML (sklearn) = feasible after install; deep model = NOT feasible here.

### Proposed plan (PENDING user resolution — not yet implemented per CLAUDE.md directive 5)
1. Categorize all 1933 trajectories by hand-action type via task-name verb taxonomy (transparent, reproducible).
2. Mark grasp-suitable categories (core: grasp/grip/hold/lift/pick_up; partial: squeeze/pinch/take/twist_cap).
3. Define a grasp-success **proxy** from tactile pressure (no GT exists) and compute per-category success probability + per-trajectory scores; emit a markdown report + CSV.

### OPEN QUESTIONS — RESOLVED (2026-06-18)
1. **Categorization method** → **name/verb taxonomy.**
2. **"Suitable for grasp item"** → **core grasp verbs only** (grasp/grip/hold/lift + pick_up).
3. **Grasp-success forecast** → **DEFERRED.** "For now only classify, not yet for prediction success." Do not build any success metric/model yet.
4. **Deliverables** → **wait for user instruction after classification.** Classification done locally; nothing committed yet.

### Classification result (script: `scripts/categorize_actions.py`, NOT committed pending instruction)
- Method: assign each task to one action category by first known action verb in its name (verb_object convention).
- **212 tasks, 1930 trajectories** across 23 categories.
- **CORE GRASP (grasp-suitable) = `Grasp/Hold/Lift` (8 tasks, 82 traj) + `Pick-up` (64 tasks, 635 traj) = 72 tasks / 717 traj (~37% of all trajectories).**
- Largest non-grasp categories: Organize/Arrange (140), Use tool/appliance (131), Open/Close (130), Squeeze (100), Push/Pull/Drag/Slide (98), Plug/Unplug/Insert (89).
- Full per-category task assignment printed to console; reviewed edge cases (e.g. `pick_up_and_squeeze_power_bank`→Pick-up by leading verb; `grip_hand_dynamometer`→Grasp/Hold/Lift; `handle_*`,`remove_pen_cap`,`split_*`→Other).

### Pending user instruction
- Whether to commit `scripts/categorize_actions.py` and emit a report/CSV.
- When/if to proceed to the grasp-success forecast (and which success definition).

---

## Session 3 — 2026-06-18 — Grasp/Hold/Lift tactile subset + tactile→tactile forecasting plan

### Request
Download (prepare) the Grasp/Hold/Lift data; goal = train a model to predict **future tactile
from past tactile**. Produce an implementation plan: literature review, process, method choice,
validation, preprocessing. Be rigorous/constructive/precise/efficient.

### Key reasoning / decisions
- Task is **tactile→tactile** forecasting ⇒ **videos not needed**; `pressure_grids.npz` already
  on disk from the metadata pull. "Download" = prepare the subset, not re-fetch MP4s.
- Prepared subset `datasets/grasp_hold_lift_tactile/` (+ `manifest.csv`) via
  `scripts/prepare_grasp_tactile.py`. **82 traj, 31,577 frames @30fps.**

### EDA facts (ground the plan)
- Lengths skewed: min 71 / median 125 (~4.2s) / mean 385 / max 2206 ⇒ windowing required.
- Each hand 21×21 with **~50.8% structurally-NaN cells = fixed sensor mask** (~220 valid taxels);
  values in [0,1]; ~33% valid taxels active/frame (sparse).
- **Predictability probe** (`scripts/tactile_predictability_probe.py`): persistence nMSE
  0.04→0.13→0.23→0.47→0.72→1.34 at h=1/3/5/10/15/30; force autocorr 0.99→...→−0.17 at lag 30.
  ⇒ honest horizon **0.1–0.5s**; persistence is a strong baseline ⇒ judge models by **skill vs
  persistence**; **N=82 is the binding constraint**.

### Literature review (in plan)
Tactile prediction: ACTP/ACTVP (arXiv:2205.09430, Conv-LSTM, slip), DFPC strawberry
(2303.05393), Dream-Tac (2606.08737), Tactile diffusion policy (2510.13324). Backbones: OpenSTL,
ConvLSTM, PredRNN, **SimVP/TAU** (2206.05099, CVPR'22/23), PredFormer (2410.04733), survey
(2401.14718). Recommendation: **SimVP/TAU headline + ConvLSTM baseline**; transformer/generative
as extensions.

### Compute (UPDATED after user: "we have gpu, school CRC")
- Two-tier: local Windows (`touchanything` env, CPU torch) for dev/baselines; **CRC GPU cluster
  (Linux+CUDA, likely SLURM)** for training/CV/ablations. `environment.yaml` builds on cluster.
- GPU enables **pretrain on all 1,930 traj → fine-tune on 82 grasp clips** to fight small-N.

### Artifacts (NOT committed — pending approval per CLAUDE.md)
- `docs/TACTILE_PREDICTION_PLAN.md` (full plan), `scripts/prepare_grasp_tactile.py`,
  `scripts/tactile_predictability_probe.py`, `datasets/grasp_hold_lift_tactile/` (gitignored).

### DECISIONS — RESOLVED (2026-06-18, user)
1. Horizon = **0.5 s** (15 frames @30fps); report 1/3/5/10/15.
2. Hands = **both** (bimanual 2-ch primary, dominant-hand ablation).
3. Method = **ConvGRU primary** + ConvLSTM precedent baseline + SimVP/TAU headline CNN.
   GRU question answered: ConvGRU chosen for N=82 (fewer params/less overfit, keeps spatial
   structure); plain GRU rejected (flattening loses contact geometry).
4. Compute = **ND CRC**. CONFIRMED scheduler = **UGE/`qsub`** (NOT SLURM), GPU via
   `-q gpu -l gpu_card=1` (4-day limit). Conda: init once, `conda activate` in jobs.
5. **Pretrain on all 1,930 traj → fine-tune on 82 grasp clips.**
6. **Deterministic** next-frame prediction (generative deferred).
7. **CUDA** env (no local CPU torch).

### CRC env setup created (this turn; NOT committed)
- `scripts/crc/environment_tactile_cuda.yaml` (lean conda env), `scripts/crc/setup_crc_env.sh`
  (conda init + env + CUDA torch 2.5.1/cu124), `scripts/crc/train_gpu.job` (UGE GPU template),
  `scripts/crc/README.md` (rsync + setup + qsub workflow).
- Plan `docs/TACTILE_PREDICTION_PLAN.md` updated: §4 model lineup (ConvGRU), §6 compute (UGE),
  §10 decisions resolved.
- Refs: CRC GPU docs https://docs.crc.nd.edu/resources/gpu.html ; conda
  https://docs.crc.nd.edu/popular_modules/conda.html

### BUILD ("go", 2026-06-18) — `src/tactile_forecast/` package implemented
- `tactile_utils.py` (torch-free: mask/transform/window/splits/metrics) — **verified locally**
  on real data: mask 217 valid/hand (structural, 0 variance across trajs), 5,955 windows
  (Tin10/Tout15/stride5), LTO 5-fold (65/17), LOTO 8-fold, metric sanity OK.
- `data.py` (TactileWindows + trajectory-level split_train_val, mask-safe aug),
  `models/{conv_rnn(ConvGRU+ConvLSTM, scheduled sampling), simvp}`, `models.build_model`,
  `engine.py` (masked MSE + active-taxel weight, train/eval, SS schedule),
  `baselines.py` (persistence, last-velocity), `train.py`/`eval.py` (CLI: lto/loto, fold,
  grasp/full, pretrain & --pretrained finetune; outputs best.pt/train_log/test_metrics/summary).
- `configs/tactile/{convgru,convlstm,simvp}.yaml`. CRC: `scripts/crc/smoke_test.py` (synthetic
  e2e) + updated `train_gpu.job` (runs entrypoint via -v CONFIG/FOLD/SCOPE/PROTOCOL/PRETRAINED)
  + README run recipe.
- **All 11 modules byte-compile.** Torch path NOT run locally (no local torch per decision #7);
  to be smoke-tested on CRC (`python scripts/crc/smoke_test.py`).
- Headline metric = mean **skill vs persistence** (must be >0); honest horizon ≤0.5 s.

### CORRECTION (2026-06-18) — TAU not implemented
- User asked where the "TAU" training method is. **It is not in the code.** Built models =
  SimVP-lite, ConvGRU, ConvLSTM only. Plan said "SimVP/TAU" (family name) but only SimVP exists
  (`configs/tactile/simvp.yaml`; no `tau.yaml`). TAU = Temporal Attention Unit (Tan et al.,
  CVPR 2023, arXiv:2206.12126): SimVP skeleton + TAU translator (intra-frame statical + inter-frame
  dynamical attention, parallelizable) + Differential Divergence Regularization loss.
- ACTION PENDING user choice: either implement TAU (`models/tau.py` + DDR in engine + tau.yaml)
  OR edit plan wording "SimVP/TAU" → "SimVP" to match code.

### CRC RUN LOG
- 2026-06-19: Pushed full pipeline to fork (7e3bec0). User cloned on crcfe01.
- **Fix (fff1db7):** `setup_crc_env.sh` aborted on `set -u` — CRC `/etc/bashrc` has unbound
  `BASHRCSOURCED`. Removed `set -u`; now sources `$(conda info --base)/etc/profile.d/conda.sh`
  directly instead of `conda init`+`.bashrc`. Env `tactile` had not been created; gave user
  manual create commands + the fixed script.

### FIRST TRAINING RESULT + FIX (2026-06-19)
- Smoke test passed on GPU (A10, cuda True) after two harness fixes: smoke mask must be batched
  (B,C,H,W) [3a05619]; `horizon_metrics` mask cast to bool [65f1d64].
- First real run (ConvGRU, LTO fold0, grasp, no pretrain): pipeline OK end-to-end but **test
  mean-skill ≈ 0.0038 ≈ break-even with persistence** (skill@h ~0.01→0). Training loss *rose*
  over epochs under scheduled sampling. Diagnosis: models predict ABSOLUTE frames → easiest
  optimum is to copy last frame (= persistence). Persistence is very strong (probe: autocorr
  0.99@33ms).
- **Fix [390861c]: residual prediction** — models output Δ from last observed frame
  (`pred=clamp(last+Δ,0,1)`); persistence == zero delta, so skill comes from learned deviations.
  `residual` flag (default true) in ConvGRU/ConvLSTM/SimVP + configs. Also silenced torch.load
  weights_only warning. Awaiting rerun to confirm positive skill.

### RESIDUAL RESULT CONFIRMED (2026-06-19)
- ConvGRU, LTO fold0, grasp, no pretrain, residual=on: **test mean-skill = 0.174** vs persistence
  (last_vel −2.4). skill@h: h1=−0.04 (persistence near-unbeatable at 33ms), rising monotonically
  to h15=+0.25. => Future tactile IS predictable from past beyond persistence; gain grows with
  horizon over 0.5s. Caveat: single fold (test=17). Added `scripts/aggregate_results.py`
  [2e706c2] for mean±std across folds.

### LTO 5-FOLD CV RESULTS (grasp, no pretrain, residual on) — 2026-06-19
(2h runtime was recurrent models' Python time-loop, not a hang; completed fine.)
- ConvGRU : 0.138 ± 0.056 (h1=-0.090, h15=+0.207)
- ConvLSTM: 0.152 ± 0.031 (h1=+0.036, h15=+0.211)
- **SimVP : 0.192 ± 0.044 (h1=+0.065, h15=+0.235) — BEST at every horizon**
=> Conclusive: future tactile predictable from past beyond persistence (~19% error reduction
   over 0.5s). SimVP (non-recurrent) beats both recurrent models AND is far faster; recurrent
   models weak/negative at h1. **Promote SimVP to primary** (overturns prior ConvGRU pref).
   Fold 2 hardest for all (likely long grip_hand_dynamometer test split).

### LOTO RESULT (SimVP, grasp, no pretrain) — 2026-06-21
- **SimVP LOTO = +0.005 ± 0.111** (per-fold -0.131..+0.151). vs LTO +0.192.
- KEY FINDING: grasp-only learned tactile dynamics are **object-specific** — do NOT generalize
  to unseen objects (mean ≈ persistence; some folds worse). Motivates pretraining.

### PRETRAIN->FINETUNE SETUP [092ea63]
- train.py `--exclude-grasp`: drop the 8 grasp tasks during pretrain so LOTO held-out object is
  never seen (no leakage). download_egotouch.py `--pressure-only` (~1.7GB npz). env += hf_hub.
- Workflow: download --pressure-only -> pretrain SimVP --scope full --pretrain --exclude-grasp
  (~1848 traj) -> finetune --protocol loto --pretrained ... --out runs/simvp_ft_grasp_loto_fN.
- Compare `simvp_ft | grasp | loto` vs baseline `simvp | grasp | loto` (+0.005).

### PRETRAIN DONE (2026-06-22)
- First attempt ran on CPU front-end (device=cpu, 361k windows) -> stuck 13h; killed. Cause:
  ran on crcfe01 (no GPU). Fix: qsub GPU batch job + stride20/batch256 [60a49ab]; also fixed
  UGE inline-comment bug on `#$ -M` [6b93fb8].
- Pretrain (SimVP, scope full, --exclude-grasp = 1851 traj, 30 epochs, GPU) completed in ~5h
  (slow due to per-epoch val eval over ~62k windows). best.pt @ epoch23, val_skill 0.227 on
  held-out NON-grasp data => good general tactile predictor. -> runs/simvp_pretrain/best.pt

### HEADLINE RESULT (2026-06-23) — pretraining unlocks unseen-object prediction
- SimVP LOTO: scratch **+0.005 ± 0.111** -> pretrained->finetuned **+0.097 ± 0.122** (~18x,
  positive at all horizons; helped 6/8 held-out objects; fold5 stays ~-0.20 outlier).
- Full story: (1) tactile predictable from past (LTO +0.192); (2) doesn't generalize from few
  objects (LOTO scratch ~0); (3) broad multi-object pretrain enables it (LOTO +0.097).
- Documented in `docs/RESULTS.md` [aa7e49d]. CORE STUDY COMPLETE.

### OPTIONAL NEXT
- Visualization: predicted-vs-GT pressure GIFs (eval-based renderer).
- Ablations: LTO-finetune (does pretrain help seen-object too?); smaller SimVP (30.5M is
  over-param); investigate fold5 outlier object.
- Perf: speed up engine.evaluate (memory-heavy concat) if rerunning pretrain.
- rsync runs/ back to local for plots.
- Set up pretrain-on-full (1,930 traj) → finetune (needs full npz on CRC: rsync or add HF
  downloader). Minor: h1 slightly negative (model adds noise at easiest horizon) — possible
  later tweak (per-horizon loss weighting).
- Commit decision: Session 2/3 artifacts (categorize script, tactile prep/probe scripts, plan,
  src/tactile_forecast, configs, crc setup) — not yet committed/pushed to the fork.
- [ ] Set `git user.name`/`user.email` (name from gh login; email jh9141@nyu.edu).
- [ ] `gh repo fork Jianyi2004/TouchAnything` → re-point origin to fork, upstream to original.
- [ ] Commit 4 files, push to fork (publishing — proceed only after auth confirmed).

---

## Session 4 — 2026-07-01 — New direction: which ACTION CATEGORY is most predictable (cross-dataset)

### Goal (user request)
Read three tactile/force datasets, enumerate every action collected, categorize actions (by
force type / movement pattern / etc.), then run per-category prediction to find **which category
of action is easiest to predict** — or at least summarize the *traits* of an action series that
make it predictable. Priors given by user: (a) actions with a **standard procedure** are easier
to predict; (b) actions with a **repeatable/periodic pattern** are easier. Ultimate goal: use the
predictor to give the user **feedback / adaptive strategies** to improve performance.

This is a NEW research thread built on the existing tactile→tactile forecasting infra
(`src/tactile_forecast`, skill-over-persistence metric, LTO/LOTO protocols). Session 1-3
established: LTO seen-object +0.192; LOTO unseen ~0; pretrain→finetune LOTO +0.097.

### Datasets read (2026-07-01) — sources
1. **OpenTouch** (opentouch-tactile.github.io, arXiv 2512.16842). First in-the-wild egocentric
   FULL-HAND tactile dataset. Modalities @30Hz: FPC-based tactile sensor + Meta Aria egocentric
   video + Rokoko Smartgloves hand pose; 2ms sync. 5.1h recordings, ~2,900 curated clips,
   ~800 objects, 14 environments, 14 object categories. Labels per clip: object name, object
   category, environment, **action type**, **grasp type** (29 grasps from GRASP taxonomy:
   e.g. Medium Wrap, Small Diameter, Prismatic Two-Finger, Index-Finger Extension), NL caption.
   Action examples named in text: pressing, rotating, turning, button click, grasping; contact
   with chair/table/transparent objects. Full action list is in Supp. Mat. (not enumerated on
   arXiv HTML; would need supp PDF).
2. **Force–Vision / "Learning to Jointly Understand Visual and Tactile Signals"** (ICLR 2024,
   Li/Liu et al., extends GEM). Cross-modal force+vision on **articulated tools**. Sensor:
   Sundaram et al. STAG-style tactile glove (full-hand NORMAL force map) + webcam. Scale:
   **2,000,000 paired frames over 89 real object instances** (scissors, staplers, clips/clamps,
   pliers, spray bottles, ...). Manipulation types explicitly analyzed: **press, hold, squeeze**.
   Key finding they report: "press" clusters apart from "hold"/"squeeze" — press activates a
   *contiguous* hand region and is *one-directional*; squeeze/hold create *force closure* → the
   two force-closure actions are more similar to each other. (Directly relevant to our
   force-type axis.)
3. **ActionSense** (NeurIPS 2022 D&B, MIT CSAIL). Multimodal WEARABLE kitchen dataset.
   Modalities: custom conductive-thread tactile gloves + Myo EMG (forearm muscle) + 17-IMU Xsens
   body tracking + finger gloves + Pupil eye-tracking w/ first-person cam + 5 RGB + depth + 2 mic.
   **20 unique activity labels in 6 task categories** (Fig 2), ~7+ subjects. The 6 categories:
   (i) Peeling & slicing (cucumber/potato/bread; + auxiliary "clear cutting board"),
   (ii) Spreading (almond butter / jelly on bread w/ knife),
   (iii) Wiping (pan/plate w/ towel or sponge — periodic circular/linear strokes, force key),
   (iv) Open/close a jar (rotational, subtle, tactile+EMG key),
   (v) Pouring water (monotonically changing container weight; transparent liquid),
   (vi) High-level tableware sequences (set table; load/unload dishwasher; stacking).
   NOTE full 20 leaf labels live in Fig 2 / supp (not machine-readable from the main-text PDF).

### Existing repo datasets already in this taxonomy
- **EgoTouch** (21×21 pressure grid) — general in-the-wild tactile (used for pretraining).
- **grasp_hold_lift_tactile** — 8 tasks: grasp_body_lotion, grasp_cola, grasp_floral_water,
  grasp_power_adapter, grasp_sunscreen, grip_hand_dynamometer, hold_teapot, lift_towel.
  These are ALL sustained-grip/hold/lift = one corner of the proposed taxonomy (explains why
  LOTO≈0: near-static maps where persistence is already strong → low skill headroom).

### PROPOSED unified action taxonomy (draft — for user review)
Categorize along orthogonal axes; each concrete action = a point in this space.

**Axis A — Force type / contact mechanics**
- A1 Sustained force-closure grip/hold (grasp_*, hold_teapot, jar-hold, FV "hold"/"squeeze")
- A2 Impulsive one-directional press (button click, FV "press", stapler)
- A3 Cyclic surface force (wiping, spreading, slicing strokes, peeling strokes, scrubbing)
- A4 Torsional / rotational (open/close jar, turn knob, OpenTouch "rotating"/"turning")
- A5 Monotonic ramp load (pouring — weight ↓; lifting — load ↑; squeeze-to-close)
- A6 Precision fingertip (pinch, click, fine slice)

**Axis B — Movement / temporal pattern** (this is the predictability-driving axis)
- B1 Periodic / rhythmic-repeatable (wiping, slicing, peeling, spreading) → hypothesis HIGH skill
- B2 Quasi-static / near-constant (hold, grip, sustained press) → low MSE but LOW *skill* (persistence wins)
- B3 Monotonic ramp (pouring, lifting) → MEDIUM
- B4 Discrete one-shot transition (button click, jar-open "snap", pick-place onset) → LOW
- B5 Composite long-horizon sequence (set table, load dishwasher) → LOW (planning + many sub-actions)

**Axis C — Procedural standardization** (user prior)
- C-high: standardized (regular slicing strokes, standard jar twist, standard pour tilt)
- C-low: free-form/adaptive (wiping strategy, spreading adapts to substance, tableware planning)

**Axis D — Contact spatial dynamics of the tactile map**
- D1 Stable footprint (same taxels active) — grip/hold/press → spatially trivial
- D2 Migrating/sliding contact (wiping, slicing, peeling) — contact region translates
- D3 Making/breaking contact (pick-place, click) — onset/offset hardest

### Predictability hypothesis (to TEST, ranked most→least "skill-over-persistence")
1. B1 periodic surface actions (A3×B1×D2): structured motion persistence CAN'T capture → highest skill.
2. A4 rotational / B3 monotonic ramps: some learnable trend → medium.
3. A1/B2 sustained holds: low raw error but ~0 skill (persistence already near-perfect).
4. B4 discrete events / B5 long sequences: lowest (event timing / planning).
This predicts the user's priors partly REVERSE under a skill-over-persistence metric: "easy to
hold steady" ≠ "high forecasting skill." Must pin down the metric (OPEN Q1).

### OPEN QUESTIONS (must resolve before any implementation — plan-before-code)
- **Q1 — Definition of "easier to predict."** Raw accuracy (MSE/IoU/force-MAE) vs.
  **skill-over-persistence** (structured, learnable dynamics)? These rank categories differently
  (static holds win #1 on raw error but ~0 on skill). RECOMMEND: report both; headline on
  skill-over-persistence since that's where feedback/adaptive strategies have leverage.
- **Q2 — Which dataset(s) for this iteration?** ActionSense is the cleanest labeled *everyday-action*
  taxonomy with tactile time-series (best fit). OpenTouch adds action+grasp labels; force-vision
  adds press/hold/squeeze. Do we have download access? (EgoTouch was downloaded metadata-only;
  ActionSense/OpenTouch/FV not yet fetched, and their tactile sensor geometries differ from the
  21×21 EgoTouch grid → new preprocessing per dataset.)
- **Q3 — Prediction target / modality.** Continue tactile→tactile forecasting (reuse infra), or
  predict a force scalar, or cross-modal (vision/pose→tactile)? "Feedback to enhance performance"
  hints we may want to compare a user's applied force to a learned "ideal" template.
- **Q4 — Category granularity.** Use the proposed A/B/C/D axes (recommend B as primary grouping),
  or a flatter user-defined category set?
- **Q5 — Scope of THIS step.** Deliver categorization + study design only (await answers), or also
  stand up a first per-category forecasting run on whatever tactile data is already local?

### ANSWERS (user, 2026-07-01)
- Q1 = **Both, skill as headline** (report MSE/IoU/force-MAE + skill; rank by skill-over-persistence).
- Q2 = **All three datasets** — produce ONE unified categorization spanning ActionSense + OpenTouch
  + Force-Vision (+ local EgoTouch/grasp).
- Q3 = tactile / **physical-representation prediction** (predict the tactile physical signal;
  reuse tactile→tactile forecasting representation).
- Q5 = **Also prototype now.**

### CONSTRAINT REALITY CHECK
- torch NOT installable on this Windows box → cannot TRAIN here (training runs live on CRC GPU).
- Only EgoTouch (21×21 grids) + grasp_hold_lift tactile are downloaded locally; ActionSense /
  OpenTouch / Force-Vision raw data NOT local (different sensor geometries → per-dataset preprocessing later).
- ∴ "Prototype now" = a **training-free predictability probe** (numpy only) grouped BY CATEGORY over
  local EgoTouch. Directly measures "which category is most predictable" without a GPU. Existing
  `scripts/tactile_predictability_probe.py` (persistence nMSE, autocorr, smoothness) + 
  `scripts/categorize_actions.py` (verb→category) are the building blocks — MERGE them.

### PLAN (this step)
1. `scripts/predictability_by_category.py` — for every EgoTouch trajectory: categorize (verb map)
   + map to a **temporal-pattern class (Axis B)**, load pressure_grids.npz (L+R, nan→0), compute
   per-sequence: persistence nMSE (RAW hardness), constant-velocity nMSE, **velocity-skill vs
   persistence** (learnable first-order dynamics headroom = skill proxy), **periodicity score**
   (max total-force autocorr at lag 10–45 = rhythmic/repeatable evidence), **contact-migration**
   (1−IoU of active-taxel mask across h). Aggregate & RANK by category and by B-class. Write CSV.
2. `docs/ACTION_CATEGORIES.md` — unified cross-dataset taxonomy table mapping every ActionSense /
   OpenTouch / Force-Vision / EgoTouch action into Axes A(force)/B(temporal)/C(standardization)/
   D(contact-dynamics), with the per-category predictability numbers attached where measurable.
3. Interpret: does empirical velocity-skill/periodicity confirm the hypothesis (B1 periodic >
   ramps > holds > events)? Feed into the feedback/adaptive-strategy goal.

### IMPLEMENTATION (2026-07-01)
- Wrote `scripts/predictability_by_category.py` (numpy/venv only; imports `categorize` from
  `categorize_actions.py`). Per EgoTouch trajectory: persistence nMSE @h={1,5,15,30}, periodicity
  (max total-force autocorr, lag 10–45 frames), contact_migration (1−IoU active-taxel mask @h15).
  Composite `PI = z(−persH15)+z(periodicity)+z(−migr15)`. Groups by verb category AND temporal
  pattern (Axis B); ranks; writes `docs/actionsense/predictability_by_category.csv`.
- DISCARDED a constant-velocity skill proxy: `h·velocity` extrapolation blows up on impulsive
  tactile spikes (velSk ≈ −15..−37), noise-dominated → not a valid training-free skill proxy.
  A real skill-over-persistence number needs the GPU forecaster. Documented in script docstring.
- Ran `--max-per-task 12` → 1,493 sequences.

### RESULTS (probe, EgoTouch, n=1493) — ranked easiest→hardest (PI)
- Easiest: **Cut/slice** PI+6.11 (persH15 0.088, periodicity 0.968, migr 0.215) >> Take +2.84 >
  Inflate +2.57 > **Spray** +2.38 > **Wash/Clean (wipe)** +2.31.
- Hardest: **Press/Click** −7.14 (persH15 1.399, migr 0.689) < Pinch −4.51 < Plug/Insert −3.82 <
  Fold −2.62 < Push/Pull −2.27 < Squeeze −1.98 < **Grasp/Hold/Lift −1.57**.
- FINDINGS: (1) periodic surface actions (cut/spray/wipe) most predictable — CONFIRMS "repeatable
  pattern" prior + B1 hypothesis. (2) make/break-contact events (press/click, plug) hardest —
  CONFIRMS B4/D3. (3) NUANCE refuting naive view: sustained HOLDS are NOT trivially predictable
  (Grasp/Hold/Lift below median, worst persH30=1.223) — grips drift + footprint unstable; explains
  grasp-only LOTO≈0. (4) periodicity predicts forecastability better than procedural standardization.
- Small-n caveat on top categories (Cut/Spray n=10). Next: full run (`--max-per-task 0`), then
  CONFIRM by running `src/tactile_forecast` per-category on CRC GPU (probe PI = the hypothesis).

### DELIVERABLES this session
- `docs/ACTION_CATEGORIES.md` — unified cross-dataset taxonomy (ActionSense+OpenTouch+Force-Vision
  +EgoTouch) mapped into Axes A/B/C/D + empirical predictability table + feedback-target implication
  (B1-periodic actions are the good feedback targets: they have a "correct rhythm/force template").
- `scripts/predictability_by_category.py`, `docs/actionsense/predictability_by_category.csv`.
- Saved OpenTouch/ActionSense/Force-Vision paper PDFs were parsed via pypdf (pdftoppm/Read-PDF
  unavailable on Windows) to extract exact taxonomies.

### FOLLOW-UP (a) FULL-DATA PROBE + (b) PER-CATEGORY FORECASTER (2026-07-01/02)
User: "do a and b".
- (a) Ran probe `--max-per-task 0` → **1,929 sequences**. Ranking reproduces the sampled run
  almost exactly (Cut PI +6.02, Take +2.90, Inflate +2.62, Spray +2.46, Wash +2.37; bottom:
  Press/Click −6.67, Plug/Insert −4.86, Pinch −4.17, Fold −2.34, Push/Pull −2.28, Squeeze −2.11,
  Grasp/Hold/Lift −2.08). → ranking is ROBUST to sampling. Wrote `docs/actionsense/predictability_by_category_full.csv`.
- (b) Wired the REAL forecaster for per-category confirmation:
  - NEW `src/tactile_forecast/categories.py` = single source of truth (VERB_CATEGORY, CORE_GRASP,
    categorize, TEMPORAL_PATTERN, all_categories). Pure stdlib (both `src/__init__.py` and
    `src/tactile_forecast/__init__.py` are torch-free, so local scripts can import it).
  - Refactored `scripts/categorize_actions.py` to import from that module (adds repo root to
    sys.path) — removes the duplicated verb map. Verified it still runs (212 tasks / 1930 traj).
  - `src/tactile_forecast/train.py`: added `--category NAME` (filters trajectories by
    categorize(task)); run-dir now carries a slug tag (`simvp_full_<slug>_lto_f<fold>`).
  - NEW `scripts/crc/percategory_gpu.job` (UGE) takes `-v CATEGORY,FOLD,CONFIG,PROTOCOL`; trains
    LTO within one category on `--scope full`. Header has the all-categories×5-folds submit loop.
  - Verified (torch-free): `--category` filter + slug over real data → every category has ≥5
    trajectories (5-fold LTO viable); slugs clean. py_compile passes on all edited files.
- Doc `docs/ACTION_CATEGORIES.md` updated with full-data table + §5 confirmation-run instructions.
- STATE: cannot train locally (no torch). CRC run is the remaining step to turn the probe
  HYPOTHESIS (PI ranking) into MEASURED per-category skill. All artifacts uncommitted (fork not set up).

### CRC STAGING + AGGREGATION (2026-07-02)
User: "stage the CRC commands". Also confirmed prediction methods = 3 architectures.
- METHODS (verified in `models/__init__.py` build_model): **ConvGRU**, **ConvLSTM** (both
  `ConvRNNSeq2Seq`, cell gru/lstm), **SimVP** (`simvp.py`, headline). Plus 2 non-learned baselines
  in eval (persistence, last_velocity). **TAU is NOT implemented** — this SimVP is "SimVP-lite"
  (Conv translator, n_trans=4), not the gated Temporal Attention Unit. TAU would be a translator
  swap if we want it; noted as optional.
- BUG FIXED: `scripts/aggregate_results.py` DIR_RE could not parse per-category run dirs
  (`simvp_full_cut_lto_f0`) → those runs were silently skipped. Rewrote regex to capture an
  optional slug `(?:_(?P<category>[^_]+(?:-[^_]+)*))?`; unit-tested on 7 dir names incl. the
  existing `simvp_ft_grasp_loto_f5`. Aggregator now groups by category and prints a
  **PER-CATEGORY RANKING** (mean test skill) — the study headline that confirms/breaks the probe PI.
- NEW `scripts/crc/run_percategory.sh` — one-command sweep (9 categories × 5 folds = 45 SimVP
  jobs; CONFIG/CATS/FOLDS overridable). CATS list uses only space-free category names (slashes ok).
- `scripts/crc/README.md` §6 — full staging walkthrough: (A) rsync working tree, (B) rsync ONLY
  `pressure_grids.npz` of full EgoTouch to /scratch365 + symlink, (C) env+smoke, (D) submit sweep,
  (E) rsync runs back + `aggregate_results.py`. Uses NETID placeholder (CRC netid = jhao3).
- Ready to run on CRC. I cannot submit (no CRC/SSH/torch here) — user launches it.

### SCOPE EXPANSION: all 4 datasets, probe-first, OpenTouch next (2026-07-02)
User: not grasp-focused (clarified: EgoTouch sweep already ranks ALL 23 categories, grasp is
just one). Wants which category is predictable ACROSS all 4 datasets. Decisions (AskUserQuestion):
next=OpenTouch, depth=probe-first (training-free, no GPU), where=CRC.
- DATA AVAILABILITY (checked): all 4 downloadable. OpenTouch = public Google-Drive via
  `scripts/download_data.sh` (26 HDF5 shards + `final_annotations` labels). ActionSense =
  public (delpreto/ActionNet, CC-BY-NC, HDF5). Force-Vision = public Google-Drive zip.
- CROSS-SENSOR CAVEAT: EgoTouch 21x21 2-hand / OpenTouch 16x16 1-hand / ActionSense
  conductive-thread / FV STAG differ → CANNOT compare raw skill across datasets. Design: rank
  WITHIN each dataset, then pool by TEMPORAL-PATTERN axis (B1..B5) to test if the same action
  KIND wins everywhere (sensor-agnostic answer). OpenTouch `action` free-text is mapped through
  the SAME categorize() verb taxonomy → lands in the same category/pattern space as EgoTouch.
- OPENTOUCH SCHEMA (verbatim from repo build_label_data.py): HDF5 `data/<clip_id>/right_pressure`
  = (T,16,16); labels in CSV/TSV keyed by clip id, cols object_name/object_category/environment/
  action/grip_type. 30 Hz. Pressure raw up to ~3072 (scale-invariant metrics → no normalization).
- BUILT:
  - NEW `src/tactile_forecast/predictability.py` — shared numpy metrics (seq_metrics, aggregate,
    add_predictability_index) for ANY (T,C,H,W) sensor. Sanity-tested (periodic→period 1.0,
    static→migr 0.0, event→period 0.0). EgoTouch probe left with its inline copy (don't disturb
    the committed/running script); shared module is go-forward, used by OpenTouch probe.
  - NEW `scripts/opentouch_predictability.py` — `--inspect` (dump HDF5 tree + label cols + join
    rate) and probe modes; groups by temporal-pattern / mapped-category / raw-action / grip_type;
    writes docs/predictability_opentouch.csv. Robust: auto-detect HDF5 clip groups + label key col.
  - Both compile; h5py 3.15.1 present locally.
- OPEN RISK — DISK: OpenTouch HDF5 bundles RGB (`rgb_images_jpeg`) → shards likely large; CRC
  home only 35G free and /scratch365/jhao3 not provisioned. PLAN: gauge size (download labels +
  1 shard, measure), then either (a) fits → download all + probe, (b) too big → build a
  streaming driver (gdown shard → per-shard probe → delete → next) or request scratch from
  crcsupport. Decide after the size gauge.

### OPENTOUCH VALIDATED ON CRC (2026-07-02)
- GAUGE: 1 shard (office_csail_p2.hdf5) = 561 MB → 26 shards ≈ 14.6 GB → FITS in 35 GB home.
  No streaming/scratch needed. Download-all approved.
- SCHEMA CONFIRMED live: HDF5 top keys {calibration, data, transform_slam_to_rgb}; `data/<clip>`
  has right_pressure (T,16,16) f32 max=3072, camera_poses, rgb_images_jpeg, hand_landmarks,
  timestamps, plus a per-clip `labels`=(0,0) index-pair (NOT the action). Labels come from
  final_annotation.zip → `final_annotations/<scene>_merged.csv`, key col `clip_id` =
  "<scene>::demo_NNN" (globally unique), cols incl. action (gerund), grip_type (GRASP taxonomy),
  object_category, environment, description, peak_idx. One row per clip.
- FIRST PROBE (1 shard, 111/113 usable, 0 unlabeled → join works): grip-type ranking sensible
  (Prismatic-3-Finger/Medium-Wrap easiest; Prismatic-4-Finger/Index-Extension hardest). Action
  vocab = gerunds: placing/adjusting/removing/pinching/picking up/holding/pulling/pushing/moving/
  pressing/turning. Category/pattern were all "Other" (gerund mismatch) — FIXED:
  - categories.py: added `categorize_phrase()` (inflection stemmer: pulling→pull, cutting→cut,
    placing→place, picking up→pick) + new verbs (slice/peel/chop→Cut, pour/scoop→Pour[new,B3],
    wipe/scrub→Wash/Clean, adjust→Organize). Left EgoTouch `categorize()` + the `spread`→Fold/Cloth
    mapping UNCHANGED (no silent shift to the running EgoTouch sweep).
  - opentouch_predictability.py now uses categorize_phrase; verified all 20 observed/likely verbs
    map to the right category+pattern (only "removing"→Other).
  - NEW scripts/crc/download_opentouch.sh (26 shard IDs + labels, verbatim from opentouch repo).
- NEXT: download all shards, run probe → full OpenTouch per-category + temporal-pattern ranking;
  compare to EgoTouch by the B-axis.

### OPENTOUCH FULL RESULT (2026-07-02) — 26 shards, 2496 usable / 2958 clips (457 unlabeled)
- Raw-action ranking (trustworthy): TOP pouring +4.4 / serving +3.6 / eating +3.4 / stirring +3.0
  / scooping +2.5 / flipping +2.4 / wiping +1.3; BOTTOM cutting(n4) -3.0 / moving -2.6 / turning
  -2.2 / pulling -1.8. Standouts have persH15 0.26-0.39 vs pack 0.7-0.9.
- contact_migration ≈ 0.005 for ALL categories (single-hand grasp footprint never breaks) →
  DEGENERATE metric here; PI driven by persH15 + periodicity.
- CROSS-DATASET SURPRISE: OpenTouch temporal-pattern ranks B4>B2>B1 — OPPOSITE of EgoTouch (B1 high).
  Cause: a-priori verb→pattern map breaks (turning-a-latch != rhythmic turn; many OT verbs unmapped
  → "Other" holds predictable food actions; Pour mislabeled B3). LESSON: assign Axis B per-action
  from data, not a-priori.
- DURABLE FINDING (answers user's "trait" goal): predictable = smooth continuous slowly-varying
  contact force (pour/stir/scoop/serve/wipe/slice); unpredictable = abrupt onset / make-break
  engagement (press/plug/pull/move/tap/stiff-turn). persH15 = sensor-agnostic predictor.
- Documented in docs/ACTION_CATEGORIES.md §3b. docs/predictability_opentouch.csv written on CRC.
- OPEN: (a) expand taxonomy to OT vocab (stir/serve/eat/flip/examine/carry/lower/align/type/touch/
  tighten/unscrew/tilt/tap/feel/inspect/switch/detach/attach/point/rest) + re-derive Axis B
  empirically; (b) then ActionSense + Force-Vision same recipe; (c) optional GPU forecasting to
  confirm probe on OpenTouch.

### ACTIONSENSE BUILT (2026-07-02) — dataset #3
- SCHEMA (from delpreto/ActionNet parsing_data): wearables HDF5 per subject-session,
  `<device>/<stream>/{data,time_s,time_str}`. Tactile = `tactile-glove-left`/`-right` ->
  `tactile_data/data` = (N,H,W) grids. Labels = `experiment-activities/activities/data` rows
  [Activity, Start/Stop, Valid, Notes] + time_s; pair Start->Stop for intervals, drop Valid in
  {Bad,Maybe}. 20 activity phrases (Peel/Slice/Spread/Open-close jar/Pour/Clean/Set/Stack/Load/
  Unload/Get/Clear). Subjects S00-S05 wore tactile; S06-S09 did NOT.
- Continuous recording -> SEGMENT by activity intervals, RESAMPLE each clip to 30 Hz (match
  EgoTouch/OpenTouch frame-based metrics), stack L+R -> (T,2,H,W), probe. (Resample = mild
  smoothing confound; acceptable for a within-dataset ranking; noted.)
- BUILT: scripts/crc/download_actionsense.sh (12 wearables URLs, S00-S05, curl, ~small);
  scripts/actionsense_predictability.py (--inspect + probe; segment/resample/stack; groups by
  raw activity / category / temporal-pattern). Taxonomy: added tableware verbs set/stack/load/
  unload/clear/get -> Organize/Arrange (B5). Kept spread->Fold/Cloth (EgoTouch spread_bed_sheet
  is genuinely cloth; ActionSense butter-spread mislabels but raw-activity label is unambiguous).
- All 20 labels map sanely (Peel/Slice->Cut B1, Pour->Pour B3, Clean->Wash/Clean B1, jar->
  Open/Close B4, tableware->Organize B5). compile + resample verified locally.
- NEXT (user on CRC): git pull; bash scripts/crc/download_actionsense.sh; probe --inspect (confirm
  tactile shape/Fs); then full probe. Then cross-dataset synthesis (EgoTouch+OpenTouch+ActionSense).

### ACTIONSENSE RESULT + THREE-DATASET SYNTHESIS (2026-07-03)
- DISK saga: ActionSense wearables HDF5 = 2-4 GB each (embed eye-video) → ~35 GB, exceeds 100 GB
  home (66 used) → repeated curl/truncation/ENOSPC. SOLVED via streaming driver
  scripts/crc/stream_actionsense.sh (download 1 file → probe → delete → next; --jsonl accumulate
  + --report-only aggregate). OpenTouch raw data got deleted along the way (kept its earlier CSV).
- ACTIONSENSE PROBE (299 clips, S00-S05, 32x32 2-hand, 6 Hz→30 Hz resample; persH1 & migration
  degenerate from upsampling — persH15/H30 + periodicity carry signal):
  - raw activity: Slice cucumber +2.7 / Pour +2.0 / Clear board +1.7 / Clean-plate-towel +1.7 /
    Peel +1.5 / Slice bread +1.4 ... bottom: Open/close jar -2.3/-2.9, Get/replace items -4.1.
  - category: Pour +2.6 > Cut +1.8 > Wash/Clean +0.1 > Fold/Cloth(spread) -0.5 > Organize -1.4 >
    Open/Close -2.6. temporal: B3 ramp +2.5 > B1 periodic +0.8 > B5 composite -1.2 > B4 trans -2.2.
  - Here a-priori Axis B WORKS (actions match canonical mechanics), unlike OpenTouch.
- SYNTHESIS (3 sensors) — TRAIT CONFIRMED: predictable = smooth/continuous/slowly-varying force
  (pour/slice/wipe/peel/stir/scoop); unpredictable = abrupt onset/make-break (open-close jar,
  press/click, plug, stiff turn). persH15 = sensor-agnostic predictor. REFINEMENT (from
  ActionSense): monotonic ramp (pour #1) > rhythmic cycle (slice) > sustained hold > transition —
  a cycle has force-reversal turning points; a pour doesn't. Category ranking is dataset-dependent;
  the TRAIT is stable → the durable answer + basis for user feedback (smooth actions have a
  scorable "correct" force profile). Written up in docs/ACTION_CATEGORIES.md §3c + §4.
- REMAINING: Force-Vision (4th dataset, press/hold/squeeze) optional; OpenTouch improved-taxonomy
  category view optional (needs 14 GB re-download); GPU per-category forecasting to confirm probe.

### NEW DIRECTION — GENERATIVE FORECASTER FOR SMOOTH ACTIONS (2026-07-03)
User directives: (1) DOCUMENT the study thoroughly → wrote docs/STUDY_SUMMARY.md. (2) DROP
EgoTouch going forward — usable hardware is the glove behind the 3 linked datasets (ActionSense/
OpenTouch/Force-Vision); EgoTouch = historical reference only. (3) TRAIN a GPU forecaster
(ConvLSTM family) on the predictable smooth-force actions (slice, wipe/clean, pour, peel) in
ActionSense. (4) BRAINSTORM the training framework in detail (generative framework? network? loss?
latent embedding? physical latent variables? why?).
- DESIGN DOC: docs/TACTILE_FORECAST_PLAN.md. Proposal = a PHYSICS-STRUCTURED LATENT WORLD MODEL:
  β-VAE encoder → low-dim latent with NAMED physical channels [total force F, center-of-pressure
  (x̄,ȳ), contact area A, patch orientation/eccentricity, motion phase (sinφ,cosφ), force-rate
  dF/dt] + small residual → ConvLSTM/GRU latent predictor (in LATENT space, reusing our ConvLSTM)
  → decoder. Phase 2: stochastic RSSM / latent diffusion. Rationale: small data (~300 clips) +
  interpretable latent needed for feedback. Loss = masked log-space recon + total-force + contact
  support(BCE/IoU) + physical-latent supervision + temporal smoothness(jerk) + spectral/phase
  (periodic subset) + β·KL. Small-data plan: shared model conditioned on action, self-supervised
  pretrain on the full continuous stream, heavy aug (flip/rotate/speed-warp), cross-glove transfer.
- OPEN QUESTIONS Q1-Q7 in the plan doc (latent form; deterministic-first vs RSSM; shared vs
  per-action; rate/horizon; use Xsens pose?; compute/caching; which actions in v1). AWAIT user
  input before implementing.
- TODO: cache segmented ActionSense smooth-action clips as small npz (avoid 30 GB re-download).

### v1 DECIDED + STATE EXTRACTOR BUILT (2026-07-03)
- Decisions (AskUserQuestion): actions = pour+slice; target = EXPLICIT physical state vector
  (Path A, user: "decide the latent variables like CoP/velocity, not learn them" → no VAE, no
  ConvLSTM); dynamics = GRU baseline THEN compare to structured ramp/oscillator; tactile-only.
- Clarified for user: "spatial residual map" = a learned feature grid for a neural decoder —
  dropped for v1 (we go fully explicit/analytic). ConvLSTM is for spatial grids, so a vector
  state ⇒ use GRU/Kalman, NOT ConvLSTM (explained the fork; user chose the vector path).
- BUILT: `src/tactile_forecast/physical_state.py` — analytic s(t): per-hand pressure moments
  [F,x̄,ȳ,sxx,syy,sxy] + derive() (area,θ,ecc,vx,vy,dF) + phase() (numpy Hilbert). Coords in
  [-1,1] (sensor-agnostic). UNIT-TESTED on synthetic: pour→F ramps/CoP fixed; slice→CoP
  oscillates, phase advances at exactly the injected freq.
- WIRED: `actionsense_predictability.py --extract-states DIR` saves state_N.npy (T,C,6) +
  manifest.jsonl (append across streamed files); `stream_actionsense.sh` now passes it so ONE
  re-stream produces the tiny state dataset (few MB) → rsync/commit, no more 30 GB re-downloads.
- NEXT: user re-streams once to build ~/actionsense/states/ → rsync to local → I build the GRU +
  structured forecasters (train on CPU locally). Then feedback demo.

### BASELINE-OFFSET BUG FOUND + FIXED (2026-07-03)
- First state dataset (299 clips, transferred to data/actionsense_states/) was DEGENERATE:
  real F ≈ 585,000 with ±0.5% wobble, CoP_x std ≈ 0.001 (no motion). CAUSE: ActionSense
  conductive-thread gloves have a large per-taxel DC baseline (~571/taxel, untared; that's the
  `tactile-calibration-scale` device's job) → total force dominated by offset, CoP pinned to
  center. (Also retro-explains ActionSense's tiny persH in the probe — static baseline inflates
  Var.) FIX: `physical_state.baseline_correct` subtracts per-taxel 5th-percentile-over-time before
  moments; verified on synthetic (recovers CoP std 0.354 = injected amplitude). clip_states
  baseline-corrects by default.
- Must RE-EXTRACT (saved moments can't be un-baselined). To make it the LAST CRC round, added
  `--save-clips-for "Pour,Slice"` → caches raw resampled (T,C,H,W) clips (float16) so all future
  preprocessing/forecasting is LOCAL. stream_actionsense.sh updated.
- ACTION: user re-streams once → transfer states/ (now ~200 MB incl. clip_N.npy) to
  data/actionsense_states/ → then build forecaster fully locally.
- DONE: corrected states transferred. Verified REAL signals: pour F ramps 950→9800 (was flat
  585k); slice CoP moves. 299 states + 70 clips local. clip_*.npy gitignored (200M); states kept.

### v1 FORECASTER BUILT + RESULT (2026-07-03) — runs fully local on CPU torch
- BUILT: `src/tactile_forecast/state_forecast.py` (data/windows/normalize + numpy baselines
  persistence/velocity/linfit + GRU seq2seq residual rollout) and
  `scripts/train_state_forecaster.py` (per-action, skill-vs-persistence per physical variable,
  all-12 vs core F+CoP summary, --downsample to native ~6 Hz).
- RESULT (pour, slice; downsample5→6Hz, t_in6/t_out12=1s→2s): GRU ≈ persistence, slightly BELOW
  on aggregate (all=-0.09..-1.1) and HIGH VARIANCE across seeds (pour 0_F once +0.40, once -0.13).
  velocity/linfit MUCH worse (-3..-19). Core F+CoP no robustly positive.
- INTERPRETATION (honest): this CONFIRMS the study's central principle at the physical-state level
  — smooth, slowly-varying signals are "predictable" precisely because PERSISTENCE predicts them
  well ⇒ little skill-over-persistence headroom. The smoothness that makes them predictable makes
  them hard to BEAT persistence on. (Same reason EgoTouch holds ≈ 0 skill.) Not a bug; a property.
- REFRAME (for the feedback GOAL): skill-vs-persistence is the WRONG target for smooth actions.
  Pivot to a NORMATIVE model: build the expected physical-state trajectory (mean±band, phase/DTW
  aligned across clips) per action; feedback = deviation of a user's F/CoP/dF-dt from the expert
  band ("force jerky", "rhythm irregular"). The extracted state is used DESCRIPTIVELY, not to beat
  persistence. DECISION PENDING with user: (a) pivot to normative feedback model; (b) push
  forecasting (pool pour+slice+peel+clean for 3-4x data, longer horizons, phase-explicit
  oscillator, regularization); (c) accept finding + write up.

### v2 SLOW/FAST + PROBABILISTIC — BREAKTHROUGH (2026-07-03)
User chose: separate slow+fast & model the FAST action component; probabilistic (mean+band).
- DIAGNOSIS confirmed: fast (high-pass) component of F/CoP decorrelates within ~2 s (autocorr
  ~0..-0.3) → persistence-of-fast is a WEAK baseline (headroom exists), and fast is 0.33-0.55 of
  slow amplitude (real signal). The slow grip/postural part is what made raw-state persistence
  unbeatable.
- BUILT `scripts/train_action_dynamics.py`: low/high-pass split (scipy butter, cut 0.4 Hz) of
  active-hand F,x,y → target = fast [F,x,y]; input += slow F + CoP velocity; action embedding;
  probabilistic GRU (mean+logvar), Gaussian NLL; 5-fold CV by trajectory; downsample 30→10 Hz,
  predict 0.5 s from 1 s.
- RESULT (5-fold): pooled(Pour,Slice,Peel,Clean) MEAN skill **+0.725**, band coverage@2sd **0.93**
  (ideal ~0.95 → well-calibrated!). Pour+Slice only +0.736. Per-target: F_fast +0.63-0.68,
  x_fast/y_fast +0.76-0.78. STABLE across folds (std 0.01-0.05) — v1's high variance gone.
- TAKEAWAYS: (1) the redesign (slow/fast + probabilistic), NOT extra data, drove the win (pooling
  ≈ pour+slice alone). (2) We now have a calibrated model of the expected fast action dynamics +
  uncertainty band → FEEDBACK-READY: score a user's fast F/CoP against the expert mean±band.
- NEXT OPTIONS: build the feedback/anomaly demo (deviation vs band); add phase/rhythm metric;
  per-hand (not just active); write up. Committed with the v2 code.

---

## COMPREHENSIVE SUMMARY (2026-07-06) — for explaining the work to other researchers

### A. Research question
Which *kind* of hand action is easiest to predict from its own past tactile signal, and — more
usefully — **what trait makes an action series predictable**, so a predictor can give a user
feedback to improve performance. Priors under test: standardized-procedure and repeatable/periodic
actions are easier.

### B. Datasets and what we did with each
Four tactile datasets; three processed with data, one from paper only.
1. **EgoTouch** (21×21 FPC grid, 2 hands, 30 Hz) — probed (23 verb categories) AND used to build the
   pixel forecaster (SimVP/ConvLSTM/ConvGRU). *Later deprecated* per user (not the target glove).
2. **OpenTouch** (arXiv 2512.16842; 16×16 FPC, 1 hand, 30 Hz) — probed 2,496 clips. Per-clip
   action + grip labels (GRASP taxonomy) in HDF5 + CSV.
3. **ActionSense** (NeurIPS'22; 32×32 conductive-thread, 2 hands, ~6 Hz) — probed 299 clips (S00-05)
   AND used for the physical-state forecaster (v1/v2). 20 kitchen activities as Start/Stop intervals.
4. **Force-Vision** (ICLR'24; STAG glove) — categorized from the paper only (press/hold/squeeze);
   NOT downloaded/probed.

### C. Processing pipeline per dataset
- **EgoTouch**: HF download (metadata + `pressure_grids.npz`, no video). Layout scene/task/traj.
  Tasks named `verb_object` → categorized by first-verb token. Pressure = (T,2,21,21), ~50% NaN
  structural sensor mask (zero-filled), log1p amplitude transform.
- **OpenTouch**: 26 HDF5 shards (~14 GB) + `final_annotations` CSVs via gdown. Each HDF5 =
  `data/<clip>/right_pressure` (T,16,16) + labels joined from per-scene CSV on `clip_id`
  ("<scene>::demo_N"). Free-text gerund actions normalized (pulling→pull) then verb-mapped.
- **ActionSense**: wearables HDF5 (2-4 GB each, embed EMG/Xsens/eye-video) → too big for the
  home quota, so a STREAMING driver downloads one file → processes → deletes → next.
  Tactile = `tactile-glove-{left,right}/tactile_data/data` (T,32,32). Activities from
  `experiment-activities/activities` rows [Activity,Start/Stop,Valid,Notes]; pair Start→Stop
  (drop Bad/Maybe) → intervals; slice tactile per interval; resample to 30 Hz; stack both gloves
  → (T,2,32,32). **Baseline correction** (per-taxel 5th-percentile subtraction) applied before any
  physical-state computation (see Problem P4).

### D. Scripts and their functions
- `scripts/categorize_actions.py` — assign each EgoTouch task to an action category (verb taxonomy);
  print per-category task/trajectory counts. (Classification only.)
- `scripts/predictability_by_category.py` — EgoTouch per-category **training-free probe**: load
  pressure, compute predictability metrics per trajectory, group by verb category AND
  temporal-pattern axis, rank by composite index; write CSV.
- `scripts/opentouch_predictability.py` — OpenTouch probe (HDF5 clips + CSV labels); `--inspect`
  schema mode + probe; group by temporal-pattern / mapped-category / raw-action / grip; CSV.
- `scripts/actionsense_predictability.py` — ActionSense probe: segment continuous tactile by
  activity intervals, resample, stack gloves, metrics; `--jsonl`/`--report-only` (streaming
  accumulate/aggregate); `--extract-states` (save physical-state trajectory per clip);
  `--save-clips-for` (cache raw clips).
- `scripts/crc/stream_actionsense.sh` — the download→probe→delete streaming driver (bounds disk to
  one file); accumulates per-clip records; final report + state extraction.
- `scripts/aggregate_results.py` — aggregate GPU pixel-forecaster runs by (model,scope,category);
  ranked per-category test skill.
- `src/tactile_forecast/train.py` — pixel forecaster trainer (SimVP/ConvLSTM/ConvGRU), LTO/LOTO CV,
  `--category` filter, skill-vs-persistence. (EgoTouch.)
- `scripts/train_state_forecaster.py` — **v1** physical-state forecaster (GRU vs baselines).
- `scripts/train_action_dynamics.py` — **v2** slow/fast probabilistic action-dynamics model.
Shared modules: `categories.py` (taxonomy), `predictability.py` (metrics), `physical_state.py`
(analytic state), `state_forecast.py` (v1 data/model).

### E. Algorithms and WHY we chose them
1. **Verb taxonomy categorization** (rule-based, first known verb token; gerund-normalized).
   *Why:* unify heterogeneous labels across datasets into ONE comparable category space + a
   temporal-pattern axis (B1 periodic … B5 composite), enabling cross-dataset comparison.
2. **Training-free predictability probe** — per clip: `persistence_nMSE@h` = MSE(y[t+h],y[t])/Var
   (decorrelation rate), `periodicity` = max total-force autocorr at lag 0.33-1.5 s,
   `contact_migration` = 1−IoU of active-taxel mask, composite `PI` = z(−persH15)+z(period)+z(−migr).
   *Why:* measure "how forecastable" WITHOUT training/GPU — fast, sensor-agnostic, and directly
   tests the periodicity/standardization priors. PI fuses the three physical axes.
3. **Pixel forecaster (SimVP/ConvLSTM/ConvGRU)**, skill vs persistence, LTO/LOTO.
   *Why:* standard tactile spatiotemporal forecasting; establishes REAL trained-model skill
   (not just the proxy) on EgoTouch, and a per-category comparison.
4. **Analytic physical-state extraction** — per hand per frame: 0th/1st/2nd pressure moments
   [F, CoP(x,y), spread(sxx,syy,sxy)]; derived area/orientation/velocity/dF-dt; Hilbert phase;
   per-taxel baseline subtraction; coords normalized to [-1,1].
   *Why:* a low-dimensional, fully interpretable state — data-efficient for small data AND directly
   usable for feedback (named physical variables a coach can talk about). User chose explicit
   variables over a learned latent.
5. **v1 GRU seq2seq on the raw state.** *Why:* a vector state calls for a vector sequence model
   (ConvLSTM is for spatial grids). RESULT: failed (≈ persistence).
6. **v2 slow/fast + probabilistic GRU** — low/high-pass split F/CoP; model the FAST action
   component; probabilistic head (mean+variance), Gaussian NLL; action embedding; k-fold CV.
   *Why:* the slow grip component is trivially persistent (killed v1); the fast component carries
   the stroke/pour dynamics and decorrelates within ~2 s (real headroom); probabilistic output
   yields the calibrated "expert band" feedback needs.

### F. Results
- **EgoTouch probe (1,929 clips):** easiest Cut(slice)+6.0, Take, Inflate, Spray, Wash/Clean;
  hardest Press/Click −6.7, Plug/Insert, Pinch, Grasp/Hold/Lift. Holds NOT trivially predictable.
- **EgoTouch pixel forecaster:** LTO (seen-object) +0.192 skill; LOTO (unseen) ≈ 0; broad
  pretraining lifts LOTO to +0.097.
- **OpenTouch probe (2,496):** easiest pour/serve/eat/stir/scoop/wipe; hardest turn(latch)/pull/
  move. `contact_migration≈0` (single-hand grasp never breaks contact) → degenerate there.
  A-priori temporal-pattern axis INVERTS vs EgoTouch (Problem P2).
- **ActionSense probe (299):** Pour +2.6 > Cut(slice/peel) +1.8 > Wash/Clean > Fold(spread) >
  Organize(tableware) > Open/Close(jar) −2.6. Cleanest confirmation; pattern axis works here.
- **HEADLINE (3 sensors):** predictable = **smooth, continuous, slowly-varying contact force**
  (pour/slice/wipe/peel/stir/scoop); unpredictable = **abrupt onset / make-or-break** (jar,
  press, plug, stiff turn). `persH15` is the sensor-agnostic predictor. Refinement: monotonic
  ramp (pour) > rhythmic cycle (slice) > hold > transition. Category ranking is dataset-dependent;
  the TRAIT is stable.
- **v1 forecaster (raw state):** GRU ≈ persistence (mean skill ~−0.1), HIGH variance. Confirms the
  "smooth ⇒ low skill-over-persistence headroom" principle at the state level.
- **v2 forecaster (slow/fast + probabilistic):** 5-fold CV mean skill **+0.725** (pooled) / **+0.736**
  (pour+slice) vs persistence-of-fast; per-target F +0.63-0.68, CoP +0.76-0.78; band coverage@2sd
  **0.93** (well-calibrated); STABLE across folds. Pooling ≈ pour+slice alone → the REPRESENTATION
  (slow/fast + probabilistic), not extra data, drove the win. → feedback-ready.

### G. Problems encountered (scientific/methodological; version-control issues excluded)
- **P1 Cross-sensor incomparability.** Four different glove geometries/rates → raw skill numbers
  are NOT comparable across datasets. *Fix:* rank WITHIN each dataset; compare across datasets only
  by the temporal-pattern axis and the qualitative trait.
- **P2 The a-priori temporal-pattern axis breaks across datasets.** The same verb behaves
  differently by context ("turning a stiff latch" in OpenTouch is an abrupt transition, not the
  rhythmic turn EgoTouch assumed), and many verbs were unmapped → dumped in "Other". *Fix:* expand
  the taxonomy; treat the pattern label as a-priori and let the measured periodicity decide;
  emphasize the trait over the category label.
- **P3 The "predictable" actions have little skill-over-persistence headroom.** The very smoothness
  that makes pour/slice predictable in absolute terms makes them near-perfectly predicted by
  persistence → a trained forecaster on the raw state can't beat it (v1 failed). This is the
  deepest finding, not a bug. *Fix/insight:* separate the persistent slow (grip) component and model
  the fast (action) component, which does have headroom (v2).
- **P4 Sensor DC baseline offset (ActionSense).** The conductive-thread glove is not tared: every
  taxel has a large resting value (~571/taxel) → total force ≈ constant (585,000 ± 0.5%) and CoP
  pinned to center — the first physical-state extraction was DEGENERATE (no motion visible). *Fix:*
  per-taxel 5th-percentile baseline subtraction before computing moments (validated on synthetic:
  recovers the true CoP oscillation). Also retro-explains the inflated (tiny) persH in the probe.
- **P5 Resampling artifact.** ActionSense native ~6 Hz upsampled to 30 Hz → adjacent frames are
  near-duplicates → persistence artificially strong at short horizons. *Fix:* forecast at native
  rate (downsample back to ~6-10 Hz).
- **P6 Noisy higher-order features dilute the metric.** The 2nd-moment shape terms (orientation/
  covariance) are unpredictable jitter and irrelevant to feedback, but equal-weighting dragged the
  mean skill negative. *Fix:* focus targets/metrics on the core feedback variables (F, CoP).
- **P7 Small data / high variance.** ~15-30 clips per activity, ~18 train trajectories per action →
  a single train/val split gave wildly unstable skill (+0.40 vs −0.13). *Fix:* k-fold CV
  (report mean±std); action pooling; probabilistic model; strong regularization.
- **P8 Disk/logistics (ActionSense).** Wearables files are 2-4 GB each (~35 GB total) vs a limited
  home quota → download failures/truncation. *Fix:* the streaming download→probe→delete driver +
  caching only the tiny states (and a small set of raw clips) locally so no dataset is re-downloaded.
- (Excluded per user: GitHub auth/divergent-branch/CRC-code-sync issues — real time sinks but
  not scientific.)

### H. One-paragraph narrative (for a researcher)
We built a sensor-agnostic, training-free probe to rank how forecastable each action's tactile
signal is, applied it across three tactile gloves, and found a stable, sensor-independent trait:
smooth continuous-force actions (pour, slice, wipe) are predictable, abrupt make/break actions
(jar, press) are not — but the very smoothness that makes them "predictable" means a naive
forecaster only matches persistence. Reducing each pressure field to interpretable physical
variables (force, center of pressure), then **separating the trivially-persistent grip from the
fast action component and modeling that component probabilistically**, yields a calibrated
forecaster (skill +0.73 over persistence, 93% band coverage) whose interpretable, bounded outputs
are exactly what is needed to give a user actionable feedback.

### PER-ACTION v2 COMPARISON → ACTION CHOICE (2026-07-06)
Ran v2 (slow/fast probabilistic, 5-fold) per single action. Fast-component skill / band coverage:
- Peel (n=30): +0.754 / 0.92 ; Slice (n=75): +0.738 / 0.92 ; Clean/wipe (n=60): +0.618 / 0.89 ;
  Pour (n=25): +0.610 / 0.81 (miscalibrated, small n).
RANKING: rhythmic repeated-stroke actions (peel, slice, wipe) > ramp (pour).
INSIGHT: this REVERSES the raw-signal trait ordering (which put pour/ramp top). Reason: for the
FAST component, rhythmic strokes have a clean oscillation (structured, predictable, calibratable),
whereas pour's fast part is unstructured tremor once the ramp is removed. Rhythmic actions also
have a well-defined "correct rhythm" → natural feedback template.
DECISION: target the REPETITIVE-STROKE family (slice + peel, then wipe) for the forecaster +
feedback demo; pour is the weakest fit despite being "most predictable" in the raw sense.
Horizon sweep (pooled): skill 0.74/0.68/0.62/0.58 at 0.5/1.0/1.5/2.0 s (gentle decay; 1 s a good
operating point). 0.73 "skill" is dimensionless (1-MSE/MSE_persistence); CoP in [-1,1] grid units
(not mm), force uncalibrated.

### REFACTOR: library / thin-CLI separation for the v2 forecaster (2026-07-07)
User: "plot scripts should only plot"; wanted clear structural repo. Done all 4:
1. NEW `src/tactile_forecast/action_dynamics.py` = LIBRARY (single source of truth): slow_fast,
   build_features, load_pooled, windows, split_train_test, Norm, ProbGRU, train(), evaluate(),
   forecast_clip(), save()/load() checkpoint. (Convention: src/ = importable nouns; scripts/ = verbs.)
2. `scripts/train_action_dynamics.py` slimmed to a CLI: k-fold CV report + train final on all clips
   + SAVE checkpoint to runs/ (gitignored). No model code. CV reproduces: Slice+Peel MEAN +0.770.
3. `scripts/plot_action_forecast.py` slimmed to plotting only: `--ckpt` loads a saved model (no
   training) OR default sweeps past-context (1/2/3/5/10s) training via the library. No model/train
   code; no more script->script import. Sweep skills reproduce (0.69/0.71/0.72...).
4. RENAMED probes to verbs: predictability_by_category.py->probe_egotouch.py,
   opentouch_predictability.py->probe_opentouch.py, actionsense_predictability.py->probe_actionsense.py
   (git mv, history kept). Fixed the one functional ref (stream_actionsense.sh PROBE path) and
   removed probe_egotouch's script->script import (now imports categorize from the package).
Verified: all compile; train CLI +0.770 + saves ckpt; plot --ckpt loads+plots (no train); plot
sweep reproduces. runs/ gitignored. Structure now: LIBRARY in src/tactile_forecast (action_dynamics,
predictability, physical_state, categories), thin CLIs in scripts/ (train_*, plot_*, probe_*, crc/).

### PAST-CONTEXT SWEEP: data-size confound + horizon note + TODO (2026-07-07)
- Training now SWEEPS past-context (scripts/train_action_dynamics.py --pasts 1,2,3,5,10; future
  1s = t_out 10). Full-quality result plateaus ~3s: 1s +0.69 / 2s +0.71 / 3s +0.72 / 5s +0.72 /
  10s +0.71 (reduced epochs25/folds3 demo). Saves a checkpoint per past (runs/ad_<acts>_p<p>s.pt).
- CONFOUND FOUND (user question): training data size is NOT equal across history lengths. In
  action_dynamics.windows(), `win = t_in + t_out` and the loop `range(0, T - win + 1, stride)` yields
  FEWER windows as t_in grows. Measured on Slice+Peel (t_out=10, stride=2): #windows 15,529 (1s) ->
  15,154 (2s) -> 14,779 (3s) -> 14,029 (5s) -> 12,154 (10s) — the 10s model trains on ~22% LESS data.
  Also clips with T < t_in+t_out are silently dropped (none here: min clip 114 >= 110 win, by luck).
  So the plateau/decline at long history is partly LESS DATA, not only decorrelation.
- TODO (fair comparison, LATER): shared-anchor mode — only forecast at positions a >= max(t_in),
  same anchors for every t_in, so all history lengths get IDENTICAL window count + identical future
  targets; only the depth of past differs. Re-run the sweep to see if the 3s sweet spot survives.
- HORIZON / RUN DIFFERENCES (clarification): the earlier CRC single run
  `train_action_dynamics.py --actions Slice,Peel` that gave MEAN +0.770 was the pre-sweep version =
  ONE config, 1s past, **0.5s** forecast (t_out=5), full epochs80/folds5. The new sweep default
  forecasts **1s** (t_out=10) -> harder -> the 1s-past row is +0.69, not +0.77. Same-machine reruns
  are reproducible (same code+data+seed); the local demo used reduced epochs25/folds3 (slightly
  different numbers, same trend) vs a full CRC/local run at epochs80/folds5.

### PLAN (approved 2026-07-08) — causal filter + raw-vs-highpass ablation + leakage checklist
User-approved changes (implementing now):
1. CAUSAL FILTER: action_dynamics.slow_fast filtfilt -> sosfilt (butter output='sos', forward-only).
   Rationale: filtfilt is non-causal (backward pass sees the future) -> the fast component leaks
   future info into both input and target for a forecasting task. sosfilt is causal. Also make
   velocity causal (np.gradient central-diff -> backward diff). Cost: startup transient -> cut
   first 5s (=50 frames @10Hz) per clip, in BOTH train and eval.
2. RAW-vs-HIGHPASS ablation (only the INPUT changes; target always fast [F,x,y]):
   input_mode='highpass' = [F_fast,x_fast,y_fast,F_slow,vx,vy] (current);
   input_mode='raw'      = [F,x,y,vx,vy] (no decomposition).
3. REPORT by every channel (F, CoP-x, CoP-y) x every history (1/2/3/5/10s) x per-forecast-step
   (+0.1..+1.0s) x each HAND (left=ch0, right=ch1, reported separately, not just active). Print
   history x channel tables per (input_mode, hand); write a full CSV with all breakdowns.
4. Pipeline order (confirmed, unchanged): raw field -> per-frame F + CoP -> causal high-pass ->
   z-score (train stats). z-score stays AFTER the filter.
5. LEAKAGE CHECKLIST: scripts/check_leakage.py (runnable, PASS/FAIL, run before every training) +
   docs/leakage_checklist.md. Six checks: (1) filter causal (impulse test), (2) norm stats
   train-only, (3) split by trajectory/no clip overlap, (4) input strictly before target,
   (5) baseline sees same past-only input, (6) pipeline order (CoP/force before filter; z-score
   train-only + consistent train/test).
FILES: action_dynamics.py (sosfilt, causal velocity, build_features input_mode+hand+warmup,
evaluate per-step), train_action_dynamics.py (--input-mode, --hands, per-step, CSV),
plot_action_forecast.py + plot_test_results.py (input_mode/hand passthrough), NEW check_leakage.py,
NEW docs/leakage_checklist.md. NO re-extraction needed (filter applied at load from committed states).
EXPERIMENT: rerun sweep raw vs highpass, both hands, on CRC -> compare.

### IMPLEMENTED: causal filter + ablation + leakage checklist — RESULTS (2026-07-08)
All 6 leakage checks PASS (scripts/check_leakage.py) after filtfilt->sosfilt + causal velocity.
Reduced demo (epochs15/folds3, pasts 1&3, both hands, raw&highpass) — full run needed to finalize:
- BIG FINDING: causal filter DROPS skill from ~+0.70 (old non-causal, epochs25) to ~+0.51-0.54
  (causal, epochs15). Part is fewer epochs, but the direction confirms filtfilt was LEAKING future
  info into the fast target and inflating skill. Full epochs80/folds5 run (CRC) will quantify the
  honest causal skill.
- RAW ~= HIGHPASS: raw-input vs highpass-input give nearly identical skill (left 0.510 vs 0.513;
  right 0.543 vs 0.540) -> the explicit slow/fast INPUT decomposition is UNNECESSARY; the model
  predicts the fast target equally well from raw signals. (Ablation answered.)
- HANDS: right hand slightly > left (~+0.54 vs ~+0.51) for Slice/Peel (right = dominant/tool hand).
- PER-STEP: skill RISES with forecast horizon (+0.11 @0.1s -> +0.59 @1.0s). Correct + expected:
  skill is vs persistence-of-fast, which is strong at 0.1s (autocorr ~0.8) but collapses by 1s
  (fast reverses), so the model's ADVANTAGE grows with horizon.
- Full per-(input_mode,hand,history,step,channel) breakdown -> docs/actionsense/action_dynamics_results.csv.
NEXT: run full matrix on CRC (python scripts/check_leakage.py && python scripts/train_action_dynamics.py
--actions Slice,Peel) for the honest causal numbers; compare raw vs highpass definitively.

### FULL CAUSAL RESULT — Slice (CRC job 1169778, 2026-07-09)
Honest causal run (sosfilt, warmup 5s, all 6 leakage checks PASS). NOTE: ran SLICE ONLY (45 clips)
— the qsub `-v ACTIONS="Slice,Peel"` comma was split by UGE, dropping Peel; rerun `qsub
scripts/crc/train_state_gpu.job` (no -v, defaults to Slice,Peel) to add Peel. Results ->
docs/actionsense/action_dynamics_results.csv (per input_mode x hand x history x forecast-step x channel).

Pooled MEAN skill vs persistence-of-fast (per history, from the .o tables):
             1s     2s     3s     5s    10s
raw/left   +0.402 +0.371 +0.313 +0.305 +0.259
raw/right  +0.401 +0.384 +0.385 +0.345 +0.294
hp /left   +0.376 +0.364 +0.335 +0.301 +0.232
hp /right  +0.392 +0.390 +0.371 +0.331 +0.314

FINDINGS (causal, Slice):
1. HONEST SKILL ~+0.40 at 1s history (was ~+0.70 with the leaky filtfilt) — the leak inflated by
   ~0.3. This is the number to report.
2. RAW ~= HIGHPASS confirmed at full scale: raw 0.40 vs highpass 0.38 (left), 0.40 vs 0.39 (right).
   The slow/fast INPUT decomposition does NOT help -> can simplify the model to raw input.
3. MORE HISTORY HURTS: skill DECLINES monotonically 1s->10s (e.g. raw/right 0.40->0.29). Opposite
   to the old non-causal (which plateaued). Confounded by fewer training windows for longer history
   (see fair-comparison TODO) — but honest pipeline shows no benefit from >1-2s of past.
4. RIGHT HAND > LEFT, esp. at long history (10s: right ~0.29-0.31 vs left ~0.23-0.26). Right =
   dominant/tool hand for slicing; its CoP-x (stroke) is the most predictable channel (x_skill ~0.45-0.48).
5. PER-STEP: skill starts NEGATIVE at +0.1s (persistence near-perfect that close; model slightly
   worse), crosses 0 by ~+0.2s, rises to ~+0.48 at +1.0s. Model only beats persistence at >0.2s lead.
6. CALIBRATION WORSE: coverage@2sd ~0.73-0.85 (was ~0.92 leaky), drops with history (0.83@1s->0.73@10s)
   -> bands overconfident on the honest (harder) task; a calibration fix / CRPS is warranted.

### FULL CAUSAL RESULT — Slice+Peel (CRC job 1170576, 2026-07-10)
Honest causal (sosfilt, warmup 5s, leakage checks pass), pooled Slice(45)+Peel(30)=75 clips.
docs/actionsense/action_dynamics_results.csv (per input_mode x hand x history x step x channel).

avg-over-steps MEAN skill by (mode, hand, history):
             1s     2s     3s     5s    10s
raw/left   +0.365 +0.334 +0.317 +0.304 +0.245
raw/right  +0.404 +0.409 +0.399 +0.364 +0.322
hp /left   +0.374 +0.348 +0.343 +0.296 +0.257
hp /right  +0.427 +0.408 +0.394 +0.369 +0.328

FINDINGS (consistent with Slice-only, adding Peel):
1. Honest skill ~+0.40 (right hand); RAW ~= HIGHPASS again (hp marginally > raw on right; ~tie on
   left) -> slow/fast INPUT decomposition still unnecessary. Can simplify to raw input.
2. RIGHT HAND >> LEFT, and the RIGHT-HAND CoP_x (stroke direction) is by far the most predictable
   channel: +0.47 @1s and still +0.435 @10s (vs F +0.20-0.37, CoP_y +0.33-0.39). The knife-stroke
   left-right motion is the dominant, most-forecastable signal.
3. HISTORY: right hand ~flat 1-3s (+0.40) then declines; left declines from 1s. ~1-3s past optimal
   (data-size confound still applies; fair-comparison TODO).
4. PER-STEP: rises with horizon (raw/right 3s: 0.19@0.1s -> 0.50@1.0s); +0.1s now positive for the
   right hand (Peel helps short-horizon).
5. COVERAGE ~0.76-0.86 (overconfident, drops with history) -> calibration fix next (post-hoc sigma
   scaling to hit ~0.95).

### CALIBRATION FIX implemented (2026-07-10)
Post-hoc sigma-scaling to fix overconfident bands (coverage was ~0.76-0.86 << 0.95 ideal).
- action_dynamics.calibrate_sigma(model, norm, val_clips, t_in, t_out, target=0.95): s =
  percentile(|Y-mu|/sd, 95)/2 -> scaling sd by s makes +/-2sd contain ~95%. Fit on a VAL set.
- evaluate(..., sigma_scale) applies it to coverage; forecast_clip(..., sigma_scale) to the bands.
- train_action_dynamics CV: per fold hold out a VAL subset of TRAIN, calibrate on it, report BOTH
  covRaw and covCal on the test fold; CSV gains coverage_raw + coverage_cal. Final checkpoints train
  on 85%, calibrate sigma on 15%, store sigma_scale in meta; plots apply it.
- VERIFIED (reduced raw/right): covRaw 0.90-0.91 -> covCal 0.95-0.96. On the low-coverage configs
  (~0.76) the scale is larger and still lifts to ~0.95. Skill unchanged (calibration only rescales
  uncertainty, not the mean).
- NEXT: rerun full matrix on CRC (qsub) to get calibrated coverage in the CSV.

### RIGOROUS CODE + RESULTS REVIEW (2026-07-10, on request: "is F prediction too good?")

Scope: full review of action_dynamics pipeline + docs/actionsense/action_dynamics_results.csv +
forecast_F/CoPx/CoPy figures. New diagnostic: scripts/tmp_diag_predictability.py (torch-free).

VERIFIED CORRECT (no leakage found):
- Causal filter (sosfilt) + backward-diff velocity; warmup trim (action_dynamics.py:39-51,77).
- Window construction: inputs strictly before targets, within-clip only (windows(), :100-113).
- Split by clip; norm stats from train clips only; sigma calibrated on val held out of train.
- forecast figures are honest: seeded from TRUE value at each 1 s anchor, then autoregressive
  (plot_forecast_overlay.py:37-41); test clips excluded from training (test_ids by clip, seed=1).
- check_leakage.py assertions are sound and match the code.

KEY NEW FINDING - THE SKILL BASELINE IS TOO WEAK; HEADLINE NUMBERS ARE INFLATED:
The fast target is ANTI-correlated with itself at 1 s lag (variance-weighted AC over the 75
Slice+Peel clips, both hands, ds=3/cut=0.4/warmup=5: F_fast rho(1.0s) = -0.19, x = -0.18,
y = -0.14). Persistence MSE at 1 s = 2(1-rho)*var ~ 2.4*var, so trivial baselines score high
skill-vs-persistence:
  - predict CONSTANT ZERO @1.0s: F +0.59, x +0.58, y +0.56  (>= GRU's ~0.50-0.54 in the CSV!)
  - damped persistence (best scalar a*last, a ~ -0.2): +0.57..+0.61 @1.0s
  - linear ridge (past 1 s of 3 channels): +0.62 @1.0s  (beats the GRU at the far step)
Per-step zero-baseline skill (mean-channel, all clips): -3.0 @0.1s, -0.38 @0.2s, +0.14 @0.3s,
+0.34 @0.4s, +0.44 @0.5s ... +0.57 @1.0s. The GRU beats BOTH trivial baselines only in the
~0.2-0.5 s band; by 0.6 s+ shrink-to-zero matches/exceeds it. The avg-over-steps "+0.40 honest
skill" headline mostly reflects (a) GRU ~ persistence at short steps where zero is terrible and
(b) GRU ~ shrink-to-zero at long steps where persistence is terrible. Conclusions like
"right-hand CoP-x most predictable, +0.47" must be re-checked against the zero/damped baselines
(x @1.0s zero-baseline ~ +0.58 > CSV +0.54).
ACTION: add zero + damped-persistence + linear-ridge baselines to evaluate() and report skill
vs the STRONGEST trivial baseline per step.

WHY F LOOKS "TOO GOOD" IN forecast_F.png (mechanism, not leakage):
1. Re-anchoring: every 1 s segment restarts from the TRUE last value; errors never compound
   beyond 10 frames. The eye reads 50 concatenated 1 s forecasts as one great long forecast.
2. F_fast is the smoothest, highest-SNR channel: F = sum over all taxels (spatial averaging),
   spectral centroid 0.56 Hz => typical period ~1.8 s, so a 1 s forecast is ~half a period of a
   smooth quasi-periodic stroke cycle seeded from truth. AC(0.1s)=0.91.
3. Causal-filter group delay near the 0.4 Hz cutoff leaks lagged SLOW trend into the "fast"
   target - an extra smooth predictable component (affects target definition, not causality).
4. MSE/NLL training => amplitude shrinkage toward 0 at uncertain times; since the target
   oscillates around 0, a shrunk forecast still LOOKS close. Visible in the figures (orange
   amplitude < black).
Why CoP looks worse: CoP = moment ratio (divide by F) => noise amplified at light contact,
heavy-tailed spikes that an MSE model rightly ignores; centroid 0.67-0.74 Hz (faster content).
So: F prediction is NOT suspiciously good - quantitatively it is no better than trivial
shrinkage at the 1 s step; the visual impression comes from 1-3 above.

OTHER ISSUES (secondary):
- ACAUSAL PREPROCESSING: physical_state.baseline_correct() subtracts the per-taxel 5th
  percentile over the WHOLE clip (physical_state.py:68) - uses future frames. Per-clip constant
  => negligible for the high-passed F target, but changes CoP nonlinearly and is not deployable.
  Fix: percentile over the first N seconds or a running percentile.
- hand="active" (action_dynamics.py:63-64) picks the hand from the WHOLE-clip mean force
  (future info). Default in plot_action_forecast.py only; CSV runs use explicit left/right. OK
  offline, flag for any online claim.
- STALE CSV: docs/actionsense/action_dynamics_results.csv header has `coverage` (one col) but the current
  train_action_dynamics.py writes coverage_raw+coverage_cal - the CSV predates the calibration
  fix; regenerate.
- NO SUBJECT ID in the manifest (probe_actionsense.py:234-237): clip-level split mixes
  subjects/sessions => results = within-corpus generalization, not new-user.
- POOLED MSE in raw sensor units => skill dominated by high-amplitude clips (persistence MSE
  ~1.8e6 a.u.^2). Report per-clip skill median/IQR too.
- CSV writes fold MEANS only; cross_validate has per-fold arrays - add +/- std.
- First-step F skill negative (left): decoder gets y_last yet does worse than copying it =>
  parameterize the decoder as residual from y_last (predict delta).
- Fragile-but-correct: plot_forecast_overlay derives test_ids from the hand=left clip list and
  reuses them for hand=right; only safe because load_pooled ordering/length filter is
  hand-independent. Add an assert.

CONCLUSION for the user question: no implementation bug/leakage makes F "too good"; the figures
are honest but flattering (1 s re-anchoring). The real problem is the opposite direction: the
skill-vs-persistence metric OVERSTATES the model because persistence-of-fast is anti-correlated
at 1 s. The model has genuine (small) value only at ~0.2-0.5 s lead.

---

## Session (2026-07-13) — Method clarifications, horizon plot, two open design decisions

### User's 7 questions — answers (with code evidence)
1. **Prediction method / "0.1 s steps" / aggregation.** Clarified a conflation:
   - The 1/2/3/5/10 s are **five independent models** (different past-context), NOT aggregated.
     Panel (a) compares them; only one history is used per model.
   - The 0.1 s steps are how **one** model emits its 1 s: an **autoregressive** seq2seq GRU
     decoder rolls out t_out=10 steps, feeding its OWN prediction back
     (`action_dynamics.py:156-162`, key line 161 `inp = mu.unsqueeze(1)`).
   - User wants **one-shot / direct** multi-horizon (single forward pass emits all 10 frames,
     no feed-back). This is practical (swap decoder for a t_out*3 head), avoids rollout error
     compounding. **OPEN DECISION #1: switch to one-shot direct?** (would re-run sweep + figures).
2. **Skill.** `skill = 1 - MSE_model/MSE_persistence` (`action_dynamics.py:210-212`). Persistence =
   last observed value repeated over all 10 future steps. 0 = tie, +1 = perfect, <0 = worse.
3. **Raw WAS trained.** The sweep trains both input_modes every run (`train_action_dynamics.py`
   default `--input-modes raw,highpass`, driven by `train_state_gpu.job:34`). CSV has raw+hp rows;
   raw/highpass only changes the INPUT, both predict the same high-pass target.
4. **Horizon plot delivered** — `scripts/plot_horizon.py` -> `docs/actionsense/horizon_highpass.png`.
   Calibrated highpass, right hand, 3 s history -> 1 s ahead on test clip 6. Per channel shows:
   history window consumed (grey), true future (black), forecast mean+calibrated +/-2sigma (blue),
   persistence (dashed). sigma_scale=1.966. Forecast bends toward truth (F, CoP-x); persistence flat.
   NOTE: uses the CURRENT autoregressive method; will be re-rendered if we adopt one-shot.
5. **Persistence baseline — meaningfulness.** For the zero-mean FAST target, persistence is a WEAK
   floor (last fast value ~0, easy to beat), which flatters skill. **OPEN DECISION #2: add stronger
   baselines** — zero/mean baseline and AR(1)/linear extrapolation — and report skill vs all three.
6. **Input = tactile-only.** Confirmed: only the tactile pressure map -> [F, CoP_x, CoP_y]
   (`action_dynamics.py:67`). NO EMG (Myo), NO Xsens IMU/motion, though ActionSense records them.
7. Full multi-day documentation pass to be done at session end (this entry is the running log).

### Open questions awaiting user
- **#1** Switch model from autoregressive rollout to one-shot direct 1 s forecast?
- **#2** Add zero and AR(1) baselines alongside persistence?

### Commits this session
- `195fc70` regenerate results_summary with calibrated coverage (dashed ~0.95 vs raw ~0.80)
- horizon plot + script (this session)

### PLAN (2026-07-13) — one-shot vs autoregressive + AR(1) baseline  [awaiting resolution]
User decisions: (1) build BOTH decoders and compare; (5) add AR(1)/linear-extrapolation baseline.

Planned changes (NOT yet implemented — rule 5):
1. `action_dynamics.py`: add a `decoder` mode to the model.
   - `autoregressive` (current): decoder GRU feeds its own prediction back (t_out steps).
   - `oneshot` (new): single forward pass emits all t_out*3 means + logvars directly
     (direct head from encoder hidden state; no feed-back -> no rollout error compounding).
2. `evaluate()`: add an AR(1) baseline forecast; report skill vs persistence AND vs AR(1).
3. `train_action_dynamics.py`: add `decoder` to the sweep (doubles configs); add `*_ar1` skill
   columns; keep persistence columns.
4. `plot_results_summary.py` / `plot_horizon.py`: show both decoders / both baselines.
5. Re-run sweep, regenerate figures.

OPEN QUESTIONS (need user answers before coding):
- Q1 One-shot head: direct MLP from encoder hidden (recommended) vs non-autoregressive GRU
  with a fixed input token?
- Q2 AR(1) coefficient phi: estimated per-channel on the TRAIN set (stable, recommended) vs
  per-window from each clip's own history (adaptive)? Or do you want plain linear least-squares
  slope extrapolation instead of AR(1)?
- Q3 Report skill vs BOTH persistence and AR(1) (recommended) vs REPLACE persistence with AR(1)?
- Q4 Re-run where: local (sweep now doubles) vs CRC batch job?

---

## Session (2026-07-14) — COLD-START ONBOARDING SNAPSHOT (read this to catch up fully)

Purpose: a self-contained state-of-the-project dump. A fresh Claude that reads this section
(plus the COMPREHENSIVE SUMMARY at ~line 687) should know essentially everything we know.

### 0. One-paragraph orientation
We forecast the near-future TACTILE dynamics of the hand during smooth manipulation, to later give
real-time feedback. Data = ActionSense conductive-thread gloves (32x32 taxels/hand, two hands).
We picked the smoothest, most continuous-force actions — SLICE (cutting) and PEEL — and train a
small probabilistic GRU to predict the next 1 s of the hand's tactile "physical state" from a few
seconds of history. Input is TACTILE ONLY (no EMG/Myo, no Xsens IMU). We report skill vs a
persistence baseline, with calibrated uncertainty bands.

### 1. Data inventory (Slice + Peel) — computed 2026-07-14 from data/actionsense_states/manifest.jsonl
RAW (full recording length, before warmup cut):
  Slice: 45 clips, 1705 s (~28.4 min), mean 37.9 s, min 11.3 s, max 220.1 s
  Peel : 30 clips, 1536 s (~25.6 min), mean 51.2 s, min 31.5 s, max  71.4 s
  TOTAL: 75 clips/trials, 3241 s (~54 min)
Trial breakdown (15 reps each): Slice cucumber/potato/bread; Peel cucumber/potato (5 dishes x 15).
USABLE after 5 s warmup cut/clip: Slice ~1480 s, Peel ~1386 s, ~2866 s (~48 min).
Each clip is used for BOTH hands separately -> 150 hand-trajectories. Downsample ds=3 -> 10 Hz
(changes sample count, not seconds). CAVEAT: shortest Slice clip is 11.3 s, so 10 s-history configs
drop the short clips -> the long-history skill rests on fewer/longer trials (training-size confound).

### 2. End-to-end forecasting pipeline (ORDER MATTERS)
(a) Upstream (already done, stored in state_N.npy): 32x32 taxel pressure map -> baseline-corrected
    (per-taxel 5th-percentile subtraction, fixes the untared-glove DC offset) -> per-hand physical
    MOMENTS [F (total force), CoP_x, CoP_y, sxx, syy, sxy].
(b) build_features (action_dynamics.py:54): read F/CoP moment channels FIRST, THEN causal high-pass
    them (slow_fast, butter+sosfilt, CAUSAL). Order = MOMENTS-THEN-HIGHPASS (NOT highpass taxels
    then moments — that would make CoP, a ratio, blow up when the fast denominator crosses 0).
    - target = fast [F_fast, x_fast, y_fast] (always).
    - input_mode highpass -> [F_fast,x_fast,y_fast,F_slow,vx,vy]; raw -> [F,x,y,vx,vy].
    - velocity vx,vy = causal backward difference. warmup_sec=5 s dropped (filter transient).
(c) windows (line 100): sliding X=feat[s:s+t_in], Y=targ[s+t_in:s+t_in+t_out]; input strictly
    before target (no leakage). fps=10 Hz -> t_in = history_s*10, t_out = 1 s = 10 frames.
(d) Model ProbGRU (line ~140): encoder GRU -> decoder GRU rolled out t_out steps, AUTOREGRESSIVE
    (line 161 inp = mu.unsqueeze(1) feeds its own prediction back), + action embedding, mu/logvar
    heads -> Gaussian per step. Loss = Gaussian NLL.
(e) evaluate/_predict (line 186): skill = 1 - MSE_model/MSE_persistence per channel & per step;
    coverage@2sd = fraction of truth inside mu +/- 2*sigma_scale*sd.
(f) calibrate_sigma (line 197): post-hoc scalar so coverage -> ~0.95 on a validation slice.

### 3. THREE code paths & the re-anchoring insight (critical for reading the figures)
There are three places a forecast is produced; only the OVERLAY PLOT re-anchors:
  Path A - the model (action_dynamics.py forward): rolls out 1 s from ONE anchor using its OWN
           predictions; never sees ground truth after the anchor.
  Path B - the skill metric (_predict): one model call per window, full 1 s rollout, NO mid-forecast
           refresh -> the reported skill numbers are HONEST 1-second-ahead.
  Path C - the overlay figure (plot_forecast_overlay.py:37-41): tiles the whole clip into
           consecutive 1 s blocks, RE-ANCHORS each block to a fresh GROUND-TRUTH window + true seed
           every 1 s. This makes forecast_F/CoPx/CoPy.png visually HUG the real curve (error can't
           accumulate past 1 s) — a flattering VISUAL only; it does NOT change the skill numbers.
  => docs/actionsense/horizon_highpass.png (single anchor, Path A/B) is the HONEST picture; forecast_*.png is
     flattering. This resolved the "why do these plots look so different" question.

### 4. Results — skill per config (calibrated CSV docs/actionsense/action_dynamics_results.csv)
Mean skill over the 10 forecast steps (0.1..1.0 s); skill = 1 - MSE_model/MSE_persistence:
  mode/hand/hist    F     CoP-x  CoP-y  mean   cov
  highpass/right/1s 0.394 0.455  0.390  0.413  0.948   <- best
  raw/right/1s      0.378 0.461  0.392  0.410  0.942   <- tied (raw ~= highpass)
  highpass/left/1s  0.355 0.350  0.362  0.356  0.949
  raw/left/1s       0.318 0.346  0.351  0.338  0.948
  ... skill DROPS monotonically 1s->10s history for every config (e.g. right/highpass 0.413->0.310).
Takeaways: (i) raw ~= highpass (input decomposition buys nothing); (ii) right hand > left (+0.05-0.07);
(iii) CoP-x on the right hand is the standout channel (0.42-0.46, barely decays with history — the
stroke direction); (iv) more history HURTS (causal filter: old context = noise); (v) coverage
~0.94-0.95 everywhere (calibrated). Per-step: skill RISES with lead time (persistence degrades faster
than the model). Full 200-row per-step table is in the CSV.
CAVEAT: skill is vs PERSISTENCE, a WEAK floor for the zero-mean fast target -> absolute levels are
flattered; the RANKING should survive a tougher baseline but magnitudes will drop. (-> AR(1) plan.)

### 5. Calibration status
Coverage@2sd was overconfident (~0.80) -> post-hoc sigma-scaling lifted it to ~0.947 across the
whole matrix, skill UNCHANGED (calibration rescales the band, not the mean). Two CSVs kept:
  docs/actionsense/action_dynamics_results.csv        = calibrated (coverage_raw + coverage_cal)
  docs/actionsense/action_dynamics_results_precal.csv = pre-calibration (recovered from git)
results_summary.png panel (d): solid=raw coverage (~0.80), dashed=calibrated (~0.95), red=ideal.

### 6. File map (what each script is)
  src/tactile_forecast/action_dynamics.py  = THE library (single source of truth): slow_fast,
      build_features, load_pooled, windows, Norm, ProbGRU, train, evaluate, calibrate_sigma,
      forecast_clip, save/load. Plot/CLI scripts import from here; they must not redefine logic.
  scripts/train_action_dynamics.py = sweep CLI (input_mode x hand x history), k-fold CV, writes CSV.
  scripts/check_leakage.py         = 6 leakage checks; run before every training (job aborts if fail).
  scripts/plot_results_summary.py  = 4-panel summary from the CSV (no training).
  scripts/plot_forecast_overlay.py = whole-clip rolling overlay (Path C, re-anchored; flattering).
  scripts/plot_horizon.py          = NEW: single-anchor honest view (history + 1 s forecast + band +
      persistence + truth), per channel -> docs/actionsense/horizon_highpass.png.
  scripts/plot_test_results.py     = predicted-vs-true scatter per channel.
  scripts/probe_*.py               = per-dataset predictability probes (categorization phase).
  scripts/crc/train_state_gpu.job  = UGE batch job (git pull; check_leakage; run sweep; writes to
      runs/ which is gitignored to avoid pull collisions). CRC netid jhao3, conda env tactile.
  data/actionsense_states/         = state_N.npy (committed) + manifest.jsonl; clip_*.npy gitignored.

### 7. This session's Q&A (2026-07-13..14) — condensed
  - Prediction method: 5 independent history models (NOT aggregated); each emits 1 s via
    autoregressive rollout. (See Session 2026-07-13 entry #1.)
  - Skill definition: 1 - MSE_model/MSE_persistence (#2).
  - Raw WAS trained; raw/highpass only changes input, same target (#3).
  - Horizon plot delivered (#4). Persistence is a weak baseline (#5). Tactile-only input (#6).
  - "Why do forecast_*.png and horizon look so different?" -> the three-paths / re-anchoring insight
    (Section 3 above).
  - "F/CoP first or highpass first?" -> MOMENTS-THEN-HIGHPASS (Section 2b).
  - "How much Slice/Peel data?" -> Section 1.
  - "Skill per entry?" -> Section 4 table.

### 8. CONFIRMED decisions (this session) vs STILL-OPEN questions
CONFIRMED by user:
  - Build BOTH decoders (autoregressive + one-shot direct) and compare.
  - Add an AR(1)/linear-extrapolation baseline (stronger than persistence).
STILL OPEN (user has NOT answered; do not code until resolved — rule 5):
  - Q1 One-shot head design: direct MLP from encoder hidden (recommended) vs non-autoregressive GRU
    with a fixed input token.
  - Q2 AR(1) coefficient phi: per-channel on TRAIN (stable, recommended) vs per-window (adaptive);
    or plain linear least-squares slope extrapolation instead of AR(1).
  - Q3 Report skill vs BOTH persistence and AR(1) (recommended) vs REPLACE persistence.
  - Q4 Re-run location: CRC batch (recommended) vs local (sweep doubles with 2 decoders).
PLAN for the implementation is at ~line 1112 (PLAN 2026-07-13).

### 9. Previously-offered but NOT-yet-requested next steps (backlog)
  - Fair-comparison history run (equal window counts across history lengths).
  - Add calibrated +/-2sigma bands to the overlay figures.
  - Simplify model to raw input only (since raw ~= highpass).
  - Feedback demo on the calibrated right-hand CoP-x model.

### 10. Guardrails / gotchas learned (do not repeat)
  - CRC: git pull + grep coverage_cal BEFORE every qsub (else the job runs stale code).
  - Job writes CSV to runs/ (gitignored) to avoid overwriting the tracked docs/ CSV on pull.
  - filtfilt is NON-CAUSAL (leaks future) -> we use sosfilt (causal). Never reintroduce filtfilt.
  - Action matching uses label.startswith(action) (substring match wrongly pooled "bread slice").
  - Windows local shell = PowerShell (.venv\Scripts\python.exe); CRC = bash (plain python).

---

## Session (2026-07-16) — PLAN: frozen evaluation harness + classical baselines  [AWAITING RESOLUTION]

User task (also written into CLAUDE.md): build a FROZEN eval harness + 3 classical baselines
(persistence, seasonal-naive, AR) for tactile forecasting. Do NOT touch/retrain the probGRU.
Hard constraints: causal-only filtering; no cross-split leakage (fit on TRAIN, select on VAL, touch
TEST once); global train-derived normalization; CoP masking by force threshold; target-time indexing.
SE: single YAML config w/ hash, deterministic (identical tables across runs), pytest on synthetic
signals, modular (metrics/masking/baselines/evaluate/config/tests).

### Repo grounding (verified 2026-07-16) — 5 spec-vs-repo contradictions
A. RATE: spec "~15 Hz, 15 steps" but manifest fps=30 (all 299 clips), pipeline ds=3 -> 10 Hz
   effective -> 1 s = 10 steps. (Whether true hardware rate is 30 Hz is a separate open uncertainty.)
B. TARGET: spec "6-dim raw [F,CoPx,CoPy] x both hands"; probGRU actually predicts 3-dim HIGH-PASS
   (fast) [F_fast,x_fast,y_fast] for ONE hand (per-hand models). action_dynamics.py:28.
C. COLLISION: src/tactile_forecast/{baselines.py, eval.py} already exist but belong to the SEPARATE
   PIXEL-map forecaster (operate on (B,t_in,C,H,W) images; LTO/LOTO/grasp). Must NOT overwrite them.
D. DEPS: pytest MISSING, statsmodels MISSING (yaml/torch/scipy/numpy OK). Plan: pip install pytest;
   implement AR with numpy (Yule-Walker/OLS), no statsmodels dependency.
E. SPLIT: only 2-way split_train_test (seed=1, frac=0.25, by-clip). No VAL, no splits.json. Need a
   FROZEN 3-way split. Existing probGRU results used the 2-way test set (seed=1).
GOOD: causal-clean already (only filtfilt is a comment saying not to use it); Norm is global/
train-derived. Constraints 1 & 3 already satisfied upstream.

### Proposed design (NOT implemented yet)
- New package `src/tactile_forecast/eval_harness/` (avoids the pixel eval.py/baselines.py collision):
  metrics.py, masking.py, splits.py, baselines/{persistence,seasonal,ar}.py, evaluate.py, __init__.py
- `configs/eval_harness.yaml` (repo already uses configs/*.yaml): horizon, history, mask_threshold_pct,
  ar_orders, seasonal_period_range, ds/fps, paths, split_file. evaluate records sha256(config).
- `tests/` (new): pytest synthetic tests (sine seasonal, AR(2) recovery, masking, causality).
- Frozen split -> `data/actionsense_states/splits.json` (train/val/test lists of clip idx).
- Metrics indexed by TARGET time (t+h) with (t,h) metadata; documented in module docstring.
- Masking: one function; frame masked for CoP metric iff RAW total force < train per-hand 5th pct;
  force channels never masked.
- Determinism: fixed seeds; assert two runs produce byte-identical table.

### OPEN QUESTIONS (blocking — need answers before coding)
- Q1 TARGET: freeze harness on (a) 6-dim RAW [F,CoPx,CoPy]x2 hands [matches spec text; probGRU not
  scorable until re-scoped], (b) 3-dim FAST 1-hand [matches current probGRU], or (c) configurable?
- Q2 RATE/HORIZON: confirm 10 Hz -> 1 s = 10 steps (spec's "15" is wrong for this repo)?
- Q3 SPLIT: keep the existing seed=1 TEST set unchanged and carve VAL out of its TRAIN (past probGRU
  test numbers stay comparable), vs define a brand-new frozen 3-way split?
- Q4 PLACEMENT/DEPS: new `eval_harness/` package (don't touch pixel eval.py/baselines.py) + pip
  install pytest + AR via numpy (no statsmodels) — OK?
- Q5 (minor) MASK KEY: mask CoP by RAW total force even when the target is the fast component
  (recommended — masking reflects physical contact, not the zero-mean fast signal). OK?

### RESOLUTION + IMPLEMENTATION (2026-07-16) — frozen eval harness BUILT

User answers to the 4 open questions: Q1 target = 6-dim RAW both-hands; Q2 rate = 10 Hz / 10
steps; Q3 split = fresh 3-way 60/20/20 by recording, stratified by (action,object); Q4 = new
`eval_harness/` package + pip install pytest + AR via numpy (no statsmodels). ("What are steps"
and "how is the current split" were answered in-chat: a step = one predicted sample; 1 s horizon
= 10 samples at 10 Hz; old split was 2-way 75/25 seed=1, no val.)

BUILT (all imported from ONE place; harness is frozen):
- configs/eval_harness.yaml            single source of truth; sha256 stamped into results.
- src/tactile_forecast/eval_harness/
    config.py     Config + config_hash (sha256 of the yaml bytes).
    splits.py     stratified 60/20/20 by recording -> data/actionsense_states/splits.json
                  (COMMITTED; n=75 -> train 45 / val 15 / test 15). By recording => both hands
                  same split. Only partitions indices, never reads signals -> cannot leak.
    dataset.py    RAW 6-dim target [F_L,CoPx_L,CoPy_L,F_R,CoPx_R,CoPy_R] from state_N.npy moments
                  0..2 of each hand, downsampled x3 -> 10 Hz. Global TRAIN z-score (Norm). Per-hand
                  force thresholds = TRAIN 5th pct.
    masking.py    ONE CoP mask: a CoP target frame is dropped iff that hand's RAW force < TRAIN
                  threshold; force channels never masked. Keys off the TARGET-frame force.
    metrics.py    masked per-channel / per-horizon MSE, MAE, skill=1-MSE/MSE_persistence, nRMSE.
    baselines/    base.Baseline (predict(hist,H) reads only hist=past<=t) + predict_series
                  (rolling-origin, structurally causal). persistence, seasonal (causal
                  same-phase-k-periods-back; period ranked on TRAIN autocorr, selected on VAL),
                  ar (per-channel OLS AR(p) on normalized signal, fit TRAIN, order selected on
                  VAL, recursive causal forecast; numpy lstsq, no statsmodels).
    evaluate.py   fit(TRAIN)->select(VAL)->score(TEST once); determinism assert (two runs
                  identical); writes docs/actionsense/harness_baselines.csv with config_hash.
- tests/test_harness.py (pytest, 6 tests, ALL PASS): seasonal exact on sine; seasonal selects
  true period; persistence MSE matches analytic 1-cos(2*pi*h/T); AR(2) coeff recovery + beats
  persistence; CoP masking excludes low-force frames from CoP but keeps force; causality (future
  corruption never changes a forecast issued at t) + non-vacuous past-dependence sanity.

RESULTS (docs/actionsense/harness_baselines.csv, config_hash ccb0d9c5, TEST split, mean skill vs persistence):
  persistence  nRMSE 0.517  (reference, skill 0)
  seasonal(3)  nRMSE 0.556  skill -0.13..-0.18  -> WORSE than persistence: raw aggregate force/CoP
               is NOT cleanly periodic at a single global period; copying a period back loses to
               copying the last value. (Seasonal picked the min period 3, near-persistence.)
  ar(16)       nRMSE 0.470  skill +0.12..+0.23  -> BEATS persistence; right-hand CoP-x highest
               (+0.23), echoing the probGRU finding that right-hand CoP is most predictable.
Constraints honored: causal-only (no filtfilt; baselines structurally causal + tested); no
leakage (fit TRAIN / select VAL / TEST once); global TRAIN normalization; CoP masking; target-time
indexing. pytest + the harness both green. Deterministic (assert passes).

CONTRADICTIONS SURFACED (did not silently work around): (A) spec 15 Hz/15 steps vs repo 10 Hz/10
steps; (B) spec 6-dim raw both-hands vs probGRU 3-dim fast one-hand; (C) existing pixel
eval.py/baselines.py belong to a DIFFERENT stack (left untouched); (D) pytest+statsmodels missing;
(E) only a 2-way split existed. All resolved with the user before coding.

NOTE: this harness scores the RAW 6-dim target. The current probGRU predicts the 3-dim FAST 1-hand
target, so it is NOT directly scorable here yet (by design: task said don't touch it). To later
plug the probGRU in, it must be re-scoped to predict raw 6-dim both-hands, then call the same
metrics/masking/splits.

### Step 0 (2026-07-16) — exploration for the REFINED harness spec + contradictions

Explored existing preprocessing (do-not-duplicate; reuse loaders):
(a) SEGMENTATION: probe_actionsense.py cuts per-subject ActionSense HDF5 into per-activity clips
    by Start/Stop markers, resamples each to 30 Hz, -> PS.clip_states -> state_idx.npy + manifest.
    Files processed S00..S05 in order; idx runs across subjects but SUBJECT IS NOT RECORDED.
(b) SPLIT: old split_train_test (2-way 75/25 seed=1, no val); NEW frozen eval_harness/splits.py
    -> splits.json (3-way 60/20/20 by recording, stratified action x object). Treat as FROZEN.
(c) FILTERING: raw 6-dim target has NO temporal filter, BUT physical_state.baseline_correct
    subtracts a WHOLE-CLIP 5th-pct DC offset per taxel (physical_state.py:68, percentile over the
    full time axis incl. future) => NON-CAUSAL upstream offset. Reported per instructions.
(d) NORMALIZATION: upstream baseline_correct (DC); harness Norm = global per-channel z-score TRAIN
    only. No per-window normalization.

CONTRADICTIONS / BLOCKERS (surfaced, not worked around):
  1. "per subject x activity" impossible as written: manifest has activity, NO subject. Subject only
     recoverable by re-streaming 88GB HDF5 on CRC (upstream change + data regen). Available now:
     activity x object (5 groups). 
  2. baseline_correct NON-CAUSAL (whole-clip percentile). Making it causal needs re-streaming.
  3. Deps: statsmodels MISSING (spec wants AutoReg; escape clause allows numpy), pyarrow/fastparquet
     MISSING (parquet), pandas OK.

RETRAIN? No GRU retraining in any case (harness never touches it). Baseline fits are cheap/local.
ONLY heavy recompute = RE-STREAM states on CRC, needed ONLY if we choose subject-grouping (#1) or
causal baseline-correction (#2). Otherwise everything runs locally in seconds.

PROPOSED PLAN (deltas on existing eval_harness/, reuse loaders/splits/masking/metrics):
  seasonal -> per-group period from TRAIN autocorr peak (0.3-3 s config range), fallback persistence
    + warning, store T in results. AR -> statsmodels AutoReg (numpy fallback), orders {2,5,10,15,20,30},
    order on VAL, fit-scope config (default = chosen grouping). metrics -> skill vs persistence AND
    seasonal AND ar on identical masked frames. output -> tidy long CSV + parquet
    [model,channel,hand,horizon_step,metric,value,n_frames,config_hash]. plots -> per-channel
    full-horizon skill bars + per-step skill curves. evaluate -> score external model predictions in a
    standard indexed format. README usage section.

OPEN QUESTIONS (blocking; awaiting user):
  Q1 grouping for seasonal-T / AR-fit: activity x object (now) vs re-stream for subject vs global?
  Q2 non-causal baseline_correct: accept DC offset as-is vs make causal (needs re-stream)?
  Q3 deps: pip install statsmodels + pyarrow, or numpy AR + CSV-only?

### IMPLEMENTATION (2026-07-16) — refined harness delivered (statsmodels AR, per-group, tidy output)

Decisions: grouping = activity x object; baseline_correct accepted as-is (non-causal DC, documented);
installed statsmodels 0.14.6 + pyarrow 25. NO GRU retrain, NO re-stream.

Reworked eval_harness/ (reused loaders/splits/masking; changed baselines + output):
- config: fit_scope (group|global), ar_orders {2,5,10,15,20,30}, seasonal range in SECONDS (0.3-3s),
  seasonal_min_autocorr floor. dataset.group_keys -> 'action-object' (or 'ALL' global).
- baselines now GROUP-aware (fit(train,groups)/select(val,groups,H)/predict(hist,H,group)).
  * seasonal: period per group from TRAIN autocorrelation FUNDAMENTAL peak (smallest-lag local max
    within 95% of tallest, >= floor); no peak -> fallback persistence + warning; T stored.
  * ar: statsmodels AutoReg (trend='c', numpy OLS fallback) per group per channel, order selected
    per group on VAL by iterated H-step nMSE; recursive causal multi-step forecast.
- metrics: added masked_horizon_mae. evaluate: skill vs persistence AND seasonal AND ar on identical
  masked frames; tidy LONG table [model,channel,hand,horizon_step,metric,value,n_frames,config_hash]
  -> docs/actionsense/harness_baselines.csv + .parquet (990 rows); sidecar _fitparams.csv (seasonal T + AR order
  per group). External-model scoring: --model-preds preds.npz (standard target-time-indexed format).
  Determinism asserted (two runs identical).
- scripts/plot_harness.py (plots only): docs/actionsense/harness_skill_bars.png + harness_skill_curves.png.
- tests/test_harness.py: 7 pytest (added seasonal-fallback; group-aware). ALL PASS.
- src/tactile_forecast/eval_harness/README.md: how to score any future model.

RESULTS (TEST, config_hash b0194860): persistence nRMSE 0.517 (ref); ar nRMSE 0.467, skill vs
persistence +0.14..+0.25 (right-hand CoP-x best +0.25), AR order 20-30 per group; SEASONAL fell
back to persistence for ALL 5 groups (skill 0). Masking active: CoP n_frames 46647-47205 vs force
49670 (~2.5-3k low-force CoP frames removed).

FINDING / OPEN QUESTION: seasonal-naive is inert because RAW aggregate force/CoP has NO
autocorrelation peak in 0.3-3 s under its slow trend (autocorr monotonically decays -> no local
max). Options to make seasonal engage: estimate the period on a CAUSAL detrended signal (first
difference or causal high-pass; TRAIN-only, still causal per constraint 1). NOT added silently ->
awaiting user decision. Deliverables 1-5 complete; no retrain.

### Session (2026-07-20) — repo reorganization: file-by-file dataset categorization (NO move yet)
User asked to group TouchAnything-dataset files vs ActionSense files, but FIRST document what each
file does + its dataset, rigorously from code + SESSION_LOG. Wrote docs/REPO_ORGANIZATION.md.
FINDING: not a clean 2-way split — THREE dataset bodies + shared:
  A TouchAnything upstream (video+pose+tactile-pixel; DINOv2/MANO/WiLoR; EgoDex/EgoPressure)
  B ActionSense (ours: physical_state, state_forecast v1, action_dynamics v2 probGRU, eval_harness)
  C EgoTouch/OpenTouch (ours: predictability study + tactile-PIXEL forecaster; EgoTouch deprecated)
  D shared/infra/root.
COUPLING: B and C are interleaved in ONE package src/tactile_forecast/ (shared __init__/categories/
predictability) -> main import-risk for any physical move. Proposed target dirs touchanything/,
tactile_pixel/, actionsense/, shared/. OPEN QUESTIONS (awaiting user): Q1 2 vs 3 buckets (recommend
3); Q2 physically move (staged+tested, high-risk) vs adopt the doc as logical map; Q3 keep src/ root
vs top-level per-dataset dirs. NOTHING MOVED — plan-before-code.

### Reorg STAGE 1 DONE (2026-07-20) — ActionSense -> src/actionsense/  [verified]
git mv physical_state.py, state_forecast.py, action_dynamics.py, eval_harness/ from
src/tactile_forecast/ -> src/actionsense/ (+ new __init__). Rewrote imports in scripts
(check_leakage, train_action_dynamics, train_state_forecaster, probe_actionsense, plot_*),
tests/test_harness.py, and prose path refs (configs/eval_harness.yaml, docs). probe_actionsense
still imports categories/predictability from tactile_forecast (group C, moves in stage 2).
VERIFIED locally: pytest 7 pass; `python -m src.actionsense.eval_harness.evaluate` -> identical
config_hash b0194860 + determinism PASS; all B imports OK. src/tactile_forecast/ now = pure group C.
Fixed a `git add -A` slip that swept AGENTS.md + tmp_diag (untracked again, local kept).
REMAINING: stage 2 (C: rename src/tactile_forecast -> src/tactile_pixel), stage 3 (A: src/{data,
models,losses,utils,datasets,resources} -> src/touchanything/). Both are NOT runtime-testable locally
(training deps absent) -> will be static/compile + import-checked only. Plus scripts/configs/docs/data
grouping. Target: src/{touchanything,tactile_pixel,actionsense}/ + shared.

### Reorg STAGES 2-4a DONE (2026-07-20)
Stage 2: src/tactile_forecast -> src/tactile_pixel (group C: EgoTouch/OpenTouch pixel + predictability).
  Verified: C imports+compile; src.tactile_pixel.{train,eval} --help; probe_actionsense B+C cross-imports.
Stage 3: src/{data,models,losses,utils,datasets,resources} -> src/touchanything/ (group A upstream).
  A independent of B/C. Verified: 25 files py_compile; B+C+pytest still pass. (Full A runtime = CRC-only.)
Stage 4a: configs/ grouped -> configs/{actionsense,tactile_pixel,touchanything}/. Updated eval_harness
  DEFAULT_CONFIG + wilor mapping paths + doc/CRC refs. Harness re-verified (determinism PASS).
FINAL src/: actionsense/ tactile_pixel/ touchanything/ (+ __init__). configs/ grouped likewise.
DEFERRED (documented in docs/REPO_ORGANIZATION.md): scripts/ grouping (needs per-script sys.path depth
fix + invocation-ref updates in docs/CRC jobs; only ActionSense scripts testable locally); docs/ + data/
(referenced by path from the FROZEN harness config -> moving breaks outputs). Awaiting user go on those.
Commits: 3c13dfe (s1) + s2 + s3 + s4a. AGENTS.md/tmp_diag kept untracked throughout.

### OPEN QUESTIONS CLOSED (2026-07-21) — clean slate before new work
1. SEASONAL-NAIVE inert (raw aggregate force/CoP has no autocorrelation peak under its slow trend)
   -> RESOLVED: close as a documented finding. Leave seasonal as fallback-to-persistence. Rationale:
   causal detrending would fix period DETECTION but not ACCURACY (seasonal copies the RAW value one
   period back; for a slow-drifting signal "one period ago ~ now" -> stays ~persistence). AR is the
   strong baseline that beats persistence. NO code change.
2. probGRU one-shot-decoder comparison + AR(1) baseline (approved 2026-07-13, never built)
   -> RESOLVED: close as SUPERSEDED. The frozen eval_harness AR baseline replaces AR(1); the one-shot
   vs autoregressive probGRU comparison is moot because the probGRU predicts the OLD target (fast
   3-dim 1-hand) which the harness redefined (raw 6-dim) -> the probGRU would need re-scoping first.
3. Reorg scripts/docs/data grouping -> CLOSED earlier (user: "stop here"; src/ + configs/ grouped).
STATUS: no open questions remain. Ready to start new work.

### Session (2026-07-21) — PLAN APPROVED: forecast F/CoP from the tactile MAP (flatten vs CNN)

Full plan: C:\Users\haoji\.claude\plans\cheeky-meandering-wigderson.md. Summary:

GOAL: does the raw tactile pressure MAP carry extra spatially-structured signal for forecasting the
next 1 s of the 6-dim F/CoP target? Two encoders feeding an IDENTICAL GRU+one-shot head (only the
per-frame encoder differs): (a) FLATTEN (2x32x32 -> Linear -> d), (b) CNN (conv -> d). If CNN > flatten,
spatial structure of the contact patch contributes. Scored on the FROZEN eval harness (vs persistence/
seasonal/AR) via --model-preds.

LOCKED DECISIONS: target = future 6-dim F/CoP (harness, unchanged); RE-STREAM CRC for all 75 Slice/Peel
maps (only 45/75 local now); per-taxel baseline = FIRST-N-FRAMES (causal, replaces non-causal whole-clip
5th-pct); map amplitude = log1p + single GLOBAL train scale (same all taxels); time scale/split/target
MATCH the harness (ds=3 ->10Hz, horizon 10, origins min_history=40); SWEEP history t_in in {1,3,10 s}
(10/30/100 frames) for both encoders; left-pad early windows with zeros (post-baseline no-contact) so
predictions exist at every harness origin (fair-comparison caveat).

STEPS: (1) CRC re-stream probe_actionsense --save-clips-for "Slice,Peel"; verify state_N.npy identical
(preserves splits/target); scp clip_N.npy local. (2) new src/actionsense/tactile_map/data.py
preprocessing. (3) reuse eval_harness dataset.load_target / splits / baselines.origins for target+windows.
(4) models.py FlattenEncoder / CNNEncoder / shared Seq2Seq. (5) 2 enc x 3 hist = 6 runs -> export
preds_<enc>_<hist>.npz -> harness --model-preds -> compare (skill vs history, flatten vs cnn vs baselines).
FILES: src/actionsense/tactile_map/{__init__,data,models,train,export_preds}.py; configs/actionsense/
tactile_map.yaml; scripts/train_tactile_map.py; tests/test_tactile_map.py. REUSE: eval_harness/*,
tactile_pixel/tactile_utils.make_transform. VERIFY: unit tests (baseline causal, origin-alignment,
log1p invertible, train-only norm) + CPU smoke train + harness scoring + flatten-vs-cnn plot.
NOTE: target keeps whole-clip-baseline F/CoP (comparability); new causal baseline affects MAP INPUT only.
IMPLEMENTATION: build code + smoke-test on the 45 local maps first; full 75-map run after CRC re-stream.

### IMPLEMENTATION (2026-07-21) — tactile-map -> F/CoP forecaster BUILT + smoke-verified
New pkg src/actionsense/tactile_map/: data.py (causal first-N-frame per-taxel baseline + log1p +
global TRAIN scale + LAZY harness-aligned windows with zero left-pad), models.py (FlattenEncoder,
CNNEncoder, shared GRU + one-shot head -> (B,H,6)), train.py (fit TRAIN / early-stop VAL / export
TEST preds npz in RAW units). CLI scripts/train_tactile_map.py; configs/actionsense/tactile_map.yaml
(baseline_frames=10, alpha=10, d=64, hidden=64, epochs=60, sweep encoders x histories{1,3,10 s}).
Reuses eval_harness dataset.load_target / splits / baselines.origins / Norm. Deviation: inlined
log1p compress in data.py (avoid actionsense->tactile_pixel cross-group dep instead of make_transform).
tests/test_tactile_map.py: 8 pass (window causality, origin alignment, left-pad, log1p, baseline
first-N, model shapes, real-map load). 15/15 across both harness suites.

HARNESS BUG FIXED: baselines/__init__.py now exports `origins` (score_external used BL.origins; the
--model-preds path had never been exercised until now).

SMOKE (available maps, 1s hist, 2 epochs, 27 train / 9 test): train+export+HARNESS SCORING works
end-to-end. Skill vs persistence: flatten -1.82, cnn -3.03 (both < persistence) -- EXPECTED for a
2-epoch undertrained model (must infer current F/CoP level from the map while persistence gets the
last value free). Mechanics proven; real training will show the actual flatten-vs-cnn result.

DATA GAP FOUND + FIX: only 45/75 slice/peel had maps because stream_actionsense.sh used
`--save-clips-for "Pour,Slice"` (Peel never saved -> all 30 Peel missing). Fixed to
"Pour,Slice,Peel". RE-STREAM NEEDED on CRC to cache Peel maps (per approved plan). available_idxs()
checks file existence, so after copying clips locally the full sweep just runs (no manifest edit needed).

NEXT: user runs CRC re-stream (git pull -> bash scripts/crc/stream_actionsense.sh), verify state_0.npy
identical (idx mapping preserved), scp clip_*.npy to local, then run the full 6-run sweep + harness score.

### CRC LOGIN — reference (2026-07-21)  (source: scripts/crc/README.md §0 + session notes)
NetID = jhao3. UGE scheduler (qsub/qstat/qrsh). Front-end has NO GPU (torch.cuda False there) ->
GPU work goes through qsub batch jobs.
- On campus / ND VPN:      ssh -Y jhao3@crcfe01.crc.nd.edu     (crcfe02 also works)
- Off campus (no VPN):     ssh -Y jhao3@bastion.crc.nd.edu     then on bastion:  ssh crcfe01
- Auth: NetID password + 2FA (donGoogle Authenticator passcode, per account setup).
- Passwordless (optional): ssh-keygen -t ed25519 ; ssh-copy-id jhao3@crcfe01.crc.nd.edu (2FA may still apply).
- After login: echo $0 ; if not bash -> run `bash` ; then `conda activate tactile`.
- Repo on CRC: cd ~/TouchAnything && git pull   (fork: github.com/Jiayi459/TouchAnything).
- Pull results back to local: rsync -avz jhao3@crcfe01.crc.nd.edu:~/TouchAnything/runs/ ./runs/
  (CRC cannot push to GitHub without a PAT; use scp/rsync to move files).

### RESULT (2026-07-22) — tactile-map -> F/CoP: CNN beats flatten (spatial structure helps)
Full 75-recording map coverage (re-streamed Peel maps from CRC; state_0 identical -> frozen split
intact). Sweep: 2 encoders x history{1,3,10 s}, 40 epochs, scored on the frozen harness TEST (15/15).

CRITICAL FIX first: absolute-level map models MEAN-REVERTED (pred ~ train-mean, not current level)
-> deeply negative skill (-1.9..-2.2, worse than persistence). Diagnosed (corr(model,true)=0.78 so
maps carry signal, but net hedges to mean on a target where persistence is very strong). FIX =
predict RESIDUAL over persistence (delta vs last observed value; anchor added back at export). At
worst the model predicts delta=0 -> matches persistence. data.py target = Y[t+1:]-Y[t]; train.export
adds tnorm-space anchor tn[t]+delta. 9 unit tests pass.

RESIDUAL RESULT (mean skill vs persistence, harness TEST):
  persistence  0.000 (ref)      ar  +0.180  (still the ceiling; classical AR on aggregates)
  flatten  1s -0.044  3s -0.030  10s -0.020   (~= persistence; flattening loses the spatial info)
  cnn      1s +0.054  3s +0.064  10s +0.084   (POSITIVE; beats persistence; rises with history)
Headlines: (1) CNN > flatten at EVERY history -> the contact-patch SPATIAL STRUCTURE contributes to
predicting the CHANGE in F/CoP (answers the core question). (2) CNN improves with history (+0.05->+0.08).
(3) flatten stuck at persistence. (4) AR on aggregates (+0.18) STILL beats map+CNN (+0.08) -> spatial
structure helps vs flattening but hasn't beaten the strong aggregate baseline yet.
Artifacts: docs/actionsense/tactile_map_results.csv (9 models, tidy), docs/actionsense/tactile_map_skill_vs_history.png,
docs/actionsense/tactile_map_skill_bars.png, scripts/plot_tactile_map.py. Ran on CPU locally (~25 min/sweep;
models tiny). NEXT ideas: give CNN more capacity / combine map+aggregate (hybrid) to try to beat AR;
probabilistic head; or CRC GPU for a bigger sweep.

### UPGRADE (2026-07-22) — tactile-map model now matches the probGRU protocol (5-fold CV + probabilistic)
Per user ("exactly what we did for F/CoP" -> add 5-fold CV + probabilistic head), ported the
action_dynamics protocol onto the tactile-map CNN:
- models.py Seq2Seq: now PROBABILISTIC one-shot head -> (mu, logvar) each (B,H,6); lv clamped [-6,4].
- train.py: Gaussian NLL loss on the RESIDUAL; _predict/evaluate/calibrate_sigma/cross_validate.
  5-fold CV by recording (norms+model fit on TRAIN; sigma calibrated on a VAL subset of TRAIN; skill
  vs persistence + coverage@2sd measured on the held-out TEST fold). Persistence == residual 0, so
  skill = 1 - MSE(mu)/MSE(resid). Uses ALL 75 map recordings (not the frozen 15-rec test).
- scripts/train_tactile_map.py: CV sweep (encoder x history x folds) -> docs/actionsense/tactile_map_cv_results.csv
  [encoder,history_s,forecast_step_s, 6x <ch>_skill, mean_skill, coverage_raw, coverage_cal].
- scripts/plot_tactile_map.py: skill-vs-history + coverage(raw/cal) from the CV CSV.
- scripts/crc/train_tactile_map_gpu.job: symlinks ~/actionsense/states/clip_*.npy into data dir,
  runs the CV sweep on GPU (30 trainings = 2 enc x 3 hist x 5 folds; GPU now genuinely warranted).
- tests: 9 pass (model shape now (mu,lv) clamped). Local smoke (2 folds, 3 ep, 1s): flatten -0.049,
  cnn +0.032; coverage 0.90->0.95 after calibration (works).
SUPERSEDES the earlier deterministic frozen-split map result (docs/actionsense/tactile_map_results.csv). Full CV
run to be done on CRC GPU (qsub train_tactile_map_gpu.job) with all data + 80 epochs, then pull the
CSV + plot locally.

### CRC LOGIN — UPDATED (2026-07-22): direct crcfe01 times out off-campus -> use bastion ProxyJump
Direct `ssh jhao3@crcfe01.crc.nd.edu` times out unless on campus/VPN (crcfe01 is firewalled). FIX:
always go through the bastion. Set up ONCE in ~/.ssh/config (done on this machine):
    Host crc
        HostName crcfe01.crc.nd.edu
        User jhao3
        ProxyJump jhao3@bastion.crc.nd.edu
        ForwardX11 yes
        ServerAliveInterval 60
Then simply:
    ssh crc                                   # front-end via bastion (2FA at bastion, then password)
    scp crc:~/TouchAnything/docs/x.csv docs/  # copy FROM crc to local (uses the config)
    scp local docs/x crc:~/TouchAnything/...  # copy TO crc
Manual equivalent (no config): ssh -Y -J jhao3@bastion.crc.nd.edu jhao3@crcfe01.crc.nd.edu
Auth: bastion asks for Google Authenticator code, then crcfe01 asks for the NetID password.
On CRC: conda activate tactile ; cd ~/TouchAnything && git pull.

### CRC LOGIN — AUTHORITATIVE (2026-07-22, per docs.crc.nd.edu/new_user/connecting_to_crc.html)
Destination is ALWAYS the front-end crcfe01/crcfe02.crc.nd.edu. Off-campus you MUST tunnel, two
OFFICIAL ways (docs quote: off-campus "first need to connect to the campus VPN"; alternative "use
the server bastion.crc.nd.edu as your login host"):
  A) ND VPN then direct: connect ND campus VPN (ND OIT / vpn.nd.edu) -> `ssh -Y jhao3@crcfe01.crc.nd.edu`
     (this is the plain "previous" command; it only works on-campus or on VPN).
  B) Bastion (no VPN): `ssh -Y -J jhao3@bastion.crc.nd.edu jhao3@crcfe01.crc.nd.edu`
     Automated by ~/.ssh/config Host `crc` (ProxyJump bastion) -> just `ssh crc`.
The earlier timeout was: off-campus WITHOUT VPN and using the direct command -> blocked; use A or B.
Auth: NetID password + Google Authenticator 2FA (Okta/authenticator) at the prompt(s).

### OVERFITTING CHECK (2026-07-22) — F/CoP probGRU overfits badly at 80 epochs (no early stopping)
scripts/plot_fcop_loss_curve.py: reuses action_dynamics (load_pooled/Norm/windows/ProbGRU + NLL),
splits clips 70/15/15, logs train/val/test NLL + MSE per epoch. Config raw/right/3s, 80 epochs.
FINDING (docs/actionsense/fcop_loss_curve.png): classic overfitting on BOTH metrics.
  MSE (mean, drives skill): min-val @epoch 10 (0.729) -> rises to 0.937 by epoch 80 (train 0.430).
  NLL (mean+variance):      min-val @epoch 10 (0.220) -> rises to ~1.0 by epoch 80 (train -0.236).
  val & test track each other closely (no distribution mismatch); both diverge from train after ~ep 10.
IMPLICATION: action_dynamics.train runs 80 epochs and returns the FINAL model (NO early stopping) ->
the F/CoP sweep results (docs/actionsense/action_dynamics_results.csv, the +0.40 skills) come from OVERFIT models;
early stopping at ~epoch 10 would improve skill AND calibration. The NEW tactile_map train.py already
early-stops (keeps best-val), so the CRC tactile-map CV run is NOT affected -- only the old probGRU is.
RECOMMENDATION: add early stopping (keep best-val weights) to action_dynamics.train + re-run the F/CoP
sweep; or lower epochs to ~15-20. (Login methods already documented above, commit e22231f/prior.)

### OVERFITTING FIXED (2026-07-22) — early stopping in action_dynamics.train
Added early stopping: train(..., val_clips=) keeps the lowest-VAL-NLL weights (over all epochs);
cross_validate + the final-model path pass the val subset. No leakage (split by clip; norm train-only).
DEMONSTRATION (raw/right/3s, same 70/15/15 clip split, 80 epochs, held-out TEST):
  early-stop (best-val):  TEST mean skill +0.546, coverage 0.93  (per-ch F/x/y 0.54/0.55/0.54)
  overfit (80ep final):   TEST mean skill +0.401, coverage 0.81  (per-ch 0.30/0.49/0.42)
=> early stopping lifts skill +0.40->+0.55 (~36% rel) AND calibration 0.81->0.93; the force channel
(most overfit) recovers most (0.30->0.54). The old F/CoP sweep numbers (docs/actionsense/action_dynamics_results.csv)
were depressed by overfitting. TODO: re-run the full F/CoP sweep with early stopping to refresh the CSV.
Baselines reminder: persistence = last-value (skill-0 reference, per-split); linear AR (harness, raw
6-dim) = +0.18 mean skill (best right-CoP-x +0.25). Split confirmed leak-free (by clip, norm train-only).

ssh -Y -J jhao3@bastion.crc.nd.edu jhao3@crcfe01.crc.nd.edu
### RESULT (2026-07-22) — tactile-map CV on CRC GPU: CNN > flatten confirmed (5-fold, probabilistic)
Full CRC GPU run of the 5-fold probabilistic CV (train_tactile_map_gpu.job, all 75 map recordings,
80 epochs, early-stopped). Mean skill vs persistence + coverage (raw->calibrated):
  cnn      1s +0.052  3s +0.050  10s +0.063   cov 0.93-0.94 -> 0.95
  flatten  1s -0.040  3s -0.025  10s -0.026   cov 0.93      -> 0.95
=> CNN beats flatten at EVERY history (spatial contact-patch structure contributes), now under the
rigorous 5-fold + probabilistic + sigma-calibration protocol; bands calibrate cleanly to 0.95. CNN
best at 10s (+0.063); flatten stays just below persistence. Consistent with the earlier deterministic
frozen-split result (cnn +0.05..+0.08). Artifacts: docs/actionsense/tactile_map_cv_results.csv, _skill_vs_history.png,
_coverage.png. (AR on aggregates +0.18 still the ceiling -- map+CNN helps vs flatten but not vs AR yet.)
Added scripts/plot_tactile_map_loss_curve.py (train/val/test NLL+MSE per epoch for both encoders) to
check whether the map model overfits like the F/CoP one (running).

### LOSS CURVES (2026-07-22) — tactile-map models overfit almost immediately (flatten ep1, cnn ep4)
scripts/plot_tactile_map_loss_curve.py (3s history, 60 ep, split by clip 52/11/12, train-only norm,
eval cap 2500). docs/actionsense/tactile_map_loss_curve.png:
  flatten: min-val NLL @epoch 1; final NLL tr/va/te = -0.956/1.258/1.411 (memorizes instantly, val
           rises from epoch 1 -> NO generalization window).
  cnn:     min-val NLL @epoch 4; final NLL tr/va/te = -0.926/0.969/1.407 (val MSE dips/holds ~4 epochs
           before rising -> a real learning window). 
INTERPRETATION: both overfit MUCH faster than the F/CoP probGRU (ep 1-4 vs ep 10) -- the 2048-dim map
is over-parameterized for only 45 train recordings. This IS the mechanism behind CNN>flatten: the CNN's
spatial inductive bias extracts a little GENERALIZABLE signal before overfitting; flatten memorizes at
once. Modest CV skill (+0.05-0.06) reflects DATA SCARCITY, not encoder failure; early stopping (CV keeps
best-val) is essential (else val NLL ~1.3-1.4 by ep60). NEXT to widen the CNN lead: more data
(activities/subjects), regularization (dropout/weight-decay/smaller d), or glove augmentation.

### F/CoP EARLY-STOPPED SWEEP (2026-07-23, CRC) — cross-validated, big improvement + corrects a finding
Pulled runs/action_dynamics_results.csv -> docs/actionsense/action_dynamics_results_earlystop.csv. Compared to the
OLD overfit docs/actionsense/action_dynamics_results.csv (mean skill vs persistence-of-fast, 5-fold CV):
  Early stopping improved skill in EVERY config by +0.10..+0.23. Examples:
    raw/right 1s   +0.410 -> +0.513   highpass/right 3s +0.369 -> +0.519   raw/left 10s +0.219 -> +0.449
  New best ~+0.51-0.52 (right hand), ~+0.46 (left). Coverage stayed ~0.94-0.95 (calibration handled both).
CORRECTION: the earlier "more history HURTS" finding was an OVERFITTING ARTIFACT. Old: skill fell with
history (right 1s +0.41 -> 10s +0.31). New (early-stopped): skill is ~FLAT across history (right 1s +0.51
-> 10s +0.50); the gain is LARGEST at 10s (+0.18..+0.23), exactly where overfitting was worst. So longer
history was only losing because more input -> more overfitting without early stopping.
NOTE: these probGRU skills are vs persistence-of-fast on the FAST 3-dim 1-hand target -- NOT directly
comparable to the harness AR (+0.18, raw 6-dim target). docs/actionsense/action_dynamics_results_earlystop.csv is the
honest result; docs/actionsense/action_dynamics_results.csv (overfit) kept for the before/after diff.

### BEFORE/AFTER early-stopping loss figure (2026-07-23)
docs/actionsense/fcop_earlystop_comparison.png (raw/right/3s): same loss curve, two DEPLOYED checkpoints marked.
AFTER early-stop (deploy min-val ep10): test NLL 0.217, test MSE 0.737. BEFORE (deploy final ep80):
test NLL 1.036, test MSE 0.951. Early stopping recovers +0.818 NLL / +0.215 MSE on the held-out test
by deploying the min-val checkpoint instead of the overfit final one. (Early stopping = checkpoint
selection; the training trajectory is identical -- one curve, different deployed point.)

### MSE SUMMARY (2026-07-23) — all models on one scale (raw/right/3s, normalized FAST target, same 70/15/15 split seed 0)
"var explained" = 1 - MSE/0.978 (0.978 = fast-signal variance = predict-mean MSE) = the HONEST R^2.
"skill vs pers" = 1 - MSE/1.624 (persistence-of-fast MSE) = the flattering metric.

  model                     | norm test-MSE | skill vs persistence | var explained (R^2)
  --------------------------|---------------|----------------------|--------------------
  probGRU early-stop (ep10) |     0.737     |        +0.55         |       +0.25
  AR (fast target, p=20)    |     0.763     |        +0.53         |       +0.22
  probGRU overfit (ep80)    |     0.951     |        +0.41         |       +0.03
  predict-mean (0)          |     0.978     |        +0.40         |        0.00 (= variance)
  persistence-of-fast       |     1.624     |         0.00         |       -0.66

KEY POINTS:
1. Persistence-of-fast (1.624) is a WEAK baseline -- WORSE than predicting the mean (0.978) -- because
   the high-pass component oscillates around 0, so "repeat last fast value" is anti-correlated. So
   skill-vs-persistence is inflated: even predict-mean scores +0.40.
2. HONEST metric = variance explained vs the mean: early-stop probGRU +0.25, AR +0.22, overfit +0.03.
3. The LINEAR AR nearly TIES the probGRU (0.763 vs 0.737, ~3%). The GRU's edge over a simple linear
   autoregression on the fast target is marginal.
4. The OVERFIT probGRU (0.951 ~= predict-mean 0.978) had ~ZERO real skill; its flashy +0.41 was entirely
   the weak-persistence illusion. Early stopping (0.737) gives genuine skill (+0.25 R^2).
SEPARATE CONTEXT: the harness AR on the RAW 6-dim both-hands target = +0.18 skill vs persistence -- a
DIFFERENT target/scale, not comparable to the fast-target MSEs above.

### DEFINITIVE COMPARISON (2026-07-23) — forecasting the raw 6-dim F/CoP, all same target/split/protocol
Added aggregate-F/CoP encoder (neural AR: GRU on the 6-dim history) scored by the same 5-fold
probabilistic CV as the map models. FOUR-WAY mean skill vs persistence (docs/actionsense/forecaster_comparison.png):
  history | linear-AR | GRU-aggregate | CNN-map | flatten-map | persistence
     1s   |  +0.180   |   +0.120      | +0.052  |  -0.040     |   0
     3s   |  +0.180   |   +0.138      | +0.050  |  -0.025     |   0
    10s   |  +0.180   |   +0.142      | +0.063  |  -0.026     |   0
RANKING: linear AR > GRU-aggregate > CNN-map > flatten-map > persistence.
CONCLUSIONS:
 1. LINEAR AR IS BEST. Neither nonlinearity (GRU) nor a richer input (map) beats a simple per-channel
    linear autoregression on the raw F/CoP -- these dynamics are essentially linear-autoregressive.
 2. THE MAP IS AN INFERIOR INPUT to the aggregate for THIS target: map models (+0.05-0.06) << aggregate
    models (+0.12-0.14). Going through the pixel representation loses info -- the net must reconstruct
    the aggregate (F=sum, CoP=centroid) from pixels imperfectly.
 3. WITHIN the map, spatial structure still helps (CNN +0.05 > flatten -0.03) -- consistent with the
    earlier finding -- but not enough to reach the aggregate, let alone AR.
 4. GRU-aggregate improves slightly with history (+0.12->+0.14) then plateaus, still below AR.
Coverage ~0.94-0.95 (calibrated) for all learned models. Fast-component probGRU (action_dynamics)
kept separate/untouched. Artifacts: docs/actionsense/forecaster_comparison.png, tactile_map_cv_results_aggregate.csv,
scripts/plot_forecaster_comparison.py.

### RIGOR CHECK (2026-07-23) — GRU-aggregate vs linear AR: same target/input, protocol now matched
Verified: BOTH use load_target = RAW 6-dim both-hands F/CoP (NOT the fast/high-pass component;
dataset.py:43-52). GRU-aggregate input=past of load_target, target=future of load_target
(train.py:143-144, AggWindows); AR operates on the same load_target. Same data (Slice+Peel, 75 recs),
same autoregressive input representation. THE ONE MISMATCH: the four-way plot's AR (+0.180) came from
the FROZEN split (harness fit_and_forecast on splits.json, test 15), while the GRU used 5-FOLD CV.
FIXED: ran AR on the IDENTICAL 5-fold folds (same recs order, same seed=0 fold_of, same val carve) ->
AR mean skill = +0.166 (per-fold 0.15-0.19, stable). So protocol-matched AR = +0.166 (vs +0.180 frozen).
Ranking UNCHANGED and now fully apples-to-apples: AR +0.166 > GRU-aggregate +0.12-0.14 > CNN-map
+0.05-0.06 > flatten-map -0.03 > persistence 0. Updated docs/actionsense/forecaster_comparison.png (AR line 0.166).

---

## CONSOLIDATED RESULTS & METHODS (2026-07-24) — 1-second tactile forecasting on Slice/Peel

Rigorous write-up of the current forecasting results. Self-contained: a reader should be able to
reproduce and interpret everything from this section. (Supersedes the scattered incremental entries
above for the purpose of a clean summary; those remain as the running log.)

### 0. Common setup (shared by all raw-target experiments)
- **Data**: ActionSense wearables, conductive-thread gloves, 32x32 taxels/hand, 2 hands. Activities
  **Slice (45 recordings: cucumber/potato/bread x15) + Peel (30: cucumber/potato x15) = 75**. Each
  recording is one activity interval segmented by the HDF5 Start/Stop markers, resampled to 30 Hz,
  then **downsampled x3 -> 10 Hz** (`cfg.downsample=3`). Stored as `state_<idx>.npy` (T,2,6) physical
  moments (per-taxel 5th-pct DC baseline removed) + raw map `clip_<idx>.npy` (T,2,32,32).
- **Target (raw 6-dim)**: `eval_harness.dataset.load_target` -> [F_L, CoPx_L, CoPy_L, F_R, CoPx_R,
  CoPy_R] = moments 0..2 of each hand, at 10 Hz. This is the FULL raw signal (NOT high-pass filtered).
- **Horizon**: 1 s = **10 steps** (100 ms each). Forecast origins from `min_history=40`, stride 1;
  predictions indexed by TARGET time t+h.
- **Frozen split**: `data/actionsense_states/splits.json` = 60/20/20 by RECORDING, stratified by
  (activity, object): **train 45 / val 15 / test 15**. (Also used: 5-fold CV by recording, seed 0.)
- **Causality/leakage**: all filtering causal (sosfilt, no filtfilt); normalization fit on TRAIN only;
  split by clip so no window crosses splits (verified by `scripts/check_leakage.py`, 6 checks PASS).
- **Metric**: skill = 1 - MSE_model/MSE_persistence (0=tie persistence, 1=perfect, <0=worse), per
  channel/step + mean; CoP channels masked where that hand's raw force < TRAIN 5th-pct; coverage@2sd
  for probabilistic models (target 0.95 after sigma-calibration).

### 1. Classical baselines on the raw 6-dim target (frozen harness)
Method: `eval_harness/evaluate.py` fits on TRAIN, selects hyperparameters on VAL, scores TEST once.
- **persistence** y_hat(t+h)=y(t): the 0-skill reference. STRONG here (raw F/CoP drifts slowly).
- **seasonal-naive**: period estimated per (activity,object) from TRAIN autocorrelation. **Falls back
  to persistence for ALL groups** -> the raw aggregate has NO autocorrelation peak in 0.3-3 s (slow
  trend dominates). Documented finding, not a bug.
- **linear AR**: per-channel statsmodels AutoReg, trend='c', order p in {2,5,10,15,20,30} selected on
  VAL, per (activity,object). **Mean skill = +0.180** (frozen split); **+0.166** re-scored on 5-fold
  CV (protocol-matched, per-fold 0.15-0.19). Best channel right-hand CoP-x (+0.25). nRMSE 0.467 vs
  persistence 0.517.

### 2. Tactile-MAP forecasters (raw 6-dim target)  [scripts/train_tactile_map.py, src/actionsense/tactile_map/]
Question: does the raw pressure MAP carry extra 1-s predictive signal vs the aggregates?
Method (identical for both encoders; only the per-frame encoder differs):
- Input = past `t_in` frames of the map (2,32,32); preprocessing: causal per-taxel first-N-frame
  baseline (N=10), log1p compression (alpha=10), global TRAIN scale. History swept t_in in {1,3,10 s}.
- Encoders: **flatten** Linear(2048->64); **CNN** 3-conv -> global-avg-pool -> 64. Both -> shared GRU
  (hidden 64) -> one-shot PROBABILISTIC head (mu, logvar) -> (10,6).
- Target = RESIDUAL over persistence (predict change vs last value; at worst matches persistence).
- Training: Gaussian NLL, **5-fold CV by recording**, **early stopping** (best-VAL-NLL checkpoint),
  post-hoc sigma-calibration on VAL. (CRC GPU run for the map models.)
Results (mean skill vs persistence, 5-fold CV):
    history |  CNN(map) | flatten(map)
      1s    |  +0.052   |   -0.040
      3s    |  +0.050   |   -0.025
     10s    |  +0.063   |   -0.026     (coverage 0.93 raw -> 0.95 calibrated)
- **CNN > flatten at every history** -> the contact-patch SPATIAL structure contributes to predicting
  the change in F/CoP. Flatten sits at/below persistence.
- Loss curves (`plot_tactile_map_loss_curve.py`): both encoders **overfit almost immediately** (flatten
  min-val ep1, cnn ep4) -- the 2048-dim map is over-parameterized for 45 train recordings. CNN's brief
  generalization window (val holds ~4 epochs) IS the mechanism behind CNN>flatten. Early stopping is
  essential (else val NLL ~1.3-1.4 by ep60).

### 3. Aggregate-F/CoP GRU (neural AR, raw 6-dim target)  [encoder="aggregate"]
Method: identical protocol to sec.2 (5-fold CV, probabilistic, residual, early-stop, calibration) but
input = past `t_in` frames of the raw 6-dim F/CoP itself (autoregressive; the neural counterpart of AR).
Results (mean skill vs persistence, 5-fold CV): **1s +0.120, 3s +0.138, 10s +0.142** (coverage ~0.95).

### 4. FOUR-WAY comparison (raw 6-dim target, IDENTICAL 5-fold CV, same input/target/data)  [docs/actionsense/forecaster_comparison.png]
    linear-AR +0.166  >  GRU-aggregate +0.12..+0.14  >  CNN-map +0.05..+0.06  >  flatten-map -0.03  >  persistence 0
RIGOR: verified GRU-aggregate and AR use the SAME target (load_target, raw 6-dim both hands, NOT fast),
SAME autoregressive input, SAME Slice/Peel data, and (after fixing) the SAME 5-fold folds (AR re-scored
5-fold = +0.166 vs +0.180 frozen-split). Caveat: AR is not swept over t_in (it selects its own order as
history) -> a single history-agnostic number.

### 5. Fast-component probGRU (SEPARATE experiment, different target)  [src/actionsense/action_dynamics.py]
This is the ORIGINAL v2 model on a DIFFERENT target: the **high-pass FAST component**, 3-dim, ONE hand
([F_fast,x_fast,y_fast]). NOT comparable to the raw-target results above (different target/baseline).
- **Overfitting fix**: `action_dynamics.train` originally ran 80 epochs and returned the FINAL model
  (no early stopping) -> badly overfit (train/val/test loss curve: val bottoms ~epoch 10 then rises;
  `docs/actionsense/fcop_earlystop_comparison.png`). Added early stopping (keep best-VAL). Effect (5-fold CV):
  skill improved **+0.10..+0.23 per config**; best ~**+0.51** (right hand); and skill became **~flat
  across history** (the earlier "more history HURTS" finding was an OVERFITTING ARTIFACT -- longer
  history overfit more without early stopping). Coverage ~0.95 (calibration handled both).
- **MSE decomposition** (raw/right/3s, normalized fast target, one split): probGRU-earlystop 0.737,
  AR-on-fast 0.763, predict-mean 0.978 (=variance), probGRU-overfit 0.951, persistence-of-fast 1.624.
  Honest R^2 (=1-MSE/0.978): early-stop +0.25, AR +0.22, overfit +0.03. -> (a) persistence-of-fast is
  a WEAK baseline (worse than the mean) so skill-vs-persistence is inflated; (b) the honest signal is
  ~25% variance explained; (c) linear AR ~TIES the GRU on the fast target too.

### 6. ANALYSIS / CONCLUSIONS
1. **A linear autoregression is the best 1-s forecaster of the raw F/CoP.** Neither nonlinearity (GRU)
   nor a richer input (the tactile map) beats per-channel linear AR. The predictable part of this
   signal over 1 s is the (near-)linear autoregressive trend.
2. **The tactile MAP is an inferior input to the aggregate** for this target (map +0.05 << aggregate
   +0.14): passing through the pixel representation loses information -- the net must reconstruct
   F(=sum) and CoP(=centroid) from pixels, imperfectly, when those aggregates were available directly.
3. **Within the map, spatial structure still helps** (CNN +0.05 > flatten -0.03), robust across history
   and CV -- but not enough to reach the aggregate, let alone AR.
4. **Overfitting silently depressed the neural results** (both the fast-target probGRU and the map
   models). Early stopping is decisive: it turned a near-useless overfit fast-probGRU (R^2 +0.03) into
   a genuinely predictive one (R^2 +0.25), and it corrected the spurious "more history hurts" trend.
5. **Baseline choice matters for interpretation.** Skill-vs-persistence flatters models on the FAST
   target (persistence-of-fast is worse than the mean); variance-explained (R^2 vs mean) is the honest
   cross-target metric.

### 7. CAVEATS / LIMITATIONS
- **Data-scarce**: only 45 training recordings (Slice+Peel); the 2048-dim map models overfit by ~epoch
  1-4. Conclusions about "map doesn't help" are for THIS data regime; more data/regularization/augmentation
  could change the map's standing.
- **1-s horizon only**; linear AR dominates at 1 s -- longer horizons (where linear extrapolation breaks)
  were not tested and could favor the map/nonlinear models.
- **Two targets in play**: raw 6-dim both-hands (harness, sec.1-4) vs fast 3-dim one-hand (probGRU, sec.5).
  Their skill numbers are NOT comparable (different persistence baselines). R^2-vs-mean is comparable.
- AR uses its own order as history (not swept over t_in); it is a single number in the four-way plot.

### 8. ARTIFACTS
- Figures: `docs/actionsense/forecaster_comparison.png` (four-way), `tactile_map_skill_vs_history.png`,
  `tactile_map_coverage.png`, `tactile_map_loss_curve.png`, `fcop_earlystop_comparison.png`,
  `fcop_loss_curve.png`, `harness_*curves.png`.
- Tables: `docs/actionsense/tactile_map_cv_results.csv` (cnn/flatten), `tactile_map_cv_results_aggregate.csv`,
  `harness_baselines.csv` (persistence/seasonal/AR), `action_dynamics_results{,_earlystop}.csv` (fast).
- Code: `src/actionsense/{eval_harness, tactile_map, action_dynamics.py}`; `scripts/train_tactile_map.py`,
  `plot_forecaster_comparison.py`, `plot_tactile_map*.py`, `plot_fcop_loss_curve.py`, `check_leakage.py`.
- Tests: `tests/test_harness.py` (7), `tests/test_tactile_map.py` (10) -- all pass.

---

## PORTABILITY / CONTINUE ON ANOTHER COMPUTER (2026-07-24)
Repo: https://github.com/Jiayi459/TouchAnything (origin/main @ 0071021). Working tree CLEAN, nothing
unpushed. All ActionSense RAW TACTILE MAPS are now ON GITHUB (un-gitignored + committed).

### PUSHED to GitHub (a plain clone has all of this)
- All code: src/ (actionsense/{eval_harness,tactile_map,action_dynamics.py}, tactile_pixel/,
  touchanything/), scripts/, configs/, tests/. SESSION_LOG.md, CLAUDE.md, AGENTS.md.
- data/actionsense_states/: state_*.npy (299), manifest.jsonl, splits.json, AND all 100 clip_*.npy
  (raw tactile maps, ~401MB) -> covers all 75 Slice/Peel recordings + 25 others. (Un-gitignored 2026-07-24.)
- All docs/: every result CSV (harness_baselines, action_dynamics_results{,_earlystop},
  tactile_map_cv_results{,_aggregate}, ...) and every figure (forecaster_comparison.png, loss curves,
  skill plots, ...).
- scripts/tmp_diag_predictability.py.
=> Everything needed to REPRODUCE the forecasting work travels with the repo (states + maps + results).

### NOT pushed (gitignored) -- and why / how to get them
- datasets/ (~15GB): EgoTouch/OpenTouch/grasp_hold_lift_tactile = the OLDER PIXEL work. Too big for git,
  NOT needed for the current ActionSense forecasting. Re-download (scripts/download_egotouch.py,
  scripts/crc/download_opentouch.sh) only if returning to that thread.
- .venv/ (~1.3GB): Python env -> regenerate with pip (below).
- runs/ (~125MB): model npz + CRC logs -> regenerable; the results are already captured in docs/.

### SETUP on the new machine
  git clone https://github.com/Jiayi459/TouchAnything.git && cd TouchAnything
  python -m venv .venv
  .venv/bin/pip install numpy torch scipy pandas statsmodels pyarrow pytest matplotlib pyyaml   # (+ h5py only for CRC streaming)
  pytest tests/            # expect 17 passing (7 harness + 10 tactile_map)
Then you can immediately: view all results (docs/), re-run harness scoring, the aggregate-GRU, the
tactile-map CV (maps are present), and all plots. GPU training uses CRC (see CRC LOGIN section above:
`ssh crc` via bastion ProxyJump).

### SECOND COPY on CRC
~/TouchAnything (git repo; `git pull` to sync) + ~/actionsense/states/ (state_*.npy + clip_*.npy +
manifest.jsonl from the re-stream). The ActionSense HDF5 were deleted (streaming-delete); re-stream via
`bash scripts/crc/stream_actionsense.sh` (KEEP=1 to retain) only if regenerating states from scratch.

---

## Session (2026-08-05) — PROJECT CONCLUSIONS document

### Request
Read SESSION_LOG.md thoroughly; conclude all results so far as points, covering project detail with
high clarity/logic/completeness; write it as a separate file in the documents folder; refer to
specific code and documents throughout; list references at the end; ask questions if any arise.

### Work done
- Read all 1,927 lines of SESSION_LOG.md (Sessions 1-4 + COMPREHENSIVE SUMMARY + COLD-START SNAPSHOT
  + rigorous review + CONSOLIDATED RESULTS + PORTABILITY).
- **VERIFIED every quantitative claim against the committed artifacts** rather than trusting the log:
  recomputed mean skills directly from `docs/actionsense/harness_baselines.csv` (AR +0.1886 over 66 rows;
  seasonal 0.0 == persistence, confirming the documented fallback), `docs/actionsense/tactile_map_cv_results.csv`
  (cnn +0.052/+0.050/+0.063, flatten -0.040/-0.025/-0.026), `_aggregate.csv` (+0.120/+0.138/+0.142),
  and both `action_dynamics_results{,_earlystop}.csv` (per-config means + calibrated coverage). All
  match the log. Also confirmed `splits.json` = train 45 / val 15 / test 15, n=75, seed 0.
- Checked the harness config hash: `sha256(configs/actionsense/eval_harness.yaml)[:16]` =
  **8afc249f260894fd**, which MATCHES the hash stamped in `docs/actionsense/harness_baselines.csv`. The
  `b0194860` cited in the 2026-07-16 log entry is stale — it predates commit `ee8d097` (reorg stage
  4a) which MOVED the yaml into `configs/actionsense/`, changing the file bytes and hence the hash.
  Results are current; only the log's hash string is historical. (No action needed; noted so a future
  reader isn't confused by the mismatch.)
- Verified the code line references cited in the new document actually exist (physical_state.py:68
  percentile baseline; action_dynamics.py slow_fast/build_features/windows/ProbGRU/train/
  calibrate_sigma/evaluate; tactile_map/data.py:51 log1p and :140 residual-over-persistence target;
  eval_harness/metrics.py:45 skill).

### Deliverable
**NEW `docs/PROJECT_CONCLUSIONS.md`** — the concluding document. Structure: (0) what the project
became; (1) the five phases; (2-6) per-phase results as points, each with code/doc citations;
(7) nine cross-cutting methodological conclusions; (8) caveats/limits; (9) status + open items
ranked by scientific value; (10) full reference list (log sections, project docs, code modules with
line anchors, scripts, result artifacts, external datasets/papers).

Content decisions (stated for review):
- Positioned as a CONCLUSION of the whole project, not a replacement for the existing
  docs/STUDY_SUMMARY.md (predictability study) or docs/RESULTS.md (EgoTouch pixel forecaster) —
  it links to both and does not duplicate their tables beyond the headline numbers.
- Deliberately foregrounded the SELF-CORRECTIONS (filtfilt leak -0.3 skill; early stopping turning
  R^2 +0.03 into +0.25 and retracting the "more history hurts" finding; the weak-baseline audit
  showing predict-zero scores +0.57) as first-class results, since they change how every earlier
  number must be read.
- Recorded the original grasp-success goal as formally unanswerable on EgoTouch (no success labels)
  — this is the reason the project pivoted, and it belongs in a conclusion.
- Flagged that the feedback/adaptive-strategy application (the ultimate stated goal) is still
  UNBUILT despite all its prerequisites now existing — open item #1.

### Open questions for the user (non-blocking; document delivered)
1. Audience: is this for a supervisor/collaborator handoff, or a draft skeleton toward a paper? A
   paper draft would want the four-way comparison and the metric audit promoted to the front and the
   EgoTouch phase compressed to a paragraph.
2. Should `docs/STUDY_SUMMARY.md` and `docs/RESULTS.md` now be marked as superseded-in-part by this
   document (they predate the causal-filter and early-stopping corrections), or left standing as
   phase-local records?
3. Should the honest R^2-vs-mean metric be recomputed for the raw-6-dim four-way comparison (§6.4)?
   Currently only the fast-target experiment (§5.5) has it; without it the +0.166 AR headline still
   rests on skill-vs-persistence, which the audit showed can flatter.

---

## Session (2026-08-06) — DATA AVAILABILITY AUDIT (ActionSense content/format; OpenTouch; Force-Vision)

### Request
Point out the ActionSense data and its content/format; and for OpenTouch
(opentouch-tactile.github.io) and the ICLR Force-Vision submission site, report whether they are
downloaded, how much, and where. Refer to SESSION_LOG for history.

### Method — verified on the filesystem, not from the log
Did not trust prior log statements; re-derived every number from the working tree
(`/Users/haojiayi/TouchAnything`, branch main, tree clean) with `du`, `numpy` loads of every
`.npy`, and a parse of `manifest.jsonl`. Also ran a `find ~ -maxdepth 4` sweep for
`*opentouch*`/`*force*vision*`/`*actionnet*` dirs and any `*.hdf5`/`*.h5` under `$HOME`.

### FINDING 1 — ActionSense: the ONLY dataset present on this machine (416 MB, git-tracked)
Location: `data/actionsense_states/` — 401 tracked files (`git ls-files data | wc -l` = 401),
416 MB, all committed to GitHub (un-gitignored 2026-07-24, see PORTABILITY section).
Two artifact kinds, disjoint index sets, plus two metadata files:
- **`state_*.npy` x 299** — physical-state time series, `float32`, shape **(T, 2, 6)**
  = (frames, {left,right} glove, `[F, xbar, ybar, sxx, syy, sxy]`), definitions in
  `src/actionsense/physical_state.py:26,37` (F = Σp; CoP = Σp·x/Σp; second moments). Resampled to
  **30 Hz** from the native ~6 Hz. T range 82 / median 763 / max 6602; **320,309 frames total**
  (per manifest). Derived AFTER per-taxel 5th-percentile baseline subtraction
  (`physical_state.py:64-68`).
- **`clip_*.npy` x 100** — raw tactile maps, `float16`, shape **(T, 2, 32, 32)** = (frames,
  {L,R}, 32x32 taxels), ~401 MB of the 416 MB, value range ~499-1016 (uncalibrated sensor units,
  DC-offset NOT removed at this stage). 422,696 frames across the 100 clips.
- **`manifest.jsonl`** (299 lines, one per state clip): `{idx, label, cat, fps, T, features,
  has_clip}`.
- **`splits.json`**: train 45 / val 15 / test 15 (n=75, seed 0) — the Slice/Peel forecasting subset.
- **STALE FIELD FOUND:** `manifest.jsonl` reports `has_clip=true` for only **70** clips, but **100**
  `clip_*.npy` exist on disk. The manifest was written before the second map-extraction pass; every
  clip id on disk does resolve to a manifest row, so nothing is orphaned — but code must not use
  `has_clip` as the source of truth for map availability. Glob the directory instead.
- **Which clips have raw maps** (all 100, by label): Pour water 25, Peel cucumber 15, Slice cucumber
  15, Peel potato 15, Slice potato 15, Slice bread 15. I.e. the 75 Slice/Peel recordings used by the
  tactile-map CV + 25 Pour. No maps exist for Clean/Spread/Organize/Open-close-jar.
- **Full label inventory (299 state clips, 22 raw labels / 6 categories)**: Clear cutting board 28,
  Pour water 25, Get items from fridge/cabinets/drawers 24, Peel cucumber 15, Slice cucumber 15,
  Peel potato 15, Slice potato 15, Slice bread 15, Spread almond butter 15, Spread jelly 15, Clean
  plate w/ sponge 15, Clean plate w/ towel 15, Clean pan w/ sponge 15, Clean pan w/ towel 15,
  Get/replace items 15, Open/close jar 9, Open jar 6, Get items from cabinets 6, Set table 6, Stack
  on table 5, Load dishwasher 5, Unload dishwasher 5. Categories: Cut / Pour / Wash-Clean /
  Fold-Cloth(spread) / Organize-Arrange / Open-Close.
- **NOT present locally:** the source ActionSense wearables HDF5 (~2-4 GB each x 12, ~35-88 GB) —
  deleted by the streaming driver `scripts/crc/stream_actionsense.sh` on CRC. Re-fetchable via
  `scripts/crc/download_actionsense.sh` (12 public URLs, data.csail.mit.edu/ActionNet, S00-S05).
  Consequence already noted in PROJECT_CONCLUSIONS §8: subject ids are unrecoverable without a
  ~88 GB re-stream.

### FINDING 2 — OpenTouch: downloaded once on CRC, since DELETED; nothing on this machine
- Was fully downloaded on CRC (2026-07-02): **26 HDF5 shards ~14 GB (561 MB/shard) + labels**, via
  `scripts/crc/download_opentouch.sh` (26 gdown file IDs + `final_annotation.zip` ID, copied
  verbatim from OpenTouch-MIT/opentouch), default dest `~/opentouch/data`. Probed successfully:
  2,496 usable of 2,958 clips.
- **Then deleted** during the ActionSense disk crisis (SESSION_LOG:571-573: "OpenTouch raw data got
  deleted along the way (kept its earlier CSV)"). Confirmed 0 bytes on this Mac: no `~/opentouch`,
  no `.hdf5`/`.h5` anywhere under `$HOME`, no `datasets/` dir in the repo.
- **Its derived result CSV is also not local:** `docs/predictability_opentouch.csv` was written on
  CRC and never pushed. `docs/` only holds `predictability_by_category{,_full}.csv`, which I
  verified are the **EgoTouch** probe outputs (grouping/group/n/pers_nmse_h1/h15/h30/periodicity/
  contact_migration/predictability_index; B5 composite n=379/411 — EgoTouch trajectory counts, not
  OpenTouch's 2,496). The OpenTouch numbers survive only as prose in
  `docs/ACTION_CATEGORIES.md` §3b and `docs/PROJECT_CONCLUSIONS.md`.
- Recovery cost if needed: `pip install gdown && bash scripts/crc/download_opentouch.sh` -> ~14 GB.

### FINDING 3 — Force-Vision (sites.google.com/view/iclr-submission-force-vision): NEVER downloaded
- 0 bytes, on any machine, at any point. Confirmed by log (SESSION_LOG:385, 588, 703) and by
  `docs/PROJECT_CONCLUSIONS.md:368` ("Force-Vision was never downloaded — its contribution is
  taxonomic only"), and by the disk sweep (no `*force*` dirs under `$HOME`).
- No download script exists for it (`scripts/crc/` has only `download_actionsense.sh` and
  `download_opentouch.sh`); `scripts/download_egotouch.py` covers EgoTouch.
- Availability was checked on 2026-07-02 (SESSION_LOG:483): **public Google-Drive zip**.
- Its entire contribution to the project is taxonomic, read from the paper: STAG glove
  (Sundaram et al.), 2,000,000 frames, 89 articulated tools, manipulation types press / hold /
  squeeze, mapped into Axes A/B/C/D in `docs/ACTION_CATEGORIES.md` §2. No tactile array from it has
  ever been loaded.

### Summary table (state as of 2026-08-06)
| Dataset | Downloaded? | How much | Where |
|---|---|---|---|
| ActionSense | Yes, derived form only | 416 MB (299 state + 100 map .npy + manifest/splits) | `data/actionsense_states/` (git-tracked, on GitHub); raw HDF5 deleted from CRC |
| OpenTouch | Yes once, now deleted | was ~14 GB / 26 shards; now 0 | ex-`~/opentouch/data` on CRC; nothing local; result CSV also only on CRC |
| Force-Vision | **No, never** | 0 | n/a — paper-only taxonomy |
| EgoTouch (4th, deprecated) | Yes, on CRC | part of gitignored `datasets/` ~15 GB | not local; deprecated 2026-07-03 |

### Conclusions / risks worth acting on
1. `manifest.jsonl:has_clip` is stale (70 vs 100). Low severity today because the forecasting code
   globs, but it is a live trap for any future consumer. Cheap fix: regenerate the field.
2. `docs/predictability_opentouch.csv` is the only quantitative OpenTouch artifact and it exists on
   exactly one machine (CRC) with no backup. If OpenTouch conclusions are to survive, that CSV
   should be pulled into `docs/` and committed — far cheaper than the 14 GB re-download.
3. Force-Vision remains the one clean way to add a 4th sensor geometry to the cross-dataset trait
   claim, and it is a single public Google-Drive zip. Still open, still cheap.

### No code changed this session (audit only).

---

## SESSION (2026-08-06) — PORT THE FORECASTER TO OPENTOUCH: dataset analysis + PLAN (AWAITING USER)

### Request
User: "look at https://opentouch-tactile.github.io/ in detail; check whether the predictability
algorithm could also be used on this dataset — right now it was only trained on ActionSense.
Analyze the dataset: format? actions? which actions should be downloaded? Reference SESSION_LOG
for how we downloaded before (I believe part of OpenTouch is already on CRC). Ask questions until
100% sure, then WAIT for permission before downloading."

### CLARIFICATION OF WHAT "THE PREDICTABILITY ALGORITHM" MEANS HERE
Two distinct artifacts exist in this project and only ONE of them is ActionSense-only:
1. **Training-free predictability probe** (persH1/15/30, periodicity, contact_migration, PI) —
   ALREADY RUN ON OPENTOUCH (2,496 usable of 2,958 clips, 2026-07-02; SESSION_LOG:530-543,
   docs/ACTION_CATEGORIES.md 3b). Nothing to port.
2. **Trained physical-state forecaster + frozen eval harness** (Phase D/E: F/CoP target, 1 s
   horizon, persistence / seasonal-naive / linear AR / GRU-aggregate / CNN-map / flatten-map,
   frozen 60/20/20 split, sigma-calibration) — ActionSense ONLY. **This is what the user means.**
   Porting it to OpenTouch = the first CROSS-SENSOR test of the four-way ranking
   (AR +0.166 > GRU +0.14 > CNN-map +0.05 > flatten-map -0.03 > persistence 0).
This is also PROJECT_CONCLUSIONS 9 open item #7 ("GPU per-category forecasting on OpenTouch to
convert the probe hypothesis into measured skill").

### DATA STATE (re-verified against the 2026-08-06 audit above)
- OpenTouch raw HDF5: downloaded once on CRC (~14 GB), **DELETED** during the ActionSense disk
  saga. Nothing on this Mac. `docs/predictability_opentouch.csv` also exists ONLY on CRC.
- => A full re-download is required. `scripts/crc/download_opentouch.sh` (26 shard IDs + labels)
  was re-verified today against the upstream `OpenTouch-MIT/opentouch/scripts/download_data.sh`:
  **27 IDs (26 shards + `1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ` labels) — still current, unchanged.**

### DATASET ANALYSIS — OpenTouch (arXiv 2512.16842; opentouch-tactile.github.io)
**Capture**: Meta Aria glasses (egocentric RGB) + FPC full-hand tactile glove + Rokoko Smartgloves
(pose), synchronized at **30 Hz**, ~2 ms mean latency. 5.1 h total, ~3 h densely annotated,
14 environments, ~800 objects / 14 object categories. Single **RIGHT** hand.

**Distribution format**: Google Drive via `gdown`; **26 HDF5 shards** (~561 MB each, ~14.6 GB
total, one shard per scene+participant, e.g. `office_csail_p2.hdf5`, `fablab_ml_p1.hdf5`) plus
`final_annotation.zip` (small).

**HDF5 schema (confirmed live on CRC 2026-07-02, SESSION_LOG:507-523)**
```
<shard>.hdf5
  calibration, transform_slam_to_rgb
  data/<clip_id>/
      right_pressure     (T,16,16) float32, max 3072        <- THE ONLY THING WE NEED
      rgb_images_jpeg    <- the reason shards are 561 MB
      camera_poses, hand_landmarks, timestamps
      labels             (0,0) index-pair — NOT the action
```
**Labels**: `final_annotations/<scene>_merged.csv`, one row per clip, key `clip_id` =
`"<scene>::demo_NNN"` (globally unique), columns: `action` (free-text gerund), `grip_type`
(29-class GRASP taxonomy), `object_name`, `object_category`, `environment`, `description`,
`peak_idx`. Join rate verified 100% on the pilot shard; 457/2,958 clips carry no label row.

**Actions (observed vocabulary, gerunds)**: placing, adjusting, removing, pinching, picking up,
holding, pulling, pushing, moving, pressing, turning, pouring, serving, eating, stirring,
scooping, flipping, wiping, cutting, plus examining, carrying, lowering, aligning, typing,
touching, tightening, unscrewing, tilting, tapping, feeling, inspecting, switching, detaching,
attaching, pointing, resting. Mapped into our taxonomy by `categorize_phrase()`.

**Our own probe ranking on these actions (PI, easiest -> hardest)**:
`pouring +4.4 - serving +3.6 - eating +3.4 - stirring +3.0 - scooping +2.5 - flipping +2.4 -
wiping +1.3` ... `pulling -1.8 - turning -2.2 - moving -2.6 - cutting (n=4) -3.0`.
=> The forecasting experiment has a PRE-REGISTERED PREDICTION: skill should track this order.

### GAP ANALYSIS — what breaks when the ActionSense harness meets OpenTouch
| # | Gap | Detail | Proposed handling |
|---|---|---|---|
| G1 | **One hand** | target is 6-dim `[F,CoPx,CoPy] x {L,R}`; OpenTouch has right only | 3-dim target `[F_R,CoPx_R,CoPy_R]`; compare against the ActionSense RIGHT-hand columns only |
| G2 | **Clip length** | 5.1 h / 2,958 clips = **~6.2 s mean**. Harness `min_history: 40` @10 Hz = 4 s + 1 s horizon = 5 s floor -> most clips yield 0-12 origins | lower `min_history`; report retained-clip count; see OQ3 |
| G3 | **Rate** | OpenTouch is genuinely 30 Hz (ActionSense was 6 Hz *up*sampled then down to 10) | see OQ3 |
| G4 | **Baseline/DC** | FPC pressure is probably ~0 at rest (unlike the conductive thread's ~571/taxel DC that caused bug P4). But clips are segmented around `peak_idx`, so a causal first-N-frames baseline may already be IN contact | measure the rest-value distribution in `--inspect` BEFORE choosing; do not assume |
| G5 | **Force units** | 0..3072 FPC vs uncalibrated thread -> `F` is NOT comparable across corpora | zero-shot transfer would need per-corpus z-scoring; CoP (already [-1,1]) is the only natively transferable channel |
| G6 | **Split axis** | ActionSense manifest has **no subject id** -> its results are within-corpus only (open item #5) | OpenTouch `clip_id` encodes **scene + participant** (`_p1`/`_p2`) -> we CAN hold out environment/participant. **This is a genuine upgrade over ActionSense.** |
| G7 | **Fit grouping** | harness `fit_scope: group` = action x object | OpenTouch has `action` + `object_category` -> maps cleanly |
| G8 | **Map encoder** | flatten branch is `Linear(2048->64)` for 2x32x32 | 1x16x16 = 256 -> re-instantiate input dim (trivial); CNN branch is shape-agnostic |
| G9 | **Disk** | 14.6 GB of shards, of which we need only `right_pressure` (~276 MB fp16 for the WHOLE corpus) | stream-extract (download shard -> extract -> delete), same trick as `stream_actionsense.sh`; result is a permanent ~300 MB local cache, never re-download again |

### WHICH CLIPS/ACTIONS TO DOWNLOAD — the key point
Actions are **scattered across scenes**, so shards cannot be pre-filtered by action *blindly*.
BUT `final_annotation.zip` is small and independent -> **download labels FIRST (minutes), build the
action x scene contingency table, THEN decide which shards to pull.** This turns "what should be
downloaded" from a guess into a measurement. Recorded as the mandatory Step 0.

### PROPOSED PLAN (pending answers)
- **Step 0 (cheap, ~1 min, no commitment)**: `gdown 1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ` -> unzip ->
  action x scene x participant x object_category counts + clip-duration histogram if derivable.
  Decide shard subset from real counts.
- **Step 1**: download the selected shards; stream-extract `right_pressure` + `timestamps` + label
  row -> `data/opentouch_states/{state_N.npy, clip_N.npy(fp16), manifest.jsonl}` mirroring the
  ActionSense layout so the harness loads it with a config swap, not a rewrite; delete each shard
  after extraction. Verify G4 (rest baseline) here.
- **Step 2**: `configs/opentouch/eval_harness.yaml` (3-dim target, rate/min_history per OQ3,
  split per OQ4) + minimal generalizations in `eval_harness/dataset.py` + `splits.py`.
- **Step 3**: classical baselines (persistence / seasonal-naive / linear AR) on the frozen split.
- **Step 4**: GRU-aggregate + CNN-map + flatten-map -> reproduce the four-way comparison figure.
- **Step 5**: per-action skill sweep -> test the probe's pre-registered ordering.
- Report BOTH skill-vs-persistence AND R^2-vs-mean (methodological lesson #4).

### OPEN QUESTIONS (asked of the user 2026-08-06; NOT resolved yet — no code, no download)
- **OQ1 — primary experiment**: in-domain refit of the four-way comparison (sensor-independence
  test) / zero-shot transfer of the ActionSense-fitted models / both / per-action skill sweep?
- **OQ2 — download scope**: all 26 shards, or a label-driven subset, or a 2-3 shard pilot first?
- **OQ3 — rate & history**: downsample 30->10 Hz (exactly matches the ActionSense harness:
  horizon = 10 steps, AR order grid 2..30) vs native 30 Hz (3x the windows, horizon = 30 steps,
  AR order grid must be rescaled) vs both. Couples to G2: at 10 Hz, `min_history: 40` = 4 s and
  most 6 s clips die; proposed default `min_history: 20` (2 s) with retention reported.
- **OQ4 — split axis**: hold out scene/participant (new-environment claim, impossible on
  ActionSense) vs clip-level stratified (apples-to-apples with the ActionSense protocol) vs both.
- **OQ5 — where**: CRC (download script already there, 14 GB fits) then rsync the ~300 MB cache
  local for CPU modelling — vs entirely local.
- **OQ6 — free rider**: while on CRC, also pull back `docs/predictability_opentouch.csv`
  (the only quantitative OpenTouch artifact, single-copy on CRC)? It is a file copy, not a rerun.
  *(Note: it is on CRC only if that file survived; if not, Step 1 regenerates it for free.)*

### STATUS: WAITING for user answers to OQ1-OQ6. Nothing downloaded, no code written.

### DECISIONS (user, 2026-08-06, via AskUserQuestion) — OQ1-OQ5 RESOLVED
| OQ | Decision | Consequence |
|---|---|---|
| OQ1 experiment | **Both**: in-domain refit of the four-way comparison **+** per-action skill sweep | One training pipeline; the sweep is a second reporting pass over the same TEST predictions, grouped by `action`. Pre-registered prediction = the probe order (pouring/serving/eating/stirring/scooping easiest; cutting/moving/turning/pulling hardest). |
| OQ2 download | **Labels first, then decide** | Step 0 = `gdown 1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ` only. Build action x scene x participant x object_category counts. Shard selection is then EVIDENCE-DRIVEN, not guessed. No 14 GB commitment until the table exists. |
| OQ3 rate/history | **10 Hz (downsample 3), `min_history: 20`** | horizon = 10 steps = 1.0 s and AR order grid `[2,5,10,15,20,30]` stay IDENTICAL to the ActionSense harness -> numbers are directly comparable. min_history 40->20 (4 s -> 2 s) so ~6 s clips survive; retained-clip count is a REPORTED number, not a silent filter. Native 30 Hz deferred (not refused) as a later sensitivity check. |
| OQ4 split | **Both; scene/participant hold-out is the HEADLINE**, clip-level stratified reported alongside | Headline = a real new-environment/new-participant generalization claim, which ActionSense structurally could not make (no subject id; PROJECT_CONCLUSIONS 9 open #5). Clip-level = apples-to-apples with the existing ActionSense +0.166. Expect headline < clip-level; that gap is itself a result. |
| OQ5 where | **CRC download+extract, rsync ~300 MB cache local, model locally on CPU** | Mirrors the ActionSense workflow. User runs the CRC commands (I have no SSH); I do everything downstream. |
| OQ (map) | **DEFER the tactile-map branch.** First pass = persistence / seasonal-naive / linear AR / GRU-aggregate | The cache still stores raw 16x16 clips (fp16), so CNN-map/flatten-map can be added later WITHOUT re-downloading. **Recorded honestly: the four-way comparison is therefore NOT completed in this pass** — the refit tests AR vs GRU vs the two classical baselines only. G8 (Linear(2048->64) -> 256) stays open. |

### DEFAULTS I AM SETTING (stated, not asked — override any of these if you disagree)
- **D1 per-action sweep threshold**: report per-action skill only for actions with **n >= 30 clips**
  in the corpus; everything below is pooled into `other (n<30)` and named. Prevents a repeat of the
  probe's `cutting (n=4) -3.0` entry being read as a real ranking position.
- **D2 target**: 3-dim `[F_R, CoPx_R, CoPy_R]` (G1). When quoting ActionSense for comparison I will
  use its **right-hand** channels only, not the 6-dim mean.
- **D3 baseline/DC (G4)**: **measured, not assumed.** Step 1 dumps the per-taxel rest-value
  distribution and the value at frame 0 vs `peak_idx`. Only then do I pick between (a) no
  correction, (b) causal first-N-frames, (c) whole-clip percentile. If clips start already in
  contact, (b) is invalid and I will say so rather than apply it.
- **D4 metrics**: report BOTH skill-vs-persistence AND **R^2 vs the mean** (methodological lesson
  #4 — skill-vs-persistence is only meaningful when persistence is strong).
- **D5 clip retention**: every filter (unlabeled 457 clips, clips shorter than history+horizon,
  low-force CoP masking) reports its drop count in the results table.
- **D6 layout**: cache mirrors ActionSense (`data/opentouch_states/{state_N.npy, clip_N.npy,
  manifest.jsonl}`) + adds `scene`, `participant`, `action`, `object_category`, `grip_type` fields
  to the manifest so the harness loads it via a config swap rather than a rewrite.
- **D7 OQ6**: I will attempt to copy `docs/predictability_opentouch.csv` back from CRC opportunistically;
  if it is gone, Step 1 regenerates it for free from the same extracted cache. Not a blocker.

### STATUS: PLAN COMPLETE, AWAITING EXPLICIT GO. Nothing downloaded, no code written yet.
Next action on GO = Step 0 ONLY (labels zip, ~few MB) -> action x scene table -> return to user
with shard selection before any 14 GB transfer.

### FOLLOW-UP (2026-08-06): "why only right hand?" — VERIFIED FROM THE PAPER, + A PLAN-BREAKING FINDING
User challenged the single-hand claim. I re-verified against arXiv:2512.16842 (HTML) rather than
re-quoting the log, since my claim rested on ONE live shard inspection from 2026-07-02.

**Answer — it is a HARDWARE FACT, not a design choice of ours.** Paper, verbatim:
> "We instrument only the right dominant hand to simplify hardware and standardize annotations."
There is no `left_pressure` array in OpenTouch. Confirmed independently by our own live schema dump
(SESSION_LOG:507-523: `data/<clip>` = right_pressure / camera_poses / rgb_images_jpeg /
hand_landmarks / timestamps / labels — no left field) and by the upstream loader, which reads
`right_pressure` only. So D2 (3-dim target) is forced by the data, not chosen.
Mitigating fact: on ActionSense the RIGHT hand was already the BETTER hand (+0.05 skill over left;
right-hand `CoP_x` is the single most predictable channel, PROJECT_CONCLUSIONS 5.4) because it is
the dominant/tool hand. So we lose the weaker half of the ActionSense target, not the stronger.

**NEW FACT 1 — 169 taxels, not 256.** Paper: "a 16x16 electrode grid around a commercial
piezoresistive film, forming **169 taxels** that uniformly cover the fingers and palmar surface",
palmar side only. So the 16x16 array carries a **structural dead-taxel mask** (~87 dead of 256),
exactly like EgoTouch's 217-of-441. Consequence: dead taxels read ~0 and contribute zero weight to
the F/CoP moments, so the moments stay valid WITHOUT a mask — but any map-model normalization and
any per-taxel baseline must exclude them, or the dead cells will dominate the statistics.
Added to the Step-1 inspection list.

**NEW FACT 2 — CLIPS ARE ~1.9 s, NOT ~6.2 s. THIS BREAKS THE OQ3 DECISION.**
Paper: "Each recording session lasted 5 to 25 minutes, yielding short clips **averaging 57 frames**"
(= 1.9 s @30 Hz), sampled by pressure dynamics (lowest-pressure pre-peak / peak / lowest-pressure
post-peak). My earlier ~6.2 s figure was a bad inference (5.1 h / 2,958 clips) that wrongly treated
total RECORDING time as clip time — **corrected here.** (Residual tension: the paper also says
"2,900 clips (3 hours)" = 3.7 s/clip. Either way the true value is 1.9-3.7 s, not 6.2 s.)
Consistency check: our probe computed `persH30` on these clips, so they are >= 30 frames. OK.

**Why this breaks OQ3 (10 Hz, min_history 20, horizon 10):** a 57-frame clip downsampled 3x is
**19 frames total**. Required = min_history 20 + horizon 10 = 30 frames. **Every clip is discarded.**
Even a 3.7 s clip gives 37 frames = 7 origins. The 10 Hz / 1 s protocol that made OpenTouch
directly comparable to ActionSense is **not viable on this corpus.** Options, none yet chosen:
  (a) **native 30 Hz**, history 0.5 s (15 fr) + horizon 1 s (30 fr) = 45 fr floor -> ~12 origins on a
      57-frame clip. Keeps the 1 s horizon (the physically meaningful quantity); loses step-count
      comparability with the ActionSense AR order grid (rescale 2..30 -> 6..90).
  (b) **shorten the horizon to 0.5 s** at 10 Hz -> 20+5=25 fr floor; still kills a 19-frame clip.
      Only works combined with (a). => 0.5 s horizon @30 Hz = 15+15 = 30 fr floor, comfortable.
  (c) accept the loss and keep only the longest clips — must first MEASURE the length histogram.
**This is now the first thing Step 0/1 must measure**, and it is a cheap measurement: clip lengths
are derivable from the labels + timestamps without pulling all 26 shards.
=> OQ3 IS RE-OPENED. The 10 Hz answer stands only if the measured length histogram supports it,
which the paper says it will not. I will bring the histogram back before committing.

**Net effect on cost (favourable):** at ~57 frames/clip the extracted pressure cache is
2,958 x 57 x 256 x 2 B ~= **86 MB**, not the ~300 MB I estimated. The 14.6 GB of shards is almost
entirely `rgb_images_jpeg` (~87 KB/frame x ~169 k frames), which we discard.

### FOLLOW-UP 2 (2026-08-06): "why is history capped at 0.5 s when ActionSense swept 1/2/3/5/10 s?"
User is right to challenge. Answer, with the numbers measured rather than asserted.

**The cap is NOT a consequence of the 30 Hz rate. It is a consequence of CLIP DURATION.**
History is a quantity of *real time*. Sampling rate changes the frame COUNT inside a clip, never
its DURATION. Going 10 Hz -> 30 Hz triples the samples in a 1.9 s clip; it does not create a 4th
second that is not there. Usable history is capped at `clip_duration - horizon`, full stop.

**MEASURED — ActionSense recording durations** (299 local `state_*.npy`, 30 Hz manifest rate):
`min 2.7 s | p25 11.4 s | median 22.0 s | p75 45.9 s | max 220.1 s | mean 35.7 s | total 178 min`
=> a 10 s history costs `(10+1)*30 = 330` frames, affordable against a 660-frame median recording.
Even so it dropped 24% of recordings (227/299 fit). THAT is why the 1/2/3/5/10 s sweep existed.

**OpenTouch clips are ~10-20x shorter** (paper: mean 57 frames = 1.9 s; the alternative reading
2,900 clips / 3 h = 112 frames = 3.7 s). Window budget @30 Hz, 1 s horizon, stride 1:

| history | frames needed | clip = 57 fr (1.9 s) | clip = 112 fr (3.7 s) |
|---|--:|---|---|
| 0.5 s | 45 | 13 origins -> 38,454 total | 68 origins -> 201,144 total |
| 1.0 s | 60 | **DEAD** | 53 origins -> 156,774 total |
| 2.0 s | 90 | **DEAD** | 23 origins -> 68,034 total |
| 3.0 s | 120 | **DEAD** | **DEAD** |
| 5 / 10 s | 180 / 330 | **DEAD** | **DEAD** |

So 3/5/10 s history is **physically impossible on OpenTouch at any sampling rate** — it would
require 4-11 s clips from a corpus whose own paper reports 1.9-3.7 s. 0.5 s is safe under BOTH
readings; 1 s and 2 s are viable only under the longer reading. **Hence: sweep 0.5 / 1 / 2 s,
with 1 and 2 contingent on the measured length histogram.** I chose 0.5 s earlier as the only
value guaranteed to survive the pessimistic reading, not as a rate-imposed ceiling.

**Why this costs us almost nothing scientifically:** the ActionSense ablation already settled
history length as a non-variable — after early stopping, skill is FLAT across the sweep
(0.513 @1 s -> 0.502 @10 s, PROJECT_CONCLUSIONS 5.4), and the earlier "more history hurts" claim
was retracted as an overfitting artifact. 0.5/1/2 s spans the region where any real effect would
live. The OpenTouch run therefore loses the *dead* end of a sweep already known to be flat.

**THE ONE FACT THAT WOULD OVERTURN THIS** (must be checked in Step 0/1, cheap): the paper says
recording *sessions* ran 5-25 min. If the HDF5 shards contain those long sessions rather than only
the curated short clips, the full 1/2/3/5/10 s sweep becomes available and this whole constraint
evaporates. Current evidence says they do NOT: our live inspection found 113 clips in
`office_csail_p2.hdf5` and 2,958 across 26 shards, matching the paper's ~2,900 *curated* clips
(i.e. shard clips == curated clips). Not yet proven — Step 1 will dump the true length histogram
and settle it. If the long sessions ARE present, I will re-open the history sweep in full.

**Revised OQ3 recommendation** (pending the histogram): native **30 Hz**, horizon **1 s (30 steps)**,
history sweep **{0.5, 1, 2} s**, AR order grid rescaled 2..30 -> ~6..90 frames. Report the retained
clip count at each history length as a first-class number (D5).

---

## SESSION (2026-08-07) — STEP 0 EXECUTED: label reconnaissance. TWO PLAN-CHANGING FINDINGS.

### Pre-download decisions (user, AskUserQuestion)
| Q | Decision |
|---|---|
| What to extract | **pressure + Rokoko pose + timestamps + calibration**; discard ONLY `rgb_images_jpeg`. Rationale: pose costs ~15 MB now and a 14.6 GB re-download later; it is also an input channel ActionSense never had. |
| Shard pass | **stream-extract, delete as it goes** (peak disk ~1 GB, not 14.6 GB) |
| Drive quota | **log failures, continue, retry pass** (extraction is per-shard + append-only -> resumable) |
Also measured: **this Mac has only 4.4 GB free** -> the shards physically cannot land here.
CRC-for-shards / local-for-modelling is now forced, not merely preferred.

### STEP 0 DONE — labels only (459 KB, `final_annotation.zip`). No shards downloaded.
25 CSVs, **2,958 label rows**, 16 columns:
`clip_id, object_name, object_category, environment, action, grip_type, description,
ts_start, ts_end, model, onset_idx, onset_ts, peak_idx, peak_ts, post_idx, post_ts`.
Annotations are LLM-generated (`model` = `gpt-5`) then human-reviewed.
**`ts_start`/`ts_end` (ns) let us settle clip length WITHOUT downloading a single shard.**

### FINDING A — clip lengths are FINE. My pessimistic read was wrong; OQ3 resolves favourably.
| p0 | p5 | p25 | **p50** | p75 | p90 | p99 | max | mean | total |
|---|---|---|---|---|---|---|---|---|---|
| 0.53 s | 1.53 s | 2.03 s | **2.80 s** | 4.16 s | 6.33 s | 15.99 s | 45.99 s | 3.65 s | **3.00 h** |
3.00 h over 2,958 clips reproduces the paper's "2,900 clips (3 hours)" exactly => the paper's
"averaging 57 frames" refers to its own sampled-frame benchmark, NOT to clip length. My 1.9 s
figure was wrong; **the correct median is 2.80 s (84 frames @30 Hz)**.
**History budget @30 Hz, 1 s horizon, stride 1** (clips surviving / origins):
`0.5 s -> 2829 (96%), 191k · 1 s -> 2251 (76%), 153k · 2 s -> 1298 (44%), 101k ·
3 s -> 788 (27%), 70k · 5 s -> 328 (11%), 38k · 10 s -> 90 (3%), 12k`
=> **history sweep {0.5, 1, 2} s is comfortable; 3 s is viable (70k origins).** 5/10 s are not.
CAVEAT to carry: restricting to long clips SELECTS on action (long clips skew to eating/serving/
scooping), so the history sweep must report its action mix or the trend confounds with content.

### FINDING B — THE ACTION DISTRIBUTION BREAKS THE PER-ACTION SWEEP, AND IMPEACHES OUR OWN PRIOR RESULT
66 distinct actions, brutally long-tailed. **Only 14 have n >= 30.** The head is:
`picking up 974 · placing 253 · pulling 247 · pressing 237 · pushing 154 · holding 119 ·
grasping 111 · adjusting 89 · turning 84 · touching 82 · moving 78 · removing 57 · sliding 55 ·
inspecting 32`.
**Every one of those is in the ABRUPT / make-break class** — the hard end of our trait axis.
The actions our probe crowned as most predictable are near-empty:

| action | probe PI (2026-07-02) | **n clips** |
|---|--:|--:|
| pouring | **+4.4** | **7** |
| serving | +3.6 | 10 |
| eating | +3.4 | 8 |
| stirring | +3.0 | **5** |
| scooping | +2.5 | 8 |
| flipping | +2.4 | 12 |
| wiping | +1.3 | 20 |
| cutting | -3.0 | 4 |

**=> CORRECTION TO OUR OWN DOCS.** `docs/PROJECT_CONCLUSIONS.md` 4 and
`docs/ACTION_CATEGORIES.md` 3b quote "pouring +4.4 / serving +3.6 / eating +3.4 / stirring +3.0 /
scooping +2.5" as the OpenTouch headline supporting the cross-dataset trait claim. We flagged
`cutting (n=4)` but NOT the top of the ranking, which rests on **5-10 clips per action**.
OpenTouch's contribution to the trait claim is therefore **far weaker than documented** — it is
suggestive, not confirmatory. The trait claim still stands on EgoTouch (1,929 clips) and
ActionSense (299 clips); OpenTouch should be demoted to corroboration pending the measured rerun.
This must be fixed in both docs. (Not yet edited — flagged to user first.)

**Smooth-class total = 108 clips**, scattered across 21 of 25 scenes (office_ml_p1 21,
eat_ygf_p2 14, home_kitchen_p1 11, home_kitchen_p3 9, ...) => **no shard subset can capture them**;
a targeted download is impossible. This settles OQ2: **download all 26 shards.**

### Consequence for OQ1 — the per-action sweep must be REFRAMED (needs user call)
- The **in-domain refit** (persistence / seasonal / AR / GRU on all clips) is **UNAFFECTED** —
  it does not depend on the action mix. That half of the plan stands as approved.
- The **per-action sweep** cannot test the probe's ordering as designed: with D1 (n>=30) it would
  cover 14 actions that are ALL in the hard class = a restricted-range test with little spread.
  Options: (a) sweep the 14 n>=30 actions and report it honestly as a hard-class-only test;
  (b) pooled TRAIT contrast, smooth (108) vs abrupt (2,850), which tests the actual durable claim;
  (c) both. Recommendation: **(c)**, with (b) as the headline since it matches the real hypothesis.

### Other reconnaissance facts
- 26 shards vs 25 CSVs: `grocery_target_p3_p4_merged_by_ts` merges two shards into one annotation
  file. **This is almost certainly the source of the 457 "unlabeled" clips in the 2026-07-02
  probe** (its join used `<shard_stem>::<group>`, which cannot match the merged prefix).
  `extract_opentouch.py:label_lookup` handles it with a prefix-containment fallback ->
  recovers ~15% of the corpus that the old probe silently dropped.
- Clip-id prefixes: 25 distinct, all exactly matching their CSV stem (verified, no other surprises).
- `environment`: store 1107 · office 674 · kitchen 293 · workshop 232 · supermarket 229 · home 162
  · restaurant 108 · bedroom 55 · ... (21 values) -> supports the scene-level held-out split (OQ4).

### BUILT (compile-checked + unit-tested; nothing run on real shards yet)
- `scripts/extract_opentouch.py` — one shard -> cache (`state_N.npy` (T,1,6) moments,
  `clip_N.npy` (T,1,16,16) fp16, `pose_N.npy`, append-only `manifest.jsonl` carrying scene/action/
  grip/object/environment/peak_idx/T/fps_est). Hand axis kept at extent 1 so harness indexing
  matches ActionSense's (T,2,6). `--taxel-stats` dumps per-taxel activity + p5 rest level.
  **NO baseline correction applied** — deliberately deferred (D3): the ActionSense DC bug must not
  be blind-fixed on a different sensor when clips are segmented around a pressure peak.
- `scripts/crc/stream_opentouch.sh` — labels -> per shard {gdown -> extract -> `rm`} -> summary;
  `done_ids.txt` / `failed_ids.txt` make it resumable; `--taxel-stats` on shard 1 only.
- **Unit tests on synthetic (all pass)**: single hot taxel -> CoP equals that cell's coordinate
  exactly; empty frame -> F=0, CoP=0, all finite (no NaN); sliding blob -> CoPx strictly monotonic
  with CoPy constant (the stroke signature); 169-live-of-256 dead-taxel grid -> finite moments.

### STATUS
Step 0 complete. Extraction code ready and tested. **The 14.6 GB shard pass has NOT started** —
it must run on CRC (this Mac has 4.4 GB free) and I have no SSH, so the user runs one command.
Awaiting the user's call on the FINDING-B reframe (OQ1) and on fixing the two docs.

### CHALLENGE (user): "download all actions, even though the predictable ones are a small amount?"
Answered with measurement, and the measurement INVERTS the rationale.

**The predictable subset is not a dataset.** Trait-class totals (hist 0.5 s + 1 s horizon @30 Hz):
| class | clips | duration | windows |
|---|--:|--:|--:|
| SMOOTH / predictable | 108 | **8.2 min** | 9,847 |
| ABRUPT / hard | 2,850 | 171.8 min | 181,286 |
| ALL | 2,958 | 180.0 min | 191,133 |
| *(ActionSense for scale)* | *299* | *177.9 min* | — |
=> 8.2 min is **22x less** than ActionSense. A forecaster cannot be TRAINED on it. So
"download only the predictable actions" is not a cheaper experiment, it is **no experiment**.
The targeted download is also a poor trade on its own terms: the best 7 shards = 3.9 GB (27% of
the bytes) return only 68% of the smooth clips, because they are spread over 21 of 25 scenes.

**The correct rationale for taking all 26 shards — OpenTouch is the HARD-POLE COMPLEMENT.**
- ActionSense: 178 min, overwhelmingly SMOOTH (slice/peel/pour/clean).
- OpenTouch: 172 min, overwhelmingly ABRUPT (picking up/pressing/pulling/grasping).
- Near-identical duration, opposite poles => together they span the trait axis with ~175 min at
  EACH end. This is a much stronger position than "more smooth data", which OpenTouch can never
  supply. It also retires the earlier framing that OpenTouch's value is its predictable actions.

**NEW FALSIFIABLE PREDICTION (the payoff):** the trait hypothesis predicts OpenTouch's overall
1 s forecast skill should land **clearly below ActionSense's +0.166**, because 96% of its clips
are abrupt/make-break. If a 96%-abrupt corpus forecasts as well as a smooth one, **the trait claim
— the project's most portable finding — is falsified.** ActionSense could not run this test (its
hard actions, jar / get-replace, were a minority). Recorded BEFORE the data is seen.

**Design consequence (resolves the FINDING-B reframe):** train on the FULL corpus, then EVALUATE
stratified by trait class. 108 clips is too few to train a per-class model but 9,847 windows is
ample to evaluate on. Holding sensor, pipeline and model constant makes this a CLEANER trait test
than any cross-dataset comparison. => Option (c) from FINDING B, with the trait contrast as
headline and the 14-action n>=30 sweep as the secondary, restricted-range view.

**RISK, named in advance:** if OpenTouch skill lands ~0 across the board, the AR-vs-GRU ranking
test is underpowered (models all near zero are hard to order). Mitigation: 191k windows give power
at small effect sizes; report CONFIDENCE INTERVALS, not bare point estimates. If the CIs overlap,
the honest conclusion is "the ranking does not replicate at measurable resolution", NOT a re-rank.

**DECISION CONFIRMED: download all 26 shards.**

### CORRECTION (2026-08-09, prompted by user): "train on full corpus" IS CONFOUNDED FOR THE TRAIT TEST
User asked why training on the full corpus is reasonable and what stratified evaluation buys.
The first question exposes a genuine flaw in the design I logged on 2026-08-07. Recording it.

**THE FLAW.** I wrote "train on the FULL corpus, then EVALUATE stratified by trait class". With a
corpus that is 96% abrupt, a model trained pooled and then scored on the 108 smooth clips gives a
number with TWO indistinguishable causes: (i) smooth actions are intrinsically harder, or (ii) the
model hardly saw smooth actions in training. That is a TRANSFER measurement mislabelled as a TRAIT
measurement. The trait claim cannot be tested that way.

**THE FIX — the two experiments need DIFFERENT training designs. They were conflated.**
- **E1 sensor-independence** (does AR > GRU > CNN-map > persistence replicate on a 2nd sensor?):
  question is about MODEL FAMILIES, not action classes. Fit on the FULL corpus; the action mix is
  simply what OpenTouch is. **Pooled training is CORRECT here.** Unchanged from the approved plan.
- **E2 trait test** (are smooth actions more forecastable?): pooled training is the WRONG tool.
  Use only estimators with no cross-class contamination:
  1. **Training-free**: persistence nMSE @1 s and R^2-vs-mean, per class. No fitting => no confound.
     (This is exactly why the Phase-C probe result was clean, and it is the honest primary metric.)
  2. **Linear AR fit PER CLASS** (`fit_scope`): each class gets coefficients from its own data.
     Smooth = 9,847 windows, ample for AR(90) x 3 channels; abrupt = 181,286. Unconfounded, and AR
     is our best model anyway (it won the four-way comparison).
  3. **Pooled GRU is reported for E1 ONLY**, flagged as confounded by class imbalance for E2.
- SCOPE NOTE: the ActionSense config uses `fit_scope: group` = action x object, which is too fine
  for OpenTouch's rare actions (stirring n=5). **E2 fits at TRAIT-CLASS scope (2 groups)**; the
  n>=30 per-action sweep fits at action scope where the counts support it.

**WHY STRATIFY AT ALL (answering the second question).** The pooled average is structurally
incapable of expressing the finding. If the truth were smooth +0.22 / abrupt +0.06, the aggregate
is `0.96*0.06 + 0.04*0.22 = 0.067` — indistinguishable from the abrupt number alone. A real 3.7x
class difference vanishes into one digit because the 96% majority swamps it. Every claim in this
project is about DIFFERENCES BETWEEN KINDS OF ACTION, so on the trait question the stratification
is not post-hoc analysis layered on the result — **it IS the result**.

**Net effect on the download decision: NONE.** Both E1 (needs maximum data) and E2 (needs both
poles present, per-class fitting) want the full corpus. All 26 shards still stands.

### 2026-08-10 — FIRST EXTRACTION RUN PRODUCED A CORRUPT CACHE. TWO BUGS, BOTH MINE. FIXED.
User's scene table (cache vs labels) exposed it: `have` is an exact multiple of `want` for most
scenes — grocery_plant 140/70, office_ml_p1 662/331, sports_dicks 460/230 (2x);
hardware_homedepot_p5 552/184 (3x) — while home_bedroom 138/138 and office_csail_p2 113/113 are
correct. **Summing `have` = 4,511 manifest rows for a corpus of 2,958 clips.** More clips than
exist => definitive over-extraction. Meanwhile only 2,666 `state_*.npy` files exist on disk.

**BUG 1 — CONCURRENT RUNS (no lock).** The user legitimately started the script more than once
while working out screen/tmux/nohup. Consequences, all three of which the script permitted:
  (a) each instance read an empty/stale `done_ids.txt` -> re-extracted the same shards;
  (b) `next_index()` read the same manifest in both -> both wrote the SAME `state_N.npy`
      filenames -> **files overwrote each other, so the manifest row -> file mapping is broken**
      (4,511 rows vs 2,666 files). This is why no dedupe can repair the cache;
  (c) `rm -f "$SHARDS"/*.hdf5` at the top of each loop deleted the OTHER instance's in-flight
      download -> **the 19 "failures" were self-inflicted, not Drive quota.**
**BUG 2 — LABEL JOIN COLLISION (`p1`/`p2`).** The log shows shards `sports_dicks_p1` AND
`sports_dicks_p2`, but there is only ONE `sports_dicks` CSV. My prefix-containment fallback let
BOTH `p1::demo_000` and `p2::demo_000` claim the same annotation row, so one of them carried a
WRONG action/grip label. Independent of the concurrency bug, and worse: it is silent. My
2026-08-07 note claimed this fallback "recovers ~15% of the corpus" — it was also corrupting it.

**FIXES (committed; compile-checked + unit-tested).**
1. `stream_opentouch.sh`: atomic `mkdir` lock (`$WORK/.stream.lock`, NFS-safe) with EXIT/INT/TERM
   trap and an explicit stale-lock message; **per-instance shard dir `shards.$$`** so no run can
   delete another's download; `sort -u` on `done_ids.txt` to collapse earlier duplicates.
2. `extract_opentouch.py`: **join by TEMPORAL OVERLAP, not by name.** The CSV carries
   `ts_start`/`ts_end` (ns) and every clip carries `timestamps`, so the join is exact and
   name-independent — evidently how the authors merged p3/p4 (`..._merged_by_ts`). Rules: exact
   key accepted only if it genuinely overlaps (>=50% of the clip span); otherwise best-overlap row;
   **a label row may be claimed by at most one clip**; no overlap => reported MISS, never a
   silent wrong label. Manifest now records `label_cid` + `join` method + overlap fraction.
3. `extract_opentouch.py`: **idempotent** — `read_manifest()` returns the set of already-extracted
   `<shard>::<group>` keys and claimed label cids; re-running after an interrupt skips instead of
   duplicating.
**Unit tests (pass):** p1/p2 same group name at different times -> each gets its OWN correct row
(`pulling` / `pouring`); a claimed row is never handed out twice; a non-overlapping clip returns
MISS rather than a wrong label. Plus the earlier moment tests still hold.

**CONSEQUENCE: the cache must be REBUILT from scratch (~14.6 GB re-download).** Unavoidable —
(b) means row->file identity is lost, and (2) means an unknown subset of labels is wrong. Deleting
`~/opentouch/cache` and re-running is the only sound path. Cost is time, not data.
**Confirmed good so far:** `fps est` median **30.01 Hz** on every shard (min 30.00, max 30.01) =>
the native-30 Hz / 1 s-horizon plan stands. The `taxels:`/DC-offset line was truncated in the
terminal and is still OUTSTANDING.
**METHOD NOTE:** the scene-vs-label table is what caught this; a bare clip count (2,666 of 2,958)
looked merely incomplete. Any future cache build must be verified per-scene against the labels,
not by total count. Added as a standing check.

---

## SESSION (2026-08-10续) — CONFIG + LOADER 架构定稿,TRAIN PLAN 完整记录,新增开放问题

### 用户的强约束(覆盖了我此前的"零风险机械替换"提案)
用户明确要求:**`src/actionsense/` 下任何文件一个字节都不准改**,哪怕改动在数学上对 ActionSense
可证明是 no-op(字面量 6 -> `len(cfg.channels)`,ActionSense 自己 channels 长度就是 6)。
理由未强制要求,但与本项目一贯的"frozen harness"文化一致(参见 `categorize_phrase()` 新增而不
改 `categorize()`、EgoTouch/OpenTouch/ActionSense 用独立 probe driver 而非共享一份改出分支)。
**采纳,不再论证"风险可控",按用户决定执行。**

### 逐文件核实结果(读完 `src/actionsense/eval_harness/` 全部源码后)
| 文件 | 硬编码 6? | 处理方式 |
|---|---|---|
| `config.py` (`Config`/`load_config`) | 无 | **import 复用**(零 ActionSense 专属逻辑) |
| `dataset.py::Norm` / `force_thresholds` | 无(形状由数据推导) | **import 复用** |
| `dataset.py::load_target` / `group_keys` | 硬编码双手 / 解析 `"Slice a cucumber"` 字符串 | 不复用,OpenTouch 自己重写(数据格式本就不同,不是"改 ActionSense") |
| `baselines/persistence.py` | 无(`np.repeat` 形状随输入) | **import 复用** |
| `baselines/__init__.py` | 无(纯注册表) | 新建 OpenTouch 版,内部 import 上面两个未改的 + 下面 fork 的 |
| `masking.py` | 有,1 处:`np.ones((N,6))` | **fork** -> `np.ones((N, C))` |
| `metrics.py` | 有,4 处:`.reshape(-1,6)` | **fork** -> 从 `mask.shape[-1]` 推导,不需要传 cfg(保持这些函数纯数组运算的设计) |
| `baselines/base.py` | 有,1 处:空语料兜底 `np.zeros((0,H,6))` ×2 | **fork** -> `len(cfg.channels)` |
| `baselines/ar.py` | 有,4 处:系数矩阵形状、`range(6)`、`buf` 补零、输出数组 | **fork** -> `len(self.cfg.channels)` |
| `baselines/seasonal.py` | 有,1 处:输出数组 `np.empty((H,6))` | **fork** -> `len(self.cfg.channels)` |
| `evaluate.py` | 有,6 处:mask reshape、`range(6)`、shape 校验消息 | **fork** -> `len(cfg.channels)` |

### 新建文件清单(全部路径)
```
configs/opentouch/eval_harness.yaml     新建,结构对齐 configs/actionsense/eval_harness.yaml,数值重算
src/opentouch/__init__.py               新建,空
src/opentouch/dataset.py                新建(非 fork)—— load_target(单手 (T,1,6)->(T',3))、
                                         group_keys(按 object_category)、eligible_clips(时长过滤)
src/opentouch/masking.py                fork of src/actionsense/eval_harness/masking.py
src/opentouch/metrics.py                fork of src/actionsense/eval_harness/metrics.py
src/opentouch/evaluate.py               fork of src/actionsense/eval_harness/evaluate.py
src/opentouch/baselines/__init__.py     新建 —— import Persistence(未改,来自 actionsense)+
                                         本地 SeasonalNaive/AR(fork)+ base 的 Baseline/predict_series/origins(fork)
src/opentouch/baselines/base.py         fork of .../baselines/base.py
src/opentouch/baselines/ar.py           fork of .../baselines/ar.py
src/opentouch/baselines/seasonal.py     fork of .../baselines/seasonal.py
tests/test_harness_opentouch.py         新建 —— 把 tests/test_harness.py 的 5 类单测(seasonal精确
                                         恢复/AR系数恢复/masking/causality/seasonal fallback)在 3
                                         通道下对 fork 版本重跑一遍,证明 fork 与原版行为一致
src/opentouch/splits.py                 尚未建 —— 阻塞于 p1/p2 语义未定(见下)
```
**代价明确记录**:fork 出的 6 个文件与 ActionSense 原文件短期内存在重复代码;若日后在
`src/actionsense/eval_harness/` 发现 bug,这 6 个 fork 不会自动同步,需要人工对照修。用户已确认
接受这个代价,换取"绝不触碰已发表结果的代码路径"。

### CONFIG 最终数值(`configs/opentouch/eval_harness.yaml`)
| 字段 | 值 | 依据 |
|---|---|---|
| `channels` | `[F_R, CoPx_R, CoPy_R]` | 只有右手(arXiv:2512.16842 verbatim) |
| `force_idx` / `cop_idx` | `[0]` / `[1,2]` | 同上 |
| `fps_raw` / `downsample` | `30.0` / `1` | 实测 fps_est 全语料中位数 30.01(2026-08-10 corrupted-but-rate-valid 那次运行确认),原生使用 |
| `horizon_s` | `1.0`(=30步) | 与 ActionSense 物理时长一致 |
| `ar_orders` | `[6,15,30,45,60,90]` | ActionSense `[2,5,10,15,20,30]`@10Hz 的物理秒数(0.2~3.0s)按 30Hz 等比换算 |
| `seasonal_period_min/max_s` | `0.3` / `3.0` | 保持不变(人手动作周期的先验假设,与传感器无关) |
| `fit_scope` | `object_category` | Q2 用户确认(14 类,样本比 action×object 均衡) |
| `min_history` | `15` 帧(0.5s@30Hz) | Q3 用户确认,覆盖 96% clip |
| `actions` | `[]`(不过滤) | 训练语料 = 全部 2,958 条(此前"用全部的2958"决定) |
| `split` | 占位,**不生效** | 阻塞见下 |

### SPLIT 现状:仍然阻塞
`p1`/`p2` 到底是participant还是session,论文/仓库均未回答。用间接证据(同地点 p 变体间
`object_name` 重叠率仅 1%-17%;`hardware_homedepot_p5` 一个 shard 内 `environment` 标签跨
store/kitchen/garage,说明 shard=一次连续外出行程而非固定地点)双向都说得通,**无法判定**。
后续路径:下载完成后查 HDF5 `calibration` 字段;仍不行则邮件作者 `rayxsong@mit.edu`。
`src/opentouch/splits.py`(分层 train/val/test 划分)**不写**,直到本项解决。

### HISTORY SWEEP(第三件事,今天不建,仅记录已确认的参数)
不属于 frozen harness(AR 自选阶数,不需要 history 字段)。属于未来的 GRU-aggregate/
action-dynamics 移植脚本。用户确认:sweep 用固定子集(hmax=3s,788 条,91.4 分钟,动作构成
top5 picking-up/placing/pulling/pressing/holding,未坍缩成单一动作)+ history 值 **{1,2,3}s**
(我的理解:0.5s 不进 sweep,只作为 frozen harness 的 min_history floor,已在下方开放问题中列出待
确认)。ActionSense 原 sweep 是 {1,2,3,5,10}s;5s/10s 在 OpenTouch 上会把子集砍到 328/90 条,
动作多样性坍缩,弃用是有意为之,不是遗漏。

### TRAIN PLAN —— 三个待验证结论 G1/G2/G3,以及各自的验证方法
"验证 generalizability" 拆成三个强度递减的主张:

**G1 — 排序可跨传感器复现**:AR > GRU-aggregate > persistence(> seasonal,预期 inert)在 OpenTouch
上是否成立。**验证方法**:在全部 2,958 条(pooled,不分 trait class)上重跑同一套 frozen harness
+ 一个新训练的 GRU-aggregate(权重从零开始,不借用 ActionSense 权重——G1 测的是"算法排序",不是
"权重可迁移")。AR 按 `object_category` 分组拟合。指标:skill-vs-persistence 为主(这是 ActionSense
原协议,此处目的就是复现同一协议,合理),R² 作为交叉验证的次要读数。

**G2 — trait 成立**(平滑力比突变力更好预测)。**验证方法**:不能用 pooled 训练的模型评估
(2026-08-09 已记录这个混淆的修正)。改用:(a) training-free 指标(persistence nMSE、R²-vs-mean),
按 trait class 直接算,零拟合零混淆;(b) 按 trait class **分别拟合**的 AR(smooth 9,847 windows,
abrupt 181,286 windows,样本都够 AR(90)×3通道)。Pooled GRU **不用于 G2**,只用于 G1。
**预注册的可证伪预测**(2026-08-09 记录,早于看到任何数据):OpenTouch 整体 skill 应明显低于
ActionSense 的 +0.166,因为 96% 的 clip 是 abrupt 类;若两者相当,trait 主张证伪。

**G3 — 权重迁移**:**尚未获得用户批准**。我在 2026-08-10 建议从"zero-shot 直接推理"改为
"ActionSense 预训练 + OpenTouch 微调 vs 从零训练"对比(理由:力的量纲、传感器几何、动作分布都不
同,zero-shot 数字失败了也无法归因;微调对比是 EgoTouch 阶段 pretraining 把 LOTO 从 ≈0 拉到
+0.097 的同一设计),但这个改动从未得到用户明确 yes/no,**记为开放问题,见下**。

### REPRESENTATION / MODELS / METRICS 迁移的完整表述
- **Representation**:`[F, CoP_x, CoP_y]` 解析物理量,公式与 `src/actionsense/physical_state.py`
  完全一致(已在 `extract_opentouch.py::moments` 里独立实现并单测验证等价)。CoP 已归一化到
  `[-1,1]`,跨传感器几何可比;**F 是未标定的任意单位**(OpenTouch FPC 0–3072 vs ActionSense 导电
  线另一套标度),任何跨语料的 F 数值比较必须先各自 z-score,原始单位不可比。
- **Models**:persistence/seasonal/AR 结构完全可迁移,不存在"迁移"的概念——就是同一算法在
  OpenTouch 自己的 TRAIN 上重新拟合。GRU-aggregate 架构可迁移(输入维度 6→3 是平凡改动),G1 用
  从零训练的权重,G3(如果批准)才会真正用到 ActionSense 训好的权重做起点。
- **Metrics**:见下方开放问题——不是简单"迁移",这里有一个此前对话里被我低估的真实问题。

### 开放问题清单(未解决,按重要性排序,需要用户逐一确认)

**OQ-A(重要,影响 G2 结论怎么写)—— skill-vs-persistence 对 G2 是结构性偏的,R² 应为 G2 的主指标。**
`PROJECT_CONCLUSIONS §7` 第4条已经记过:skill-vs-persistence 只有在 persistence 是"强" baseline
时才有意义;当年 ActionSense 的 fast-target 上 persistence 比 predict-mean 还差(§5.5),skill 被
系统性抬高。这个陷阱在 OpenTouch 的 G2 上**更致命**,因为 smooth/abrupt 两个 class 的 persistence
强度**本来就不同**(smooth = 力变化慢 = persistence 天然强;abrupt = 突变 = persistence 天然弱,
这就是"abrupt"的定义本身)。也就是说:即使模型在两个 class 上的**真实**预测能力毫无差别,
skill-vs-persistence 也会显示 smooth 更高——因为分母(persistence 的 MSE)结构性更小。
**这不是"也报告一下 R²"能解决的**(我 08-09 的说法不够精确),而是:**G2 的结论必须以 R²-vs-mean
为准,skill-vs-persistence 只能作为诊断性的次要数字**,否则整个 trait 论证可能是在测量
persistence 强度的差异,而不是模型能力的差异。这个问题此前完全没讨论过,需要你确认这个判断。

**OQ-B(新发现,此前完全没讨论过)—— OpenTouch clip 是围绕力峰值构造的,均匀滚动窗口可能被
"无聊片段"稀释。** 论文原话:"we sample frames by pressure dynamics: lowest pressure pre-peak
(approach), peak pressure (manipulation), and lowest pressure post-peak (release)"——即每条 clip
天生包含一次力的剧烈变化,而且标注里就有 `onset_idx`/`peak_idx`/`post_idx` 三个位置。如果按现在
的设计(stride=1 均匀滚动取 origin),大多数窗口会落在峰值前后的"平淡"区间,只有少数窗口真正跨越
那次转变——预测能力的信号可能被平淡窗口稀释、模糊了我们真正关心的"转变时刻附近可预测吗"这个问题。
这和本项目此前几次"想当然的设计被数据打脸"(resampling artifact、DC offset、非因果 filtfilt)是
同一类风险,值得同等重视。**要不要按 `peak_idx` 邻近程度对 origin 分层/加权?还是先不处理,留作
诊断项?** 这个问题需要你决定,我没有默认值。

**OQ-C(工程细节,影响聚合方式)—— per-window 还是 per-clip 聚合?**
现有 `metrics.py` 是把所有 (origin, horizon-step) 对等地汇总(`.reshape(-1,C).sum(0)`)。OpenTouch
clip 长度差异比 ActionSense 大得多(最短 16 帧,最长 1380 帧),等权重汇总意味着**长 clip 主导
指标**,短 clip(往往是最"abrupt"的那类)权重被稀释。是否要改成先按 clip 算指标、再对 clip 取平均
(每条 clip 权重相等)?这个选择会实质影响 G2 的数字,需要你决定。

**OQ-D(方法论)—— 置信区间怎么算?**
2026-08-09 已提过"如果 OpenTouch skill 接近 0,AR vs GRU 排序检验的统计功效不足,要报 CI 不能只
报点估计"。具体怎么算 CI 还没定:我建议 **bootstrap 按 clip 重采样**(不是按 window),因为同一
clip 内的 window 高度自相关——这正是"可预测性"这个概念本身的含义,按 window 重采样会低估方差、
给出过窄的 CI。这是我的建议,需要你确认。

**OQ-E(轻量,建议但非阻塞)—— seasonal-naive 要不要算?**
ActionSense 上 seasonal-naive 被发现完全 inert(5 组全部 fallback 到 persistence)。OpenTouch clip
更短,大概率更 inert。建议仍然计算(fork 已经顺带做了,零额外成本),把"是否复现同一个 null
result"当作一个确认性检查而不是主结果。默认会做,除非你反对。

**OQ-F(未解决,此前只是我单方面提议)—— G3 到底做不做,做哪个版本?**
见上方 G3 段落。需要你明确 yes/no:(a) 不做迁移实验,只做 G1+G2;(b) 做"ActionSense 预训练 +
OpenTouch 微调 vs 从零训练"对比;(c) 其他方案。

**OQ-G(小,确认我的理解)—— GRU-aggregate 是点预测还是概率预测?**
我的理解:G1 的 GRU-aggregate 应该是**点预测**(和 AR/persistence 同协议,只算 skill,不算
coverage),对应 `PROJECT_CONCLUSIONS §6.4` 描述的四路对比里的那个 GRU,而不是 `action_dynamics.py`
里 mean+logvar 的概率版本(那是另一条线,且已经决定不迁移 tactile-map 分支)。如果理解有误请纠正
——这决定了 `src/opentouch/metrics.py` 要不要顺带移植 coverage 计算(目前判断不需要)。

**OQ-H(小,确认 history sweep 细节)**:上方"HISTORY SWEEP"一节里我的理解是 sweep = {1,2,3}s,
0.5s 只当 min_history floor 不进 sweep——如果你的意思是 sweep 也包含 0.5s(即 {0.5,1,2,3}),
请指出。

### 本次会话代码状态
以上均为设计与记录,**代码尚未开始写**——下一步开始按已获批准的部分(configs yaml + `src/opentouch/`
新建/fork 文件 + 对齐单测)动手实现,实现后会用 `tests/test_harness_opentouch.py` 证明 fork 版本
与原版数值行为一致(仿照 `tests/test_harness.py` 的 5 类测试,通道数从 6 改成 3 重跑)。

### 2026-08-10续2 —— CONFIG + LOADER 已实现并验证;发现一个新的开放问题(OQ-I)

**已建文件**(均已写完,`src/actionsense/` 确认零改动 —— `git status --short src/actionsense/`
输出为空):
```
configs/opentouch/eval_harness.yaml     数值见上方 CONFIG 表格
src/opentouch/__init__.py               空
src/opentouch/dataset.py                新建:load_target/group_keys/eligible_clips
                                         + import 复用 Norm/force_thresholds(未改)
src/opentouch/masking.py                fork,1 处 6->C(从 mask.shape 推导)
src/opentouch/metrics.py                fork,4 处 6->mask.shape[-1]
src/opentouch/evaluate.py               fork,6 处 6->len(cfg.channels);main() 因
                                         splits.py 未建而 NotImplementedError,但
                                         fit_and_forecast/build_rows/score_external
                                         均可独立调用测试
src/opentouch/baselines/__init__.py     新建:import 未改的 Persistence + 本地 fork
src/opentouch/baselines/base.py         fork,1 处(空语料兜底形状)
src/opentouch/baselines/ar.py           fork,4 处
src/opentouch/baselines/seasonal.py     fork,1 处
tests/test_harness_opentouch.py         新建:tests/test_harness.py 的 5 类单测在 3
                                         通道下对 fork 重跑
```

**验证**:`pytest tests/test_harness.py tests/test_harness_opentouch.py` —— **14/14 通过**
(原 7 个字节不动,新 7 个在 3 通道下复现同样的性质:seasonal 精确恢复/AR 系数恢复/masking 正确/
因果性/seasonal fallback)。另外用合成 manifest(12 条,4 个 `object_category`,状态数组随机
生成)跑通 `dataset.eligible_clips`/`group_keys`/`load_target` + `evaluate.fit_and_forecast`
端到端,determinism check(两次跑结果逐字节相同)通过。

**OQ-I(新发现,写分层抽样的 `splits.py` 时必须处理)—— AR 对"只在 VAL/TEST 出现、TRAIN 没见过
的组"会直接 `KeyError` 崩溃。**
第一次合成测试故意让 `"jar"` 只出现在 VAL、不出现在 TRAIN,复现了这个崩溃:`_best_order` 会给
陌生组临时设置 order,但 `self.coef` 里从未 fit 过该组,`predict()` 里 `self.coef[group][p]`
直接 `KeyError`。**这不是 fork 引入的新 bug**——`src/actionsense/eval_harness/baselines/ar.py`
的 `_best_order`/`predict` 逻辑完全一样,同样输入会同样崩溃;只是 ActionSense 用 `action×object`
分组只有 5 组,标准分层抽样几乎不可能让某组只出现在 VAL/TEST。**OpenTouch 换成 `object_category`
(14 组,部分类别样本很少,如未来若改用更细的 `action` 分组,`stirring` 只有 5 条)之后,这个边界
情况现实存在**。补一份"每个类别在 train/val/test 都至少出现一次"的合成数据后,pipeline 端到端跑
通、determinism 校验通过——证明问题只在"组未覆盖"这个边界,不在 fork 本身。
**需要在设计 `src/opentouch/splits.py` 时解决,两个方向,需要你选:**
- (a) **分层抽样时强制保证** train 覆盖 val/test 出现的每一个 `object_category`(样本数极少的
  类别,例如只有 1-2 条的,直接归并到一个 `"other"` 类别,不单独分组);
- (b) **让 AR 对陌生组更健壮**——遇到 `self.coef` 里没有的组,退回全局 pooled 系数或退回
  persistence,而不是崩溃。
我倾向 (a),因为它更简单、更透明(崩溃总比静默退化成 persistence、不留痕迹地稀释结果更安全),
但这个决定应该在写 `splits.py` 之前定,现在先记录,不阻塞今天的 config/loader 工作。

### 待用户确认的问题总清单(截至本次)
OQ-A(G2 应以 R² 为主指标,skill-vs-persistence 结构性偏)、OQ-B(clip 围绕力峰值构造,均匀
窗口是否稀释信号)、OQ-C(per-window 还是 per-clip 聚合)、OQ-D(bootstrap 按 clip 还是按
window)、OQ-E(seasonal-naive 要不要算,默认做)、OQ-F(G3 zero-shot 改成 fine-tune,是否批准)、
OQ-G(GRU-aggregate 点预测还是概率预测,我倾向点预测)、OQ-H(history sweep 是否含 0.5s)、
**OQ-I(新增,split 未覆盖组的处理方式,倾向方案 a)**——均未拍板,继续讨论。

### 2026-08-10续3 —— OQ-I 落地(方案a),顺带纠正 Q2 的一个事实错误

用户确认 OQ-I 用方案 (a)。实现前先测了真实的 `object_category` 分布(此前从未测过)——发现
**Q2 当时"14 类,样本均衡"的说法是错的**:`object_category` 实际有 **110 个不同值**,长尾程度
不亚于 `action`(81/110 类 n<30,39 类 n<10,若干 n=1 如 `oven`/`sock`/`joystick`)。当时是把
"14 个 n≥30 的 action"这个数字错套到了 object_category 上,从未单独测过。已在
`configs/opentouch/eval_harness.yaml` 的注释里纠正。

**阈值权衡**(实测):n≥10 保留 71 类、`other` 仅 6%;n≥30 保留 29 类、`other` 32%;n≥50 保留
17 类、`other` 47%。选 **n≥30**(与项目已有的"可靠动作"门槛一致),写成 `min_group_size: 30`,
可配置,不是写死的常量。

**实现**(`src/opentouch/dataset.py`):
- `category_counts(cfg, field)` —— 对**全量 manifest**(不是当前查询的 idx 子集)统计每个类别的
  clip 数。刻意设计成与子集无关:否则同一个稀有类别在只查小子集时可能被错误地当作"未出现过"而不是
  "已知稀有",分类结果会随调用方式漂移。
- `group_keys` 新增合并逻辑:corpus-wide 计数 < `min_group_size` 的类别一律并入 `"other"`;
  空值仍归 `"unknown"`(不与 `"other"` 混淆——空值是标注缺失,稀有类别是标注存在但样本少,两者
  语义不同,合并会掩盖数据质量问题)。
- `missing_groups(train_groups, other_groups)` —— OQ-I 方案 (a) 的运行时校验工具,和 split 轴
  (scene/clip/participant)完全无关,`splits.py` 定下轴之后可以直接调用,不用改这个函数。

**单测**(合成数据:5 条 handle / 2 条 cup / 1 条 oven / 1 条空值,`min_group_size=3`):
`category_counts` 正确统计;`group_keys` 下 handle 保留、cup+oven 并入 other、空值归 unknown;
**稳定性检验**——只查询那 2 条 cup 中的 1 条,分类结果仍是 `other`(证明用的是全量计数,不是
子集计数);`missing_groups` 在覆盖完整时返回空集,在故意制造缺口时正确返回 `{"other"}`。全部
通过。回归:`pytest tests/test_harness.py tests/test_harness_opentouch.py` 仍 14/14 通过,
`src/actionsense/` 仍零改动。

`src/opentouch/splits.py` 本体依然阻塞于 p1/p2(split 轴未定),不受本次影响。

---

## SESSION (2026-08-11) — OQ-A~H 全部拍板;TRAIN PLAN v2(完整操作化,待 review,未动代码)

### OQ 拍板结果(用户 2026-08-11 逐条回答,原样记录)
| OQ | 结论 |
|---|---|
| A | **G2 指标三层**:primary = class-specific R²(`R²=1-SSE_model/SSE_mean`)+ ΔR²=R²_smooth-R²_abrupt;secondary = raw MSE/MAE;diagnostic-only = skill-vs-persistence。G2 的结论以 R² 为准,MSE/MAE 佐证,skill-vs-persistence 仅供参考不参与推断。 |
| B | **不做** peak-proximity 加权;记入"可继续改善"章节,效果不理想时再启用。 |
| C | 同意 per-clip 等权聚合(不按 window 数加权)。 |
| D | 同意 **clip-level bootstrap**;模型对比(G1)用 **paired** clip bootstrap;理由:海量 rolling window 不是独立样本,近似独立的单位是 clip,按 window 重采样会把高度自相关的观测当独立样本,严重低估方差。 |
| E | seasonal-naive 做,但非主结果(诊断性复现检查)。 |
| F | G3(迁移/微调)**暂不做**。 |
| G | GRU-aggregate 用**点预测**,不做概率版本。 |
| H | history sweep 保持 **{1,2,3}s**(0.5s 不进 sweep,只作 frozen harness 的 min_history floor)。 |

以下是把这些答案操作化时浮现的具体设计问题——**这些不是新的开放式讨论,是把已拍板的结论落成
可执行步骤时必须钉死的实现细节**,逐一列出、逐一给出我的建议,请你确认或修正。

---

### 实操细节 1 —— R²/ΔR² 的精确定义(答案里的公式没写全)

**"SSE_mean"里的 mean 是谁的均值?** 沿用本项目一贯的"no-leakage"约定(`Norm` 全部用 TRAIN 统计量,
`force_thresholds` 同理)——`SSE_mean` 必须是**用 TRAIN 集算出的均值**去预测 TEST,不能用 TEST
自己的均值(那样会泄漏 TEST 的信息进基线,数字会虚高)。**建议**:R² 的 mean baseline = 每个
channel 在 TRAIN 上的均值(逐 channel,不是全局一个数)。

**R² 要不要按 channel 分开报?** F(力,大数值)和 CoP(位置,`[-1,1]`)量纲完全不同,混在一起算
一个 R² 没有意义。**建议**:R²/ΔR² 逐 channel 报(`F_R`/`CoPx_R`/`CoPy_R` 各一个数),外加一个
三通道算术平均作为"headline 一个数",方便快速读——但结论以逐 channel 的表格为准,不能只看那个
平均数(万一 CoP 有 trait 效应、F 没有,平均会把这个故事抹掉)。

**问你(Q1)**:这两条约定(TRAIN-mean baseline;逐 channel 报告 + 均值作 headline)是否同意?

### 实操细节 2 —— per-clip 聚合具体怎么实现(OQ-C 定了"要不要",没定"怎么算")

现状:`predict_series`(`src/opentouch/baselines/base.py`)把所有 clip 的所有 window **拼成一个大
数组**返回,一旦拼完,clip 的身份就丢了——`metrics.py` 拿到手时已经不知道哪几行属于哪条 clip。
要做到"每条 clip 权重相等",必须在拼接之前保留 clip 归属。

**建议的实现路径**(三层,R²/MSE/MAE/skill 全部复用同一套底层聚合,不是四套各自实现):
1. `predict_series` 增加返回一个 `clip_ids: (N,)` 数组,与 `ytrue`/`yhat` 的 origin 轴对齐,
   记录每个 window 来自哪条 clip(不改变现有调用方式,新增一个返回值/或新函数,不破坏已有单测)。
2. 新增 `per_clip_sse(ytrue, yhat, mask, clip_ids, train_mean) -> dict[clip_id, {"model":..,
   "mean":.., "persistence":..}]`——每条 clip 先各自算出对模型/对 TRAIN-mean 基线/对 persistence
   的 SSE(逐 channel),这是整个聚合体系的"充分统计量"。
3. **class-level 数字 = 这些 per-clip SSE 的等权平均**,不是重新跑一遍模型:
   `R²_class = 1 - mean_clip(SSE_model_clip) / mean_clip(SSE_mean_clip)`
   (逐 clip 先各自算好、再对 clip 取平均,不是先把所有 clip 的 SSE 加总再除——这里特意用
   "平均的比值"而不是"比值的平均":单条 clip 的 R²_clip 分母可能接近 0(如果那条 clip 的
   target 本来就很贴近 TRAIN 均值),对比值取平均会被这类退化 clip 的爆炸值主导;先在 SSE 层
   面等权平均、再取一次比值,数值稳定得多)。这一步产出的 per-clip SSE 数组,后面 bootstrap
   直接复用,不需要重新跑模型——这是这套设计的关键收益(见实操细节 4)。

**问你(Q2)**:这个"per-clip SSE 优先、class 数字是 SSE 均值之比、不是逐 clip R² 的均值"的设计,
是否同意?这是一个真实的统计选择,答案里没有指定,需要你确认。

### 实操细节 3 —— trait class(smooth/abrupt)标签目前不是正式代码产物

到目前为止我在 G2 里说的"108 条 smooth / 2,850 条 abrupt",判定标准是我 2026-08-07 在临时脚本里
手打的一个动作集合(从未写成正式模块,也从未请你确认过是"最终版"):
```python
SMOOTH = {"pouring","stirring","scooping","serving","eating","wiping","flipping","cutting",
          "drinking","spreading","cleaning","scraping","drawing","writing","carrying","lowering"}
```
G2 现在要正式跑,这个集合必须变成一个有版本、可追溯的代码文件,而不是继续散落在临时脚本里。
**建议**:新建 `src/opentouch/trait.py`,固化上面这个集合 + `trait_class(action) -> "smooth"/"abrupt"`。

**还有一个必须提醒的点**:108/2,850 这两个数字,是在**join bug 修复之前**、用旧的前缀匹配方法
数出来的(2026-08-07 那次)。后来 2026-08-10 把 label join 换成了时间戳重叠匹配,理论上会救回一部分
之前被 miss 掉的 clip、也可能纠正一些之前被错误配对的 action 标签——**这两个数字下载完成后必须
用最终 manifest 重新数一遍,不能直接沿用**。

**问你(Q3)**:上面这个 `SMOOTH` 集合是否确认(还是要增删)?确认后我会建这个文件,并在下载完成
后用真实 manifest 重新核实 108/2,850 这两个数字。

### 实操细节 4 —— bootstrap 的精确操作,和"paired"到底配对什么

你说"model comparison 用 paired clip bootstrap"。这里有两种不同的"配对",需要分开定义,因为
G1(比较模型)和 G2(比较 trait class)配对的对象不一样:

- **G1(模型对比,如 AR vs GRU-aggregate)**:两个模型是在**同一批 TEST clip** 上打分的,天然可配对
  ——每次 bootstrap 迭代:对 clip 有放回重采样一组,**同一组 clip 同时喂给两个模型**各自算出
  R²(或 skill),取差值;重复 B 次,得到"差值"的分布 → 95% CI(取 2.5/97.5 分位数)。这是标准
  paired bootstrap,"配对"指"同一次重采样、两个模型都在这组 clip 上评分"。
- **G2(class 对比,ΔR²=R²_smooth-R²_abrupt)**:smooth 和 abrupt 是**互斥的两组 clip**,不存在
  "同一条 clip 既是 smooth 又是 abrupt"这种天然配对。这里只能是**两个独立样本的 bootstrap**:
  smooth 组内部有放回重采样、abrupt 组内部独立有放回重采样,各自算 R²,取差值;重复 B 次。仍然是
  **clip-level**(不是 window-level),但不是"paired"意义上的配对,是两个独立总体各自重采样。

**问你(Q4)**:我的理解是——G1 用真正的 paired bootstrap(同一重采样同时评两个模型);G2 的 ΔR²
用 clip-level 但两组独立重采样(没有"paired"这个概念,因为两组 clip 不重叠)。这个区分对吗?
如果你说的"paired"另有所指(比如按某种方式把 smooth clip 和 abrupt clip 强行配对),请指出具体
怎么配对。

**问你(Q5,次要)**:bootstrap 重采样次数,建议 **B=2000**(常见默认值,95% CI 用 2.5/97.5 分位数),
可调。有偏好的话告诉我。

**这一步依赖实操细节 2**:有了"每条 clip 的 SSE(对模型/对均值/对 persistence)"这个中间产物,
bootstrap 只是对这个小数组重采样再算比值,不需要重新跑模型或重新滚动 origin——这也是为什么
per-clip SSE 要作为一个显式的、独立的中间产物而不是内嵌在某个大函数里。

### 实操细节 5 —— GRU-aggregate 放在哪里写,和"暂缓 map 分支"的决定有一个没说清的边界

回去确认"点预测 GRU-aggregate"这个模型到底该怎么建的时候,发现一件事需要摊开说:

`PROJECT_CONCLUSIONS §6.4` 四路对比表里的 "GRU-aggregate",代码上很可能**不是独立脚本**,而是
`src/actionsense/tactile_map/` 模块里的 `AggWindows` 数据集分支(`tactile_map/data.py`)——即
map 模块内部本来就有一条"跳过原始 16×16 图、直接吃聚合信号 (T,6)"的路径,专门用来在四路对比里
当 GRU-aggregate 用。也就是说 **CNN-map/flatten-map(吃原始 tactile map 的两个模型)和
GRU-aggregate(吃聚合信号的模型)是同一个 Python 包里的兄弟分支,不是三个独立的东西**。

这和"此前决定暂缓 map 分支"的范围产生了一个交叉:当时"暂缓"的理由是"不需要原始 16×16 map",
但 GRU-aggregate 根本不吃原始 map,只吃已经在缓存里的 `state_N.npy`——按暂缓的原意它不该被一起
搁置。但代码物理上和被暂缓的东西长在同一个文件里,要不要现在就把 GRU-aggregate 这一小块从
`tactile_map/` 里单独 fork 出来(只 fork 聚合信号路径,不碰 CNN/flatten 那两条),还是把整个
`tactile_map/` 一起视为"暂缓",G1 先只跑 persistence/seasonal/AR,GRU-aggregate 也一起推迟?

**问你(Q6)**:(a) 现在就单独 fork GRU-aggregate 这一条路径(不动 CNN-map/flatten-map);还是
(b) G1 先只做 persistence/seasonal/AR 三个,GRU-aggregate 和整个 map 分支一起延后?我倾向 (a)
——G1 的核心问题正是"GRU 是否复现比 persistence 强、AR 是否复现最强",少了 GRU 这个问题答不全
——但这是范围决定,想让你拍板而不是我自己认定。

### 实操细节 6 —— 一个关于"G2 到底要不要等 splits.py"的逻辑澄清

这一点此前的表述不够精确,借这次机会说清楚,**这不是问题,是我要更正的一处逻辑**:

G2 有两层,依赖不一样:
- **training-free 半层**(persistence 的 R²/MSE,零拟合)——理论上语料下载完就能跑,不需要
  `splits.py`。但"SSE_mean"的 mean 如果没有 TRAIN/TEST 区分,只能用**全量语料**的均值,这样算出
  的数字只能当**探索性的早期信号**,不是可以正式报告的最终结果(因为用全量均值本质上是又把
  "测试"数据的统计量泄漏回了基线)。
- **AR 半层**(按 trait class 分别拟合 AR,算 R²)——AR 需要在 TRAIN 上拟合、在 TEST 上打分才算
  诚实,这一层和 G1 一样**必须等 `splits.py`**。

也就是说:**下载一完成,G2 的探索性版本立刻能跑(全量均值、无 AR、只看 persistence 的 R² 趋势),
但 G2 的正式、可报告版本和 G1 一样卡在 p1/p2**。此前的表述容易让人以为"G2 完全不需要 split",
这里更正为"G2 的training-free部分不需要 split 才能跑探索版,但正式版仍然需要"。

### 实操细节 7 —— 建议的分阶段执行顺序(整体逻辑)

把上面几点串成一个有依赖关系的执行顺序:

**阶段 1(现在就能写,零数据依赖,和之前一样用合成数据单测)**:
- `src/opentouch/trait.py`(SMOOTH 集合 + `trait_class()`,待 Q3 确认)
- `predict_series` 的 `clip_ids` 追踪 + `per_clip_sse()`(待 Q2 确认)
- 新的 R² metric 函数(逐 channel + TRAIN-mean baseline,待 Q1 确认)
- clip-level bootstrap 工具函数(paired 版本给 G1,双独立样本版本给 G2,待 Q4/Q5 确认)
- GRU-aggregate 的 fork(待 Q6 确认要不要现在做)

**阶段 2(下载完成、`splits.py` 仍未解决时就能做)**:
- 用真实 manifest 重新核实 108/2,850(或 trait.py 确认后的新集合对应的数字)
- G2 探索性版本(persistence-only R²,全量均值,明确标注"非正式结果")

**阶段 3(`splits.py` 解决之后)**:
- G1 正式跑(persistence/seasonal/AR + 视 Q6 决定是否含 GRU-aggregate),paired bootstrap 出 CI
- G2 正式版(按 class 分别拟合 AR,R² 为主指标,双独立样本 bootstrap 出 ΔR² 的 CI)

### 本次待确认问题清单(Q1–Q6,均不涉及已拍板的 OQ-A~H,是操作化过程中新浮现的实现细节)
Q1:TRAIN-mean baseline + 逐 channel 报告 R²/ΔR²(+均值作 headline)——确认?
Q2:per-clip SSE 优先、class 数字是"SSE 均值之比"而非"逐 clip R² 的均值"——确认?
Q3:`SMOOTH` 动作集合是否确认(还是要增删)?确认后 108/2,850 会用新 manifest 重新核实。
Q4:G1 用真正 paired bootstrap(同一重采样评两模型);G2 的 ΔR² 用两组独立重采样——这个区分对吗?
Q5:bootstrap 次数 B=2000,是否有偏好?
Q6:GRU-aggregate 现在单独 fork,还是和整个 map 分支一起继续暂缓?

**状态:以上均为计划,未写任何代码。等待你 review 后再动手(阶段 1 的部分不依赖下载,可以先做)。**

---

### 2026-08-12 — 重试下载,24 小时冷却不够,仍然 0/26

`scripts/crc/stream_opentouch.sh` 在 crcfe02 上单实例干净跑完(锁机制验证有效——`pgrep -fa`
一度显示 2 个进程,查证后确认是 `pgrep -f` 自我匹配的误报,`ps -eo pid,ppid,lstart,cmd` 核实
实际只有 1 个真实进程,脚本本身跑完后进程自然退出,锁目录也被 EXIT trap 正常清理)。

结果:**26/26 shard 依然全部失败**,报错文字和 2026-08-10 那次逐字相同("Too many users have
viewed or downloaded this file recently... may take up to 24 hours")。`cache` 仍是空目录
(4.0K,0 clips)。确认这是同一个 Google Drive 共享文件配额池,24 小时冷却这次不够。

三个对策摆在用户面前,尚未选择:
1. 挂自动重试循环(每 3 小时一次,`failed_ids.txt` 空了自动停),继续等配额恢复;
2. 转存到用户自己的 Google Drive(浏览器手动复制 26 个文件,换新 file ID),绕开共享文件配额;
3. 邮件联系作者 `rayxsong@mit.edu` 问有没有非 Drive 的镜像(如 HuggingFace)。

**状态:等待用户选择下载对策。同时 Q1-Q6(2026-08-11 记录)仍等待用户回复,期间不再改动任何
代码/配置。**

---

## OPENTOUCH 下载操作日志(独立追踪,与上面 harness/训练计划的讨论无关)

本节只记录"怎么把 26 个 shard + labels 弄到手"这件事的过程和现状。Q1-Q6(eval harness 设计问题)
和 GRU-aggregate fork 的代码工作**在此期间全部暂停**,等用户回复后再继续——这是操作性等待,不是
放弃或改变了之前的计划。

### 时间线
- **2026-08-10**:首次尝试全量下载(26 shard),**26/26 失败**,Google Drive 报错
  "Too many users have viewed or downloaded this file recently... may take up to 24 hours"。
- **2026-08-12(24 小时后重试)**:同样命令重跑,**依旧 26/26 失败**,报错文字逐字相同,确认
  24 小时冷却这次不够(该提示本就是"up to 24 hours"的估计上限,不是保证)。中途出现过
  `pgrep -fa stream_opentouch.sh` 显示 2 个进程的疑似双开警报,用 `ps -eo pid,ppid,lstart,cmd`
  核实后确认只有 1 个真实进程(`pgrep -f` 对自身命令行的误匹配),脚本本身单实例运行正常,
  锁机制未被触发,不是并发问题。
- **官方渠道复核**:重新完整读了 `OpenTouch-MIT/opentouch` 的 README 全文 + 3 个 issue——
  确认**没有任何替代下载渠道**(无 HuggingFace/Zenodo/S3/torrent),只有 Google Drive 一条路;
  仓库自己的 issue 里也从未有人报过这个配额问题。作者联系方式:`rayxsong@mit.edu`(论文一作
  Yuxin Ray Song),邮件求镜像这条路仍然开放、未执行。

### 探索过的绕过方案
1. **浏览器手动测试**:用户登录 Google 账号后手动点开一个 shard 链接,**成功下载(~560MB,
   和已知单 shard 大小 561MB 吻合)**——证实匿名 `gdown` 请求和已登录浏览器访问很可能是分开
   计算配额的。
2. **cookies 认证方案(部分执行,中途触发安全事件)**:
   - 设计:用一个"不常用"的小号,在隔离环境(无痕窗口/独立 Chrome 资料/换浏览器)登录,导出
     该账号在 `drive.google.com`/`google.com` 的会话 cookies,放到 `~/.cache/gdown/cookies.txt`
     (`gdown` 默认会自动读取此路径,无需改代码/改脚本),让 `stream_opentouch.sh` 用已登录身份
     重跑。
   - **安全事件**:用户把导出的 `cookies.txt` **完整内容直接贴进了对话**,其中包含
     `SID`/`SSID`/`__Secure-1PSID`/`__Secure-3PSID` 等实时会话凭证——已提醒这些值等同于该账号
     的登录状态,已泄露进对话记录。**用户已立即修改该账号密码**,使当时贴出的那份 cookies
     失效。这份已经失效的旧内容不再使用;如果之后要用 cookies 方案,必须在改密后重新走一遍
     隔离窗口登录 + 重新导出的完整流程,且**新文件绝不能再贴进对话**,只能通过 `scp` 等命令行
     方式直接传输。
   - 无痕模式下插件默认不可用(Chrome 默认禁止扩展在无痕窗口运行,需要在
     `chrome://extensions/` 手动开启"在无痕模式下允许",或改用独立 Chrome 资料/换浏览器规避)。
   - **未完成/未验证**:是否要恢复走这条路,用户尚未决定;即使走通,自动化连续请求 26 次
     是否会被 Google 的异常行为检测单独限流,也未经验证,不是"保证成功"的方案。
   - **风险分级讨论**:贴进聊天(暴露面广、不可控)明显重于 `scp` 到 CRC(点对点加密传输到
     用户自己有账号的机器,残余风险是 CRC 管理员权限/home 目录权限配置)。进一步提出更干净的
     替代:cookies 全程不离开本地 Mac——直接在本地跑同一个 `stream_opentouch.sh`(脚本本身是
     "下一个删一个"的流式设计,peak 磁盘占用约 560MB,本地 4.4GB 可用空间完全够用,`gdown`/
     `h5py` 本地已装好),完全避免把凭证放到共享文件系统上。
3. **"制作副本"到用户自己的 Drive(讨论中,建议但未测试)**——理论上最优的方案:
   - 机制:Google Drive 的"制作副本"(Make a copy,**不是**"添加快捷方式")是服务器内部直接
     复制,不经过请求方的网络下载,大概率不占用"过多用户下载"这个配额,且几乎瞬间完成。
   - 复制后的文件是用户自己账号名下的**全新文件 ID**,配额与原文件无关,之后用 `gdown` 下载
     新 ID 应该不会再撞到限流。
   - 两个未验证点:(a) OpenTouch 的分享设置是否开放了"复制"权限(下载能用不代表复制一定能用,
     虽然 Drive 通常把下载/打印/复制这三个权限绑在一起);(b) 用户自己 Drive 的可用空间是否够
     14.6GB(免费版 15GB,可能需要分批复制→下载→删除→下一批)。
   - **建议的验证步骤**(已给用户,尚未执行):先对 1 个文件测试"制作副本"选项是否存在、
     复制后能否用新 ID 正常 `gdown`,确认可行再批量处理剩下 25 个 + labels。
4. **rclone(OAuth 走官方 Drive API)**——提及但未深入,作为比 cookies 更"正规"的备选,配额池
   通常与 `gdown` 的匿名导出链接分开。用户想试的话可以再展开具体步骤。

### 当前执行中的方案
用户已启动**自动重试循环**(anonymous gdown,不涉及 cookies/账号,零安全风险):
```bash
cd ~/TouchAnything
nohup bash -c 'for i in $(seq 1 24); do
    bash scripts/crc/stream_opentouch.sh >> ~/opentouch_retry_loop.log 2>&1
    [ -s ~/opentouch/failed_ids.txt ] || { echo "ALL DONE" >> ~/opentouch_retry_loop.log; break; }
    sleep 10800
done' > /dev/null 2>&1 &
```
每 3 小时重试一次,`failed_ids.txt` 空了(全部成功)自动停止,最多循环 24 轮(合计 72 小时)。
已运行约 3 小时,尚未确认本轮结果。

**检查命令**(状态未知时随时可跑,不影响后台任务):
```bash
ps aux | grep -E "opentouch_retry_loop|10800" | grep -v grep   # 循环进程是否还活着
tail -20 ~/opentouch_retry_loop.log                             # 最近发生了什么
sort -u ~/opentouch/done_ids.txt | wc -l                        # 累计成功 shard 数(目标 26)
ls ~/opentouch/cache/state_*.npy 2>/dev/null | wc -l             # 已提取 clip 数
wc -l < ~/opentouch/failed_ids.txt                               # 本轮仍失败数(目标 0)
grep "ALL DONE" ~/opentouch_retry_loop.log || echo "尚未完成"
```

### 状态:自动重试循环运行中,结果未知。cookies 方案暂停(等重新导出),"制作副本"方案未测试。
### Q1-Q6(eval harness)与 GRU-aggregate fork 的代码工作全部暂停,等用户先处理完下载。

### 2026-08-12续 — 为什么 07-02 成功、现在反复失败:脚本比对 + 根因分析

用户提问促成的排查。直接 `git show` 对比 07-02 成功时用的初代 `download_opentouch.sh`
(commit `4d38218`)与现在的 `stream_opentouch.sh`:**下载机制逐字相同**——同一份 26 个
file ID、同样是无延迟无退避的 `gdown "$ID"` 循环、失败仅警告不中断。初代脚本注释里
就写着 `(Drive quota?)`,说明这个风险从一开始就存在,只是当时没有真正触发。**排除"这次
脚本改坏了什么"这个假设。**

**根因是外部条件 + 自身重复请求的叠加,不是代码问题:**
1. 论文自 2025-12 挂出后到现在一个多月,该共享文件的**历史累计访问量**很可能持续上升,
   Google Drive 对"任何人持链接"文件的配额大概率是滚动/累积判定,不是逐日重置。
2. **我们自己短期内对同一批 26 个 ID 发起了多轮完整请求**:07-02(成功)、08-10(失败,
   且因双开 bug 部分 shard 被实际重复请求 2-3 次)、08-12(失败)、以及本轮自动循环的
   第 1 轮(失败)——几天内反复扫同一批文件,可能在持续刷新 Drive"最近被下载过多"的
   判定窗口,阻止其自然冷却。

**行动建议(已给用户,待确认)**:自动重试循环(每 3 小时打一遍同样 26 个 ID)可能是在
南辕北辙——建议停掉或大幅拉长间隔,转向完全不同的配额池:**"制作副本"到用户自己的
Drive**(用户存储配额,与这个被打爆的共享文件配额无关,不受这几天历史请求影响),这是
当前最值得优先验证的路径。

### 2026-08-12续2 — 新建 download_own_copies.sh("制作副本"路径的下载脚本),本地充分测试

用户确认走"制作副本到自己 Drive"这条路,问"下完之后怎么在 CRC 上用"。设计:不改
`stream_opentouch.sh` 本体(它的 26-ID 硬编码数组假设的是原始共享文件;副本产生的是全新、
无法预知的 ID,数量相同但值不同),新建 `scripts/crc/download_own_copies.sh`——同样的
加锁 + 单 shard 落盘 + 下载后即删的安全设计,改成从**外部 ID 列表文件**读,而不是硬编码
数组。顺序无关紧要,因为抽取认的是 HDF5 内部的 scene 名字,不是 Drive 文件 ID 或文件名。

**本地测试**(用假的 `gdown`/`extract_opentouch.py` 替身,不发真实网络请求,专测控制流):
1. ID 解析:支持完整分享链接(`.../file/d/<ID>/view...`)、`uc?id=<ID>` 形式、纯 ID、带前后
   空白的纯 ID。**测试中发现一个真实 bug**:纯 ID 分支原本用 `tr -d '[:space:]'` 去空白,
   这个命令连 `echo` 自带的换行符也一并删掉,导致连续两个纯 ID 会被错误拼接成一行——已改用
   `read -r <<< "$line"` 的写法修复,重新测试确认。
2. **发现第二个环境问题**:脚本最初用 `mapfile` 读文件到数组,本地(macOS)的 `/bin/bash`
   是苹果因授权协议原因冻结在 3.2(2007 年版本)的老版本,`mapfile` 是 bash 4.0(2009)才有
   的内建命令,本地直接报错。CRC 是 Linux,大概率是新版 bash,但**没有 SSH 权限无法直接验证
   CRC 的 bash 版本**——与其假设,不如换成 `while IFS= read -r; do ...; done` 这种 bash 3.2
   起就支持的写法,两边都保证兼容,不留隐患。
3. 端到端跑通:3 个测试 ID(2 个成功、1 个故意模拟"配额失败"),确认成功的被正确抽取、失败的
   被记入 `failed_own_ids.txt` 且不中断后续。
4. **断点续传**:同一份 ID 列表重跑一次,之前成功的 2 个被跳过("already done"),之前失败的
   1 个（这次让它模拟成功)被重试并成功,最终 3/3、无重复抽取(marker 文件里每个 shard 只
   出现一次)。
5. **并发锁**:模拟一个"下载中"的慢速 gdown,同时启动第二个实例——第二个立即被拒绝
   (`FATAL: another run holds .../.stream.lock`,退出码 1),第一个正常跑完 3/3。直接复现并
   验证了修复 2026-08-10 那次 cache 损坏事故的同一个安全机制,不是假设它"抄对了"就算数。

**用法**:
```bash
bash scripts/crc/download_own_copies.sh IDS_FILE [WORKDIR]
# IDS_FILE: 一行一个 Drive 文件 ID 或完整分享链接,支持 '#' 注释和空行
```
Labels 不需要走"制作副本"——labels 文件很小(459KB),此前从未撞到配额问题,继续用原始
labels ID 通过 `stream_opentouch.sh`(或手动 `gdown` 该 ID)获取一次即可,`final_annotations`
目录存在时脚本会跳过重新下载。

**状态**:代码已写完、本地全部测试通过、已 commit。用户仍需完成:(a) 在自己 Drive 里对
26 个 shard 做"制作副本",(b) 收集新文件 ID 存成文件传到 CRC,(c) 在 CRC 上跑这个脚本。

### 2026-08-12续3 — 重新评估下载路径:用 claude.ai Google Drive 连接器直接自动化"制作副本"(执行中)

用户要求重新评估下载流程(转存自己 Drive vs 官方 gdown vs 其他)。本机核查:之前记录的
"3 小时自动重试循环"在本地 Mac 上**不存在**(无进程、无 ~/opentouch_retry_loop.log、无
done_ids.txt)——该循环应是跑在 CRC 上的,从本机无法核实其状态。

**新事实:本会话有 claude.ai Google Drive 连接器(登录身份 jh9141@nyu.edu),"制作副本"
不需要用户在浏览器手动点 26 次,Claude 可以直接调 Drive API 的 files.copy 完成。**

已执行(全部服务器端复制,不占本地网络/磁盘):
1. 建目标文件夹 `opentouch_own_copies`(ID `11w1KgqxWFI7hbmfO35QhxooH7EQlsb1s`)。
2. 测试性复制 shard-01 原始 ID → **成功**(587MB 秒级完成,新文件 owner=jh9141@nyu.edu),
   证实:(a) 分享设置开放了复制权限;(b) 复制操作可绕过 shard-01 当时的下载配额标记。
3. 批量复制其余 shard:**shard-05(192MB, home_kitchen_p3.hdf5)成功**,其余全部报
   "The caller does not have permission"。
4. **对照实验**:把几分钟前刚复制成功过的 shard-01 原始 ID 再复制一次 → 现在也失败。
   **结论:失败不是"每个文件被永久锁",而是连续快速 files.copy 触发了针对本账号的临时
   限流。** 顺带发现:shard 文件大小差异很大(192MB / 587MB / 1.72GB 不等,原名如
   office_ml_p1.hdf5、home_kitchen_p3.hdf5),之前"每个 shard 约 561MB"的假设不准,
   14.6GB 是总量、不是 26×561MB。
5. 副本默认权限 = 仅 owner。**匿名 gdown 下不了私有副本**——计划:26 个副本集中在同一
   文件夹,之后用户对文件夹做一次"知道链接的任何人可查看"共享,文件继承权限,
   `download_own_copies.sh` 即可用新 ID 匿名下载。(NYU Workspace 若禁止对外链接共享,
   备选 rclone OAuth,见前文方案 4。)

**进度:2/26 已复制。**明细与待办 ID 列表在 scratchpad `copy_state.md`(会话临时文件),
最终成功后会把 26 个新 ID 写成 repo 内的 IDS_FILE 供 `download_own_copies.sh` 使用。
另有一份多余的 shard-01 副本在用户 Drive 根目录(ID `1adjDn3pyRs0IRqWg7DS1cNf9A40A_iP-`,
第一次测试产物),全部完成后可移入回收站。

**当前策略:改为单个、低频(≥10 分钟间隔)复制,失败则 10m→30m→60m 退避,由后台定时器
自动唤醒继续,直到 26/26。**同时建议:CRC 上那个每 3 小时打原始 26 ID 的重试循环应停掉
(见"续"节根因分析,它可能在阻止原始文件配额冷却,且"制作副本"路线已不再需要它)。

### 2026-08-12续4 — 限流层级判定:账号级复制限流,读操作不受影响

退避重试记录:+10min 失败、+40min 失败。区分实验:复制**自己拥有的文件**(shard_05 副本,
无共享配额标记)也报同样的 "The caller does not have permission" → **限流作用于本账号的
files.copy 操作本身**,与源文件是谁的无关;get_file_metadata 读操作正常 → 连接器没坏,
只是写/复制被限。该类服务器端复制限流(rclone 社区亦有记载)通常数小时内解除,最坏 24h。
策略:每 60 分钟用真实目标 shard_02 探测一次(成功即继续推进,不造垃圾文件)。
进度:2/26。

### 2026-08-12续5 — 用户问"要不要换方法,比如手动下载":评估与建议

核实:本地 Mac 磁盘余 4.6GB(总需 14.6GB);rclone 未安装,gdown 已装;限流仍未解除
(每小时探测继续)。给出的建议排序:
1. **浏览器手动"制作副本"(推荐先试)**——与被限流的 API 不同通道;先对 shard_02 测试
   预览页 ⋮ 菜单是否有该选项,可用则手动复制其余 24 个并挪入 opentouch_own_copies,
   文件名无关紧要(抽取认 HDF5 scene 名),完成后 Claude 经连接器扫文件夹核对+生成新
   ID 列表。
2. **rclone backend copyid(备选,兼做区分实验)**——用户自己的 OAuth client 走同一
   files.copy API;能成 → 限流在 client 层,不能成 → 在账号层。已给完整命令。
3. **手动直接下载(最后手段)**——虽经验证可行,但 4.6GB 本地空间 vs 14.6GB 数据须
   分批下载→scp→删除循环,14.6GB 过两遍家用网络,且登录态浏览器下载配额连下 26 个
   有中途被掐风险。
剩余 24 个原始 ID 已写入 `data/opentouch_remaining_ids.txt`(repo 内,含说明注释)。
Claude 的每小时自动探测与用户手动进度在同一文件夹汇合,不会重复。

### 2026-08-12续6 — "今天必须拿到":镜像搜索无果,确定今日路径,用户网络受限的应对

用户报告当前网络完全打不开 shard 链接,要求今天拿到数据。行动与结论:
1. **镜像搜索(两轮 WebSearch)**:确认 OpenTouch 无任何第三方镜像(无 HuggingFace/
   Zenodo/torrent),Google Drive 是唯一渠道——"绕开 Google"的幻想正式排除。
2. **download_file_content 工具排除**:其返回形态是内联 base64,不是临时授权链接,
   192MB-1.7GB 的 shard 无法经此通道传输。
3. **用户确认(AskUserQuestion)**:(a) 手机蜂窝流量可以访问 Google Drive;(b) 可正常
   SSH 登录 CRC。→ 今日路径成立:限流解除后 Claude 自动复制完 24 个 → 用户手机上把
   `opentouch_own_copies` 文件夹设为"知道链接的任何人:查看者"(子文件自动继承)→
   CRC 上跑 `download_own_copies.sh`。
4. **已请用户立即在手机上执行文件夹共享**(不必等复制完,先共享后复制的文件同样继承)。
   已启动后台轮询(每 2 分钟 curl 探测 shard_05 副本是否已公开,上限 2h),检测到公开
   后将立即用本地 gdown 实测一个副本,端到端验证链路。
5. `data/own_copy_ids_partial.txt`(现有 2 个副本的新 ID)已写入 repo,可供 CRC 提前
   冒烟测试;最终 26/26 后会生成完整 IDS_FILE。
6. 复制限流仍未解除(距触发约 2h40m,又一次探测失败),每小时探测继续。**若今晚仍未
   解除,兜底方案**:用户在手机浏览器(桌面模式)登录 drive.google.com 手动对剩余 24 个
   原始链接逐个"制作副本"——繁琐但不依赖 API 限流解除。

### 2026-08-12续7 — 链路端到端验证成功;NYU 账号存储不足,切换到用户另一账号的方案

1. **用户已在手机上完成文件夹公开共享**(探测第 4 次命中,共约 8 分钟)。本地立即实测:
   匿名 gdown 下载 shard_05 副本成功,192,755,577 字节与 Drive 完全一致,h5py 打开正常
   (top-level: calibration/data/transform_slam_to_rgb)。**"制作副本→文件夹公开→匿名
   gdown"整条链路今日已验证打通。**测试文件已删(本地仅 4.6GB 空闲)。
   另:本机 curl/gdown 均能正常触达 Google——用户浏览器"完全打不开 shard 链接"的问题
   不是本机网络层面的封锁,具体原因未查明(可能是浏览器/账号态问题)。
2. **用户告知:NYU 账号 Drive 存储不够,有另一个存储充足的 Google 账号可用。**这使
   切换账号成为必选项——剩余 24 个 shard 约 13-14GB,NYU 账号装不下,与限流何时解除
   无关。
3. **新方案**:用户在 claude.ai 设置中把 Google Drive 连接器重新授权为大容量账号
   (设置→连接器→Google Drive→断开→重连,用手机蜂窝流量登录新账号)。之后 Claude 在
   新账号下重建流程:建文件夹→复制剩余 24 个(新账号大概率不带限流状态)→用户手机上
   共享一次新文件夹→生成完整 IDS_FILE→CRC 下载。注意:连接器切换后本会话能否直接
   拿到新凭证未知,若工具报错则需新开会话(本日志已含全部状态,可冷启动续做)。
4. **用户今天可立即做的**:scp `data/own_copy_ids_partial.txt` 到 CRC,跑
   `download_own_copies.sh` 先把已公开的 shard_01/05 拿下;完成后把 NYU 账号里的
   2 个副本+根目录多余的 shard_01 副本移入回收站,释放约 1.4GB。

### 2026-08-12续8 — 用户嫌换账号麻烦,问不换账号需清理多少空间:给出批次方案

剩余 24 shard ≈ 13.8GB(14.6 总量 − 已复制 0.78GB)。方案表:一次性≈14GB;2 批≈7GB;
3 批≈5GB;极限逐个≈2GB(已知最大单文件 shard_02 1.72GB)。推荐 5-7GB 分 2-3 批。
分批流程:Claude 复制一批 → CRC 跑脚本(自动跳过已完成)→ 用户手机删该批副本并**清空
回收站**(不清空不释放配额)→ 下一批。文件夹共享已生效,后续文件自动继承。
提醒:(a) 根目录多余 shard_01 副本(0.59GB)可立即删+清回收站;(b) 查空间:Drive App
"存储空间"或 drive.google.com/settings/storage,用户报数后按实际定批次;(c) 不换账号
则复制限流仍挂在本账号上,清空间与等限流解除两件事并行,若限流久拖不解,换账号仍是备选。

### 2026-08-12续9 — 用户决定换账号;流程与"对话是否受影响"的说明

用户确认切换到大容量 Google 账号。已澄清:切换的只是 Google Drive 连接器的授权账号,
claude.ai 登录不动 → 对话 history 全保留、其他运行中的对话不受影响;切换后所有对话的
Drive 工具指向新账号。流程(已发给用户):(1) claude.ai 设置→连接器→Google Drive→断开
→重连,用大容量账号授权;(2) 告知 Claude 验证身份(本会话若拿不到新凭证则新开对话,
凭本日志冷启动);(3) Claude 在新账号建文件夹+复制剩余 24 shard(新账号预计无限流);
(4) 用户手机共享新文件夹(anyone with link: viewer);(5) Claude 生成完整 26 ID 列表
(NYU 旧 2 + 新 24),用户 scp 至 CRC 跑 download_own_copies.sh;(6) 26/26 落地后删
两个账号的副本+清空回收站。NYU 已公开的 shard_01/05 不受切换影响,无需重做。
**等待:用户完成连接器切换。**

### 2026-08-12续10 — 🎉 26/26 shard 全部复制完成(新账号);总量修正为 21.09GB

用户完成连接器切换(新账号 haojiayi459@gmail.com,本会话直接拿到新凭证,无需换对话)。
执行记录:
1. 新账号建文件夹 `opentouch_own_copies_v2`(ID `1EuMbRkMNRczlvgvY_i4gHJmYZoPrgCYp`)。
2. 测试复制 shard_02(1.72GB,旧账号上反复失败的那个)→ 秒级成功 → **确认限流是账号级,
   新账号干净**。
3. 应用户要求,把 NYU 的 shard_01/05 副本也复制进新文件夹(从 NYU 公开副本复制,不碰
   被标记的原始文件)→ 26 个全部集中在一个文件夹。
4. 剩余 shard 按**每批 3 个 + 批间 120 秒**(后台 sleep 定时器自动唤醒)分 8 批复制,
   10:08Z 开始、10:24Z 结束,**零失败、零限流**。上次触发限流的教训(一次并发 7 个)
   得到验证:限速分批是正确策略。
5. **数据总量修正:26 shard 实际合计 21,094,206,555 B ≈ 21.09GB**,此前记录的"14.6GB"
   有误(单文件 137MB~1.97GB,分布很不均匀)。幸而换了账号——留在 NYU 清 14GB 也不够。
   CRC 流式脚本峰值磁盘占用 = 最大单文件 ≈ 2GB,依然安全。
6. 完整 26 个新 ID 已写入 repo `data/own_copy_ids_full.txt`(含大小注释,可直接作为
   download_own_copies.sh 的 IDS_FILE)。`own_copy_ids_partial.txt` 和
   `opentouch_remaining_ids.txt` 作废(文件头已注明 superseded)。
7. NYU 账号的 opentouch_own_copies 文件夹 + 根目录多余 shard_01 副本已全部冗余,用户
   可删除并清空回收站(约 1.96GB)。

**待办**:(a) 用户手机上对 `opentouch_own_copies_v2` 做"知道链接的任何人:查看者"共享
(已挂 2 分钟间隔的公开探测,命中后 Claude 会本地 gdown 实测 shard_22 验证继承);
(b) 用户把 `data/own_copy_ids_full.txt` scp 到 CRC,跑
`bash scripts/crc/download_own_copies.sh <ids文件>`;(c) labels 不变(459KB,原始 ID,
脚本在 final_annotations 缺失时自动获取);(d) 26/26 落地 CRC 后删两个账号的副本+清空
回收站,继续 eval harness 的 Q1-Q6 与 GRU-aggregate fork 工作。

### 2026-08-12续11 — 共享生效+链路验证;发现并修复两个"副本改名"引入的 bug(会让 CRC 全盘失败)

**1. 新账号文件夹共享已生效并验证通过。**探测第 1 次即命中;本地匿名 gdown 实测下载
shard_22 副本(eat_ygf_p1.hdf5)成功,137,436,536 字节与 Drive 一致,h5py 打开正常
(calibration/data/transform_slam_to_rgb)。测试文件已删。旧账号的每小时限流探测已停止
(26/26 已完成,不再需要)。

**2. 发现两个 bug——根因都是我把副本命名为 `opentouch_shard_NN`(无扩展名):**
- **Bug A(会让 26 个全部失败)**:`download_own_copies.sh` 靠 `ls "$SHARDS"/*.hdf5` 找
  下载产物(第 82 行),而 gdown 用 Drive 标题命名文件。副本标题没有 `.hdf5`,glob 必然
  落空 → 每个 ID 都判为 "no hdf5 produced" 记入 failed。
- **Bug B(比 A 严重,是静默数据损坏)**:`extract_opentouch.py:195`
  `stem = splitext(basename(--shard))[0]`,该 stem 是 shard 身份,构成
  `shard_key = f"{stem}::{group}"`(第 210 行),既是 manifest 的 `shard` 字段(255 行),
  也是幂等去重键(211 行 `if shard_key in seen: continue`)。所以"把下载文件统一命名成
  shard.hdf5"这种看似简单的修法会让 26 个 shard 的 stem 全部相同 → 不同 shard 中同名
  group 的 clip 被静默丢弃,正是 2026-08-10 那类 cache 损坏。**原始文件名是承载语义的。**
- **额外理由**:用回原始名,clip id 与 07-02 那次成功抽取留在 CRC 缓存里的记录保持一致,
  续跑时幂等判断才正确;若改名,同一份数据会被当成新 shard 重复抽取。

**3. 修复:**
- 用 `get_file_metadata` 取回全部 26 个**原始文件名**(只读操作,无限流风险),并用文件
  大小与我记录的副本大小逐一交叉验证 → **26/26 字节数完全吻合,编号映射确认无误**。
  真实场景名:office_csail_p1/p2、office_ml_p1/p2、home_kitchen_p1/p2/p3、home_bedroom、
  hardware_homedepot_p1..p5、sports_dicks_p1/p2、grocery_plant、grocery_target_p1/p2/p3、
  grocery_tj、eat_mcdonalds、eat_ygf_p1/p2、fablab_ml_p1/p2/p3(26 个,互不重复)。
- `data/own_copy_ids_full.txt` 改为两列格式 `<副本ID>  <原始文件名>`(带大小注释)。
- `download_own_copies.sh` 增加可选文件名列:`read -r IDFIELD NAME <<< "$RAW"`(bash 3.2
  兼容),有名字则 `gdown "$ID" -O "$NAME"` 并直接按该路径取文件,无名字则回退原 glob;
  行内 `#` 注释用 `sed 's/#.*//'` 剥离;清理改为 `rm -f "$SHARDS"/*`(该目录是本实例专属,
  只可能存放一个下载产物);汇总的 `done:` 改为统计**本次 ID 文件**的完成数(原先是
  `wc -l done_own_ids.txt`,跨多次运行会串数)。

**4. 本地测试(假 gdown/假抽取器,不发网络请求):**
- 真实的 26 行 ID 文件全跑通:26/26,**记录到的 stem 26 个全部唯一**,无 `#`/空格污染
  (证明行内注释解析正确)。
- 断点续传:重跑 26 行全部 "already done",**重复抽取 0 次**。
- 混合格式:带名字 / 完整分享链接+名字 / 纯 ID 无名字 / 故意失败 → 前两者正常,纯 ID
  无名字的那条正确判失败并记入 `failed_own_ids.txt`(**这正是安全的失败模式**:宁可报错,
  不静默抽错),故意失败的也记录且不中断。
- 并发锁:第二实例被拒(FATAL + 退出码 1)。
- bash 3.2 兼容(无 mapfile)、`bash -n` 语法检查通过。
- 真实 `gdown ID -O NAME` 的调用形式已在本次两次真实下载中验证过(经过 Drive 的
  virus-scan confirm 重定向,正常工作)。

**5. 清理**:删除 `data/own_copy_ids_partial.txt` 和 `data/opentouch_remaining_ids.txt`
(均已作废;partial 指向即将被删的 NYU 副本,留着会导致 scp 错文件白跑一趟)。

**下一步(用户在 CRC 执行)**:
```bash
scp data/own_copy_ids_full.txt <CRC>:~/            # 从本地 repo
ssh <CRC>; cd ~/TouchAnything
bash scripts/crc/download_own_copies.sh ~/own_copy_ids_full.txt
```
若报 `FATAL: ~/opentouch/final_annotations missing`,先取一次 labels(459KB,从未撞配额):
`cd ~/opentouch && gdown 1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ && unzip -o final_annotation.zip`。
峰值磁盘 ≈ 最大单文件 1.97GB(边下边抽边删)。全部落地后可删两个账号的副本+清空回收站。
代码改动尚未 commit(等用户确认后再提交)。

---

## SESSION (2026-08-12续12) — Q1–Q6 全部拍板;阶段 1 全部实现并通过单测;含一处对我 08-11 论断的方向性更正

本节是 eval-harness/G2 线的工作,与上面的下载线并行、互不依赖。下载线状态见续11(26/26 副本
已就位、链路已验证,等用户在 CRC 跑脚本)。

### 用户 2026-08-12 逐条回答(原样记录,不做转述)
| Q | 用户结论 |
|---|---|
| Q1 | 逐 channel 分开报。但 **R² 的 baseline 用 TEST mean**——"这不会造成 leakage,因为 y_test(mean) 没有被模型用于产生预测,这一步计算是在预测之后"。→ **Primary standard R² = 对应 TEST subset 的 clip-balanced channel mean**;TRAIN mean 另算 **R²_OS / train-mean skill 作为 robustness metric,但不是 standard R²**。 |
| Q2 | 修改后确认:**不平均 per-clip R²;也不能直接平均 raw SSE。必须 clip 内除以有效点数**,使每条 clip 总权重相等,再聚合 SSE/SST。 |
| Q3 | **暂不确认原集合为 final。** 先定义 smooth/abrupt 的物理判据(rubric),清单是 rubric 的推论。三层结构:①语义 rubric(先验)②量化 manipulation check(后验验证,**不用于重新分类**)③预注册的敏感性分析(剔除争议子集重算)。明确裁定:cutting→abrupt、flipping→abrupt、scraping→smooth、pouring/stirring/spreading/drawing/writing/carrying→smooth(无争议);scooping/serving/eating/drinking 含混合子事件,走第三层。rubric 写进 `trait.py` docstring。 |
| Q4 | 确认:**G1 paired clip bootstrap;G2 smooth/abrupt 独立 stratified clip bootstrap,不做人为 pairing。** |
| Q5 | **B=5000 formal / 500–1000 dev**;seed 冻结**升级为** `default_rng` + 记 numpy 版本 + paired 索引共享写进单测。 |
| Q6 | **现在单独 fork deterministic GRU-aggregate**;CNN-map / flatten-map 继续暂缓。 |

Q3 的适用范围经 AskUserQuestion 追问后由用户选定:**"全词表审计"**——rubric 是定义,清单是推论,
对已知 action 逐个给 rubric 判定,`trait_class` 对未审计 action 抛错而非默认 abrupt。用户在选择时
已知情该选项会把 `holding`/`sliding` 判入 smooth、smooth 类从 ~108 扩到 300 量级。

---

### Q1 —— 我同意用户,并且必须更正我自己 08-11 的判断:方向搞反了

08-11 我写的是"用 TEST 均值会泄漏 TEST 信息进基线,数字会**虚高**"。**这句话的方向是错的**,现在
更正并留档(不是悄悄删掉):

样本均值**最小化**它自己那个样本上的平方误差,所以
`SSE_base(class_mean) ≤ SSE_base(train_mean)`,分母更小 ⇒

    R²(class_mean) ≤ R²(train_mean) ≡ R²_OS

即 **TEST-mean 分母是两者中更严格的那个,不是更好看的那个**。我当时把"基线用了 TEST 信息"直接
等同于"结果偏乐观",没有算方向。用户的论证在自己的层面上也成立:均值是在模型出完预测之后才用到
的,模型从未见过它,因此不构成预测意义上的 leakage。两条合起来:用户的选择既是标准 R² 的教科书
定义,又恰好是更保守的一侧。**这条不等式已写成单测**(`test_train_mean_r2_is_never_stricter_than_class_mean_r2`),
不是靠 docstring 声明。

三层落地(`aggregate.py`):
- **primary `class_mean`** —— 被打分的那个 TEST subset 自己的 clip-balanced per-channel 均值 = standard R²;
- **secondary `train_mean`** —— R²_OS(Campbell-Thompson 式),真正的样本外预测力陈述,报作 robustness,**不叫 R²**;
- **另备 `clip_mean`** —— 每条 clip 对**自己的均值**,即 within-clip variance explained。见下面 OQ-J:这不是多余选项,它对本语料有实质影响。

逐 channel 报告 + 算术平均作 headline:已实现为 `R2Result.per_channel` / `.headline`,docstring 与
单测都写明**结论以 per-channel 表为准,headline 不得单独报**(万一只有 CoP 有 trait 效应,平均会抹掉)。

### Q2 —— per-clip 等权的精确实现,和一个必须一起定死的连带选择

实现为 `mean_over_clips(SSE_model_k/n_k) / mean_over_clips(SSE_base_k/n_k)`(ratio of means,
不是 mean of ratios)。三处细节是在写的时候才浮出来、必须一起钉死的:

1. **`n_k` 逐 channel 计数。** CoP 在低力时被 mask、force 从不被 mask,所以同一条 clip 可能贡献
   900 个有效 force 点、只有 40 个有效 CoP 点。若用统一的 n,CoP 的权重会被 force 的点数决定。
   某 channel 有效点为 0 的 clip **只从该 channel 掉出**,且分子分母用同一批 clip(否则比值在比
   两个不同总体)。
2. **"clip-balanced mean"必须是"各 clip 自身均值的无权平均",不是点加权总均值。** 这不是口味问题:
   对目标 `J(μ)=mean_k[(1/n_k)Σ(y-μ)²]` 求导得 `μ* = (1/K)Σ_k mean_k(y)`。用点加权总均值会让
   "均值基线"在我们实际聚合的加权下**不是最小值点**,于是一个什么都没解释的模型也能"战胜均值"、
   R² 虚假为正。单测 `test_clip_balanced_mean_is_the_exact_minimizer` 用长度 2 vs 8 的两条 clip
   把两种均值拉开(5.0 vs 8.0)并验证扰动 μ 一定使目标上升。
3. **计数单位是 (origin, horizon-step) 对,不是唯一帧。** stride=1、H=30 时同一帧会在最多 30 个
   窗口里作为 target 出现。这是 harness 既有约定(`metrics.py` 就是在 N*H 上聚合),保持一致才能
   让 R² 和 MSE 描述同一个点集。

**比"存 SSE"更进一步:存充分统计量。** 每 (clip, channel) 存 `(n_valid, Σy, Σy²)` + 每个预测器一列
SSE。对任意常数 μ,`SSE = Σy² − 2μΣy + nμ²` 闭式可得。三个收益:
- **bootstrap 每个 resample 可以重算它自己的 clip-balanced 均值**。均值是被 resample 的统计量的
  一部分,当成固定值会**低估方差**。有了充分统计量这件事是免费的(`test_r2_bootstrap_recomputes_the_mean_inside_each_resample`)。
- 第三层敏感性分析(剔除争议 action 重算)退化为几百行表的代数,**不重训模型、不重滚 origin**。
- per-clip MSE、三种分母的 R²、ΔR²、skill 全部出自同一遍前向。

用户"不能直接平均 raw SSE"的理由已被单测固定(`test_long_and_short_clips_get_equal_weight`);
"不平均 per-clip R²"的理由也被固定:构造一条自身均值恰等于类均值的近常数 clip,它的 per-clip R²
爆到 >1e4,而 ratio-of-means 保持有限(`test_ratio_of_means_survives_a_degenerate_clip...`)。

### Q3 —— rubric 冻结 + 全词表审计的结果(**有需要用户签字的判断,见 OQ-L / OQ-M**)

`src/opentouch/trait.py` 已建,docstring 即预注册文本。原 2026-08-07 那个临时集合**被取代,不是
搬家**。

**应用 rubric 时我不得不加两条细化,都写进了 docstring**,因为不加就自相矛盾:
- **(R1) 事件算在"手的机械耦合"上,包括经由握持工具传递的冲击。** 手套测的是手。刀砍到砧板并不
  改变手-刀的接触状态,但冲量经刚性工具传到手、在 F 上一目了然。这正是用户裁定 cutting→abrupt
  所隐含的判据;同一条也让 scooping(勺撞碗)可疑。
- **(R2) 构成性事件才算;框住持续段的附带 onset/offset 不算。** 几乎每个动作都以建立接触开始,
  所以"有接触 onset"不能当判据,否则全都是 abrupt。要问的是:离散事件**就是**这个动作(去掉它
  动作就没发生:press 没有致动就不是 press,place 没有 release 就不是 place),还是它只是一个
  持续段的前后铺垫(倒水前要握住壶,但 pouring 是持续倾倒,握持是准备动作、通常落在标注窗之外)。
  **没有 (R2),pouring 和 carrying 会被判成 abrupt,与用户的明确裁定冲突。**

**审计表(30 个 action = n≥30 的 14 个 ∪ 旧 SMOOTH 集合的 16 个)。`[U]`=用户明确裁定,`[R]`=我按
rubric 推出,`+`=列入争议子集:**

| SMOOTH(12) | n | 依据 | | ABRUPT(18) | n | 依据 |
|---|--:|---|---|---|--:|---|
| holding | 119 | [R] 持续接触、零构成性事件 | | picking up | 974 | [R] grasp+离面即动作本身 |
| sliding | 55 | [R] 与 wiping 同构 | | placing | 253 | [R] 触面+release 即动作 |
| wiping | 20 | [R] 沿用 | | pulling | 247 | [R]+ |
| pouring | 7 | [U] | | pressing | 237 | [R] 致动转换即动作 |
| stirring | 5 | [U]("无争议") | | pushing | 154 | [R]+ |
| cleaning | ? | [R] wiping 类 | | grasping | 111 | [R] |
| scraping | ? | [U] | | adjusting | 89 | [R]+ 含 re-grip |
| spreading | ? | [U] | | turning | 84 | [R]+ 旋钮 vs 翻页 |
| drawing | ? | [U] | | moving | 78 | [R]+ 推移 vs 取放 |
| writing | ? | [U] | | touching | 82 | [R] 建立接触即动作 |
| carrying | ? | [U] | | removing | 57 | [R] 分离即动作 |
| lowering | ? | [R]+ | | inspecting | 32 | [R]+ |
| | | | | flipping / serving / eating / scooping / cutting | 12/10/8/8/4 | [U]+ |
| | | | | drinking | ? | [U]+ |

**争议子集 = 13 个**:lowering, pulling, pushing, adjusting, turning, moving, inspecting,
cutting, flipping, scooping, serving, eating, drinking。已知 clip 数合计 ≈ 726(约占语料 25%),
剔除后 smooth 仍剩 ≈ 215、abrupt ≈ 1710,**功效充足**。

**三个必须点名的后果:**
1. **`108 / 2,850` 这两个数字正式作废**(既因 join bug 修复,更因分类变了)。下载落地后用最终
   manifest 重数。按已知计数粗估:smooth ≈ 206+(holding 119 + sliding 55 + wiping 20 + pouring 7
   + stirring 5,另 7 个 action 计数未知)、abrupt ≈ 2,440+。smooth 占比从 ~4% 升到 ~8–9%,
   **G2 的功效比原计划好,不是差**。
2. **预注册的可falsify预测仍然成立**:语料仍是 ~92% abrupt,"OpenTouch 整体 1s skill 应明显低于
   ActionSense 的 +0.166"这条不受影响。但"96% abrupt"这个具体数字必须在文档里改成实测值。
3. **争议子集恰好覆盖了历史 probe 与新 rubric 打架的全部动作**(serving/eating/scooping 在 07-02
   probe 里 PI 排名最高,却在新 rubric 下是 abrupt)。这不是巧合造成的麻烦,而是设计自洽的证据:
   **唯一真实的威胁刚好被预注册的敏感性分析正面覆盖**。

**第二层统计量已冻结(定义写死,执行等 splits)**:每条 clip 上,因果滤波后 force 的 |ΔF| 的 95
分位;按 action 取中位数;另报 >3 Hz 能量占比佐证;**只在 TRAIN split 上算**。两处必须说明的:
- **用户写的"15 Hz"与冻结配置不符**:`configs/opentouch/eval_harness.yaml` 是 `fps_raw: 30.0`、
  `downsample: 1`,即有效 30 Hz,所以 |ΔF| 是 **33 ms 帧间跳变**,不是 67 ms。代码从 cfg 取率,
  不写死;此处按 30 Hz 记录。
- "因果滤波"我冻结为**严格因果的 3 帧滑动平均**(100 ms @30 Hz,左侧 padding,只用当前与过去),
  并且**会同时报 k=1(不滤波)的版本**,证明结论不依赖这个选择——滤或不滤都是一个自由度,与其
  单选一个,不如把两个都报。
- clip 太短(一阶差分 < 20 个)的**排除并计数**,不是补 0(补 0 会把短 clip 伪装成"无事件")。
- `per_action_stat` 同时输出 action **内部**的 p25/p75/min/max,直接服务于 docstring 里写明的
  粒度局限:同一 action 跨被试可能横跨两类,这一点报告但**不改分类**(按 clip 分类会引入"用信号
  性质选样本、再测该信号可预测性"的更严重循环)。

**硬纪律已落进代码而不只是口头**:`trait_class` 对未审计 action 和空标签**抛 `UnauditedAction`**,
不默认 abrupt——默认会让 66 个 action 里未审计的 36 个尾巴悄悄填满多数类;`partition()` 把
`unlabeled`/`unaudited` 作为**可计数的掉出项**返回,任何 G2 数字旁边都必须报这两个数。

### Q4/Q5 —— bootstrap:paired 的共享做成结构性的

- **G1 paired**:`bootstrap_paired(stat_fn, ...)` 每轮只生成**一个** row-index 数组交给 `stat_fn`,
  由 `stat_fn` 自己在这批 clip 上给两个模型打分并取差。**共享是结构性的,不可能被"不小心抽两次"
  破坏**。单测构造了两个误差高度相关(共享 clip 难度)的模型:paired 区间宽度 < 独立重采样的 20%,
  且 paired 检出真实的 0.5 差距而独立重采样漏检——Q4 的 G1 设计理由就此被钉成可执行断言。
- **G2 two-sample**:两组各自独立、**各自在层内**重采样。stratified 的层由调用方给,默认意图是
  **按 action 分层**——理由:全词表审计后 smooth 类被 holding/sliding 主导,不分层的抽样可能返回
  一个几乎全是 holding 的 resample,把"类的区间"变成"单个 action 的区间"。层内抽样**精确保持
  每层 clip 数**(单测逐层核对)。分层轴见 OQ-K。
- **B**:`B_FORMAL=5000` / `B_DEV=500`。`default_rng`(PCG64),绝不用 legacy 全局 `np.random.*`;
  `rng_provenance()` 输出 numpy 版本 + bit generator 名 + seed + B,**冻结 seed 只有在生成器也被
  记录时才真的能复现**。区间报 `bias = mean(samples) − point`,偏斜时看得见,不藏在对称区间里。

### Q6 —— GRU-aggregate 单独 fork(已完成)

`src/opentouch/gru_aggregate.py` + `configs/opentouch/gru_aggregate.yaml`。CNN-map / flatten-map
**刻意不在其中**,继续暂缓。与 ActionSense 原版的三处**有意**差异:
1. **确定性点预测**,MSE 损失。原版的 Gaussian NLL 头 + logvar + sigma 校准**移除而非停用**——
   从不做概率评分的概率头是不可falsify的装饰,且 MSE 才是 harness 实际打分的口径。
2. **channel 数取自数据**(3 而非硬编码 6)——正是其它 OpenTouch fork 修的同一类 bug。
3. **走 harness split 协议**(TRAIN 拟合 / VAL 选参 / TEST 只碰一次),不是原版的 5-fold CV;
   因此**不依赖 `splits.py`**(由调用方给 id 列表),这就是它能在 splits 仍阻塞时写完并测完的原因。

保持不变的约定(为了让数字与 ActionSense 四路表可比):左零 padding 使**每个 harness origin 都有
预测**(`score_external` 对齐)、residual-over-persistence 目标(输出 0 即复现 persistence)、
origin 来自同一个 `baselines.origins()`。确定性:`configure_determinism` 固定 torch seed、
`use_deterministic_algorithms`、shuffle 用显式 `torch.Generator`;docstring 注明 **CUDA 上 cuDNN
的 RNN kernel 不保证确定性**,GPU 跑必须记这条 caveat 而不能声称逐位可复现。

### 新增/修改文件
| 文件 | 说明 |
|---|---|
| `src/opentouch/trait.py` | 新建。rubric + 30 action 审计表 + 13 个争议子集 + 第二层统计量。预注册artifact。 |
| `src/opentouch/aggregate.py` | 新建。per-clip 充分统计量、clip-balanced 均值、三种分母的 R²、ΔR²、skill。 |
| `src/opentouch/bootstrap.py` | 新建。paired / stratified two-sample clip bootstrap、percentile CI、rng provenance。 |
| `src/opentouch/gru_aggregate.py` | 新建。deterministic point-prediction GRU-aggregate fork。 |
| `configs/opentouch/gru_aggregate.yaml` | 新建。模型/优化超参;history sweep {1,2,3}s(OQ-H)。 |
| `src/opentouch/baselines/base.py` | 新增 `predict_series_by_clip`(返回 `clip_ids`);`predict_series` 改为其薄包装,**既有调用方与单测零改动**。 |
| `src/opentouch/baselines/__init__.py` | 导出新函数。 |
| `tests/test_opentouch_g2.py` | 新建,29 测试(trait / aggregate / bootstrap,纯 numpy)。 |
| `tests/test_opentouch_gru_aggregate.py` | 新建,torch 相关,本机 skip。 |

`src/actionsense/` **仍然零改动**;`metrics.py` 仍是逐字 fork(新的聚合逻辑放在独立的
`aggregate.py`,不污染"scoring 定义与原版一致"这个声明)。

### 测试结果
`pytest tests/test_harness.py tests/test_harness_opentouch.py tests/test_opentouch_g2.py
tests/test_opentouch_gru_aggregate.py` → **43 passed, 1 skipped**(skip = torch 文件)。
其中既有 14 个回归(原 harness 单测全绿),新增 29 个。

**过程中被单测抓出的两个真实问题,记录下来:**
1. 我最初两个测试 fixture 自己是错的(近常数 clip 的自身均值并不等于类均值;全零 target 让分母
   消失)——不是模块 bug,而是**分母消失时模块正确地返回 NaN 而没有悄悄给 0 或 1**,守卫按设计
   工作。fixture 已修。
2. `pytest.importorskip("torch")` **只捕 ImportError**;本机 torch 是半装状态(缺
   `libtorch_global_deps.dylib`),`import torch` 抛 **OSError**,会让整个 pytest **collection 中断**,
   连纯 numpy 的 G2 测试一起拖死。改成 `try/except Exception` + `allow_module_level=True` 的显式
   skip。→ **GRU-aggregate fork 本机只做了 AST 编译检查 + 逻辑审查,未经 torch 实际执行**,这一点
   不得在后续记录里被含糊掉;真正的验证要在 CRC(有可用 torch)上跑那 8 个测试。

---

### 新的 OPEN QUESTIONS(阶段 1 代码已就绪,但这几条会改变**报什么数**,需要你拍板)

- **OQ-J(最重要):standard R² 的分母该用 `class_mean` 还是 `clip_mean`?**
  你选的 `class_mean` 是一个类一个常数(逐 channel)。本语料 clip 短(中位 2.80 s)而各 clip 的
  DC 力水平差异大,于是 **SST 会被 clip 之间的水平差主导**;任何只要能跟住"当前水平"的模型
  (persistence 免费就能做到)都会拿到很高的 R²,却没有解释任何**动态**。而 trait 假说问的正是
  clip 内部的动态可预测性。`clip_mean`(每条 clip 对自己的均值)隔离的就是这部分。
  两者代码同一条路径、成本相同,已都实现。**我的建议**:primary 仍按你的裁定用 `class_mean`
  (它是标准 R² 且更严格),但把 `clip_mean` 版本作为**并列主表**报出而不是附录——如果两者结论
  方向不一致,那本身就是关于"效应来自水平还是来自动态"的关键证据。请裁定要不要这样报。
- **OQ-K:G2 stratified bootstrap 的分层轴是什么?** 我按"action"实现(理由见上:防止 resample
  被 holding 吞掉),但代码接受调用方给任意层标签,改成 scene/participant 零成本。若 split 轴最终
  定为 scene 级留出,分层轴与 split 轴的关系需要一起想清楚。
- **OQ-L:`eating`/`drinking`/`scooping`/`serving` 的主标签定为 abrupt——确认?** 你说这四个
  "含混合子事件、建议用第三层处理、而不是硬塞"。第三层需要它们各自有一个主标签,我按 rubric 严格
  执行(R1:勺-碗碰撞、食物 release)判为 **abrupt**,同时全部列入争议子集。这与 2026-08-07 旧集合
  **相反**(旧集合把四个都算 smooth),也与 07-02 probe 的 PI 排名相反。**这是我的推论,不是你的
  裁定,需要签字。**
- **OQ-M:`pulling`/`pushing`/`adjusting`/`turning`/`moving`/`inspecting` 判 abrupt + 列入争议
  子集——确认?** 这 6 个共 684 clip。它们在 (R2) 下是真正的边界:持续位移是动作主体(偏 smooth),
  但典型实例含抓取接合与终止碰撞(偏 abrupt)。我选 abrupt(与既有文档一致、且 OpenTouch 场景多为
  货架/抽屉/门)并全部列入争议子集,使敏感性分析覆盖它们。
- **OQ-N:ΔR² 的两个分母不同类。** `class_mean` 下 smooth 与 abrupt 各用自己类内方差作分母,这正是
  "class-specific R²"的定义,但也意味着 **ΔR² 同时受"可预测性差异"和"类内方差差异"影响**。
  OQ-A 已把 raw MSE/MAE 定为 secondary 佐证,可部分对冲;`clip_mean` 版本(OQ-J)也能帮助分离。
  是否还要加一个"两类共用同一 pooled 均值"的第三种诊断?我倾向**不加**(又一个自由度),但要你知道
  这个解释性限制存在,而不是让它在审稿时才被指出。

### 状态
阶段 1(零数据依赖)**已全部完成并通过单测**,准备 commit(独立 commit,日期即预注册时间戳)。
阶段 2(下载落地后):用最终 manifest 重数两类 clip 数、跑 G2 探索版。阶段 3(splits.py 解决后):
G1/G2 正式版。**OQ-J~N 不阻塞 commit,但阻塞"报哪张表"**;OQ-L/OQ-M 若被否决,只需改
`trait.py` 的表并重跑单测(改动局部,但必须在**看到任何 G2 数字之前**改,否则就成了 post-hoc)。

**补记(同日)**:`pytest tests/`(整个目录)在本机原本就是**中断**的——`tests/test_tactile_map.py`
在模块层 `import torch`,撞上同一个半装 torch 的 OSError,导致整个 collection 失败。这是**既有
状况、与本次改动无关**,但它会让未来的会话把这个中断误读成"新代码坏了",所以给它加了与
`test_opentouch_gru_aggregate.py` 相同的 `try/except` + `allow_module_level=True` 守卫(torch 正常
时行为不变)。现在 `pytest tests/` → **43 passed, 2 skipped**(两个 torch 文件)。

### 2026-08-13 — CRC 首次实跑 download_own_copies.sh:25/26 成功,2876 clips 入 cache

用户在 crcfe01(ND CRC 前端节点,UGE 集群)上完成首次实跑。前置检查:无残留进程、无残留锁
(之前记录的"每 3 小时自动重试循环"在 CRC 上并不存在或早已退出——SESSION_LOG 里那条记录
无法在任何机器上被证实,今日确认 crcfe01 干净)。传输方式:从本地 Mac `scp -J` 经 bastion
跳板传两个小文件(ID 清单 + 修好的脚本),21GB 数据全程不经过本地。

**结果**:
```
cache:  /users/jhao3/opentouch/cache  (227M, 2876 clips)
done:   25 / 26 ids
FAILED: 1 -> /users/jhao3/opentouch/failed_own_ids.txt
```
- **"制作副本→文件夹公开→匿名 gdown"整条路线在真实环境下跑通**,配额问题彻底解决。
- 227MB cache / 21GB 原始数据 ≈ 1% —— 流式抽取的价值得到验证(峰值磁盘约 2GB)。
- 2876 clips vs 论文宣称的 2,900,差额大概率来自失败的那 1 个 shard。
- 环境注意点(已发现):`gdown` 不在 `environment_tactile_cuda.yaml` 里(当初手动 pip 装),
  跑之前需 `conda activate tactile`;h5py/numpy 在 yaml 中。

**待办**:重跑同一条命令(脚本会跳过已完成的 25 个,只重试失败的 1 个),并确认失败的是哪个
shard。若重复失败则重新"制作副本"换新 ID。之后即可恢复 eval harness(Q1-Q6)与
GRU-aggregate fork 的代码工作。

### 2026-08-13续 — 26/26 全部落地(2904 clips);并更正我本会话一处过时论断

**下载彻底完成**:用户重跑同一条命令,脚本跳过已完成的 25 个、自动重试失败的第 19 个
(`grocery_tj.hdf5`)并成功。最终:
```
cache:  /users/jhao3/opentouch/cache  (232M, 2904 clips)
done:   26 / 26   FAILED: none    (failed_own_ids.txt 为空)
```
2904 clips 与论文宣称的 2,900 相符。**断点续传逻辑在真实环境得到验证**(不是只在本地
假 gdown 下测过)。232MB cache / 21GB 原始数据 ≈ 1%。

**自我更正(重要)**:我在本会话多次说"下载完成后继续 Q1-Q6",这是**错的**——Q1-Q6 已于
2026-08-12 全部拍板,阶段 1 代码(trait.py / metrics / bootstrap / GRU-aggregate fork)已实现
并通过单测(`pytest tests/` → 43 passed, 2 skipped)。我引用的是 3003/3088 行那两条早已被
后续会话取代的旧状态,没有先读到 3360 行之后的内容就下了结论。**当前真正的阻塞是 OQ-J~N**,
它们不阻塞 commit,但阻塞"报哪张表"。

**下一步的正确顺序(有一个不可颠倒的约束)**:
1. **先做描述性重数**(已写好 `recount.py`,只输出 clip 计数,**不含任何 R²/模型输出**):核实
   26 个 shard 全在 manifest 里、clip_id 无重复(即 Bug B 的 stem 冲突在生产环境确实没发生)、
   并用**真实 manifest** 刷新 OQ-L/OQ-M 论证所依赖的过时数字(108/2,850 来自 join bug 修复前;
   OQ-M 的"684 clip"同样是旧数)。计数是设计信息、不是结果,不构成 post-hoc。
2. **再请用户裁定 OQ-J / OQ-L / OQ-M / OQ-N**。SESSION_LOG 已明确记录:OQ-L/OQ-M 若要改,
   **必须在看到任何 G2 数字之前改**,否则就成了 post-hoc。因此**在用户签字前不跑 G2 探索版**,
   哪怕它技术上现在就能跑。OQ-K(G2 分层轴)与 splits.py 的 split 轴耦合,可以等 splits 一起定。

### 2026-08-13续2 — recount 报 ModuleNotFoundError:根因是阶段 1 代码从未推送;已 commit + push

用户在 CRC 上跑重数脚本报 `ModuleNotFoundError: No module named 'src.opentouch'`。排查:
本地 `git status -sb` 显示 **main 领先 origin/main 2 个 commit**——`e60d299`(阶段 1:trait.py /
metrics / bootstrap / GRU-aggregate fork)和 `8084640`(test_tactile_map torch 守卫)**只在本地**,
CRC 是从 GitHub 拉的,自然没有这些文件。**不是环境或 sys.path 问题。**

处置(用户在三个选项中选择"commit + push + CRC 上 git pull",理由是保持 CRC 与 git 历史一致、
可复现):
- 新 commit `25129bf`:`download_own_copies.sh` 的文件名列修复 + `data/own_copy_ids_full.txt`
  + 本会话的 SESSION_LOG。commit message 完整记录了两个 bug 的机制与本地测试结果。
- `git push origin main` 成功(`8687440..25129bf`),三个 commit 全部上到 fork。
- CRC 拉取时的坑:此前 scp 过去的 `download_own_copies.sh` 相对旧 HEAD 算本地修改,会挡住
  pull;其内容与新 commit **完全一致**(scp 发生在全部编辑与测试之后),故指示用户
  `git checkout -- scripts/crc/download_own_copies.sh` 后再 pull,无内容损失。

**顺带的收尾建议(已告知用户)**:26/26 已安全落在 CRC,Google Drive 的
`opentouch_own_copies_v2` 文件夹**应取消公开共享**。刚推送到 GitHub fork 的 ID 清单在取消
共享后即成为无效链接,既避免陌生人消耗用户账号配额,也降低把可下载链接留在公开仓库的暴露面。
CRC 上的数据不受影响。

### 2026-08-13续3 — 真实 manifest 的重数结果:冻结词表覆盖不全,且类别极度不平衡(阻塞 G2)

CRC pull 到 `25129bf` 后重跑重数(纯计数,无任何 R²/模型输出)。**脚本在 `lifting` 处按设计
中止**——`trait.trait_class` 抛 `UnauditedAction`,守卫正确阻止了未裁定动作进入打分。这不是
bug,是预注册保护起作用。

**真实数字(vs 2026-08-11 引用的旧数 108/2,850,后者来自 join bug 修复前):**
```
smooth 233 | abrupt 2393 | contentious 728 | unclassified 278   (合计 2904 ✓)
```
**三个必须在任何 G2 数字之前解决的问题:**
1. **冻结的 30 词审计表没覆盖真实语料**:36 个动作词从未审计,共 278 clip(9.6%)。词表:
   aligning, attaching, bending, connecting, detaching, dropping, examining, feeling, folding,
   inserting, lifting, measuring, passing, pinching, plugging in, pointing, probing,
   resting hand, rotating, scanning, screwing, scrolling, spraying, switching off/on, taping,
   tapping, testing, tightening, tilting, twisting, typing, unfolding, unplugging, unscrewing,
   zipping。其中若干(rotating/twisting/folding/scrolling)按 rubric 很可能落 smooth,直接影响
   smooth 类的规模。
2. **类别极度不平衡:smooth 233 vs abrupt 2393 ≈ 1:10**。旧数字给人的印象(108 vs 2,850)虽也
   不平衡,但当时 smooth 更小;现在 smooth 主要由 holding(115)+sliding(53)撑着。这直接影响
   G2 的统计功效和 stratified bootstrap 的分层设计(OQ-K)。
3. **争议子集 728 clip > 整个 smooth 类 233**。第三层敏感性分析的设计是"剔除争议子集重算",
   但当被剔除的比主类还大时,该设计的解释力需要重新审视——pulling(245)/pushing(154)/
   adjusting(88)/turning(84)/moving(78)/inspecting(29) 六个 DISPUTED 全在 abrupt 侧,合计 678。

**已发现的 per-action 计数(截至中止点)**:picking up 951, placing 249, pulling 245[D],
pressing 233, pushing 154[D], holding 115(smooth), grasping 107, adjusting 88[D], turning 84[D],
moving 78[D], touching 78, removing 55, sliding 53(smooth), inspecting 29[D]。

**状态:仍未跑任何 G2 数字(正确)。**下一步:用容错版脚本取回完整 per-action 表(未审计者标
UNAUDITED 而非抛异常),然后与用户一并裁定:36 词的 rubric 补裁、不平衡对 G2 的影响、争议子集
大于主类时敏感性分析是否还成立。原有的 OQ-J/L/M/N 仍未裁定,且 OQ-L/OQ-M 的论证依赖的旧数字
(684)现已被真实数据取代(六个 DISPUTED 实为 678)。

### 2026-08-13续4 — 用户确认三步走计划;数据格式核实;新增 OpenTouch F/CoP 绘图脚本

**用户的计划(原话转述)**:1) 按动作类型分 smooth/abrupt(明确:**只分两类**);2) split data
开始 training;3) 训练后用 R² 算 metrics。并要求先列出所有动作及各自 clip 数,再确认数据格式、
画 F/CoP 随时间的变化。

**计划与既有设计的对照(我的核实结果)**:三步与 2026-08-11 的 G1/G2 计划一致,无冲突。
"只分两类"也与冻结设计相符——**争议子集(contentious)不是第三类**,而是第三层敏感性分析的标记,
两类划分不受影响。两个前置阻塞仍在:(a) 36 个动作词未审计(278 clip)导致第 1 步无法完成;
(b) `splits.py` 未解决,第 2 步的正式 split 轴仍未定(OQ-K 的分层轴与之耦合)。

**数据格式核实(读 `scripts/extract_opentouch.py` 确认,不是推测)**:
- **F 和 CoP 已在抽取时算好**,无需再碰 hdf5(hdf5 已随流式下载删除)。
- `state_<N>.npy` 形状 `(T, 1, 6)` = `[F, CoPx, CoPy, sxx, syy, sxy]`;hand 轴 extent=1
  (OpenTouch 只instrument右手,保留该轴是为了与 ActionSense 的 harness 索引完全一致)。
- `clip_<N>.npy` `(T,1,16,16)` float16 原始压力也在 cache 里;`pose_<N>.npy` 可能缺失。
- CoP 坐标归一化到 `[-1,1]`;16×16 中仅 169 个活 taxel,死格读数≈0 对矩不贡献权重,
  故 F/CoP 无需 mask(但任何 per-taxel 统计需要)。
- **未做 baseline correction**(刻意推迟:clip 围绕压力峰切分,前 N 帧可能已在接触中)。
- manifest 每条含 `fps_est`、`onset_idx`/`peak_idx`/`post_idx`,可直接用于时间轴与事件标注。

**新增 `scripts/plot_opentouch_fcop.py`(commit `bfff4cf`,已推送)**:行=clip(标注 action/
object),列=F / CoP-x / CoP-y,并画出 onset/peak/post 竖线使事件结构可见。已用合成 cache
本地测通(含"动作不存在时列出可用动作"的错误路径)。**docstring 里写死了一条约束**:这是
Layer-2 验证,**不得用于重新分类动作**,否则类别定义就成了它后来要解释的那份数据的函数。
理由:用户自己在 2026-08-12 对 Q3 的裁定即"②量化 manipulation check(后验验证,**不用于重新
分类**)"。因此**建议顺序**:先用 rubric 补裁 36 个词(纯语义、先验)→ 再画图验证 → 最后跑 G2。
若用户希望先画图探索,那也可以,但必须在日志里明确记录该顺序,并且分类仍须严格由 rubric 决定。

**仍未跑任何 G2 数字。**待用户跑容错版重数脚本给出完整 per-action 表(我无 CRC 访问权限,
无法自行读取 manifest)。

### 2026-08-13续5 — 新增原始 tactile map 绘图脚本(commit `4495425`,已推送)

用户要求画出与 F/CoP 同一批 clip 的**原始 tactile map**。核实:`clip_<N>.npy` `(T,1,16,16)`
float16 原始压力确实在 cache 里(`download_own_copies.sh` 调用 `extract_opentouch.py` 时未加
`--no-clips`;cache 232MB 的体量也与"2904 clip × 约 43KB 原始 map + state + pose"吻合),
因此**无需重新下载 hdf5**。现有 `scripts/plot_tactile_map.py` 画的是 CV 结果表、不是 map,
不可复用。

**`scripts/plot_opentouch_tactile_map.py`**:行=clip,列=帧(等距 N 帧 ∪ onset/peak/post,
peak 列标题标红),每帧 `imshow` 16×16 压力场并叠加**青色 CoP 圈**;逐 clip 用 99.5 分位数作
色标上限(防单个热 taxel 压平整个场)。**clip 选取直接 import 自 `plot_opentouch_fcop.py` 的
`pick()`/`load_manifest()` 而非重写**,保证 `--actions/--n` 选到的是同一批 clip、两张图能对上;
也支持 `--idx`(取前一个脚本打印的索引)。缺 `clip_<N>.npy` 时给出原因(`--no-clips` 抽取)而非
裸 FileNotFoundError。

**CoP 叠加的坐标约定**(依据 `extract_opentouch.moments()` 逐行核对,写进 docstring):
`cx` 加权 `xs=linspace(-1,1,W)` 沿**列**轴、`cy` 加权 `ys=linspace(-1,1,H)` 沿**行**轴,故
`col=(cx+1)/2*(W-1)`、`row=(cy+1)/2*(H-1)`,配合 imshow 默认 `origin="upper"`。
**验证**:用合成 cache(其 state 由 `moments()` 的同一套数学独立算出)测试,青色圈在每一帧都
精确落在压力团中心 → 坐标映射正确,不是"看起来差不多"。同时测通 `--idx` 路径与缺文件报错路径。

**约束不变**:该脚本同样是 Layer-2 验证,docstring 明确写明**不得用于重新分类动作**。
**仍未跑任何 G2 数字。**36 个未审计动作词的补裁仍是第 1 步的阻塞项。

### 2026-08-13续6 — 传感器/标注语义的权威核实(论文),并发现 peak_idx 基准可能错位(待诊断)

用户看图后提问:深浅代表什么、16×16 对应手的哪些部位、F 怎么算、peak/onset 是什么、
为什么 peak 那帧反而很浅。**从论文(arXiv:2512.16842 全文 HTML)核实**,不是推测:
- 硬件:**169 个 taxel,覆盖手指 + 掌侧面**(不只是手掌);16×16 电极栅格 + 商用压阻薄膜;
  **30 Hz**;标定范围 **0.02–50 kPa**,超出会饱和。169 与我们抽取代码记录的"169 live taxels"
  **完全吻合**(交叉验证成立);256-169=87 个格子恒为 0,即图上永远发黑的部分。
  **论文未给出"哪个格子对应哪根手指"的映射表** —— 可由 `--taxel-stats` 实测活跃度分布推断。
- `F = p.clip(0).sum(axis=(1,2))`,是**压强总和的代理量,不是牛顿**;无面积加权、无标定、
  **无 baseline 扣除**(刻意推迟)。CoP 为压强加权重心,归一化到 [-1,1]。
- 标注三索引的论文定义:`onset_idx`=peak 前压力最低(approach);`peak_idx`=**压力峰值**
  (manipulation);`post_idx`=peak 后压力最低(release)。
- 绘图色标是**逐 clip** 99.5 分位数归一 → **行内可比,行间不可比**(已向用户说明)。

**发现的疑点(可能是真 bug,尚未确证)**:按论文定义,`peak_idx` 那一帧应当是整段**最亮**的,
与用户观察相反。`extract_opentouch.py:262` 把 `peak_idx`/`onset_idx`/`post_idx` **从标注 CSV
原样抄入 manifest,未做任何重新基准化**;而 clip 与标注行是靠**时间戳重叠**匹配的,两者起点
不必对齐。若 CSV 中的索引相对于原始录制流(或标注窗口)而非我们切出的 clip,则它是错位索引,
`plot_opentouch_tactile_map.py` / `plot_opentouch_fcop.py` 画的竖线位置就是错的,且**任何后续
用 onset/peak/post 做窗口切分的分析都会被污染**。

**已给用户决定性诊断命令**:对前 600 条 clip 比对 `peak_idx` 与 `argmax(F)` 的帧距,并统计
越界比例。判读标准:差≤2 帧占绝大多数 → 索引为 clip 本地、图无误,"peak 浅"是信号真实现象;
中位差很大或大量越界 → 基准错位,必须修绘图脚本并复查所有依赖该索引的下游逻辑。
**结果未知,等待用户运行。在确证之前不修改代码(避免按猜测改)。**

### 2026-08-13续7 — peak_idx 疑点排除(我的怀疑是错的);"peak 帧偏暗"的真实成因

**诊断结果(用户在 CRC 运行,600 条 clip)**:
```
peak_idx 在 [0,T) 内: 600 | 越界: 0
|peak_idx - argmax(F)|: 中位 0.0 均值 0.0 最大 0 | 完全一致 100.0%
```
**`peak_idx` 与 `argmax(F)` 逐帧精确相等,零越界。续6 中"索引可能相对原始录制流而错位"的怀疑
被证伪,现予撤回。**标注索引是 clip 本地的,两个绘图脚本的竖线位置一直是对的,无需修改。
(教训:该疑点提出时即标注为"待确证、在确证前不改代码",这个处理方式是对的——若当时按猜测
"修复",反而会把正确的索引改坏。)

**"peak 帧看起来更暗"的真实成因(物理解释,非 bug)**:`F = Σ_taxels p`,是**空间求和**,
不是单格最大值。总力取峰值可以来自**接触面积铺开**(整掌+手指轻压,每格值不高但格子多),
而指尖单点重压会产生极亮的单格却给不出大的 F。又因绘图色标上限取该 clip **全部帧**的 99.5
分位数,只要 clip 内存在某个集中受力瞬间,vmax 就被抬高,于是"面积大但每格不亮"的 peak 帧
显得偏暗。
**对项目的实质含义:模型预测的 F 是空间聚合量,其峰值与视觉上最亮的 map 不是同一回事。**
这一点在解释 G1/G2 结果时需要记住(例如"预测 F 好"不等于"预测接触模式好")。

**改进(commit `f797f19`,已推送)**:`plot_opentouch_tactile_map.py` 现在把每帧的 **F 值和
受载格子数 n**(>5% 色标上限)直接印在该帧标题上,并加大行距避免标题与上一行图重叠。这样
"F 最大但看起来暗"是可读的事实,而不需要用眼睛推断。已用合成 cache 重新渲染验证版式。

**已给用户的下一步验证命令**:对 8 条 clip 对比"peak 帧"与"单格最亮帧"的 F/maxcell/面积,
若 peak 帧面积显著更大而 maxcell 更小,则上述解释在真实数据上成立。

### 2026-08-13续6 — probGRU 预测流程图的设计方案(计划,未实现)

**用户要求**:仿照参考论文 Fig.2(object embedding | forward prediction 双栏示意图),画出我们
prediction pipeline 的预测模型图,**明确表示 input history / probGRU / output**;要求我先设计
"该包含哪些内容、怎么排列"。

**我锁定的建模对象(核实过代码,不是推测)**:`ProbGRU`
([action_dynamics.py:142](src/actionsense/action_dynamics.py#L142))。逐行确认的结构事实:
1. 编码器 `enc = nn.GRU(din, hid)`,`din` = 6(highpass:`[F_fast,x_fast,y_fast,F_slow,vx,vy]`)
   或 5(raw:`[F,x,y,vx,vy]`);`hid=48`。
2. 解码器 `dec = nn.GRU(3, hid)`,**输入维度是 3(fast target),不是 din**;`h` 由编码器末态初始化。
3. **自回归 rollout**:`inp = mu.unsqueeze(1)`([:161](src/actionsense/action_dynamics.py#L161)),
   每步把自己的预测均值喂回去,**不是 one-shot**。这是它与另外两个变体的关键区别。
4. **action embedding 在 head 处 concat,不在 GRU 输入处**:`oc = torch.cat([o[:,-1], e], -1)`
   ([:158](src/actionsense/action_dynamics.py#L158)),`emb = nn.Embedding(n_act, 8)`。
5. 双 head:`mu`(3) 与 `lv`(3),`lv.clamp(-6,4)`。训练 Gaussian NLL;VAL NLL 早停;
   事后 `calibrate_sigma` 标量缩放使 coverage@2σ→0.95。
6. seed 帧是 `y_last = norm.ny(Yin)[:,-1]`,即**最后一帧已观测的 fast target**。
7. 预处理:30 Hz →`ds=3`→ 10 Hz;`cut=0.4 Hz` 二阶 Butterworth **因果** `sosfilt`;
   丢弃前 `warmup_sec=5 s` 瞬态;`t_in ∈ {1,2,3,5,10} s`;`t_out = 1 s = 10 步`。

**图的内容清单(两栏,仿 Fig.2)**

(a) 左栏「因果特征构造」——**替代**参考图的 object-embedding 栏。理由:参考图那栏之所以独立成栏,
是因为 embedding 由对比损失学出来、是方法贡献;我们的 action embedding 只是 4 个离散动作 id 的
查表(8 维),**独立成栏会严重夸大它**。我们真正对应"方法贡献"的是 causal slow/fast 分解
(因果性修正曾把 skill 从 +0.70 拉回 +0.40,是项目最重要的方法结论之一)。故左栏内容:
tactile map →(离线、非学习的 `physical_state`)→ s_t=[F,x̄,ȳ] → 降采样 10 Hz → 因果低通
0.4 Hz → slow/fast 分流 → 特征 x_t(D=6)与目标 y_t(3);标注"丢弃前 5 s 瞬态"。
action label → Embedding → e_a(8) 作为**小侧输入**画在左栏底部,箭头引向右栏的 head。

(b) 右栏「概率性 rollout」——三段式,时间一律左→右:
  B1 历史块:t_in 帧特征列向量堆 → Encoder GRU(48) → h_t;下方叠一条 3 通道 fast 信号缩略曲线。
  B2 解码链:3 个解码 cell + "…",h 左→右传递;**虚线弯箭头**表示 μ 回喂(自回归);e_a 从下方
  扇出**只进 head 框**(不进 GRU cell)——这条必须画对,否则与代码不符。
  B3 输出:μ 曲线 + ±2σ 带 + 真值 + persistence 灰虚线;标注 skill 与"报告 R² vs mean"。
  贯穿全高的粗虚线 = 预测原点 t(左=已观测,右=待预测),这是全图最重要的一条线。
  底部细条:Gaussian NLL → VAL NLL 早停 → σ 标定(coverage@2σ→0.95) → 5-fold CV by clip。

**刻意不画的东西(避免不诚实)**:(i) 不能画成"tactile map 直接进网络"——probGRU 吃的是 6 维
state,不是 map(参考图是 map 进 encoder,照抄就是错的);(ii) 不画 one-shot head(那是
tactile_map/models.py 和 opentouch/gru_aggregate.py 的结构,不是 probGRU);(iii) 不把
persistence 画成模型的一部分,它是基线。

**OPEN QUESTIONS(待用户裁定,未定前不动手实现)**
- **OQ-P1 画哪个模型?** (1) ActionSense probGRU(上述,自回归+概率头);(2) OpenTouch
  `Seq2SeqPoint`(确定性、one-shot、residual-over-persistence,是当前 G1/G2 的在用模型);
  (3) 一张图统一表示、用分支标注差异。用户说"probGRU"→默认 (1),但当前活跃工作在 OpenTouch,
  需确认这张图是给哪份稿子用。
- **OQ-P2 产出格式?** matplotlib 脚本(可复现、进 `scripts/`)/ TikZ(投稿最清晰)/ SVG。
- **OQ-P3 范围?** 是否含左栏 (a) 与底部训练条,还是只画 input→probGRU→output 的最小版。
- **OQ-P4 版面?** 双栏 7in×2.9in(仿 Fig.2)还是单栏窄图。

### 2026-08-13续8 — 【重大】真实 tactile map 图暴露两个问题:F 被直流偏置主导、10/26 shard 标注疑似错配

用户把 `docs/opentouch/exploratory/opentouch_tactile_map.png` 取回本地,Claude 直接读图分析(不再靠转述)。

**问题 1:F 中 95%+ 是直流偏置,CoP 近乎失效。**
- 每帧 F ≈ 700,000–770,000,**整段变化幅度仅约基线的 ±4%**;平均每格读数 ≈ 750000/256 ≈ 2930。
- `n=256`(所有格子都在色标 5% 以上)**包括本应是"死格"的 87 个** → docstring 里"dead cells
  read ~0"的假设在真实数据上**不成立**。图面大片饱和于 magma 高值端,仅零星暗格。
- **CoP 的青色圈在所有帧、所有 clip 上都钉在正中央几乎不动** —— 均匀场的压强加权重心必然趋于
  几何中心。**当前 CoP 携带的接触位置信息极少。**
- 因此"peak 帧看起来和别帧无异"是必然的:真实接触信号只占 1–5%,被基线淹没。
- **续7 中给出的"F 是空间求和、面积铺开导致 peak 帧偏暗"的解释虽然在物理上成立,但不是本图的
  主因;主因是直流偏置。该解释的适用性被本次实测修正。**
- `extract_opentouch.py` docstring 明确写着 baseline correction 被**刻意推迟**,待 `--taxel-stats`
  测出静息水平后再定。**本图即该测量:基线不是小偏移而是主导项,这个决定现在必须做。**
  影响面:预测一个 95% 为常数的信号会让 persistence 天然占优、R² 虚高 —— 正是 OQ-J 中
  "SST 被 clip 间水平差主导"担忧的极端形态。

**问题 2:10/26 个 shard 的 `peak_idx` 与 `argmax(F)` 不一致(用户全量诊断结果)。**
```
office_ml_p2      271  100.0%      sports_dicks_p1   90  92.2% (最大差 68 帧)
sports_dicks_p2   102   70.6% (最大差 81 帧)        有不一致的 shard: 10 / 26
```
`sports_dicks_p1/p2` 正是 docstring 点名的**共用同一份 CSV** 的那一对。**关键推论**:若不一致
源于 clip 认领了错误的标注行,则错的不只是 peak 位置,**`action` 标签同样是错的**;而 action
是 smooth/abrupt 分类的唯一依据 → 直接污染 G2。**我此前"前 600 条 100% 一致"的诊断因只取
manifest 前 600 条(未覆盖被绘图选中的 clip)而不具代表性,该方法学缺陷已向用户承认。**

**已给用户三段式诊断命令**:(A) 逐 clip 的 raw 分位数 + 扣基线(每格 5 分位)前后 F 的变异系数
与 argmax 变化;(B) 扣基线前后 CoP 的活动范围;(C) peak 不一致率按 manifest 的 `join` 方式交叉
分组,以定位标签错配机制。

**状态:未改任何代码。**baseline correction 怎么扣(每格 5 分位 / 前 N 帧 / 全局)是需要用户拍板的
方法学决定,且必须在看到任何 G2 数字之前定死,否则又是 post-hoc。G1/G2 在此之前不得启动。

### 2026-08-13续9 — 待决事项清单(应用户要求)与 Claude 的建议

**时间敏感警告(已置顶告知用户)**:**暂缓删除 Drive 副本、暂缓取消共享**。若 D2 确认标签错配,
修复需重新抽取受影响 shard,而时间戳只存在于原始 hdf5(cache 未保存),即需**重新下载那 10 个
shard**。清理须等全部定稿之后。

**第 0 层——数据本身(最先,因其改变数据):**
- **D1 baseline correction 做不做/怎么做。** 建议:**必须做**;方案倾向**按 shard(≈一次录制
  session)估计每 taxel 静息水平**(该 shard 全部帧的 5% 分位),逐格扣除并截断到 0。理由:clip
  围绕压力峰切分,**单 clip 内可能全程在接触**,用 clip 内分位会连真实接触一起扣掉(docstring
  原已警告);shard 级样本量大得多。同时**需重新识别死格**("死格读 0"假设已被证伪,改用
  "时间方差≈0"定义)。**不需要重新下载**——`clip_*.npy` 全在 cache,F/CoP 可直接重算。
- **D2 10/26 shard 标注错配的处置。** 建议:若确认为 join bug 则**修 join,不丢 shard**
  (丢 10/26 ≈ 丢 40% 语料);代价是那些 shard 需重新下载+重抽。并建议**把每 clip 的
  ts_start/ts_end 写入 manifest**,使此类问题今后可直接从 cache 审计。

**第 1 层——类别定义(必须在任何 G2 数字之前):**
- **D3 36 个未审计动作词(278 clip)的 rubric 补裁**:建议现在就逐词裁定(Claude 给依据、用户
  签字)。`rotating/twisting/folding/scrolling/typing` 等很可能落 smooth,而 smooth 现仅 233,
  补裁可能显著改变类规模。
- **D4 1:10 不平衡 + 争议子集(728) > smooth 类(233)**:建议**不做人为再平衡**(下采样丢数据);
  并把第三层从"剔除争议子集"改为"**全量 / 仅无争议 两张并列表**",使结论对争议裁定的依赖程度
  可见,而非只给剔除后的单一数字。
- **D7 OQ-L/OQ-M 签字**:维持 Claude 的 rubric 推论,但裁定权在用户(OQ-L 与 2026-08-07 旧集合
  相反)。OQ-M 的"684"已被真实数据取代,实为 678。

**第 2 层——指标与统计:**
- **D5 OQ-J(class_mean vs clip_mean)**:建议**等 D1 完成后再定**(扣基线后"clip 间水平差主导
  SST"会大幅缓解);无论如何建议两者**并列主表**。
- **D8 OQ-N**:建议**不加** pooled-mean 第三诊断,但在论文写明解释性限制。
- **D6 OQ-K 分层轴**:与 D9 的 split 轴一起定,现在不单独决策。
- **D9 splits.py 的 split 轴**:建议**按 scene 或 participant 留出**(manifest 有 `scene`/
  `environment`),**绝不可随机切 clip**(同场景同物体的 clip 高度相关,随机切严重泄漏)。

**建议执行顺序**:D2 定性 → D1 定方案并重算 → D3/D7 签字 → D4 定报表形式 → D9/D6 定 split
→ D5/D8 定指标 → 才跑 G1/G2。**最划算的起步是 D1**(无需重新下载,只用 cache 重算),且做完后
D5 的判断会自然清晰。已提议先写"扣基线前后对照"的**纯描述性**分析脚本(只出统计与图,
不出任何 R²),供用户看过效果再定正式方案。**等待用户指示从哪一项开始。**

**OPEN QUESTIONS 已裁定(用户,2026-08-13)**:OQ-P1 = **ActionSense probGRU**;OQ-P2 = **matplotlib
脚本**;OQ-P3 = **完整双栏(含左栏 (a) 与底部训练条)**;OQ-P4 随之取双栏宽版面。

**实现:`scripts/plot_model_diagram.py`** → `docs/model_diagram.png` + `.pdf`(矢量,投稿用)。
纯绘图脚本,不读数据、不依赖 cache,任何机器上都能跑。

*版面*:统一绘图坐标 130×56(figsize 13×5.6,比例 2.32),`ax.add_axes([0,0,1,1])`,inset 用
`x/W, y/H` 直接换算,故所有元素(含两张 inset 曲线)共用一套坐标,改一个数就整体对齐。

*落实的设计决定*:
- 预测原点 t 的粗虚线贯穿 y=10.8..50.5(**刻意不穿过底部训练条**,否则像把协议也切成两半)。
- `e_a` 用**参考图 Fig.2 的惯例**:在每个 head 旁重画一个小深蓝条,而不是画一条总线——既避免
  与 decoder cell 的走线打架,也把"concat 发生在 head 而非 GRU 输入"画对了。
- 自回归回喂用**橙色虚线弧**从 head_k 绕到 cell_{k+1} 底部,与实线的 h 传递明确区分。
- 左栏 → 右栏有两条蓝色曲线箭头分别标 **x**(进特征条)与 **y**(进观测目标曲线),明确区分
  "编码器输入(6 维)"与"目标/seed(3 维)"这两个不同的量;这一区分在代码里就是
  `enc = GRU(din,...)` 与 `dec = GRU(3,...)` 的维度差。

*诚实性措施(写在画布上,不只写在 docstring 里)*:
- 右上角固定注记 **"inset curves are illustrative, not measured"**——两条曲线是定种子合成的
  示意信号,不是实验数据。
- 但示意曲线的**形状是真的**:μ 用 `true * linspace(0.85,0.35)` 表现 NLL 训练的**幅度向均值收缩**,
  σ 随 horizon 增大;二者都是 5.4/5.5 节记录的真实性质。输出小图下方直接写
  "skill = 1−MSE/MSE_pers **(report R² vs mean)**",把 5.5 节"skill-vs-persistence 结构性虚高"
  的结论带进图里,避免图本身诱导读者只看 skill。
- docstring 顶部列出"**刻意不画**"三条:(i) map 直接进学习型 encoder(probGRU 吃 6 维 state,
  照抄参考图就是错的);(ii) one-shot head(那是 tactile_map/models.py 与 gru_aggregate.py);
  (iii) persistence 作为模型的一部分(它是基线,只出现在输出小图里)。

*未做*:未提交(等用户看过图再定);未把该图挂进 PROJECT_CONCLUSIONS.md 的图目录。

### 2026-08-13续10 — 用户要求"先不解决那些问题,直接跑一遍预测(probGRU)";已提供探索性驱动

**两处必须先澄清的事实(已告知用户)**:
1. **OpenTouch 侧不存在 probGRU。** `src/opentouch/gru_aggregate.py` 是**确定性点预测**,概率头
   是依据用户自己 2026-08-11 对 **OQ-G** 的裁定("GRU-aggregate 用点预测")被**刻意删除**的
   (文件内注明理由:"从不被概率性评分的概率头是不可证伪的装饰")。probGRU 属于 ActionSense。
   要跑概率版必须先推翻 OQ-G。
2. **split 是不可用的占位符**:`configs/opentouch/eval_harness.yaml` 的 split 块写明
   `PLACEHOLDER -- NOT YET FUNCTIONAL`、`splits.py does not exist`、`DO NOT wire into
   evaluate.py`;`evaluate.main()` 故意抛 NotImplementedError。但 `fit_and_forecast()` /
   `build_rows()` / `score_external()` 在**调用方提供 splits** 时均可用(该 fork 的设计即
   "由调用方传 id 列表"),故无需 splits.py 也能跑探索版。

**污染控制(Claude 的设计选择,已向用户说明)**:本次运行**不做 smooth/abrupt 分组**,只对全量
语料打分。理由:trait 裁定(36 个未审计词、OQ-L/OQ-M 未签字)必须在**看到任何按类数字之前**冻结;
不分类就不可能污染它们,同时仍满足"跑一遍看看 pipeline 通不通"的诉求。

**`scripts/run_opentouch_exploratory.py`(commit `463ae10`,已推送)**:
- **按 group 整组留出**(默认 `scene`),不按 clip 随机切——同一 scene 的 clip 共享环境、物体与
  个人习惯,clip 级随机切会把近重复样本分到两侧、虚高所有分数。这是 splits.py 被阻塞期间**最弱
  但可辩护**的替代,**不等价于**真正的 split。
- **不修改冻结配置**:`config_hash` 就是该 yaml 的文件哈希,改它重定向 `states_root` 会静默破坏
  与历史运行的可比性。改用软链接:`ln -s ~/opentouch/cache data/opentouch_states`。
- 每行 CSV 与结尾横幅都带 `exploratory=True` 与 split 标签,防止日后被误当作 harness 结果。
- 结尾明确提示:**F 受直流偏置主导(D1 未决)**,故这些数字会因与动态无关的原因偏袒 persistence。
- **本地验证**(合成 cache,60 clip/6 scene → 40/10/10;本机 torch 半装故只测 baseline 路径):
  三个 baseline 全部 fit/score/落盘正常,CSV 1395 行,汇总表与 skill 表打印正确。

**待用户在 CRC 执行**;GRU 路径(60 epochs × 3 history)在前端节点较重,建议先 `--skip-gru`
拿 baseline,再用小 `--epochs`/`--max-clips` 冒烟,最后视情况走 qsub GPU 作业跑全量。

### 2026-08-13续11 — 核实用户回忆:ActionSense 确实去掉了 baseline,但机制是频率分解而非片段筛选

用户提出"记得 ActionSense 的 training input 筛掉了变化不大的部分、只留波动高的部分",要求确认。
**逐行读代码核实结果:结论方向正确,但机制需更正。**

**没有任何按方差筛选 clip/窗口的逻辑**:`action_dynamics.load_pooled()` 只过滤 `min_len>=20` 帧,
`windows()` 按 stride 取遍每个滑窗。实际做的是**对每路信号做因果频率分解**:
- `slow_fast(sig,fps,cut)`:**因果** 2 阶 Butterworth 低通(仅前向 `sosfilt`,**刻意不用
  filtfilt**,否则泄漏未来),`slow`=低通输出,**`fast = sig - slow`**;CLI 默认 `--cut 0.4` Hz。
- **`TARGETS = ("F_fast","x_fast","y_fast")` —— 预测目标恒为 fast**,与 input_mode 无关
  (docstring: "target: always fast")。**这就是"去掉 baseline"的实际含义。**
- 输入是独立开关:`highpass` → `[F_fast,x_fast,y_fast,**F_slow**,vx,vy]`(**慢分量作为输入特征
  保留**,只是不作目标);`raw` → `[F,x,y,vx,vy]`。
- `warmup_sec=5.0`:每 clip 前 5 秒在训练与评测中**都丢弃**(因果滤波启动暂态)。

**关键澄清:该路线 ≠ 当前运行的路线。** 冻结 harness 的目标是 **RAW**——
`configs/actionsense/eval_harness.yaml` 与 `configs/opentouch/eval_harness.yaml` **都逐字写着**
`RAW = the physical moments straight from state_N.npy (NO high-pass; NO warmup cut)`;OpenTouch 的
`gru_aggregate` 吃的也是 harness 的 RAW 通道。**因此今天的直流偏置问题是 Q1 决定的直接后果**:
2026-08-11 用户在 Q1 选了 (a) 6-dim RAW 而非 (b) 3-dim FAST,**而该决定作出时无人看过 OpenTouch
的真实信号**。

**D1 因此拆为两个层级不同、互不替代的决定(已向用户列出):**
- **D1a 传感器级直流**:每 taxel 静息偏置(≈2930/格),在**计算 F/CoP 之前**扣除。现状:未做
  (`extract_opentouch.py` 刻意推迟)。`slow_fast` **不解决**这一层——它作用于力矩计算之后的
  F/CoP,对每格偏置无能为力(即 docstring 所称的 "ActionSense DC-offset bug (P4)")。
- **D1b 信号级慢分量**:用 `slow_fast` 把低频漂移从**目标**中去掉。现状:未做(harness 选 RAW);
  若采纳需**重开 Q1**。
**OpenTouch 目前两层都没做,故 F 有 95% 是常数。**

**可比性提醒(已告知用户)**:ActionSense 的 probGRU 数字是在 **FAST 目标**上取得的,OpenTouch
的 GRU-aggregate 打的是 **RAW 目标**,两者**不可直接对照**;harness 层面 RAW vs RAW 一致,故 G1
的跨传感器比较本身不受影响。

### 2026-08-13续12 — 用户裁定:用与 ActionSense 完全一致的 probGRU 训练 OpenTouch(推翻 OQ-G)

**用户指令**:"不用 low pass,用所有频率的 history 作为 input 预测;改变模型,仍然使用和
ActionSense 完全一致的 probGRU、loss function、architecture 来训练 OpenTouch。"

**Claude 先摆出的三个事实(实施前澄清,未擅自决定)**:
1. **warmup 5 秒物理上不可行**:ActionSense `build_features` 默认丢弃每 clip 前 **5.0 s**
   (因果滤波启动暂态),而 OpenTouch clip **中位仅 2.80 s(84 帧)**,照搬会清空绝大多数 clip。
2. **probGRU 架构含 action embedding**(`nn.Embedding(n_act,8)`,解码每步拼进输出头),即
   **模型被动作标签条件化**;ActionSense 在少数动作上池化训练,OpenTouch 约 50 个动作且长尾,
   词表规则是新问题。
3. 该要求**明确推翻 OQ-G**("GRU-aggregate 用点预测")。故**新建 `src/opentouch/prob_gru.py`**,
   **不改** `gru_aggregate.py`(后者是预注册的确定性臂,不应就地改写)。

**用户通过 AskUserQuestion 的两项裁定(2026-08-13)**:
| 问题 | 裁定 |
|---|---|
| 预测目标 | **RAW 3 维** [F,CoPx,CoPy](非 FAST)。→ 可被冻结 harness 直接打分,与 baseline 同尺度对比;代价:严格说与 ActionSense probGRU 的**目标**不同(架构与 loss 一致)。 |
| action embedding | **保留**(架构完全一致)。 |

**`src/opentouch/prob_gru.py`(commit `494738a`,已推送)**
- **逐字复制**:action embedding、encoder GRU、**自回归** decoder(以最后观测目标 seed,把预测
  的 `mu` 喂回下一步)、`[decoder state ; action emb]` 上的 mu/logvar 双头、`logvar.clamp(-6,4)`、
  **高斯 NLL `0.5*(lv + (y-mu)^2*exp(-lv))`**、按 VAL NLL 早停、以及**其自身超参
  (hidden 48 / epochs 80 / lr 3e-3 / batch 64)**——注意**不是** gru_aggregate.yaml 的 64/60,
  那属于 tactile_map aggregate 分支(另一个模型)。
- **四处被迫的差异(均已在 docstring 说明理由)**:(1) 目标 RAW(用户裁定);(2) 输入为全频段
  `[F,CoPx,CoPy,vx,vy]`,含 ActionSense 的**因果**后向差分速度,**全文件无任何低通**;
  (3) **无 warmup 裁剪**(无滤波器即无暂态,且 5 s 会清空语料);(4) 窗口取自 harness 的
  `origins()`(训练与打分看同一组 rolling origins),而非 ActionSense 的 stride-2 采样——stride
  属采样细节,不属架构/loss。
- **归一化**:目标用 harness 的 TRAIN-fitted `Norm`(与 baseline 共享一套),输入特征另用
  TRAIN 拟合的 z-score(因含速度,harness Norm 不覆盖)——对应 ActionSense 分离的 nx/ny。
- **动作词表**:**仅由 TRAIN 构建**(用 VAL/TEST 的动作定义词表会把 split 泄漏进模型输入空间);
  TRAIN 中 n < `baselines.min_group_size`(=30,与 AR baseline 合并稀有 object_category 用的
  同一阈值)的动作并入 `"other"`(id 0),TEST 中未见过的动作也落 `"other"`。

**测试 `tests/test_opentouch_prob_gru.py`**:特征因果性(t 时刻之后的改动不得影响 t 之前的特征)、
特征布局、**仅 VAL 出现的动作不得产生新 embedding id**、预测形状与 harness `origins()` 一致、
确定性(同种子两次结果 bitwise 一致)、以及**loss 确为高斯 NLL**(固定残差下 logvar 的最优值应为
`log(残差^2)`,MSE 伪装成的 loss 不满足此性质)。**本机 torch 半装,这些测试只能 skip
(43 passed, 3 skipped),必须在 CRC 上真实运行**;驱动脚本改动后已本地复验 baseline 路径正常。

**驱动脚本**新增 `--model {prob_gru,gru_aggregate,both,none}`,默认 `prob_gru`。

### 2026-08-13续13 — CRC 实跑暴露 `KeyError: 'sports equipment'`;根因在驱动脚本未做既定检查

**报错**:`baselines/ar.py:93 KeyError: 'sports equipment'`(经 `predict_series_by_clip`)。

**根因**:AR 按 `object_category` **分组拟合**;而我的临时 split 按 `scene` 整组留出,某个
category(如 sports equipment)可能**完全存在于被留出的 scene 内**,于是 TRAIN 从未拟合过它,
AR.predict 取系数时 KeyError。**这不是 baseline 的 bug**——`dataset.missing_groups()` 的
docstring 明写:"Non-empty = AR.predict() will raise KeyError deep inside baselines/ar.py
rather than failing at split-construction time with a clear message -- so splits.py MUST call
this after building a split and assert the result is empty (OQ-I option (a))"。
**是我的驱动脚本没有调用这个既定检查。**

**修复(commit `49b4db0`)**:`adhoc_split` 现在在划分后循环检查——若某个留出单元携带 TRAIN 见不到
的 AR 组,就把**整个单元**搬回 TRAIN,直至无缺失;随后对 val/test 分别 `assert missing_groups()
为空`,使失败发生在**划分阶段并带清晰信息**,而不是运行 20 分钟后死在字典查找上。搬移的单元会
被打印。**代价**:split 向 TRAIN 偏移,已在 docstring 中记为"这些数字是探索性的"的又一条理由;
整单元搬移保证仍是 scene 粒度留出,不产生 clip 级泄漏。

**验证方式的自我修正**:第一版合成数据(4 scene、每个独占一个 category)过于病态,导致全部被搬进
TRAIN、test 为空——守卫正确拦截并给出清晰提示,但**没有走到搬移路径**;第二版把独占 category 设
为 12 clip,低于 `min_group_size: 30` 被并入 `other`,**仍未触发**。第三版设为 36 clip(≥30,保留
为独立组)后改为**对 `adhoc_split` 做单元级验证**(全流程太慢):**40 个 seed 下修复后缺失组为 0;
同样 40 个 seed 用不搬移的旧逻辑有 14 个会缺组——正好是触发搬移的那 14 个**,构成干净的反证。
(教训:两次"测试通过"其实都没覆盖目标路径,若不追查会误以为已验证。)

### 2026-08-14 — 探索性首跑结果解读 + GPU 作业脚本(含一个必须先修的缺陷)

**用户在 CRC 完成冒烟运行**(`--epochs 1 --histories 1 --max-clips 300`),pipeline 端到端跑通:
```
persistence  F_R 174,481,881 | CoPx 0.00009 | CoPy 0.00003
seasonal     F_R 176,954,686 | ...           skill vs persistence: -0.014 / -0.018 / -0.020
ar           F_R 125,562,891 | ...           skill: +0.280 / +0.441 / +0.256
prob_gru     F_R 131,372,398 | ...           skill: +0.247 / +0.377 / +0.187
```
**解读(已发用户)**:
- **定量证实直流偏置问题**:persistence 在 F 上的 RMSE ≈ 13,200,而 F 本身 ≈ 750,000 →
  **1 秒预测"假设完全不变"只错 1.8%**;所有模型都在这 1.8% 的余量内竞争。CoP 的 RMSE ≈ 0.009
  (坐标域 [-1,1]),因 CoP 几乎钉在中心。**skill +0.28 的含义是"把慢漂移拟合得比 persistence 好",
  与触觉动态的可预测性基本无关。**
- **排序 AR > probGRU > persistence > seasonal 恰好复现 ActionSense 的结论(G1 的核心问题),
  但现在不能当证据**:probGRU 只训了 **1 个 epoch** 却已拿到 +0.25,说明分数主要来自任务过易;
  且仅 300 clip、仅 1 s history、临时 split、直流主导目标。
- seasonal ≈ -0.02,与既有结论一致(无自相关峰 → 退化为 persistence)。
- **`vocab 2`**:300 clip 子集中只有 1 个动作达到 `min_group_size=30`,**action embedding 本次
  几乎未起作用**;全量下预计 10-14 个动作达标。
- val NLL(0.0604)约为 train(0.0186)的 3 倍,1 个 epoch 即有此差距——可能过拟合,也可能是按
  scene 留出导致 val 分布不同,需全量 NLL 曲线分辨。
- **全量耗时外推**:300 clip → 16,063 训练窗口 → 7 s/epoch;全量约 9.7 倍窗口且 t_in 由 30 扫到
  90 帧 → **前端 CPU 约 7-9 小时**。

**用户指示"写 qsub 用 GPU 跑"。发现并先修的缺陷**:`prob_gru.py` **原本没有任何 `.to(device)`**
——直接提交到 gpu 队列会**占着一张卡以 CPU 速度运行**。已补:`pick_device()`(有 GPU 用 cuda,
否则 CPU)、模型与每个 batch 上设备、预测经 `.cpu()` 回传、`history["device"]` 记录实际设备
(**CUDA 运行不能声称 CPU 那种 bitwise 可复现性**,因 cuDNN 的 RNN kernel 即使在 deterministic
模式下也不保证确定性——该 caveat 早在 `gru_aggregate.configure_determinism` 中写明)。
驱动新增 `--device`。

**`scripts/crc/opentouch_probgru_gpu.job`(commit `f3de7c8`)**:沿用 repo 既有 UGE 约定
(`-q gpu -l gpu_card=1 -pe smp 4`,`#$` 行内无注释,日志入 `logs/`)。两处前置校验使失败在几秒内
发生而非浪费数小时 GPU 时段:(a) 校验 `data/opentouch_states/manifest.jsonl` 存在(该 symlink 是
cache 的唯一入口,静默训练空数据会白费整个 slot);(b) **先跑依赖 torch 的两个测试文件**
(它们只在有可用 torch 的机器上真正执行,CRC 是唯一场所)。支持
`qsub -v EPOCHS=,MODEL=,MAX_CLIPS=,HISTORIES=,SEED=,SPLIT_FIELD=,OUT=` 覆盖。
本地已验证:`bash -n` 通过、`${VAR:+--flag}` 在 `set -u` 下未设变量时安全展开、baseline 路径回归
正常、`pytest tests/` 43 passed 3 skipped。

### 2026-08-15 — splits.py 完成(D9 解决);F/CoP 直流问题按用户指示挂起待结果

用户指示:"正在跑,现在来解决这次 run 的缺陷,完成 split;F 和 CoP 变化很小的问题先记录,
等结果出来之后再看。"

**一、论文核查(不解决问题,但排除了猜测)**:arXiv:2512.16842 全文——14 个环境、每场录制
5-25 分钟、clip 平均 57 帧;**未定义任何官方 split,未提泄漏**;`_pN` 的含义**论文没有正式定义**。
(注:抓取摘要中"_pN 看起来是参与者"、"参与者跨多个地点"两句是**推断而非原文**,不作为依据。)
语料本身也**没有任何参与者字段**。

**二、用户裁定(AskUserQuestion, 2026-08-15)**:
| 问题 | 裁定 |
|---|---|
| split 轴 | **按地点基名留出**(剥掉 `_pN`,如 office_ml_p1/p2 归为一个单元) |
| AR 分组冲突 | **让 group_keys 感知 TRAIN**(未在 TRAIN 出现/样本不足的类目归 `other`) |

**三、为什么"按地点"能解开死结**:无论 `_pN` 是参与者还是场次,同一地点的全部 shard 都在同一侧
→ **该未解决的问题变得无关紧要**,而不是被猜测掉。26 shard → **12 个地点**
(hardware_homedepot 5;home_kitchen/grocery_target/fablab_ml 各 3;sports_dicks/office_ml/
office_csail/eat_ygf 各 2;home_bedroom/grocery_tj/grocery_plant/eat_mcdonalds 各 1)。
**仍不能保证的**:若同一人在多个地点录制,他会出现在两侧——**manifest 无人物标识,任何基于它的
split 都无法排除**。已写入模块 docstring,并要求报告时称"按地点留出"而非"按参与者留出"。

**四、被卡住的第二个原因(此前未识别)**:粗粒度 split 与 AR 的 `fit_scope: object_category`
**结构性冲突**——物体跟着地点走,留出一个地点就带走它的类目,而 `group_keys` 用**全语料计数**,
导致"全语料常见但 TRAIN 完全没有"的类目 → 即 08-13 的 `KeyError('sports equipment')`。
**修复**:`group_keys(cfg, idxs, train_idxs=None)`,传 `train_idxs` 则按 TRAIN 计数(不传保持原行为,
既有调用与测试不受影响);`evaluate.fit_and_forecast` 传入 TRAIN。

**五、测试当场揪出的残余漏洞(重要)**:仅按 TRAIN 计数**还不够**——若 TRAIN 中每个类目都达标,
则**没有任何 TRAIN clip 落进 `other`**,被映射到 `other` 的留出类目仍然无系数(9 个测试里 5 个
当场失败)。**补救:把 TRAIN 中最小的达标类目按升序并入 `other`,直到 `other` 达到
`min_group_size`**,保证**兜底组自身永远可拟合**。代价是一两个类目失去独立 AR 拟合。

**六、另一个自造 bug**:`assign()` 原写法 `sorted(...)` 之后又 `shuffle(...)`,**排序被打乱**,
"大单元优先放置"的意图失效 → 合成数据上 **89% 的 clip 被塞进 train**。改为先 shuffle(供 seed
变化)再按大小降序 sort。注:当存在一个占比 62.5% 的巨型地点时,配额本就无法更接近——除非把地点
拆开,而那正是本设计要避免的。

**七、交付**(commit `e9e278e`,已推送):
- `src/opentouch/splits.py`:`location()` / `by_location()` / `assign()`(贪心填最缺额的桶) /
  `build()`(内含 `missing_groups` 断言) / `save()` / `load()` / `summarize()` / CLI。
- **`evaluate.main()` 不再抛 NotImplementedError**:建或载入 split → 跑冻结协议 → 写表。
- 驱动新增 `--split-mode {location,adhoc}`,**默认 location**。
- `tests/test_opentouch_splits.py` 9 个测试(后缀规则、地点不跨侧、覆盖且不重复、15 个 seed 下
  AR 永不遇到未拟合组、**反证旧的全语料规则在同一 split 上必然失败**、确定性、seed 敏感性、
  比例边界、往返序列化、地点过少报错)。**全套 52 passed, 3 skipped**,无回归。
- 合成语料端到端验证:train 55.6% / val 22.2% / test 22.2%,`other` 组含 30 个 TRAIN clip。

**八、按用户指示挂起的事项**:**F/CoP 的直流偏置问题(D1)不在本轮处理**,已在续8/续11 详细记录
(F 中 95%+ 为直流、CoP 近乎钉在中心、D1a 传感器级 vs D1b 信号级两层、ActionSense 用
`slow_fast` 只解决 D1b)。**等 GPU 作业结果出来后再评估。**

**九、对正在运行的作业的影响**:该作业已加载旧代码(ad-hoc scene split),**不受本次改动影响**;
其结果仍是探索性的。日后 `git pull` 后默认切换为 location split,两者数字**不可直接比较**。

### 2026-08-15续 — 用户三项决定:4 折分组交叉验证、action 词表说明、G2 改为"合训分评"

**一、用户决定:改用 4 折分组交叉验证(grouped k-fold,单元=地点)。**
Claude 提出的理由(用户采纳):12 个地点做单次 60/20/20 后,**TEST 只剩 2-3 个地点**,抽到
`eat_mcdonalds`(单一场景、动作少)还是 `hardware_homedepot`(5 shard、动作丰富)会让结果天差地别
——**该差异来自抽签而非模型**。轮转后既用满数据,又能给出"跨地点方差"这一本身有价值的量。
实现(`splits.folds`):把 12 个地点按 clip 数贪心划为 k 个 block(大单元优先进最空的 block);
fold i 取 block i 为 TEST、block (i+1)%k 为 VAL、其余为 TRAIN → **每个地点恰好当一次 TEST、
一次 VAL**,且任一 fold 内 train/val/test 三者地点互不相交。每个 fold 独立通过
`missing_groups` 断言。驱动新增 `--folds k`,输出含 `fold` 列,并打印**跨折的
mean [min, max]**——刻意不是只报均值:"在三个地点赢、第四个输"的模型没有证明跨环境泛化。
新增 5 个测试(每地点恰好当一次 TEST、fold 内地点不交叉、折间规模不失衡、确定性、k<3 报错),
`tests/test_opentouch_splits.py` 共 **14 passed**。

**二、回答"词表只从 TRAIN 构建"是什么意思**:probGRU 的 `nn.Embedding(n_act, 8)` 是一张查找表,
行数必须训练前定死。规则:**只统计 TRAIN 的 clip**,TRAIN 中出现 ≥ `min_group_size` 的动作各占
一行,其余(TRAIN 中罕见,或 TRAIN 中**根本没出现过**)一律映射到第 0 行 `other`。
**为何不能用全语料建表**:只在 TEST 出现的动作会得到一行**从未被训练的随机初始化向量**,且词表
本身泄漏了测试集信息(等于告诉模型"存在这些动作")。**按地点留出后的后果**:测试地点可能含训练
地点没有的动作,它们全落 `other`,**embedding 恰在最需要它的新情况下失效**——这是诚实的代价,
须在报结果时说明。

**三、用户裁定 G2 的执行方式:"无论 abrupt 还是 smooth 都一起训练,只在 test set 里分开
evaluate 两类动作来验证 G2。"** Claude 赞成(smooth 仅 233 clip,分开训练不可行)。
**但据此发现与原 G2 设计的冲突并已提请用户注意**:2026-08-11 的 G2 写的是"**按 trait class
分别拟合 AR**"。若 AR 按类分别拟合而 GRU 合并训练,**AR 白得一次按类特化的机会,比较不公平**。
按新裁定应统一为:**所有模型(persistence/seasonal/AR/probGRU)在全量 TRAIN 上拟合一次,再把
TEST 按 trait 拆成两组分别计分**。已告知用户,若无异议即照此实现。
**注意执行顺序不变**:按类的数字仍须等 D3(36 个未审计动作词)与 OQ-L/OQ-M 签字**之后**才能查看,
否则裁定变成 post-hoc。因此本轮只落地机制,不产出任何按类数字。

### 2026-08-15续2 — G2 接线完成:全量拟合一次,只在 TEST 上按 trait 分开计分(commit `273e905`)

用户确认:"AR 也一样,都只在 test set 上的 abrupt / smooth 子集上分别实现。"故
**2026-08-11 的"按 trait class 分别拟合 AR"作废**——若 AR 按类特化而 GRU 合并训练,比较测到的
是特化优势而非 trait 本身。

**发现阶段 1 已把计分机制建全**,无需重造:`aggregate.py` 的 `clip_stats()`(逐 clip 充分统计)、
`r2(st, model, baseline, rows=...)`(**`rows` 参数正是"同一拟合模型、只换子集计分"的入口**)、
`clip_balanced_mean`、`delta_r2`;`bootstrap.py` 的 paired / two-sample(含分层)。缺的只是把
harness 的逐 clip 预测接进去的胶水。

**新增 `evaluate.collect_clip_stats(cfg, splits, external=None)`**:在全量 TRAIN 上拟合每个
baseline → 用 `predict_series_by_clip` 保留 clip 归属地预测 TEST → 可并入 GRU 臂的逐 clip 预测
(`external={name: {clip_idx: (n_origins,H,C)}}`,即 `prob_gru.predict` 的返回形态)→ 收敛成一个
`ClipStats`,**所有模型在同一套 masked 点集上计分**。

**新增 `evaluate.trait_rows(cfg, st, allow_unaudited=False)`**:返回各桶到 `st.clip_ids` 的行索引,
供 `aggregate.r2(..., rows=...)` 使用。**当 TEST 中存在未审计动作时直接拒绝返回**(报出具体是哪些
动作),因为"先看数字再裁定"正是预注册要防的 post-hoc——该约束**写进代码强制执行,不靠记忆**。
截至 2026-08-13 有 36 个未审计动作、覆盖 278 clip,**因此 G2 现在会被这道守卫挡住,符合预期**。

**写测试时发现并修正了我自己的一处错误**:我原 docstring 断言 `trait.partition` 会对未审计动作
抛异常——**实际不会**,它分成 smooth/abrupt/**unlabeled**/**unaudited** 四桶,抛异常的是
`trait_class`。测试当场揭穿(断言集合不等),遂改为显式检查 unaudited 桶。3 个新测试:
按类计分用的是同一拟合模型、未审计动作拒绝计分、外部预测能并入同一份统计且不改变 baseline 的
SSE。**全套 60 passed, 3 skipped。**

**下一步的唯一阻塞**:D3——36 个未审计动作词的 rubric 补裁 + OQ-L/OQ-M 签字。做完 G2 即可出数。

### 2026-08-15续3 — 用户签字 OQ-L / OQ-M(采纳 Claude 的分类)

**用户 2026-08-15 回复**:"OQ-L 和 OQ-M 我都同意你的分类。"
- **OQ-L 确认**:`eating` / `drinking` / `scooping` / `serving` **主标签判 abrupt**,并**全部列入
  争议子集**(走第三层敏感性分析)。注:这与 2026-08-07 的旧集合(四个都算 smooth)**相反**,也与
  07-02 probe 的 PI 排名相反——该分歧已如实记录,由用户裁定采纳 rubric 推论。
- **OQ-M 确认**:`pulling` / `pushing` / `adjusting` / `turning` / `moving` / `inspecting`
  **判 abrupt 且列入争议子集**(真实数据实测合计 **678** clip,非 08-11 记录的 684)。

**必须澄清的范围问题(已告知用户)**:**OQ-L/OQ-M 的签字不等于 D3 完成。** 二者只覆盖 10 个已在
冻结表内的动作;**36 个从未审计的动作词(覆盖 278 clip)仍未裁定**——`lifting`/`rotating`/
`twisting`/`folding`/`dropping`/`tapping`/`inserting`/`typing` 等。`evaluate.trait_rows()` 的守卫
会因它们继续拒绝 G2 计分,**这是预期行为**。
**已向用户提议**:由 Claude 按 rubric 逐词给出裁定建议与依据(区分 R1 碰撞/释放判据 vs 持续位移
主体),并标注应否列入争议子集,交用户逐条签字。**等待用户确认后再动手。**

**另**:已向用户说明 GPU 作业的进度查看方式(`qstat -u $USER`;`logs/opentouch_probgru.o<JOB_ID>`;
`grep -c "] epoch "` 数累计 epoch,总数 240 = 3 histories × 80),并提醒 **epoch 计数器每换一个
history 会重置**,须结合最近的 `sweep:` 行判断真实进度。

### 2026-08-15续4 — 澄清：CoPx/CoPy 的坐标系到底是什么（用户提问，纯分析，无代码改动）

**用户问**:"COPx COPy 到底是什么？描述的是 CoP 在手上这个坐标系的位置，还是真实世界物理空间的
位置？如果 plot 出 CoP 坐标，坐标系是什么、表示什么？"

**答（已核对代码，非记忆）**:
- **定义**:压力加权质心。`physical_state.py:48-49`（ActionSense)与 `extract_opentouch.py:148-149`
  (OpenTouch) 是同一套数学:`xbar = Σp·gx/Σp`,`ybar = Σp·gy/Σp`。
- **坐标系 = 传感器阵列自身的索引坐标**,由 `physical_state._grids`(:29-33)线性归一化到 [-1,1]:
  **x = 列方向,y = 行方向,原点 = H×W 矩阵几何中心,半阵列宽 = 1.0**。
- **不是世界坐标**:sensor-fixed,随手平移旋转。手握物整体在空间移动 → CoP 不变。CoP 变化 ≡
  载荷在手面上重新分布(打滑、滚握、刀刃行程扫过掌面)。
- **也不是严格的解剖学手坐标**:轴向是传感器矩阵的行/列方向,不是近远端/桡尺侧;格子→手上皮肤
  的映射由手套走线布局决定(AS 32×32;OT 16×16 中仅 169 格活)。
- **单位是无量纲 grid units,不是 mm**;换算需 taxel pitch,且是沿贴合弯曲手面的薄膜距离,非 3D 欧氏。

**各图纵轴的实际含义(易读错,记录在案)**:
1. `docs/opentouch/exploratory/opentouch_fcop.png` — `plot_opentouch_fcop.py:27` 标注 `CoP x [-1,1]`,即原始网格坐标。
2. `docs/forecast_CoPx.png` 等 v2 图 — **不是位置**。`load_pooled(input_mode="highpass")` 的 target
   是 CoP 的高通(fast)残差,单位仍为 grid units 但零点是慢分量,不能据此推断"手上哪个位置"。
3. harness 6 维 target(`eval_harness/dataset.py:3`)是 raw 网格坐标,训练时按通道 z-norm。

**四个坑(结论性)**:
1. F→0 时 CoP 无定义 → `masking.py` 按力阈剔除;低接触时比值放大噪声(呼应 :1041)。
2. **两数据集的 CoP 不是同一个量**:AS 做了 `baseline_correct`(动态接触质心);OT 故意未做
   (`extract_opentouch.py:21-25`),含 DC 偏置故"几乎不离开传感器中心"
   (`run_opentouch_exploratory.py:20`)。**跨数据集直接比 CoP 数值是错的。**
3. **OT 的 (0,0) ≠ 有效接触区中心**:87/256 死格读 0 不贡献权重,可达域是活格凸包;除非活格集合
   关于矩阵中心对称,否则 CoP=0 无"居中"物理含义。**这是一个尚未量化的量**(见下)。
4. y 随行索引增大 + imshow 默认 origin='upper' → filmstrip 上 y 向下为正;
   `plot_opentouch_tactile_map.py:15` 的 `row=(cy+1)/2*(H-1)` 与此自洽。

**实测印证**:扫 `data/actionsense_states/state_{0,1,10}.npy`,CoP 幅度典型仅 ±0.15~0.35,即动态接触
始终集中在阵列中部一小块 —— 与"手套上的局部位置"一致,与"世界坐标"无关。

**由此浮现的待办(未动手,待用户决定是否做)**:算一次 OpenTouch 活格集合的几何中心与凸包范围,
把 CoP 的"名义零点"与"有效零点"的偏差量化;否则 OT 的 CoP 数值缺一个可解释的参考点。

### 2026-08-15续4 — 结果查看:补上"保存预测 + 画预测曲线"的缺口(commit `6dfc042`)

用户要求看:F/CoPx/CoPy 的预测曲线(横轴时间,含 history 与 prediction,含均值与方差)、
probGRU vs persistence vs AR 的对比、以及分动作类别的评估。

**发现的缺口(必须先说明的坏消息)**:**已完成的那次运行只写了指标 CSV,没有保存模型或预测**
——进程退出时模型和预测全部丢弃,因此**几小时 GPU 时间无法产出任何一条预测曲线**。这是我此前
设计驱动脚本时的疏漏。

**补救(已实现并推送)**:
- `--save-preds DIR`:逐 clip 保存 `clip_<idx>.npz`(该 clip 的真实信号 `y`、`origins`、`fps`、
  action/object、每个模型的 `mu_<model>`,以及 probGRU 的 `sigma_prob_gru`)。**baseline 的预测用
  `predict_series_by_clip` 重算以保留 clip 归属**,而不是从 `fit_and_forecast` 返回的拼接数组里
  切——后者已丢失 clip 身份。
- `prob_gru.predict_with_sigma()`:返回 RAW 单位的 (mu, sigma)。z-score 是逐通道线性变换,
  均值项抵消,故 `sigma_raw = exp(lv/2) * norm.std`。
- `scripts/plot_opentouch_forecast.py`:行=clip,列=通道;**同一时间轴上画出 history 与 horizon**,
  灰竖线标 forecast origin,黑线为真实值(历史段实线、未来段淡化),各模型不同颜色线型,
  **probGRU 叠 ±2σ 带**。已用合成的 `--save-preds` 形态数据本地验证出图正确(σ 带随 horizon 变宽
  可见)。
- **为什么要画那条 σ 带**:该方差头**参与训练(占高斯 NLL 的一半)却从不被评分**(冻结 harness 只
  测点误差)。这张图是它唯一可见之处:**若带宽不随 horizon 变宽,说明概率那一半什么也没学到**,
  而任何 MSE 表都看不出这一点。

**代价**:要出图必须**再跑一次**带 `--save-preds` 的作业(预测无法从已有 CSV 反推)。

**分类别评估仍被挡住**:`evaluate.trait_rows()` 因 **36 个未审计动作(278 clip)** 拒绝计分——
这是预期的预注册保护。**OQ-L/OQ-M 已签字不解除该阻塞**(二者只覆盖表内的 10 个动作)。
已再次向用户提议:由 Claude 按 rubric 逐词给出 36 个词的裁定建议与依据,用户签字后 G2 即可出数。

### 2026-08-15续5 — 【结果】GPU 全量首跑完成(2026-08-15 08:11 EDT):probGRU 反超 AR

**运行配置**(据输出的 split tag 判定):**全量语料、ad-hoc `scene` 级 split、seed 0、单次划分
(非 4 折)、`--model prob_gru`**。即该作业用的是**提交时的旧代码**,**早于** `splits.py`
(location split)与 4 折的落地,故与日后 location-split 的数字**不可直接比较**。

**结果原文**:
```
=== full-horizon per-channel MSE (EXPLORATORY, split=adhoc-scene-seed0) ===
model                     F_R       CoPx_R       CoPy_R
persistence      234836372.94630      0.00008      0.00004
seasonal         235574173.63953      0.00008      0.00004
ar               193205409.42370      0.00006      0.00003
prob_gru         186618481.82375      0.00005      0.00003

=== skill vs persistence ===
seasonal              -0.0031      -0.0066      -0.0056
ar                     0.1773       0.2695       0.1984
prob_gru               0.2053       0.2911       0.2405
```
换算成可读量级:persistence 的 F 上 RMSE ≈ **15,324**(约为 F≈750,000 的 **2.04%**);ar ≈ 13,900
(1.85%);prob_gru ≈ 13,661(1.82%)。**即"1 秒后假设完全不变"只错 2%,所有模型都在这 2% 的余量
内竞争**——D1(直流偏置)未解前,这些 skill 与"触觉动态可预测性"关系很弱。

**最值得注意的发现:排序相对冒烟运行发生了翻转。**
| | AR | probGRU | 排序 |
|---|---|---|---|
| 冒烟(300 clip, 1 epoch) | 0.280 / 0.441 / 0.256 | 0.247 / 0.377 / 0.187 | AR > GRU |
| **全量(80 epochs)** | 0.177 / 0.270 / 0.198 | **0.205 / 0.291 / 0.241** | **GRU > AR(三个通道全部)** |
充分训练后 **probGRU 在全部三个通道上超过 AR**。**G1 的核心问题是"ActionSense 的
AR > GRU > persistence 能否在第二个传感器上复现"——就本次(探索性)证据看,它没有复现,
方向相反。** 同时两者的 skill 绝对值都比冒烟时低,与"全量语料更难、冒烟子集偏易"一致。
seasonal 仍略负(-0.003~-0.007),与"无自相关峰、退化为 persistence"的既有结论一致。

**必须附带的限定(缺一不可)**:
1. **探索性**:ad-hoc scene split,非 location split,更非 4 折——**换一组留出地点结论可能不同**。
2. **D1 未解**:目标 95% 是直流,比较的是"谁把 2% 的漂移拟合得更好"。
3. **单次划分**:无跨折方差,无置信区间。
4. **无分类别结果**(G2 仍被 36 个未审计动作挡住)。
5. **未保存预测**:该运行无法产出任何预测曲线(见续4),要出图须带 `--save-preds` 重跑。

**因此本条记录的是一个"信号",不是结论。** 要把"GRU > AR"变成可报告的发现,至少需要:
location split + 4 折 + D1 定案后重跑。

### 2026-08-15续5 — 分析:CoP 能否投影到世界坐标(用户提问,纯分析,无代码改动)

**用户问**:"如果想 plot CoP 的 trajectory,理论上可以把 CoPxy 投影到世界坐标里吗?"

**结论:理论可行,但"投影已算好的 CoP"这个动作本身在数学上是错的,且我们目前缺最关键的一块标定。**

**(a) 数学纠正(结构性,非实现细节)**
设 φ:网格坐标 → 世界 3D。所求为 `Σpᵢ·φ(xᵢ)/Σpᵢ`;"投影 CoP"算的是 `φ(Σpᵢ·xᵢ/Σpᵢ)`。
**二者仅当 φ 仿射时相等。** φ 必不仿射:16×16 是电学行列矩阵,到手面的嵌入是 FPC 走线布局,
弯曲且几乎必然**不连续**(矩阵相邻格在手上可能相隔数厘米,如指尖与拇指尖)。87/256 死格本身
即"手形活区 + 矩形补白"的证据。
→ **正确做法:先把每个 taxel 映到 3D,再在 3D 里算加权质心。** 故 `state_*.npy` 的 6 维矩不够用,
**必须从 `clip_*.npy` 重算**。(3D 质心落在手内部是正常的,那正是刚体意义的压心。)

**(b) 需要三块信息,现状盘点**
1. **taxel→手局部 3D 表面点 —— 没有,最大缺口。** 仓库无任何 taxel 布局/几何标定。shard 顶层
   `calibration` 字段(:510)**从未被打开检查过**,可能仅是压力标定曲线。**第一个要查的。**
2. **逐帧世界系手姿态 —— 存疑,记载自相矛盾。** `extract_opentouch.py:10` 写 `(T,21,3)`,
   `probe_opentouch.py:6` 记 `(21,3)`。**若为 (21,3)(每 clip 一帧),逐帧世界轨迹直接不可行。**
   且 Rokoko Smartgloves 只给相对腕部的手指关节,绝对世界位姿须走 Aria SLAM
   (`camera_poses` / `transform_slam_to_rgb`)。
3. **手表面(非仅关节) —— 没有。** 21 landmark 是骨架点;需 MANO 类手网格蒙皮
   (`batch_process_wilor_simple.py` 是上游遗留)。
**ActionSense 侧更彻底:根本没抽 pose**(cache 仅 `state_*`/`clip_*`),要用需重新下载 Xsens。

**(c) Claude 的判断(向用户明确表态):世界系 CoP 轨迹对 G2/trait 研究大概率负价值。**
它会被**手的整体位移主导**(端物体走一米,而手上载荷分布可能纹丝不动)。传感器系 CoP 恰恰滤掉
整体运动,只留"载荷在手上如何重新分布"——正是 smooth/abrupt 要区分的量。换世界系 = 注入一个
与手法无关的巨大混杂项。**演示/可视化可用世界系;预测/trait 分析应留在传感器系。**

**(d) 提出两个更便宜、可能才是用户真实需求的替代**
1. **画活格掩膜形状**(逐格时间方差,用 `clip_*.npy`)。若呈手形(掌+五指条带),即**免费获得 taxel
   的解剖学语义**,CoP 落点立刻可解释,无需任何 3D 标定;并顺带解决续4 遗留的"活格形心 ≠ 名义零点"。
2. **在现有 filmstrip 上叠加 CoP 历史轨迹拖尾**。同一坐标系内不涉及非仿射映射,**完全正确**,
   且直接就是用户要的 "CoP trajectory"。

**(e) 若要推进 3D 版本,第一步是三个可证伪检查(CRC,只读)**:`calibration` 是否含几何;
`hand_landmarks` 形状是 (T,21,3) 还是 (21,3);腕点是否随时间移动(判世界系 vs 腕局部系);
`camera_poses`+`transform_slam_to_rgb` 是否闭合世界系链路。**任一为否,3D 方案即终止。**
已向用户提议写该只读探查脚本(不改数据、不产出任何 R²,不干扰 D1~D9 决策链)。**等待用户指示。**

### 2026-08-15续6 — 【用户裁定】D1/D2/D3 的做法被修正;执行顺序前插零成本可行性检查

用户对 Claude 的 D1/D2/D3 方案提出实质性修正,**以下为用户原意的完整记录(Claude 的原方案在相应
位置被推翻)**:

**D1 — 必须把"扣基线"与"压噪声"拆成两件事。**
- **基线用中心统计量:无接触段的 median**,**不是低分位数**。理由(Claude 原方案用 5% 分位,被否):
  低分位数确是"不被接触污染"的稳健估计器,但它**系统性低估**,而代价就是**整流偏置**——扣掉一个
  偏低的基线再截断到 0,残余噪声被半波整流,产生随噪声水平缩放的虚假正偏。
- **噪声用显式阈值处理**:估 σ̂(无接触帧的残差标准差),再做 **soft-threshold
  `X ← max(X − k·σ̂, 0)`**。如此两个参数各自可解释、可分别做敏感性分析,而不是把两件事混在一个
  分位数里。
- **Claude 需提请决定的实现细节**:"无接触帧"如何界定(可用标注的 onset/post 结构、或逐 taxel 的
  未受载帧)——这是 D1 剩下的唯一实现分歧。

**D2 — 条件同意"修 join 不丢 shard",但两点修正:**
- **"确认"的程序必须写死**,否则"若确认为 join bug 则修"是**无法证伪**的空话。可执行判据:从错配
  shard 抽若干 clip,**检查 label 与信号形态是否自洽**(typing 与 pouring 的力曲线不可能相像)。
  **若按时间戳能重新对上 → join bug,可修;若源 metadata 自身矛盾 → 改 join 也救不回来,只能丢。**
- **"丢掉 40% 语料"不是最强论据(Claude 原论据被降级)。** 更强的是:**错配 shard 大概率不是随机
  分布的**(多半集中于某次采集、某个 environment 或某批 participant)。丢掉它们等于**改变了 claim
  所针对的总体**,而且这一改变**与 D9 的 split 轴共线**——会在同一批 scene/participant 上同时做
  "删除"与"留出",**外部效度直接塌掉**。这比数据量重要得多。
- **manifest 需补的字段**(在 ts_start/ts_end 之外):**shard_id / participant / scene / session /
  源文件 hash / 实测采样率**。hash 用于重下载后仍可审计;**实测采样率**是因为超参按物理单位冻结,
  **帧率静默漂移会让继承来的 window length 变成另一个超参**。

**D3 — 必须盲裁(blind adjudication)。**
Claude 已经知道 smooth 仅 233、且补裁能显著扩大 smooth 类——**这个知识本身就构成压力**。因此:
**裁定时不得查看每个词带多少 clip**,只凭 rubric + 动作语义给判决,**签字、记 commit,然后才 join
计数**。这不是形式主义,而是把"我按物理判据裁的"从**声称**变成**可查证**。

**执行顺序(用户修正:在最前面插入零成本可行性检查,因其可能直接否决后面的决策)**
1. **三张交叉表(约 10 分钟,不碰模型)**:① 争议 × 类别;② regime × participant × scene;
   ③ clip 在 scene 下的嵌套结构。→ 分别决定 **D4 的报表方案是否成立**、**D6/D9 是否相容**、
   **bootstrap 单位怎么选**。
2. **D2 定性(按上述诊断程序) ＋ D3 盲裁**(后者完全不依赖数据,可完全并行)
3. **D1 方案定 + 用 cache 重算**
4. **D5/D8 现在就锁**(均建议"并列报表 / 写明定义",没有推迟的必要)
5. **D9/D6 → D7 签字 → 才跑 G1/G2**

**对"扣基线前后对照的纯描述性脚本"(不出任何 R²)**:用户强烈支持,并要求增加三个输出:
① **整流偏置诊断**——本应闲置的 taxel 在扣基线后非零帧的占比;
② **每 shard 的 dead / stuck / saturated 计数**;
③ **每 shard 基线的跨 shard 稳定性**。

**Claude 的执行说明**:三张交叉表中的 "participant" 字段**目前并不存在**(正是 D2 要补的),故该表
今日只能以 location(shard 基名)作代理并标注此限制。为不违反 D3 的盲裁要求,**Claude 将先完成 36
个词的盲裁并提交,再运行任何涉及计数的交叉表**。

### 2026-08-16 — 【结果】盲裁后 join 计数:词表 100% 覆盖;并发现 G2 的一个单动作主导风险

用户签字后,36 个盲裁判决已写入 `trait.py` 并**先于计数提交**(commit `7376efd`,时间戳即盲裁证据),
随后 join 真实 manifest。**结果:**
```
clips 2904 | smooth 307 (10.6%) | abrupt 2597 (89.4%) | contentious 836 (28.8%)
unlabeled 0 | unaudited 0        <- 词表完全覆盖,无 clip 落在分类之外
剔除争议子集后: smooth 277 | abrupt 1791
前 12 大动作: picking up 951 / placing 249 / pulling 245[D] / pressing 233 / pushing 154[D]
              / holding 115 / grasping 107 / adjusting 88[D] / turning 84[D] / moving 78[D]
              / touching 78 / removing 55
```

**一、盲裁的实际影响很小,这反而印证了程序的价值。**
smooth 233 → **307**(+74,+31.8%),abrupt 2393 → **2597**;比例由 1:10.3 变为 **1:8.5**。
即"补裁会显著扩大 smooth 类"这一**事前担心的压力源在事后被证明基本不存在**——长尾 36 个词总共只
带 278 clip,且多数落 abrupt 侧。盲裁因此**没有付出任何代价**,却把"我按物理判据裁的"变成了可查证
的事实。**unaudited = 0**,`trait_class` 不会再对任何真实动作抛异常,**G2 计分守卫解除**。

**二、争议子集的剔除是高度不对称的(影响 D4 的报表方案)。**
contentious 728 → **836**(+108)。剔除后:**smooth 仅 -30(-9.8%),abrupt -806(-31.0%)**;
即**争议子集的 96%(806/836)在 abrupt 侧**。因此"全量 vs 仅无争议"这对并列表**不是一次对称的
稳健性检验**——两表之差几乎全部来自 abrupt 类成分的变化。报告时必须写明这一点,否则读者会误以为
两类受到同等程度的检验。
**好消息**:剔除后 smooth 仍有 **277** clip(此前担心它会塌到几十个),**D4 的两表方案在样本量上
成立**。

**三、【新问题】G2 可能在测"holding vs picking up",而不是"smooth vs abrupt"。**
- `picking up` 单个动作 **951 clip = 全语料的 32.7%、abrupt 类的 36.6%**;
- `holding` **115 clip = smooth 类的 37.5%**。
**两个类各自被单一动作主导**,因此 ΔR² 的很大一部分可能来自"这两个具体动作的差别",而非 trait
这一抽象属性。这是此前未识别的**构念效度(construct validity)风险**,与既有的类别不平衡问题不同。
**Claude 建议的应对(待用户裁定)**:(a) 主表之外并列报**逐动作**的 R²,使单动作主导可见;
(b) 做 **leave-one-action-out** 检验(依次剔除 picking up / holding 重算 ΔR²),若结论翻转则说明
效应由该动作承载;(c) bootstrap 时按 action 分层(与 OQ-K 的分层轴决定合并考虑)。
**倾向 (a)+(b)**:两者都零建模成本,且 (b) 直接回答"结论是否只由一个动作撑着"。

**四、统计功效**:smooth 307 vs abrupt 2597。两样本 bootstrap 中 ΔR² 的精度由 smooth 侧的 307
clip(以及它们跨越多少个 location——bootstrap 单位问题,见 D6/OQ-K)决定,不是由 2904 决定。

### 2026-08-16续 — history 扫描与零填充:澄清一个误解;本轮维持 OQ-H,下轮打印填充比例

用户问"改成 {0.5,1,2} 会不会有更多 split"。**答案是否定的,且这个误解值得记录**:
`origins()` 只取决于 `min_history`(15)与 `horizon`(30),**与 `t_in` 无关**
(`lo=min_history; hi=T-horizon; arange(lo,hi,stride)`)。因此**无论 history 取多长,每条 clip
产生的窗口数完全相同**;变的只是**每个窗口中真实数据与补零的比例**(窗口取
`f[max(t-t_in+1,0):t+1]`,不足则左侧补零,而特征已 z-score,**补零 = 训练均值**)。

按 clip 长度分布测算(需 `T ≥ 30 + t_in − 1`):
| history | 需要长度 | 存在完整窗口的 clip 比例 |
|---|---|---|
| 0.5 s (t_in=15) | ≥1.47 s | **恒无填充**(min_history 恰为 15,首个 origin 即满足) |
| 1 s (t_in=30) | ≥1.97 s | 约 75% |
| 2 s (t_in=60) | ≥2.97 s | 不足 50% |
| 3 s (t_in=90) | ≥3.97 s | **约 25%** |
即 **3 秒档下约 3/4 的 clip 没有任何一个未填充窗口**;上次扫描选中 1 s 很可能源于此,而非
"1 s 信息量最优"。

**决定**:加入 0.5 s 等于**推翻 2026-08-11 对 OQ-H 的裁定**("扫描点为 {1,2,3} s;0.5 s 只是
harness 的 min_history 下限,不是扫描点"),故**本轮维持 {1,2,3} 不变**。Claude 建议改为
**打印每档的填充比例**——零风险、不推翻已有裁定,且该组数字正是日后正经决定扫描范围的证据
(可与各档 val NLL 对照)。**用户已提交本轮作业,同意下一轮再加打印。**

**本轮运行的定位(已与用户确认)**:**D1 之前的参照运行**。价值:(a) 首次跑通 location split +
4 折;(b) 产出预测曲线与跨折方差;(c) **保留一份未扣基线的结果作为 D1 落地后的对照**,以显示
"扣基线到底改变了什么"。输出文件名加 `preD1` 后缀以免混淆。
**不阻塞本轮的事项**(均可用保存的 per-clip 预测事后计算,无需再占 GPU):置信区间、分类别 skill、
逐动作 R²、leave-one-action-out。

### 2026-08-16续2 — D1 与 D2 的测量工具落地(commits `06ffe0b`, `c6200f7`)

用户已提交本轮 GPU 作业,随即要求开始实现 D1 与 D2。两者都**只出描述性统计,不出任何 R²**。

**D1 — `scripts/opentouch_baseline_report.py`**
按用户裁定实现"基线用中心统计量 + 噪声用显式阈值"的拆分:
- **基线 = 逐 taxel、逐 shard 的 median**(b+c 结合)。**"无接触帧"未引入第二个阈值**——那只会
  把任意性挪到更不显眼处;依据是"受载时间不足一半的 taxel,其静息水平就在自身 median 上",
  该前提**被测量而非假设**:`duty>0.5` 列数出违反前提的 taxel 数。
- **σ̂ 单侧估计**:仅取 ≤median 的帧算 MAD × 1.4826(接触只会把读数推高,排除上半部分可使抓握
  进不到噪声估计)。
- **soft-threshold `X ← max(X − (base + k·σ̂), 0)`,k 扫描而非固定。**
- 三个用户要求的诊断全部实现:整流偏置(闲置 taxel 在校正后仍非零的帧占比)、每 shard 的
  dead/stuck/saturated 计数、跨 shard 基线稳定性(逐 taxel 基线的 shard 间 IQR 及其相对量)。

**合成语料验证(按 2026-08-13 实测结构构造:每格≈2900、稀疏接触)复现了用户预判的失效模式**:
| k | F 均值 | F 变异系数 | CoP 活动范围 | 整流偏置 |
|---|---|---|---|---|
| raw | 742,559 | **0.0059** | 0.008 | — |
| 0 | 4,946 | **0.803** | 0.489 | **0.423** |
| 2 | 1,980 | 1.407 | 1.902 | 0.017 |
| 3 | 1,608 | 1.440 | 2.000 | **0.000** |
**只扣基线会让闲置 taxel 有 42% 的帧变成非零**;k=2 降至 1.7%,k=3 归零。同时 F 的变异系数从
0.0059 升到 0.80(135 倍)、CoP 活动范围升 60 倍。**但诊断也暴露反向张力:k=3 时 CoP 范围冲到
2.0000(整个值域),因存活格子过少使重心跳跃;k=5 又回落到 0.56。故 k 不是越大越好,必须做敏感性
分析而非拍值。**

**D2 — `scripts/opentouch_label_audit.py`**
把"若确认为 join bug 则修"中的**"确认"落成可执行程序**(用户指出该词不落地即无法证伪):
- 报告**每 shard 的错配率**(随机散布 vs 集中,含义不同——这对应用户"错配大概率非随机、丢弃会
  改变 claim 所针对的总体、且与 split 轴共线"的论据)。
- 对错配 clip,用 `trait.delta_f_p95`(冲击度)与 `hf_energy_fraction`(高频占比)**检验标签与信号
  形态是否自洽**;**参照系仅由"peak 一致"的 clip 构建**,避免拿嫌疑样本互相校准。
- **明确写出它做不到的部分**:判断能否"重新对上"需要 clip 时间戳,而 `extract_opentouch.py`
  只写了 `fps_est`/`T`、**从未写入时钟** → **重新 join 必须把那些 shard 重新下载回来**。
- 合成验证:5 个被故意错标为 `pouring` 的冲击型 clip 全部判为 **IMPLAUSIBLE(冲击度为该动作典型
  值的约 9.8 倍)**,而仅索引错位、标签正确者不被如此标记——**恰好区分"可修的 join bug"与
  "标签本身错、修不回来"**。

**待用户在 CRC 运行两者以取得真实数字**,再据此(a) 定 k 与基线方案,(b) 对 D2 作出 fix/drop 裁定。

### 2026-08-16续3 — 【D1/D2 真实数据结果】三个可信结论 + 我脚本的两处度量缺陷(已修,需重跑)

**D2(标签审计)可见部分的结论:错的是索引,不是标签。**
用户贴出的每一条判决**全部是 `consistent`**(placing / pressing / inspecting / adjusting /
picking up / touching / flipping / sliding),冲击度均落在各自动作典型值的合理倍数内。
错配集中在**多部分地点**(`hardware_homedepot_p1`、`fablab_ml_p1/p2`),且偏移量很大
(idx 2686 差 **173 帧**=5.8 s;2806 差 115 帧),与"共用一份标注 CSV、索引相对于合并流"的机制吻合。
**若整份文件的 IMPLAUSIBLE 计数确为 0/极少,则 D2 的杀伤力大幅下降**:已核查全链路,
`peak_idx`/`onset_idx`/`post_idx` **仅被绘图脚本用于画竖线**——训练(probGRU)、baseline、split、
trait 分类**均不读取**;而真正会污染 G2 的 `action` 字段是干净的。
**待确认后的处置建议**:不必重下那 10 个 shard、不丢弃、不阻塞 G1/G2;仅需在文档中记明"多部分
地点的标注事件索引基准错误,绘图竖线不可信,任何日后基于 onset/peak/post 切窗口的分析必须先修"。
(用户尚未贴出错配率表与 IMPLAUSIBLE 计数,结论待其确认。)

**D1(基线报告)三个可信结论:**
1. **扣基线是决定性的**:`eat_mcdonalds` 的 F 均值 746,325 → 4,918,即**直流占 F 的 99.3%**
   (比此前估计的 95% 更极端);变异系数 0.0084 → 0.455(**54 倍**);CoP 活动范围 0.041 → 1.24
   (**30 倍**)——CoP 由"钉在中心"变为真正在动。
2. **用户方法的前提在全语料上成立**:`duty>0.5` 在**26 个 shard 上全部为 0**,即没有任何 taxel
   受载超过一半时间,"逐格 median 即无接触段中心统计量"成立。
3. **`dead = 0`,全语料无一 taxel 读零** → `extract_opentouch.py` docstring 中"169 个活格、死格
   读 ~0"作为对本 cache 的描述**是错的**(此前仅在单图上怀疑,现为 26 shard 普查结论)。
   另:基线跨 shard 的相对 IQR 中位仅 **0.009**、p90 **0.050** → **多数格子用全语料基线亦可,
   但存在一条不稳的尾巴**;脚本原先"必须逐 shard"的措辞下得过满,已改为要求同时读 median 与 p90。

**我脚本的两处度量缺陷(因此当前不能据此定 k),已修复并推送(commit `9bb99a2`):**
- **σ̂ 在 26 个 shard 上全部退化为 0.00**。根因:**读数是整数量化的**(各 shard 基线皆为整值),
  未接触时多数 taxel 恒定,故 ≤median 的残差过半为 0,MAD 随之为 0 → **soft-threshold 完全失效**
  (这解释了 k≥2 后 F 几乎不变、整流偏置卡在 0.055 不降)。**修复:σ̂ 下限取半个量化步长**
  (量化步长由数据实测,非假设)。
- **`saturated` 判据无意义**(每 shard 报 212–256/256),它实际测量的是"池化分布有多窄"。
  **修复:改为"精确等于该格自身最大值的帧占比 > 5%"**——量化数据中真正打满的格子会反复精确停在
  轨值上,带噪声的正常格子不会。双向验证:浮点合成数据 0 个、量化+人为打满数据检出。
- `idle`(整流偏置的参照集)由"最大值最小的 5% 格"改为"**相对自身基线的超出量最小的 5% 格**"
  ——在所有格子都静息于 ~3050 时,按最大值排序排的是基线而非活动度。

**下一步**:用户在 CRC 重跑 `opentouch_baseline_report.py`(修复版)以取得可用的 k 敏感性曲线;
并从 `~/d2_audit.txt` 取错配率表与 IMPLAUSIBLE 计数以完成 D2 裁定。

### 2026-08-16续4 — 【D1 定 k = 1】【D2 结案:仅 1 个可疑 clip】【新增饱和判别工具】

**一、D2 结案(证据齐全)。**
错配率表:`sports_dicks_p2` **50/125 = 40.0%**、`sports_dicks_p1` **22/105 = 21.0%**、
其余 8 个 shard 合计 **30/2674 = 1.1%**、全语料 **102/2904 = 3.5%**。
→ **`sports_dicks` 一个地点贡献 70.6% 的错配(72/102),而它仅占语料 7.9%;其内部错配率 31.3%,
语料其余部分 1.1%,相差 28 倍。**该地点正是 `extract_opentouch.py` 点名的**共用一份标注 CSV**
的那一对,机制与"索引基准相对于合并流"完全吻合。
标签自洽性:**consistent 46 / IMPLAUSIBLE 仅 1**(`grep -c` 报 2 是因末尾说明文字含该词);
唯一一条为 `idx 1551, picking up, 3.1x 典型值`,刚过 3.0 阈值,属边缘。
**→ 裁定:错的是索引基准,不是标签。不重下、不丢弃、不阻塞 G1/G2。**
依据:(a) `action` 干净 → trait 分类与 probGRU 的 action embedding 未被污染;
(b) `peak_idx`/`onset_idx`/`post_idx` **经全链路核查仅被绘图脚本使用**;
(c) **用户的共线性论据被数据坐实**——`sports_dicks` 是一个完整地点(2 shard/230 clip),而 split
的留出单元恰为地点且总共只有 12 个,丢弃它等于同时删掉一个类别来源与一个留出单元。
**须记录的限制**:多部分地点的标注事件索引基准错误,绘图竖线不可信;任何日后基于 onset/peak/post
切窗口的分析必须先重新 join(需把 shard 重新下载回来)。

**二、D1 定 k = 1(主值),k ∈ {0, 2} 作敏感性区间。**
修复版 k 扫描(`eat_mcdonalds`):
| k | F 均值 | 较上档减少 | CoPx range | CoPy range |
|---|---|---|---|---|
| raw | 746,325 | — | 0.041 | 0.030 |
| 0 | 4,918 | — | 1.240 | 1.307 |
| **1** | **1,477** | **−3,441(残差的 70%)** | 1.415 | 1.728 |
| 2 | 1,220 | −257 | 1.867 | 1.440 |
| 3 | 1,196 | −24 | 1.867 | 1.444 |
| 5 | 1,181 | −15 | 1.867 | 1.453 |
**曲线在 k=1 处有陡峭膝点**(削掉 70% 残差),此后每档仅再削几个百分点。
**σ̂ 在全部 shard 上恰为 0.50 即所设下限** → 中位 taxel 的非接触读数**完全恒定**,MAD 真为 0。
因此 **k=1 具有干净的物理含义:阈值 = 基线 + 0.5 计数,对整数数据即"只保留严格高于静息值至少
1 个计数的读数"——量化数据上最小的有意义阈值**,而非拍值。
**k≥2 不采纳**:CoPx 冲到 1.867(接近整个 2.0 值域)后不再变化,而 CoPy 由 1.728 回落至 1.440,
**非单调**,即合成测试中"存活格子过少导致重心跳变"的征兆。

**三、整流偏置诊断在真实数据上恒为 0,已失效(需改造)。**
原因:`idle` 取"超出自身基线最少的 5% 格",而这些格子**完全恒定**,扣 median 后精确为 0。
**但这一事实本身回答了用户的担心**:整流偏置源于"低分位数低估基线",而 median 对恒定格子是
**精确正确**的,故无可整流之噪声;真正的虚假 F 是那 3,441 计数的 ±1 抖动,k=1 恰好清除。
**改造方向(未实施)**:改为报告"无接触格子对总 F 的贡献量",而非"非零帧占比"。

**四、新增 `scripts/opentouch_saturation_check.py`(commit `4b96c3f`)。**
D1 census 报 135–241/256 个格子"在自身最大值上停留 >5% 帧",**若为真实削顶,则 F 是被截断的量,
所有力相关结论都受影响**,故单独测量。判别逻辑(单格无法区分,聚合可以):轨是**同一个值**、
**跨 shard 相同**(电子学属性而非 session 属性)、**在时间上形成平台**、并**按被钉住的样本比例
吞掉力**。双向验证:植入 4095 硬件轨 → 单一共享值、跨 shard 一致、at-max 连续长度 p90=13 帧、
被标记格子承载 **99.2%** 的校正后 F;无轨(仅峰值)→ 14 个各异最大值、跨 shard 不同、95.4% 的
连续段仅 1 帧、仅占 **0.2%** 的力。
**待用户在 CRC 运行以定性。**

### 2026-08-16 — 新增只读探查脚本 probe_opentouch_geometry.py(续5 的 (e) 落地)

**触发**:用户把续5 的 Python 片段直接粘进 bash,报 `import: command not found`。遂写成脚本。

**新文件 `scripts/probe_opentouch_geometry.py`** —— 纯只读:不写文件、不拟合、不产出任何 R²,
**因此不污染 D1~D9 决策链**。分两级,因为**shard 已全部被删**
(`stream_opentouch.sh:107` / `download_own_copies.sh:110` 的 `rm -f` 是有意设计,为守 home 配额),
只有 cache 幸存:

- **TIER A(`--cache`,零下载)** 裁决检查 ②。`pose_*.npy` 本就在 cache 里。三项输出:
  1. **形状 `(T,21,3)` vs `(21,3)`** —— 直接裁定 `extract_opentouch.py:10` 与
     `probe_opentouch.py:6` 哪个 docstring 有错。**若为 (21,3),逐帧世界轨迹当场出局。**
  2. **T(pose) 是否 == T(pressure)** —— **续5 遗漏的一点,同样致命**:即便有逐帧 pose,长度不一致
     就须靠 timestamps 对齐,而 **cache 未存 timestamps** → 那是重抽,不是修补。
  3. **腕点(landmark 0)是否移动 + 坐标量级** —— 腕点恒 0 ⇒ 腕局部系(只有手指关节,无世界位置);
     量级区分归一化 / 米 / 毫米。
- **TIER B(`--shard`,需重下 1 个 shard)** 裁决 ① 与 ③:递归 dump `calibration`
  (判据:出现覆盖 256/169 taxel 的 (...,3) 形张量才是我们要的 taxel→手面映射;增益/偏置/曲线不是)
  与 `camera_poses` + `transform_slam_to_rgb`(4x4 SE(3) 则相机侧世界链路闭合)。

**给用户的执行建议**:**先只跑 TIER A**。② 若挂,①③ 无再查必要,省 587 MB 下载。
TIER B 的 shard 取最小的 `office_csail_p2.hdf5`(ID `1914FdF...`,587 MB),**须下到
`~/opentouch/probe/` 而非 stream 脚本工作目录**(否则可能被其清理逻辑连带删除),查完立即 rm。

**状态**:脚本仅在本地 mac,**尚未 commit**(已征求用户是否 commit 以便 CRC `git pull`)。**等待指示。**

### 2026-08-16续5 — 【饱和实测:传感器工作在满量程 95%】【用户裁定放弃 D1】【D2 处置实装】

**一、饱和判别结果:全部判据都指向真实硬件削顶,且比预期严重得多。**
```
天花板 3072.0 占全部 (格子, shard) 最大值的 94.8%
per-shard maximum: min 3072.0 max 3072.0  distinct 1 of 26 shards
at-max 连续长度: median 2  p90 15  max 9430 帧(30 Hz 下 314 秒)
被标记格子中 65.4% 的样本被钉在顶部;它们承载 38.6% 的校正后 F
```
**推论(独立可验证)**:单格上限 3072 → F 的理论上限 = 256×3072 = **786,432**;而实测 **F 均值
746,325 = 理论上限的 94.9%**。**即阵列静息时已跑在满量程 95%,只剩 5% 余量留给接触。**
各 shard 基线中位 2978–3072 → 每格可用余量 **0～94 计数(满量程的 0～3.1%)**;
**`fablab_ml_p2` 的基线中位恰为 3072——该 shard 的中位格子永久贴顶,不携带信息。**
更正 Claude 自己脚本里的提示语:"轨应是 2 的整数次幂"太窄——**3072 = 3×1024 = 0xC00**,
同样是固件层面的圆整值。

**二、核查:饱和不是我们抽取造成的。** `extract_opentouch.py:214` 以 **float32** 从 HDF5 读入,
**231 行 `moments(press)` 在 float32 上计算**,float16 仅用于事后保存原始地图。故 **F/CoP 未被
我们降精度**。**但连带警告**:D1 的 σ̂ 估计读的是 float16 地图,而 float16 在 [2048,4096) 的间隔为
**2.0**,故"非接触读数完全恒定、MAD=0"**部分可能是 float16 舍入的假象**;k=1 的方向不变(仍是
最小有意义阈值),但"σ̂ 真为 0"应降格为"在 float16 精度下不可分辨"。

**三、这改变了问题的性质**:此前以为 D1 是"有直流偏置、扣掉即可";实际是**传感器几乎全程工作在
饱和区,大量接触信息在采集端即已被削掉**。余量 94 计数、float16 下约 47 个可分辨等级——这解释了
F 只波动 2–4%、CoP 几乎不动、persistence 难以击败。**扣基线仍能把相对变化从 0.8% 放大到 45%,
但救不回被削掉的部分。**

**四、用户裁定:放弃 D1,直接在原始 tactile 上跑预测。**
Claude 已明确告知代价并记录:**不扣基线意味着模型仍在预测一个 99.3% 为常数的信号,skill 会好看
但与动态无关**。D1 的校正为 cache 的纯函数,**日后可随时补做,无需重新下载**。
(Claude 曾建议"写信问作者是否有未削顶的原始流"与"按余量筛 shard",用户未采纳,不再追加。)

**五、D2 处置实装(commit `02fb629`)。** 裁定所需证据已齐(见续4),唯一的操作性后果是:
**全链路中只有两个绘图脚本读取 onset/peak/post**,而在被标记的 clip 上那些竖线会把"peak"画在
并非峰值处——**比不画更糟**。故:
- `opentouch_label_audit.py --write-flags PATH` 输出被标记 clip 列表,并写明**哪些字段可信**
  (action / object_category / scene / shard / T / fps_est)与**哪些不可信**(onset/peak/post)。
- 两个绘图脚本读取该文件:被标记的 clip **不画事件竖线**,并在行标签上注明
  `[event idx unreliable]`;文件不存在时行为与此前完全一致。
**不重下、不丢弃、不阻塞 G1/G2 —— D2 到此结案。**

### 2026-08-16续6 — 【新数据集 d256.zip】远程 ZIP 清单侦察完成;新增 `scripts/crc/fetch_d256.py`

**请求**:浏览 ICLR force-vision 投稿页,开始把该 dataset 下载到 CRC。

#### 一、页面与文件事实(已核实)
- 页面 `https://sites.google.com/view/iclr-submission-force-vision/` 只给出一个 "New Link"
  的 Drive 下载链接,**无数据集名、无结构说明、无 license、无下载指引**。
- Drive 元数据:`d256.zip`,**198,849,542,248 B = 185.2 GiB**,owner `yichenl@mit.edu`
  (MIT),created 2024-11-15,modified 2025-04-20。
- **公开可下**:当前拿到的是 "Virus scan warning" 确认页(非配额拒绝页),
  即 2026-08-12 那次 OpenTouch 的 "too many users downloaded" 尚未在此文件上触发。

#### 二、关键发现:Drive 支持 HTTP Range,于是**不必下载 185 GiB 就能读到清单**
`curl -r` 实测返回 **HTTP 206 + `Accept-Ranges: bytes` + `Content-Range: .../198849542248`**。
ZIP 的中央目录记录每个成员的偏移量,故:读尾部 1 MiB 拿 ZIP64 EOCD → 定位中央目录
(offset 198,826,721,233,大小 22.8 MB)→ 只下这 22.8 MB 即得**全部 187,729 个成员**的
路径/偏移/压缩尺寸/CRC32。**总成本约 24 MB,而非 185 GiB。**

**清单结果(uncompressed 250.3 GiB / compressed 185.2 GiB):**

| group | files | uncomp GiB | 传输 GiB | 内容 |
|---|---:|---:|---:|---|
| signals | 25,473 | 3.89 | 1.10 | 触觉/EMG/姿态 pickle |
| signals1 | 28,426 | 4.34 | 1.23 | 同上 |
| signals2 | 26,922 | 4.11 | 1.17 | 同上 |
| videos | 50,942 | 75.80 | 57.64 | RGB npz,**同时含 256px 与 32px** |
| videos1 | 28,426 | 83.29 | 63.72 | RGB npz,仅 256px |
| videos2 | 26,922 | 78.88 | 60.30 | RGB npz,仅 256px |

**⇒ 这个 185 GiB 里 95% 是视频;触觉信号总共只有 12.34 GiB(传输 3.49 GiB)。**
且三个 signal group 在归档中各自**连续**,故 signals-only 只需 ~8 段长顺序读,而非 80,821 次小请求。

#### 三、内容定性(**已实际 range-取回真实成员并解开验证**,非推测)
- `signals/<split>/<subject>/<session>/<clip>.p` = pickle dict:
  `{'signal': {...}, 'label_text': str, 'label_idx': int}`。实例:`label_text='Slice a cucumber'`。
  `signal` 九路:`tactile-glove-{left,right}` **(16,32,32) f32**、`myo-emg-{l,r}` (16,8)、
  `myo-acc-{l,r}` (16,3)、`joint-position` (16,28,3)、`{left,right}-hand-pose` (16,24,3)。
  → **这是 ActionSense(MIT CSAIL)的传感器组合**,16 帧片段,数值已预缩放到 ~[0,1]。
  本仓已有 `scripts/probe_actionsense.py` 与 `data/actionsense_states`,**接得上**。
- `videos/.../video_<k>_256.npz` → `arr_0` **(16,256,256,3) uint8**;`_32` → (16,32,32,3)。
- `signals/ego_4d_{verb,noun}.npy` = **Ego4D 词表**,148 verbs / 112 nouns。
- 划分:train/val 两分(无 test);受试者 **S01–S05**;`Dataset256/` 单一顶层目录。
- ⚠️ **`_32` 低分辨率变体只有 `videos` 这一组有**(videos1/videos2 仅 256px)。

**命名解读:`Dataset256`/`d256` 的 "256" 指视频边长 256px,不是 256 taxel**
(触觉手套是 32×32=1024)。**勿与 OpenTouch 的 256/169 taxel 混淆。**

#### 四、新增 `scripts/crc/fetch_d256.py`(纯 stdlib,CRC 上无需 conda 环境)
不走 gdown。理由:(a) 单体 185 GiB 无 shard 可流,`stream_opentouch.sh` 的
"下一片→抽取→删除" 模式失效;(b) 落盘还需**第二份 250.3 GiB** 解压空间;(c) gdown 中途断线
基本等于重来。改为**按需 range 抽取**:读中央目录 → 选成员 → 合并连续区间为长顺序流 →
边收边 inflate 边切成员边写盘。

- `--groups signals`(默认)/`signals,videos --lowres` / `all`;`--include <regex>` 可再筛
  (如 `/val/` 或 `/S0[12]/`);`--plan` 干跑只报量。
- **每个成员按中央目录的 CRC-32 校验**,不符即删 `.part` 并报错 → 不会留下坏文件。
- `done.txt` 断点续传,且**只信"磁盘上确实存在该文件"**(被 kill 的任务会让日志跑在磁盘前面)。
- 每 512 MiB 一个 HTTP 请求;confirm token 过期自动重取,指数退避重试 6 次。
- 断言远端 size == 198,849,542,248,**Drive 上若重传过文件,缓存偏移全部失效时会立刻报错**而非默默写坏。

**本地实测通过**:`--include '/val/S05/(3|2)/'` 取 962 文件 / 42.07 MiB → 全部 CRC 通过,
pickle 可正常 load(`Slice a cucumber`,九路 shape 全对);重跑立即识别为已完成、0 传输。
实测速率 ~1.6 MiB/s(本机沙箱,CRC 侧应更快)——**若真取全量 185 GiB,按此速率约 33 小时,
必须走 UGE 作业而非前端节点。**

#### 五、未执行的部分与原因
**我无法从这里连 CRC**:`crcfe01.crc.nd.edu` DNS 不可解析(需校园 VPN/bastion),且
`~/.ssh` 下**只有 config 与 known_hosts,无私钥**;ND CRC 还要 Duo 二次验证,本会话非交互。
→ 下载必须由用户在 CRC 上执行,我只交付脚本与指令。**脚本尚未 commit。**

#### OPEN QUESTIONS(等用户裁定后才动)
1. **取哪个子集?** (a) 仅 signals = 3.49 GiB 传输 / 12.34 GiB 落盘(~40 min);
   (b) signals + 32px 视频 ≈ +1.2 GiB(但 32px 只覆盖 `videos` 组,视觉侧不完整);
   (c) 全量 = 185.2 GiB 传输 / **250.3 GiB 落盘**(~33 h)。
   我的建议:**先 (a)**。本项目历来只用触觉(Session 1 即主动跳过 mp4),而 (c) 的 95%
   是 RGB;真要做 force-vision 联合建模再补视频,脚本支持增量续取,不浪费已下的部分。
2. **落盘到哪、配额够吗?** 必须 `/scratch365/$USER/...`,**绝不能进 home**。
   请在 CRC 跑 `quota` 贴给我——(c) 需要 250 GiB 余量,(a) 只需 13 GiB。
3. **怎么跑?** (a) 用 `screen`/`tmux` 在前端跑即可;(c) 必须 `qsub` 长作业
   (且需确认**计算节点是否有外网出口**,ND CRC 一般有,但没验证过)。
4. **license/引用未知**:页面无任何 license 或使用条款,ICLR 匿名投稿页。取用前是否需要先
   联系作者(owner `yichenl@mit.edu`)?

#### OPEN QUESTIONS 裁定(2026-08-16,用户)
- **OQ1 = (a) 仅 signals。** 传输 3.49 GiB → 落盘 12.34 GiB,80,821 个 clip + Ego4D 词表。
  理由同建议:本项目只用触觉侧;视频日后可用同一脚本增量补取,已下部分不作废。
- **OQ3 = commit + push,用户在 CRC `git pull` 后执行。**(我无法连 CRC,见上。)
- **OQ2(scratch 配额)与 OQ4(license/引用)仍未决**,但均不阻塞 (a):12.34 GiB 对
  `/scratch365` 是小量。OQ4 在**发表/分发前**必须回答,取数自用不阻塞。

**本次 commit 范围**:仅 `scripts/crc/fetch_d256.py` + 本日志。
`probe_opentouch_geometry.py` 及 `docs/*.png` 等仍留在工作区未提交(前一轮仍在等指示,不顺手带入)。

### 2026-08-17 — 分析:probGRU 里 causal_velocity 的物理意义与三条风险(用户提问,无代码改动)

**用户问**:"probGRU 为什么要算 causal velocity?它的物理意义是什么?"

**(1) 物理意义**:vx,vy 是 CoP 在**传感器网格系**的时间导数,单位 **grid-units/s**,刻画
"载荷在手面上迁移的速度"。**不是手的速度**(握紧物体走过厨房 → vx=vy=0);只有载荷重新分布时非零
(刀刃行程扫过掌面、物体打滑、滚握)。与 smooth/abrupt trait 的目标一致,作为输入合理。
**语义歧义(须记住)**:CoP 速度**混淆真·滑移(接触斑相对皮肤材料位移)与纯·重配权(无滑动,仅食指
压重/拇指压轻即令加权质心移动)**。二者在 `Σp·x/Σp` 的一阶差分里**不可区分**。故不可读作"滑移
速度",应读作"加权质心迁移率"。

**(2) 为何显式喂 v(GRU 本已看到 CoP 历史)**:**信息论上纯冗余**(`vx[t]` 是窗口内两通道的精确
线性函数),是**归纳偏置**而非新信息。仍值得,两个理由:(i) 省掉网络必须学的差分算子(hidden=48、
小数据下不白送);(ii) **更关键是尺度** —— 差分量比水平量小数个量级(F 还压着 DC 偏置),
`FeatNorm` 对 5 通道**各自** z-score,等于把差分放大回 O(1),避免动态信息被水平项淹没。
**细节**:`features()` 在整段 clip 上算完再切窗(`prob_gru.py:139-145`),故窗口首帧的 vx 含窗口
**之前**一个样本 —— 是**过去**信息,严格因果,**不构成泄漏**,但窗口实际携带 t_in+1 个样本历史。

**(3) 为何必须 causal 后向差分 —— 最实质的一点。仓库存在两套速度实现,混用即泄漏:**
| 用途 | 实现 | 差分 | 可否作模型输入 |
| 分析 | `physical_state.py:105-106` `derive()` | `np.gradient` **中心差分** | **绝对不可** |
| 建模 | `prob_gru.py:80` `causal_velocity` | 后向差分 | 可 |
`np.gradient` **看未来**:作 t 时刻输入则输入窗末帧的 vx 含第一个 target 帧的值 = 直接喂答案。
**真实的坑**:日后复用 `derive()` 造模型特征会**静默引入泄漏,且 skill 变好看、不报错。**

**(4) 三条问题(新发现,建议纳入待决策)**
- **(a) 只差分 CoP、不差分 F,在 OpenTouch 上尤其别扭。** `FEATS_RAW` 无 dF/dt。ActionSense 的
  highpass 模式下说得通(F 已拆 F_slow+F_fast);raw 模式照抄了同一组特征。而 OpenTouch 恰是
  **F 被 DC 偏置压平**的数据集(D1 未决;`prob_gru.py:22-23` 自承 "much of the target is a
  constant") —— **唯一能承载 F 动态的量正是 dF/dt,而它恰恰没算。** 未经论证的继承,非 bug。
- **(b) mask 只管 target,不管 input。** `masking.py` 明说掩的是 TARGET 帧的 metric,且仅在
  `evaluate.py:77/130/173` 调用;`window_set` 用**未掩蔽**的 `load_target`。低力帧的病态 CoP
  (比值,低力噪声放大,见 :1041)照样进输入,而差分是高通,**再放大约 2·fps 倍**。评分端干净,
  输入端不干净。
- **(c) 与 D1 直接挂钩的副作用(此前无人记录,建议列入 D1 方案):**
  `extract_opentouch.py:157` 有 `out[F <= 0, 1:] = 0.0`。当前 OT 未扣基线,F 恒为大正数,该行几乎
  从不触发。**D1 一旦落地(按 shard 扣每 taxel 静息水平并截断到 0),脱离接触的帧整帧归零 → F=0 →
  CoP 被强制写 0**,于是紧邻帧 vx 出现 `fps·|x|` 量级**假尖峰**(纯数值伪影),且按 (b)**不被任何
  mask 拦截,直接进模型**。**D1 必须一并决定处置方式**(CoP 置 NaN 前向填充?还是给输入也加掩码?),
  否则扣基线会顺手向输入通道注入一串脉冲。

**(5) 已核实无误**:`cfg.fps = fps_raw/downsample`(`config.py:28-29`)且 `load_target` 亦
`st[::cfg.downsample]`(`dataset.py:38`) —— **降采样与帧率一致**,vx 单位确为 grid-units/s,
无隐藏常数倍错误。

### 2026-08-17 — CRC 上 `--dest /scratch365/$USER` 被拒:非脚本缺陷,是配额/分配问题

**现象**(用户在 `crcfe02` 实跑):
`PermissionError: [Errno 13] Permission denied: '/scratch365/jhao3'`,发生在 `os.makedirs(dest)`。

**判读**:报错在**创建 `/scratch365/jhao3` 自身**,即该目录**不存在**且用户对 `/scratch365`
无写权。ND CRC 的 per-user scratch 目录是**管理员预置**的,不能自建 → 要么该账号没有 scratch365
分配,要么挂载点名字不同。**与 range/ZIP 逻辑无关**,`--plan` 尚未走到网络那一步。

**顺带更正一处我的假设**:CRC 的 netid 是 **`jhao3`**,不是 `~/.ssh/config` 里的 `jh9141`
(那是 NYU torch 的账号)。README 里用 `$USER` 是对的,但我口头给的路径示例应以 `jhao3` 为准。

**改动**:`fetch_d256.py` 的 `os.makedirs` 包了 `except OSError`,向上找到第一个存在的祖先目录,
打印**可诊断的**信息(建议命令 + crcsupport 联系方式 + "13 GiB 小到可以先落 `$HOME`")而不是
裸 Errno 13。捕获 `OSError` 而非 `PermissionError`,因为只读挂载给的是 EROFS(mac 上实测 Errno 30)。
两条路径均实测:错误路径给出建议文案;正常路径 `--plan` 仍正确(且已识别出先前测试的 962 个文件)。

**待用户提供**:`df -h /scratch365`、`ls -ld /scratch365/$USER`、`quota -s` 的输出,
以定 dest。**OQ2(落盘位置/配额)由"不阻塞"升级为"当前唯一阻塞项"。**

### 2026-08-17 — 【正式 4 折结果】G2 为干净的零结果;G1 中 probGRU 略胜 AR;probGRU 方差头严重过度离散

4 折 location split 运行完成(EPOCHS=20,`--save-preds`/`--save-model` 齐备),
`scripts/opentouch_report.py` 从保存的预测事后计分(未再占用 GPU)。产物已从 CRC 推送到
GitHub 并在本地查看(commit `d72a4f5`)。

**一、G1:probGRU ≥ AR,但优势很小。**
skill vs persistence(F / CoPx / CoPy):
`prob_gru` **0.198 / 0.305 / 0.265**;`ar` **0.194 / 0.279 / 0.237**;`seasonal` ≈ −0.007。
R²:prob_gru 0.458/0.488/0.528;ar 0.456/0.469/0.510;persistence 0.325/0.263/0.358。
方向与 08-15 单折一致(**GRU > AR,与 ActionSense 的结论相反**),但**差距仅 0.004–0.029**,
在跨折方差面前很可能不显著。

**二、G2:干净的零结果——smooth 与 abrupt 的可预测性没有差别。**
ΔR² = R²(smooth) − R²(abrupt),两样本 bootstrap(B=2000):
| 模型 | F | CoPx | CoPy |
|---|---|---|---|
| prob_gru | −0.018 [−0.082, 0.039] | −0.015 [−0.097, 0.051] | +0.013 [−0.052, 0.068] |
| ar | −0.002 [−0.063, 0.050] | −0.000 [−0.064, 0.051] | −0.002 [−0.157, 0.093] |
**四个模型的置信区间全部跨过 0**;分类别 R² 几乎逐位相同(prob_gru:smooth 0.443 vs
abrupt 0.460 on F)。
**leave-one-action-out 排除了单动作主导的解释**:去掉 `picking up` 或 `holding` 后 ΔR² 仅变动
0.01–0.03,仍近于 0。即 2026-08-16 提出的构念效度风险(两类各被一个动作主导)**未改变结论**。
剔除争议子集后 ΔR² 变得略负(−0.02 ~ −0.19),方向不变。

**三、【新发现】probGRU 的方差头严重过度离散(over-dispersed)。**
预测曲线图显示 ±2σ 带极宽:F 上预测约 740,000 而带宽约 ±32,000,真值仅在 735,000–748,000
之间波动,**真值几乎总落在带子中央**。已在 `plot_opentouch_loss.py` 中实现定量检查:
±2σ 覆盖率(名义 95.4%)与"中位 σ / 中位 |误差|"之比。
**这一点任何 MSE 表都测不到**:高斯头参与训练(占 NLL 的一半)却**从不被 harness 评分**
(冻结 harness 只测点误差),因此"方差估得不对"不付出任何代价。
另注:图中 `persistence` 不可见,因被 `seasonal` 完全覆盖——后者在本语料上退化为 persistence
(既有结论)。

**四、新增 `scripts/plot_opentouch_loss.py`(commit `b315641`)**:从 checkpoint 画每折的
train/val NLL 曲线(train 每 5 轮采样,故用标记+断点而非插值)、标出被保留权重的 epoch、以及
history 扫描的 VAL 曲线与选中点;并输出上述校准诊断。
**训练曲线的意义**:早停使长训无害,但**最低点的位置有信息**——本次 VAL 在**第 2 轮**触底后
单调恶化,而 ActionSense 自己的注释写的是"约第 10 轮后严重过拟合"。**在按地点留出的 split 下
过拟合来得快 4 倍**,说明后续轮次学到的东西**不跨环境迁移**。

**五、所有结论的共同限定**:D1 被放弃 → F 仍有 ~99.3% 是直流、阵列工作在满量程 95%,故上述
R²/skill 应读作"对'常数 + 其漂移'的复现程度",而非对触觉动态的预测能力。

### 2026-08-17 — model_diagram 第二轮修改(用户五项要求)

1. **橘色自回归回喂线重连**。原来是 `arc3` 自由弧,端点悬空、看不出连到哪。改为显式折线
   `feedback()`(出 head → 走两个 decoder cell 之间的**间隙**下行 → 横穿至 y=22.4 → 上箭头进
   下一个 cell 底部),并把 "autoregressive: μ̂ fed back…" 挪到 y=19.8,使其正好成为横穿段的标题。
   走间隙(而非 head 右缘正下方)是必要的:后者会压在 cell 上。
2. **`h = 48` → `dim(h) = 48`**(用户指出原写法有歧义:h 既是隐状态又是维数)。
3. **左栏去掉 slow/fast 分叉**。三个框(低通框 + slow 框 + fast 框)与两组分叉箭头全部删除,
   主链路改为 `s_t → x_t = [F, x, y, v_x, v_y] ∈ R^5`,即代码里已有的 **`input_mode="raw"`**
   (`FEATS_RAW`, [action_dynamics.py:27](src/actionsense/action_dynamics.py#L27))。
   **未完全照办的一点(已当面说明)**:`build_features` 无论 input_mode 为何,
   **target 恒为 fast 分量**([:28](src/actionsense/action_dynamics.py#L28)、
   [:72](src/actionsense/action_dynamics.py#L72)),故因果低通不能从图上彻底消失,否则 target
   与代码不符。折中:降级为特征框下方的**一行注记**("target is the FAST component: causal
   low-pass 0.4 Hz, F_f = F − LP(F)"),流程图上不再占据步骤位。
   **依据**:5.4 节消融结论 "raw ≈ highpass 输入"(right 0.513 vs 0.513),
   即输入端的显式分解本来就无增益,删掉不损失任何已证结论。
4. **删除底部灰色训练条**(Gaussian NLL → early-stop → σ-scaling → 5-fold CV)。删后底部留白
   过多,故引入 `Y0=3.0` 裁掉画布底部并按 `(H-Y0)` 重算 figsize 与 inset 变换(inset 与主坐标
   系必须用同一套换算,否则曲线会错位)。
5. **删除右上角 "inset curves are illustrative, not measured"**。**这是本轮唯一有代价的改动**:
   该注记是两条合成示意曲线的诚实性声明。补偿措施:(i) docstring 顶部保留完整说明;
   (ii) `main()` 每次保存后 **stdout 打印提醒**,写明"该声明已按 2026-08-17 要求从画布移除,
   必须由论文 caption 承担"。**若最终 caption 未写,该图就存在把合成曲线当实测结果呈现的风险**
   ——此风险已明确记录在此,由用户承担裁定。

产物:`docs/model_diagram.png`(220 dpi)+ `docs/model_diagram.pdf`(矢量)。仍**未提交**。

### 2026-08-17续 — 【校准诊断定论】误差重尾,而非过度离散;Claude 两次读错已更正

逐通道 ±2σ 覆盖率(名义 95.4%)与 σ/中位|误差|(高斯参照 1.48):
| 通道 | 覆盖率 | 中位 σ | 中位 |误差| | 比值 |
|---|---|---|---|---|
| F_R | 94.56% | 10,399 | 5,915 | **1.76** |
| CoPx_R | 94.42% | 0.00641 | 0.00352 | **1.82** |
| CoPy_R | 94.26% | 0.00495 | 0.00275 | **1.80** |

**结论:误差分布重尾。** σ 相对中位误差偏大(1.8 > 1.48)本似"对冲",但覆盖率又略低于名义值
(94.4% < 95.4%)本似"过于自信"——**两者同时成立只能是重尾**:主体误差远小于 σ,尾部却比高斯厚,
故 ±2σ 漏掉的比例略高。高斯 NLL 只能在尖峰与厚尾之间折中出一个 σ。
**三个通道的偏差模式几乎相同**(比值 1.76/1.82/1.80,覆盖率 94.6/94.4/94.3),跨越约 6 个数量级
仍一致 → **这是任务与损失函数的性质,不是某通道的尺度问题**。

**Claude 的两次误判(如实记录)**:(1) 由预测图目测得出"方差头严重过度离散"——**错**,那是把
75 万量级上的带宽当宽度看,跨数量级的目测不可靠;(2) 见到汇总覆盖率 94.4% 后称"校准良好"——
**只对一半**,汇总中位 σ 完全由 CoP 通道决定,对 F 一无所知,且可能掩盖通道间的相互抵消。
**逐通道才是可判定的形式**,已实现(commit `60ee010`)。

**处置建议**:如实写入 limitations,**不改模型**——换重尾似然(如 Student-t)将不再"与
ActionSense 逐字一致",且**完全不影响点误差结论**(冻结 harness 只评分点误差)。

---

## 2026-08-17 — 【分条汇总】OpenTouch 端到端分析的全部结论、决定、错误与未决事项

按用户要求,把上述分散在各条目中的分析整理为一份编号清单。每条注明**性质**(实测/裁定/推论/
更正)与**依据**。这是截至本日的完整状态,可作为冷启动的唯一入口。

### A. 数据层面的实测发现(改变了对项目上限的理解)

**A1【实测】传感器几乎全程工作在饱和区。** 单格硬件轨 = **3072**,占全部 (格子, shard) 最大值的
94.8%,**26 个 shard 的最大值完全相同**(distinct 1 of 26),最长连续贴顶 **9430 帧 = 314 秒**。
推论:F 的理论上限 = 256×3072 = 786,432,而实测 F 均值 **746,325 = 上限的 94.9%**。
**阵列静息时已跑在满量程 95%,只剩 5% 余量留给接触。** 各 shard 基线中位 2978–3072 → 每格可用
余量 0～94 计数;`fablab_ml_p2` 的中位格子**永久贴顶,不携带信息**。

**A2【实测】F 中 99.3% 是直流,CoP 近乎钉在几何中心。** 扣基线后 F 均值 746,325 → 4,918;
变异系数 0.0084 → 0.455(**54 倍**);CoP 活动范围 0.041 → 1.24(**30 倍**)。

**A3【实测+裁定】D1 的 k 曲线与 k=1 的依据。** 曲线在 k=1 处有陡峭膝点(削掉扣基线后残差的
**70%**),此后每档仅再削几个百分点;k≥2 时 CoPx 冲到接近整个 [-1,1] 值域且 CoPy 非单调回落
(存活格子过少 → 重心跳变)。σ̂ 在全部 shard 恰为下限 0.50 → 中位 taxel 的非接触读数**完全恒定**,
故 **k=1 的物理含义是"只保留严格高于静息值至少 1 个计数的读数"**,是量化数据上最小的有意义阈值。
**用户裁定放弃 D1**,代价已记录:模型仍在预测一个 99.3% 为常数的信号。**校正是 cache 的纯函数,
日后可随时补做,无需重新下载。**

**A4【实测】量化限制。** cache 的 `clip_*.npy` 为 float16,在 [2048,4096) 区间间隔为 **2.0**;
配合 94 计数的余量,**接触信号只有约 47 个可分辨等级**。**但 F/CoP 未被此影响**——
`extract_opentouch.py:214` 以 float32 读入、231 行在 float32 上算矩,float16 仅用于事后存图。

**A5【实测】标签错配 3.5%(102/2904),高度集中且标签本身干净。**
`sports_dicks_p2` 40.0%、`sports_dicks_p1` 21.0%,其余 8 shard 合计 1.1%(**相差 28 倍**);
`sports_dicks` 一个地点贡献 70.6% 的错配而仅占语料 7.9%。形态自洽性检验:**consistent 46 /
IMPLAUSIBLE 1**。→ **错的是索引基准(相对合并流),不是标签。**

**A6【实测】clip 太短,history 扫描的长设置大半在喂零填充。** 中位 clip 2.80 s(84 帧);
需 `T ≥ 30 + t_in − 1` 才有完整窗口 → 1 s 档约 75% 的 clip 满足,2 s 档不足 50%,
**3 s 档仅约 25%**。补零经 z-score 后等于训练均值,即**伪造的"平均信号"**。

### B. 方法层面的裁定(均为用户拍板,Claude 执行)

**B1 split 按地点(shard 基名)整组留出**,26 shard → **12 个地点**。无论 `_pN` 是参与者还是场次
都不泄漏,**使该未解决的问题变得无关紧要**而非被猜测。**仍不能保证**同一人跨地点——manifest 无
人物标识,任何基于它的 split 都排除不了,报告须称"按地点留出"。

**B2 `group_keys` 改为 TRAIN 相对计数**,并**保证兜底组自身可拟合**(把 TRAIN 中最小的达标类目
并入 `other` 直到其达到 `min_group_size`)。前者解决"全语料常见但 TRAIN 缺席"导致的
`KeyError('sports equipment')`;后者是测试当场揪出的残余漏洞。

**B3 4 折分组交叉验证**,每个地点**恰好当一次 TEST、一次 VAL**;报告 mean [min, max] 而非仅均值。
理由:12 个地点做单次 60/20/20 后 TEST 只剩 2–3 个地点,**结论取决于抽到了哪几个**。

**B4 G2 改为"合训分评"**:所有模型(含 AR)在全量 TRAIN 上拟合一次,只在 TEST 上按 trait 拆分计分。
推翻 2026-08-11 的"按 trait class 分别拟合 AR"——否则 AR 白得一次按类特化,比较不公平。

**B5 trait 盲裁 36 个长尾动作词,commit 先于 join 计数**(commit `7376efd` 的时间戳即证据)。
词表补全至 **66 个动作**,`unaudited = 0`。事后计数显示盲裁影响很小(smooth 233→307),
**这恰好证明该程序不付代价却把"按物理判据裁的"从声称变成可查证**。

### C. 结果(4 折 location split,EPOCHS=20,`raw` 输入)

**C1 G1:probGRU ≥ AR,但优势极小。** skill vs persistence(F/CoPx/CoPy):
prob_gru **0.198/0.305/0.265**,ar **0.194/0.279/0.237**,seasonal ≈ −0.007。
**方向与 ActionSense 相反**(那边是 AR > GRU),但差距仅 **0.004–0.029**,在跨折方差前很可能不显著。

**C2 G2:干净的零结果。** ΔR² 的 bootstrap CI(B=2000)**四个模型、三个通道全部跨过 0**;
分类别 R² 几乎逐位相同(prob_gru:smooth 0.443 vs abrupt 0.460 on F)。

**C3 零结果不是单动作造成的。** leave-one-action-out:去掉 `picking up` 或 `holding` 后 ΔR²
仅变动 0.01–0.03。→ 2026-08-16 提出的构念效度风险(两类各被一动作主导:picking up 占 abrupt
36.6%,holding 占 smooth 37.5%)**未改变结论**。

**C4 剔除争议子集是高度不对称的。** contentious 836 中 **96% 在 abrupt 侧**;剔除后 smooth 仅
−30(−9.8%),abrupt −806(−31.0%)。→ 这对并列表**不是对称的稳健性检验**,须在报告中写明。

**C5 过拟合极快。** val NLL **四折全部在第 1–2 轮触底**后单调恶化(fold2 升至 1.42),train NLL
稳定降至约 −0.5,第 20 轮 train/val 差距约 **1.5 个 NLL 单位**。ActionSense 自注为"约第 10 轮后
过拟合"→ **在按地点留出下快了约 5 倍**,说明后续轮次学到的东西**不跨环境迁移**。

**C6 模型基本没有在利用历史长度。** 折内三个 history 的 val NLL 差距仅 **0.005–0.017**,
而**折间差 0.19**(fold1 −0.147 vs fold2 +0.039);且 fold0/2/3 都选了 **3 s**——正是补零最多的
那一档(约 75% 的 clip 无完整窗口)。**补零最多者反而(微弱)胜出**,与"目标 99.3% 为直流"自洽。

**C7 校准:误差重尾,而非过度离散。** 逐通道 ±2σ 覆盖率 94.56/94.42/94.26%(名义 95.4%),
σ/中位|误差| **1.76/1.82/1.80**(高斯参照 1.48)。σ 偏大似"对冲"而覆盖率偏低似"过于自信",
**同时成立只能是重尾**:主体误差远小于 σ、尾部比高斯厚。**三通道跨约 6 个数量级仍给出同一模式**
→ 属任务与损失函数的性质,非通道尺度问题。**处置:写入 limitations,不改模型**(换 Student-t 将
不再"与 ActionSense 逐字一致",且不影响点误差结论——harness 只评分点误差)。

### D. Claude 自身的错误与更正(如实记录)

**D1err 目测称"方差头严重过度离散"——错。** 那是把 75 万量级上的带宽当宽度看;**跨数量级的目测
不可靠**。实测覆盖率 94.4%,方向恰恰相反。

**D2err 见汇总覆盖率 94.4% 即称"校准良好"——只对一半。** 汇总中位 σ 完全由 CoP 通道决定,
对 F 一无所知,且可能掩盖通道间相互抵消。**逐通道才是可判定的形式。**

**D3err 怀疑 `peak_idx` 基准错位——被自己的诊断证伪,而该诊断本身有抽样缺陷。** 首次只取
manifest 前 600 条(未覆盖被绘图选中的 clip)即得"100% 一致",不具代表性;全量分组后才发现真实
错配率 3.5% 且高度集中。**教训:诊断的抽样范围必须与被解释的现象同源。**

**D4err D1 报告首版两处度量缺陷。** σ̂ 因整数量化而 MAD=0(全部 26 shard 退化为 0.00,
使 soft-threshold 完全失效);`saturated` 判据实际测量"池化分布有多窄"(报 212–256/256)。
均已修复并双向验证。

**D5err `assign()` 中 `sorted()` 后又 `shuffle()`**,排序被打乱,"大单元优先"失效 → 合成数据上
89% 的 clip 被塞进 train。

**D6err 整流偏置诊断在真实数据上恒为 0(失效)**:`idle` 取"超出基线最少的 5% 格",而这些格子完全
恒定。**但这一事实本身回答了原始担心**:median 对恒定格子精确正确,无可整流之噪声。

**D7err 两次"测试通过"其实都没覆盖目标路径**(split 的搬移逻辑):第一次数据过于病态、第二次
独占类目低于 `min_group_size` 被并入 `other`。第三次改用单元级测试才真正验证(40 seed 下修复后
缺失组为 0,旧逻辑 14 个失败)。

**D8err "省约 40%"的估计夸大**,实际约 **18%**(一个 epoch ≈ 3 单位训练 + 1 单位 train 评估 +
0.5 单位 val 评估,去掉 4/5 的中间项)。

**D9err 首版预测脚本未保存模型与预测**,导致数小时 GPU 时间无法产出任何预测曲线。已补
`--save-preds` / `--save-model`。

### E. 未决事项与下一步

**E1 D1(基线校正)被放弃**,代价已记录;为 cache 的纯函数,随时可补。
**E2 重尾似然不改**,写入 limitations。
**E3 `raw+df` 重训进行中**(唯一变量为输入增加 dF/dt)。判读:若 val 最低点后移且 train/val 差距
收窄 → 过拟合确由"记住各地点 F 直流水平"引起;若曲线形状不变 → 该假设被证伪,下一步上权重衰减
(旋钮已就位,默认关闭)。
**E4 逐动作 R² 与 LOAO 已实现并纳入主表**,不放附录。
**E5 所有结论的共同限定**:D1 未做 + 传感器满量程 95% → 上述 R²/skill 应读作"对'常数 + 其漂移'
的复现程度",而非对触觉动态的预测能力。**这必须进 limitations,它限制的是数据本身能支撑的结论
强度,不是方法问题。**

### 2026-08-18 — 【第二次 run 的全部改动清单】raw+df 臂,与第一次 run 的可比性说明

第二次 run 的**唯一实验变量是输入增加 dF/dt**。以下按"是否影响训练"分类,逐条列出自第一次 run
以来的全部改动,以便日后判定两次结果是否可比。

#### 一、影响训练的改动(只有一项)

**R2-1 输入特征 5 维 → 6 维:追加 dF/dt。** commit `5f78a20`,通过 `FEATURES=raw+df` 开启。
- **实现**:`features(Y, fps, with_df=True)` 在末尾追加 `causal_velocity(F)`,**前 5 列逐位不变**
  (有测试断言 `np.array_equal(a, b[:, :5])`),故消融不与其他输入变化混淆。
- **为何是它而不是别的**:ActionSense 的 `FEATS_RAW = ("F","x","y","vx","vy")` **只对 CoP 求差分、
  不对 F 求**,且未记录理由;其 highpass 模式已把力拆成 F_slow/F_fast,故 raw 模式更像遗漏。
- **为何它同时是过拟合的对症干预**:最可能的过拟合路径是模型记住**各训练地点的 F 直流水平**
  (D1 放弃后 F 有 99.3% 是直流),而这在换地点时立即失效;dF/dt 是 F 的**唯一无直流视图**,把这个
  可记忆量从输入中移除。解码器已由 `y_last` 提供"水平",编码器改拿"变化率",分工更清楚。
- **布局随归一化器走**:`with_df` 存在 `FeatNorm` 上而非另设平行标志,使统计量与其拟合时的特征集
  永远绑定,不可能出现"用 6 维统计量跑 5 维输入"的静默错配;checkpoint 亦记录 `features` 与
  `n_features`。

#### 二、已加入但本次**不产生任何效果**的改动(默认关闭)

**R2-2 `weight_decay`(默认 0.0)与 `dropout`(默认 0.0)。** commit `60ee010`。
测试断言 **p=0 的 dropout 不扰动任何输出**(同一输入两次前向 bitwise 相同),否则"架构与
ActionSense 逐字一致"这一表述即为假。**本次运行两者均未启用**——刻意只变一个变量,使第一次 run
可充当对照;若 dF/dt 不足以抑制过拟合,再在第三次 run 中启用。

#### 三、不影响训练、只影响分析与产物的改动

**R2-3 `scripts/opentouch_report.py`(`9dcd3f1`)**:从保存的预测事后计分(整体 skill、分类别 R²、
ΔR² + bootstrap CI、逐动作 R²、leave-one-action-out),**不需要 GPU**;split 由 `splits.folds`
按 (k, seed) 确定性重建,故每折的 TRAIN 与其力阈值可精确复原。
**R2-4 `scripts/plot_opentouch_loss.py`(`b315641`)** + 空 origins 修复(`a7dc4d1`) +
**逐通道覆盖率**(`60ee010`) + 汇总行格式修复(`4cabe66`)。
**R2-5 `FEATURES` / `WEIGHT_DECAY` / `DROPOUT` 的 qsub 透传**(`b087a3a`, `60ee010`)。

#### 四、第一次 run 的溯源(与本次对比时必须知道)

**R2-6 第一次 run(job 1364313)在运行中被 `git pull` 打断过一次。** 经日志判定:`prob_gru` 的
延迟 import **发生在 pull 之后**,故全程使用了 `a74cd14`(train NLL 每 5 轮记录一次)。
**但模型行为未变**——`features` 默认仍为 `"raw"`,`weight_decay`/`dropout` 当时尚未存在。
故 **两次 run 在"输入特征"这一点上确实只差 dF/dt**,可比性成立。判据:日志中每 5 行有 4 行
显示 `train   --`。

**R2-7 与本工作无关的并行提交**:`78c722d`、`27dfaa4`(用户提交,新增
`scripts/crc/fetch_d256.py`,D256 数据集抓取)。**仅触及 SESSION_LOG 与该新脚本,不影响训练代码。**
注意这两条向 SESSION_LOG 追加了约 1260 行,属另一条工作线的记录。

#### 五、第二次 run 的调用参数与产物路径

```
qsub -v FOLDS=4,EPOCHS=20,FEATURES=raw+df,\
        SAVE_PREDS=runs/preds_df,SAVE_MODEL=runs/models_df,\
        OUT=docs/opentouch/df/opentouch_cv4_df.csv  scripts/crc/opentouch_probgru_gpu.job
```
其余保持不变:location split、4 折、seed 0、hidden 48、lr 3e-3、batch 64、history 扫描 {1,2,3} s、
按 VAL NLL 早停、`log_train_every=5`。产物与第一次 run **分目录存放**,互不覆盖。

#### 六、判读标准(事先写定,避免事后找解释)

- **主判据 —— loss 曲线左图**:若 val NLL 的最低点**明显后移**(第 1–2 轮 → 更晚)且第 20 轮的
  train/val 差距**明显收窄**(第一次约 1.5 个 NLL 单位)→ **过拟合确由"记住各地点 F 直流水平"引起**,
  dF/dt 堵上了这条路。若曲线形状几乎不变 → **该假设被证伪**,下一步启用 R2-2 的权重衰减。
- **次判据 —— skill**:与第一次的 `prob_gru` 0.198/0.305/0.265 对比。**注意 skill 提升与过拟合
  缓解是两件事**,可能只出现其一。
- **G2 不作为判据**:第一次已是置信区间跨 0 的零结果,本次预期不变;若 ΔR² 突然显著,应先怀疑
  实现而非结论。

### 2026-08-19 — 【第二次 run 结果分析】dF/dt 假设被证伪且有害;"是不是过拟合"尚不能定论

#### 一、结果对比(raw → raw+df,4 折 location split,EPOCHS=20)

**1. 过拟合假设被证伪。** val NLL 的最低点**仍在第 1–2 轮**(星号未移动),第 20 轮的 train/val
差距几乎不变(fold2:1.87 → 1.90 个 NLL 单位),曲线形状肉眼难辨差异。
→ **过拟合并非来自"模型记住各地点的 F 直流水平"**;堵上这条路后过拟合分毫未减。
(判读标准是**事先写定的**(2026-08-18 第六节),因此这是一次真正的证伪,而非事后解释。)

**2. 加 dF/dt 使性能变差。**
| 指标 | raw | raw+df | 变化 |
|---|---|---|---|
| skill F_R | 0.1980 | 0.1607 | **−0.0372** |
| skill CoPx_R | 0.3047 | 0.2715 | **−0.0332** |
| skill CoPy_R | 0.2653 | 0.2682 | +0.0030 |
| R² F_R | 0.4583 | 0.4331 | −0.0252 |
**推测原因:dF/dt 在本数据上几乎全是量化噪声。** 依据:D1 报告显示扣基线后**70% 的残差在 1 个
计数以内**(k=0→1 削掉 3441/4918);F 为整数量化,故逐帧差分的主体是 ±1 计数抖动 × 30 Hz。
在一个只有约 3% 动态余量、47 个可分辨等级的传感器上,**"变化率"恰恰是最受量化伤害的量**。

**3. 免费的对照校验通过。** `persistence`/`seasonal`/`ar` 的**每一个数字逐位相同**(差值精确为
0.0000)——baseline 不吃 GRU 的输入,本就该如此;这同时证明 **split 重建、掩码、计分链路完全
确定性**,两轮之间除 GRU 输入外无任何变化。若这些数字动了,应先怀疑流水线而非结论。

**4. 其余结论不变。** G2 仍为零结果(全部 CI 跨 0,prob_gru 的 ΔR² 仅在小数点后第三位移动);
校准仍为重尾(94.4/94.4/94.1%,σ/|err| 1.72/1.79/1.78,与 raw 的 1.76/1.82/1.80 几乎相同)
→ **重尾是任务与高斯似然的性质,与输入无关**;history 扫描仍是折内差 0.005–0.03、折间差 0.19。

#### 二、"这是不是过拟合?"——Claude 的判断:**证据尚不能区分三种机制**

train 降、val 升确为过拟合的教科书特征,但在**按地点留出**下至少有三种机制都会产生该曲线,
**且处方完全不同**:

- **机制 1|经典记忆化**:记住训练 clip 的噪声 → 处方是正则化。
- **机制 2|地点特异性**:所学在训练地点是**真信号**,换地点即失效 → 处方不是正则化,而是更多样的
  训练地点或对地点不变的表示。
- **机制 3|方差头过度自信**(NLL 特有):高斯 NLL 含 `(y−μ)²·exp(−lv)`,在训练点压小 σ 即可降低
  train NLL,而一旦在 val 上判断错,该项会爆炸。**此时均值可能毫无退化,坏掉的只是 σ。**

**机制 3 尤其可疑**(误差已实测为重尾),且若成立还有一个更严重的连带后果:
**我们按 val NLL 早停,而 harness 只评分点误差**——若 val NLL 的恶化主要来自 σ,则第 2 轮的权重
**对 MSE 而言可能远非最优**,即一直在用一个无人报告的准则挑权重。

#### 三、已实施的判别手段(commit `9a069d9`)

**M3 判别(零成本,已实施)**:`train()` 现在**每轮同时记录 val MSE**(仅均值,不含方差头),
并记录 `best_val_nll_epoch` 与 `best_val_mse_epoch`;loss 图在孪生坐标轴上以点线画出 val MSE,
`+` 标出其自身最优轮次。**判读:若 val MSE 基本持平而 val NLL 飙升 → 机制 3 成立**,此时应改用
val MSE 早停(或同时报两者);**若 val MSE 同步恶化 → 机制 3 被排除**,问题在均值本身。

**M1/M2 判别(待跑,一次短作业)**:用**同地点的 clip 级随机划分**重跑一次。
**若 val 仍从第 2 轮开始恶化 → 机制 1(经典记忆化)**,正则化对症;
**若 val 变得平坦或持续下降 → 机制 2(地点特异性)**,正则化无用,须从数据多样性或表示入手。
这是唯一能把两者分开的实验,且只需一次短跑。

#### 四、下一步的可选措施(按信息量/成本排序)

**N1(最优先,零 GPU)**:用现有 checkpoint 重画 loss 图,看 val MSE 与 val NLL 是否分离 →
立刻判定机制 3。**注意:已完成的两轮 checkpoint 中没有 val_mse 字段**(该记录是本次新增的),
故需在下一次运行中获得;若不想等,可用保存的预测按轮次近似(不可行——预测只保存了最终权重)。
**→ 结论:M3 的判别需并入下一次运行。**

**N2**:同地点 clip 级划分的短跑(M1/M2 判别),`EPOCHS=8` 即可(最低点在第 1–2 轮)。

**N3**:若判定为机制 1,启用 `WEIGHT_DECAY=1e-3`(旋钮已就位,默认关闭);
若判定为机制 3,改用 val MSE 早停并重报。

**N4(唯一能抬高天花板者)**:**D1 重算**。目标 99.3% 为直流、传感器在满量程 95%、余量约 47 个
可分辨等级——**在这样的信号上任何模型的上限都很低,这不是模型问题**。D1 是 cache 的纯函数,
不需要 GPU、不需要重新下载。**Claude 建议优先于 N2/N3**。

**N5(不做)**:换重尾似然(如 Student-t)。会破坏"与 ActionSense 逐字一致",且**不影响点误差
结论**(harness 只评分点误差)。如实写入 limitations 即可。

#### 五、绘图能力补充(commit `9a069d9`)

`plot_opentouch_forecast.py` 新增 `--per-channel`(F 与 CoP 相差数个数量级,共用横轴会浪费两者的
纵向分辨率)与 `--compare DIR`(在**同一 clip、同一 origin** 上叠加第二臂的均值——这是唯一能看出
"输入改动把预测挪到了哪里"的方式,任何指标表都做不到)。

### 2026-08-19 — 【OQ-G 全局推翻】所有臂输出方差;新增 flatten/cnn map 臂;扫描模型不再丢弃

#### 一、用户三项裁定(AskUserQuestion)与其依据

| 问题 | 裁定 | Claude 提供的依据 |
|---|---|---|
| map 臂的 history 扫描 | **沿用 OQ-H 的 {1,2,3} s** | ActionSense tactile_map 用 [1,3,10] s,但 **10 s 在本语料只剩 90/2958 个 clip**(2026-08-07 实测),且 3 s 档已有约 75% 的 clip 无未填充窗口 |
| map 输入的基线 | **用 D1 已验证的逐 taxel median** | ActionSense 用前 N 帧均值(其录制从非接触开始);**OpenTouch 的 clip 围绕压力峰切分**(中位 2.80 s),开头 1 秒占 36% 且常已在接触中——`extract_opentouch.py` 的 docstring 早有警告。median 在本语料上被验证:`duty>0.5` 在 26 个 shard 上全为 0 |
| aggregate 臂是否改回概率式 | **改回(全局推翻 OQ-G)** | ActionSense 的三个编码器共用同一个概率头,"编码器是唯一变量"是 models.py 注释里写明的设计目的 |

**用户随后进一步指示:OQ-G 的推翻是全局的——所有臂都输出方差、所有预测都要存 σ。**

#### 二、`select_history` 不再丢弃扫描模型(commit `574923c`)

**问题**:原写法 `*_, hist = train(...)` 把每个 history 训好的模型**直接丢弃**,只取回一个标量;
选出最佳长度后再从头重训。**每折训 4 次,丢掉 3 个已训好的模型**,而 ActionSense 的
`plot_forecast_overlay` 的**行轴正是 history 长度**,因此那张图此前无法复现(除非把训练付两遍)。
**修复**:`keep=True` 返回全部模型,各自的预测写入 `runs/preds/hist_<t_in>/`(只含信号、origins
与该模型均值——baseline 不依赖 GRU 的 history 长度,逐目录重算是浪费)。

#### 三、新增 `src/opentouch/tactile_map.py`(commit `33e2137`)

照搬 `src/actionsense/tactile_map/`(models+data+train 合为一个模块):三个编码器
(Flatten / CNN / Agg)、**共用同一个 GRU 与一次性概率头**、残差对 persistence 的目标、
高斯 NLL + logvar clamp[-6,4]、log1p 压缩、全局 TRAIN 标准化、harness origins + 因果左填充、
d=64 / hidden=64 / lr=3e-3 / batch=64。

**三处被迫或被裁定的差异(均写入 docstring)**:
1. **网格 1×16×16 而非 2×32×32**(单手强制)→ FLAT 由 2048 变 **256**,CNN 走 16→8→4。
2. **基线用 D1 的逐 taxel median**(理由见上表)。
3. **扫描 {1,2,3} s**(理由见上表)。

**必须随数字一同声明的限制**:D1 的 median **按 shard 汇总**,故对被完全留出的地点,其基线是
**用该地点自身的帧**估计的——**输入上是 transductive 的,目标从不参与**。`--baseline-scope train`
可限制到 TRAIN,代价是完全留出的地点将没有基线可用。默认 `shard`。

**测试 6 项**:三臂共用一个概率头且只差编码器;网格为传感器所定;median 落在静息水平而非静息与
接触之间;窗口为残差且左填充;预测与 harness origins 对齐;**残差头置零后预测逐位等于
persistence**——这正是残差锚点的意义,锚点若丢失,该臂会比它要对比的基线更差,而这种错误在指标表上
只表现为"效果不好"。

#### 四、OQ-G 全局推翻的落地(commit `fe7cce0`)

- `tactile_map.predict_with_sigma()`:返回 RAW 单位的 (mu, sigma)。残差约定下锚点是常数平移,
  故 `sigma_raw = exp(lv/2) · norm.std`,与 prob_gru 一致。
- 驱动脚本**为任何产出 σ 的臂保存 σ**。此前 map 臂虽是概率式,**只有 prob_gru 的 σ 落盘**,
  等于有三个受训的方差头从未被记录,因而**不可证伪**(harness 只评分点误差,方差估错不付代价)。
- 校准检查改为**按模型 × 按通道**同时统计。四个臂共用一行汇总会掩盖"谁失准、往哪个方向失准"。
- **`gru_aggregate.py` 标注为被取代,不就地改写**:它的 git 历史是确定性臂的**预注册记录**,
  改写等于抹掉"在任何数字出现之前提交过什么"。其概率版就是 tactile_map 的 `aggregate` 编码器。

#### 五、下一次运行

```
qsub -v FOLDS=4,EPOCHS=20,MODEL=map_all,SAVE_PREDS=runs/preds_map,\
        SAVE_MODEL=runs/models_map,OUT=docs/opentouch_cv4_map.csv \
     scripts/crc/opentouch_probgru_gpu.job
```
`MODEL=map_all` 依次跑 aggregate / flatten / cnn 三臂。三者只差编码器,故 flatten-vs-CNN 的
对比是受控的;与 prob_gru **不可直接对比**(后者是自回归 + action embedding + 绝对目标的另一个
模型,源自 `action_dynamics.py` 而非 `tactile_map/`)。

### 2026-08-19续 — 【用户提出的 5 项 bug 排查:4 项通过,1 项属实但对称】+ D1 重启 + 三机制判别就位

#### 一、用户对 loss 曲线的 bug 假设与逐条核查结果

用户提出曲线"更像 loss 计算/checkpoint 选择/流程不一致",列出 5 项检查。**逐条查代码(附行号)**:

| # | 检查项 | 结论 | 证据 |
|---|---|---|---|
| ① | train/val 是否同一个 NLL | **是,字面同一函数** | 训练 `prob_gru.py:286 loss = nll(...)`;验证 `:238 tot_nll += float(nll(...))`,同为 216 行的 `nll()` |
| ② | train loss 是否每 epoch 重算、未累积 | **是,且根本不是累积的 minibatch loss** | 打印值由 `_val_scores(m, Xtr, ...)` **重做一次完整前向**得到,累加器为函数内局部变量,跨 epoch 累积在结构上不可能 |
| ③ | 是否都在 `eval()` 下计算;有无 teacher forcing 差异 | **是;且两边都没有 teacher forcing** | `_val_scores` 首行 `m.eval()`(232);解码器在训练与推理都把自己的 `mu` 喂回(212);无 batch norm;dropout 默认 0 |
| ④ | padding 是否被 mask | **没有——属实,是真问题** | 178 行左侧补零后作为普通输入进入损失,`prob_gru.py` 中无任何 mask。**但对 train/val 对称,解释不了曲线分叉**;它影响的是 history 扫描的可信度(3 s 档约 75% 的 clip 无未填充窗口) |
| ⑤ | checkpoint 是否误用 max | **用的是 min,方向正确** | `best = np.inf`(277),`improved = va < best`(298) |

**对用户前提的更正**:用户称"validation NLL 明明继续下降,却没有保存后面的模型"。**实际相反**:
图中实线(val)由第 1 轮约 0.0 单调升至第 20 轮 fold2≈1.43 / fold3≈fold1≈0.93 / fold0≈0.39;
**下降的是虚线(train)**,由约 −0.15 降至约 −0.5。**故星号落在第 1–2 轮正是 val 的最小值,与 `min`
的实现一致。** 已给出直接读 checkpoint 数组的核对命令,不必依赖读图。

#### 二、由这次排查逼出的一个此前未记录的真问题:**选择与报告的双重不一致**

- **损失形式不一致**:按 **val NLL** 早停,而 harness 报告 **点误差**。
- **点集不一致**:训练/早停的 NLL **不做 mask**,而 harness **在力低于 TRAIN 第 5 百分位时屏蔽
  CoP 通道**。
→ **"最优 val NLL"与"最优 harness 得分"既不是同一种损失,也不在同一个点集上。** 这不会造成
train↓/val↑ 的形状,但意味着**挑出的权重未必是 harness 意义下最好的**。判别已就位:
`best_val_nll_epoch` 与 `best_val_mse_epoch` 已逐轮记录,**下一次运行即可确认二者是否同轮取到最优**。

#### 三、D1 重启(commit `f20086f`)

用户 2026-08-16 放弃 D1,2026-08-19 决定解决。**校正是 cache 的纯函数,无需 GPU、无需重新下载。**
- **`src/opentouch/baseline.py`**:估计器从报告脚本中抽出为单一实现,**因为现在有第二个调用方
  (写校正 cache 的脚本),写入方不得与被测量过的那一版发生漂移**。报告脚本改为 import 它。
- **`scripts/opentouch_apply_baseline.py`**:逐 shard 扣除每 taxel 的 median + k·σ、截断到 0、
  **用抽取器自己的 `moments()` 重算 F/CoP**(有测试断言与 `extract_opentouch.moments` **逐位相等**,
  而非比对一份拷贝),使校正 cache 与原始 cache **通道对通道可比**。
- **写入新目录,绝不就地覆盖,也不用软链接**:未校正的运行是"校正改变了什么"的对照;而
  `config_hash` 是**配置文件**的哈希,**用软链接换掉数据会让两份不同的数据共享同一个哈希**;
  `--write-config` 生成一份仅 `states_root` 不同的兄弟配置,其哈希自然不同——这正是读者需要的信号。
- **合成语料验证**:F 均值 782,012 → 206;**变异系数放大约 3777 倍**;CoPx 活动范围 0.0001 → 0.4367。
- **未修复的部分要写进 limitations**:阵列在满量程 95%、硬轨 3072,**接触在采集端即被削顶**;
  扣基线恢复的是**幸存下来的动态范围**,不是被传感器丢掉的那部分。

#### 四、三机制判别的落地

- **机制 3(方差头过度自信)**:`val_mse` 已逐轮记录并在 loss 图上以点线画出,`+` 标其最优轮次。
  **判读:val MSE 基本持平而 val NLL 飙升 → 机制 3 成立**(此时应改用 val MSE 早停或同时报两者);
  **两者同步恶化 → 机制 3 排除**。
- **机制 1 vs 2**:新增 `--split-mode random`(**同地点的 clip 级随机划分**)。
  **若 val 仍从第 2 轮开始恶化 → 机制 1(记忆化)**,正则化对症;
  **若曲线变平 → 机制 2(地点特异性)**,正则化无用,缺的是数据多样性或对地点不变的表示。
  该模式**在打印时自我标注为"按构造即泄漏"**,只用于读曲线,**永不用于报告分数**。

#### 五、下一轮的三条命令(顺序固定)

```
# 1) 生成 D1 校正 cache + 兄弟配置(几分钟,不占 GPU)
python scripts/opentouch_apply_baseline.py --cache ~/opentouch/cache \
    --out ~/opentouch/cache_d1 --k 1 --keep-maps \
    --write-config configs/opentouch/eval_harness_d1.yaml

# 2) 机制判别(短跑,同地点 clip 级划分,只读曲线)
qsub -v EPOCHS=8,SPLIT_MODE=random,MODEL=prob_gru,SAVE_MODEL=runs/models_diag,\
        OUT=docs/diag_random.csv scripts/crc/opentouch_probgru_gpu.job

# 3) D1 之后的正式一轮
qsub -v CONFIG=configs/opentouch/eval_harness_d1.yaml,FOLDS=4,EPOCHS=20,MODEL=map_all,\
        SAVE_PREDS=runs/preds_d1,SAVE_MODEL=runs/models_d1,OUT=docs/opentouch/d1/opentouch_cv4_d1.csv \
     scripts/crc/opentouch_probgru_gpu.job
```

---

## 2026-08-19 — 问答:`best_val_nll_epoch` 与 `best_val_mse_epoch` 的确切定义

**问:** OpenTouch 里 `best_val_nll_epoch` 与 `best_val_mse_epoch` 的公式分别是什么?

**答(代码事实,非回忆):**

两者都只在 `src/opentouch/prob_gru.py:311-317` 写入,都是对**逐轮 VAL 曲线取 `np.nanargmin` 再 +1**
(转成 1-based 轮次);差别全在被取 argmin 的那条曲线上。设 VAL 有 N 个窗口、horizon H、通道 C:

- `val_nll[e] = (1/(N·H·C)) · Σ_{i,t,c} 0.5·[ lv + (y-mu)²·exp(-lv) ]`,`lv = log σ²`,在
  `ProbGRU.forward` 里被 `clamp(-6, 4)`;**常数项 0.5·log(2π) 被丢掉**(`nll()`,`prob_gru.py:216-218`)。
  `_val_scores` 按 `nll_batch × len(mu)` 累加再除以窗口数(`prob_gru.py:238,240`);因为每个窗口恰好贡献
  H·C 个元素,这个按窗口数加权的批均值**精确等于**全体元素均值(末批不满也成立)。
- `val_mse[e] = (1/(N·H·C)) · Σ_{i,t,c} (mu-y)²`,即 `tot_se / cnt`,`cnt = yb.numel()`
  (`prob_gru.py:239-240`)。**只看均值头,与 lv 无关。**
- `best_val_nll_epoch = argmin_e val_nll[e] + 1`;`best_val_mse_epoch = argmin_e val_mse[e] + 1`。

**为什么两者会落在不同轮次(这正是它们被同时记录的原因):**
`val_nll = 0.5·mean(lv) + 0.5·mean((y-mu)²/σ²)` = 校准项 + **按 1/σ² 加权**的点误差。
若 σ 在所有元素与所有轮次上是同一个常数 σ₀,则 `val_nll = 0.5·log σ₀² + val_mse/(2σ₀²)`,是 MSE 的单调函数,
两个 argmin 必然重合。**二者分离 ⟺ 方差头在动**——这就是 `_val_scores` docstring
(`prob_gru.py:222-232`)所述机制 3 的判据。

**两条必须一起讲的限定:**
1. **只有 NLL 决定权重**:早停/保存最优态用的是 `improved = va < best`(`prob_gru.py:298`),
   `select_history` 也按 `best_val_nll` 选 t_in(`prob_gru.py:344`)。`best_val_mse` / `best_val_mse_epoch`
   **纯诊断**——即使 MSE 最优轮更早,落盘的仍是 NLL 最优轮的权重。
2. **两者都在 z-score 化的目标空间**(`Norm`),**不是物理单位**,因此与报告里 RAW 单位的 test MSE
   不可直接比较。

**范围:** 这两个字段只有 `prob_gru.py` 写。`tactile_map.py:267-293` 同样逐轮记录
`train_nll/val_nll/val_mse` 并按 NLL 选权重,但**不写 `*_epoch` 字段**;`gru_aggregate.py` 是纯 MSE 臂,
只有 `best_val_mse`,没有 NLL。

### 2026-08-19续2 — 【D1 校正在全语料落地】直流占比 99.78%;26 个 shard 全部干净;并更正一处过度外推

#### 一、结果(全 2904 条 clip 校正完毕,文件数 8713 = 2904 state + 2904 clip + 2904 pose + 1 manifest)

前 400 条对照:
| | F 均值 | F cv(中位) | CoPx range | CoPy range |
|---|---|---|---|---|
| raw | 752,548.8 | 0.0133 | 0.0230 | 0.0182 |
| **D1 k=1** | **1,638.8** | **0.3090** | **0.4911** | **0.6143** |
→ **直流占 F 的 99.78%**(1,639/752,549 = 0.22%),比此前估计的 99.3% 更极端;
变异系数放大 **23 倍**,CoP 活动范围放大 **21–34 倍**——CoP 由"钉在几何中心"变为真正在动。

**逐 shard 全量检查:26 个 shard 的"全零 clip"与"零帧占比"均为 0。** F cv 中位按 shard 从
**0.244(hardware_homedepot_p3)到 0.768(home_kitchen_p2)**。

#### 二、更正一处过度外推(Claude)

此前据 D1 报告中 `fablab_ml_p2` 的 `base med = 3072`(恰为硬件天花板)推断"该 shard 的中位格子
永久贴顶、不携带信息,可能整体不可用"。**数据否定了这一推断**:该 shard 的 F cv 为 **0.4488**
(高于全语料中位),零帧占比 0。
**原因**:`base med` 是**该 shard 所有 taxel 基线的中位数**,意为"半数格子贴在 3072",而非
"该 shard 整体无信号";**基线是逐格扣除的**,另一半格子基线更低,接触仍能把它们推过各自基线。
**只有最饱和的那部分格子贡献为零。**
**这是本会话第三次从汇总统计量过度外推**(前两次:由预测图目测断言方差头"严重过度离散";由汇总
覆盖率 94.4% 断言"校准良好")。**共同模式:汇总量能支撑什么结论,取决于它是对什么取的汇总。**

#### 三、一个此前未识别的结构:地点间的动态强度差异达 3 倍

各 shard 的 F cv 从 0.244 到 0.768。**这为此前的观察提供了解释**:折间 val NLL 相差 0.19,而
折内三个 history 仅差 0.005–0.03——**按地点留出时,不同折拿到的地点在信号动态强度上本身就差 3 倍**,
故跨折离散度有相当部分**不是模型不稳,而是真实的地点异质性**。这再次支持 2026-08-15 改用 4 折的
判断:单次 split 的数字很大程度取决于抽到哪几个地点。

#### 四、事先写定的预期(避免事后找解释)

F 现有 0.31 的相对变异(原 0.013),**persistence 不能再靠"信号几乎不变"白拿高分**。故预期:
1. **所有模型的绝对 R² 大幅下降**(任务真的变难,此前是虚高);
2. **skill vs persistence 更有意义,可能反而拉大**(persistence 掉得更多);
3. **若 skill 反而缩到接近 0**,则说明**扣掉直流后剩下的部分本就不可预测**——这本身是重要结论。
无论哪种,**这一轮的数字才第一次在回答"触觉动态可不可预测",而非"常数复现得准不准"**。

### 2026-08-19 — 【根因】CRC 上 `/scratch365` **已不存在**;仓库内 11 处引用全部过时

**证据**(用户实跑):
- `ls -ld /scratch365/jhao3` → `No such file or directory`。
- `df -h /scratch365` → 返回 **`/dev/sda3  889G  ...  Mounted on /`**。
  **关键**:`df` 对不存在的路径会回退到包含它的挂载点,这里回退到了 **`/`** ——
  说明 `/scratch365` **根本不是一个挂载点**,而不是"我没有该目录的权限"。
  上一轮我判成"目录未预置 / 需向 crcsupport 申请",**判错了**:整个文件系统层已被下线或改名。
- `quota` 的 usage 自证当前存储层级:`/users/<user>`、`/groups/<user>`、**`/bluefs/<user>`**、
  **`/goldfs/<user>`**、**`/temp180/<user>`**、以及 AFS。**列表里没有 scratch365。**
  `/temp180`(180 天保留)在语义上是 `scratch365`(365 天)的继任者。
- 另:`quota` 是 CRC 的自研 wrapper,**不吃 `-s`**(我给的命令带了 `-s`,是我的错)。

**影响面**:`grep -rn scratch365` 命中 **11 处**,分布在
`scripts/crc/README.md`(6 处,含 EgoTouch 的 rsync 与 symlink 指引)、
`scripts/crc/stream_actionsense.sh:50`(注释)、`fetch_d256.py`(4 处:docstring/help/报错文案)。
**这些指引若被后续 session 照抄,会重复今天这个失败。**待定下新层级后一并改,不做半吊子替换。

**待用户提供**:各层级的配额与可写性(见下一轮给出的单条命令)。
**OQ2 仍是唯一阻塞项**,但问题已从"要不要申请 scratch"变成"选 `/temp180` 还是 `/bluefs`/`/goldfs`"。

### 2026-08-19续3 — 【机制判别结果:三个都有答案】+ 本轮(D1)运行的全部改动清单

#### 一、机制判别跑的结果:三种机制同时有了答案

诊断作业(`SPLIT_MODE=random`,同地点 clip 级划分,EPOCHS=8)**在扫描完成后崩溃**(bug 见下),
但**曲线已全部打印在日志中,无需重跑即可定案**。`t_in=90` 的完整曲线:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| val NLL | **−0.0789** | −0.0287 | −0.0418 | −0.0255 | +0.0045 | +0.0353 | +0.0921 | +0.1118 |
| val MSE | **0.4664** | 0.4683 | 0.4686 | 0.4792 | 0.4882 | 0.5166 | 0.5291 | 0.5340 |

**① 机制 1(经典记忆化)——成立。** train 与 val 来自**同一批地点**,"换地点"因素已被移除,
**val 仍从第 1 轮之后单调恶化**。→ 正则化有明确依据,不再是"试试看"。

**② 机制 2(地点特异性)——同时存在,且是放大器。** 同地点条件下 val NLL 由 −0.079 升至 +0.112
(幅度 **0.19**);而按地点留出时,同期已升至约 +0.4、最终至 1.4。**同一个记忆化过程在换地点条件下
被放大数倍。** 二者叠加,不是二选一。

**③ 机制 3(方差头过度自信)——被排除为主因。** val MSE 同样单调上升(0.4664 → 0.5340,**+14.5%**),
说明**均值本身在变差**,不只是 σ 过度自信。且 `argmin NLL` 与 `argmin MSE` **同在第 1 轮**,
即"按 NLL 早停、按 MSE 报告"的不一致在这一跑上**没有咬到**。

#### 二、Claude 的 bug:`select_history` 的返回值个数随参数变化(已修,commit `5a3111e`)

驱动写的是 `t_in, scores, kept = P.select_history(..., keep=bool(args.save_preds))`,而该诊断跑
传了 `SAVE_MODEL` 却未传 `SAVE_PREDS` → `keep=False` → 函数返回 2 个值 → **无条件按 3 个解包而
崩溃**。**且崩溃发生在整个 history 扫描完成之后**,因此代价是整个作业。
**修复:arity 恒定为 3**,`kept` 在未请求时为 `{}`;测试断言此点。
**教训:返回值长度随参数变化的 API,就是为这种失败准备的陷阱。**

#### 三、本轮(D1)运行的全部改动清单

**唯一的实验变量是 D1(基线校正)。** 逐条:

**R3-1 数据:改用 D1 校正后的 cache。** `CONFIG=configs/opentouch/eval_harness_d1.yaml`
(仅 `states_root` 不同,**哈希因此不同——这是刻意的信号,不是同一协议的重跑**)。
F 的直流占比 99.78% 被移除,变异系数放大 23 倍,CoP 活动范围放大 21–34 倍。

**R3-2 epochs 20 → 8。** 依据:**历次运行的 val 最低点全部在第 1–2 轮**(raw 四折、raw+df 四折、
同地点诊断)。8 轮覆盖该区域并留 6 轮余量,用以观察 **D1 之后最低点是否后移**。
**事先写定:若 val 到第 8 轮仍在下降,必须加长重跑**——否则"一直在降"与"被我们截断"在图上无法区分。

**R3-3 模型:只跑 `prob_gru`,输入保持 `raw`(默认)。** 用户指示"只要和之前一样的 input",
故 **不跑 flatten / cnn 两个 map 臂**(代码已就位,留待后续)。

**R3-4 不加权重衰减——Claude 收回上一条消息中"D1 与权重衰减可合并"的建议。**
理由:D1 是迄今对数据本身改动最大的一次;若同时加正则化,**之后无论过拟合是否缓解都无法归因**。
而权重衰减的依据来自机制判别跑,**不依赖本轮**,可随时单独加。

**R3-5 绘图:修复行级图例(commit `9bf84d8`)。** `fig.legend` 取 `axes[0][0]` 的句柄,而模型按
字母序排列、`ar` 居首,导致**四模型的图上只有一个写着 "ar 1 s forecast" 的图例**,读者会合理地
认为整张图只有 AR、并追问 GRU 去哪了(用户正是如此)。改为**每行在图内标注自身模型名(同色加粗)
+ 首列各自的图例**,删除图级图例而非让它保持歧义。

**提交命令**:
```
qsub -v CONFIG=configs/opentouch/eval_harness_d1.yaml,FOLDS=4,EPOCHS=8,MODEL=prob_gru,\
        SAVE_PREDS=runs/preds_d1,SAVE_MODEL=runs/models_d1,OUT=docs/opentouch/d1/opentouch_cv4_d1.csv \
     scripts/crc/opentouch_probgru_gpu.job
```

#### 四、判读标准(事先写定)

- **主判据**:val 最低点是否从第 1 轮**后移**、train/val 差距是否**收窄**。
  若明显改善 → 此前的记忆化有相当部分是在背**各地点的直流水平**;
  若曲线形状不变 → 记忆化与直流无关,下一轮再上权重衰减。
- **绝对数字不可与此前任何一轮直接比较**(config_hash 不同、目标信号已变)。
  **可比的是 skill vs persistence**,因为分子分母同在一份数据上。

#### 【OQ2 结案】落盘位置 = `$HOME/forcevision`

**配额实测(2026-08-19)**:

| 路径 | Quota | Used | Avail | 状态 |
|---|---:|---:|---:|---|
| `/users/jhao3`(home) | 100G | 68G (68%) | **33G** | 唯一可用 |
| `/temp180/jhao3` | — | — | — | **不存在** |
| `/bluefs/jhao3` | — | — | — | **不存在** |
| `/goldfs/jhao3` | — | — | — | **不存在** |

**结论**:三个大容量层级**都没有为 `jhao3` 预置目录**——它们是**须向 crcsupport@nd.edu 申请的
allocation**,不是能自己 `mkdir` 的路径。故当前只有 home 可写。
signals-only 需 **12.34 GiB < 33 GiB 可用**,**放 home 可行**,落盘后 home 约到 80/100 GB。

⚠️ **两点风险,记录在案**:
1. home 只剩 ~20 GB 余量,且 68 GB 已用中含既有 OpenTouch cache。**后续任何大动作前先 `quota`。**
2. **全量 250.3 GiB 在 home 上绝无可能。** 若日后要视频侧,**必须先申请 `/temp180` 或 `/bluefs`
   allocation**,这是有前置期的行政步骤,别等到要跑实验才发现。

**同时修掉 11 处过时路径**(README 6 处 + `stream_actionsense.sh` 1 处 + `fetch_d256.py` 4 处):
不是简单替换字符串,而是改成"**先申请 allocation,没有就用 home 并盯配额**"的正确指引,
并把 `/scratch365` 已下线这一事实与核查命令(含 `quota` **不吃 `-s`** 这个坑)写进 README 与
脚本报错文案,避免后续 session 再踩。报错路径与 `--plan` 正常路径均已实测。

---

## 2026-08-20 — 【D1 校正后 4 折结果】skill 翻倍,但预测曲线显示模型在"预测局部均值"而非动态

产出:`docs/opentouch/d1/opentouch_cv4_d1.csv` + 三张 `docs/opentouch_forecast_d1_{F,CoPx,CoPy}.png`。
**缺失**:`opentouch_loss_d1.png` 与 `opentouch_report_d1.csv` 未生成(那两步在 CRC 上失败,
报错尚未取得),故本节的分析基于驱动脚本自带的指标表与预测曲线。

### 一、结论

**C1 skill 相对 raw 几乎翻倍,事先写下的预期 ② 得到证实。**
| 模型 | raw (F/CoPx/CoPy) | **D1** |
|---|---|---|
| ar | 0.149 / 0.214 / 0.171 | **0.367 / 0.431 / 0.476** |
| prob_gru | 0.203 / 0.288 / 0.221 | **0.386 / 0.427 / 0.472** |
| seasonal | −0.019 / −0.015 / −0.016 | −0.038 / −0.033 / −0.009 |
绝对 MSE(F)从 ~1.7–2.2e8 降到 ~3.7–6.2e5,即目标本身缩小了约 350 倍——**绝对数字与 raw 不可比,
只有 skill 可比**(分子分母同在一份数据上)。

**C2 probGRU 与 AR 打平,GRU 此前的领先主要是"更擅长复现常数"。**
raw 上 GRU 领先 AR 5–7 个点;D1 后 F 上 0.386 vs 0.367、CoP 上相差不到 0.005。
**这直接修正了 G1 的结论**:此前记录的"GRU > AR,与 ActionSense 相反"在校正后的数据上变成
**"两者相当"**。另:prob_gru 的跨折离散度远小于 AR(F 上 0.0085 vs 0.0301),更稳定。

**C3【最重要】预测曲线显示:模型不是在预测动态,而是在预测局部均值。**
D1 之后真实 F 变成**快速、尖锐、约 10 Hz 量级的振荡**(在 500–2700 之间往复),而
**四个模型给出的都是近乎平直的线**:persistence 是阶梯(按定义),seasonal 与之重合,
AR 与 prob_gru 是缓慢起伏、大致跟随局部均值的曲线。**没有任何一个模型捕捉到振荡本身。**
→ **skill ≈ 0.4 的含义是"比单点持平更接近局部均值",不是"预测出了动态"。**
这一点**指标表永远看不出来**,只有把整段滚动预测画出来才可见。

**C4 skill 随 horizon 不衰减,反而上升——与 C3 一致。**
| 模型 | h=1 | h=5 | h=10 | h=20 | h=30 |
|---|---|---|---|---|---|
| ar | 0.384 | 0.440 | 0.412 | 0.432 | 0.425 |
| prob_gru | 0.374 | 0.442 | 0.415 | 0.435 | 0.433 |
**若模型真的捕捉了动态,skill 应随 horizon 衰减**(越远越难)。持平甚至上升是
"模型预测均值、而 persistence 随 horizon 迅速变差"的特征:h=1 时单点持平尚可(信号在 1 帧上仍
自相关),h=30 时它已很差,于是均值预测器的相对优势反而扩大。

**C5 seasonal 在短 horizon 上显著为负(h=1: −0.174),随 horizon 收敛到 0。**
与"seasonal 退化为 persistence、但在短滞后上带一个坏偏移"一致。

### 二、问题

**P1 loss 曲线与报表未生成**,故本轮**无法判定 D1 是否缓解了过拟合**——那是本轮事先写定的主判据。
需在 CRC 上重跑那两步并取得报错。

**P2 无法区分"快速振荡是真实触觉动态"还是"传感器噪声"。** 若是后者,它按定义不可预测,
"预测均值"就已接近最优,这将解释 C2(AR 与 GRU 打平——两者都只能做到均值)。
**判别方法**:计算校正后 F 的自相关函数;若在 1–2 帧处即跌至近零,则快分量以噪声为主。
考虑到传感器工作在满量程 95%、仅约 47 个可分辨等级,噪声主导是**高度可能**的。

**P3 G2 的零结果需要在 D1 数据上重新验证。** 此前"smooth 与 abrupt 一样可预测"是在
**99.78% 为直流**的目标上得出的,很可能只说明"两类的直流一样好复现"。本轮 `--save-preds`
已保存,**只需重跑 `opentouch_report.py`(不占 GPU)即可得到 D1 下的 ΔR²**。

**P4 map 三臂(flatten/cnn/aggregate)仍未在任何 OpenTouch 数据上跑过。** ActionSense 上
flatten 比 persistence 更差(−0.042)、CNN 明显为正(0.138)、aggregate 最好(0.181);
**该比较在 OpenTouch 上尚属空白**,而 map 的直流占比同样是 99.78%,D1 之后才值得跑。

### 三、下一步的选择(按信息量/成本排序)

**N1(零成本,最优先)**:在 CRC 重跑 `plot_opentouch_loss.py` 与 `opentouch_report.py`
(后者**必须带 `--config configs/opentouch/eval_harness_d1.yaml`**),取得 **P1 的主判据**与
**P3 的 D1 版 ΔR²**。若报错,贴出以便修复。

**N2(零成本,可判定 P2)**:计算校正后 F/CoP 的自相关函数(逐 clip,滞后 0–30 帧,取中位)。
**若 1–2 帧即跌至近零 → 快分量是噪声**,则"预测均值"已接近该数据的上限,`skill≈0.4` 就是
**这份数据所能给出的全部**,而不是模型不够好。这条结论对论文的分量大于再跑一轮模型。

**N3(一次 GPU 作业)**:在 D1 数据上跑 map 三臂(`MODEL=map_all`),填补 P4,并回答
"空间结构有没有用"——这是与 ActionSense 唯一还没做的对照。

**N4(视 N1 结果)**:若 loss 曲线显示过拟合仍严重,启用 `WEIGHT_DECAY=1e-3`(旋钮已就位)。
依据来自 2026-08-19 的机制判别(记忆化已被独立证实),不依赖本轮。

### 2026-08-20续 — 【N1+N2 结果】天花板测量给出分层结论;G2 零结果经受住 D1;选择准则不一致被坐实

#### 一、可预测性天花板(N2)——本轮信息量最大的结果

**校正暴露了真实的自相关结构。** raw 上 r(1)=0.95/0.93/0.88 看似高度自相关——**那是直流在自相关**;
D1 之后 r(1) 降到 **0.318 / 0.403 / 0.228**,即**噪声占比 68% / 60% / 77%**,且 2–5 帧内即跌破 0.2。

**观测 skill 与保守天花板的对照(D1):**
| 通道 | h | 天花板 | 实测(ar) | 占比 | 剩余 |
|---|---|---|---|---|---|
| F_R | 1 | 0.439 | 0.342 | **78%** | 0.097 |
| F_R | 30 | 0.668 | 0.373 | **56%** | 0.295 |
| CoPx_R | 1 | 0.438 | 0.391 | **89%** | 0.047 |
| CoPx_R | 30 | 0.718 | 0.454 | **63%** | 0.264 |
| CoPy_R | 1 | 0.485 | 0.420 | **87%** | 0.065 |
| CoPy_R | 30 | 0.624 | 0.478 | **77%** | 0.146 |

**结论是分层的,不是"数据全是噪声":**
- **短 horizon(h=1)已接近极限**:实测达保守天花板的 **78–89%**。一帧之后可预测的部分,模型基本
  都拿到了;剩下的 0.05–0.10 主要是帧级噪声,**任何模型都拿不到**。
- **长 horizon(h=30)仍有实质空间**:天花板随 h 升到 0.62–0.72(因为 persistence 随 h 迅速变差,
  而噪声不变),**而实测 skill 基本持平**,故占比降到 56–77%,缺口 0.15–0.30。
- **机制**:模型实际在预测"过去的局部均值"(见 2026-08-20 的预测曲线),而天花板对应的是
  "预测平滑分量在 t+h 的取值"。**平滑包络的未来演化是可预测的,但没有任何一个模型在预测它。**
→ **可行的改进方向不是更大的模型,而是显式建模慢包络**(这也正是 ActionSense 的 slow/fast 分解在做
的事,即 D1b——此前因"目标用 RAW"而未采纳)。

#### 二、G2 的零结果经受住了 D1 校正(重要)

此前的担心:"smooth 与 abrupt 一样可预测"可能只是"两类的直流一样好复现"。**校正后重测,结论不变**:
ΔR² 的 bootstrap CI **四个模型、三个通道全部跨过 0**(prob_gru:F +0.004 [−0.075, 0.073],
CoPx −0.047 [−0.126, 0.020],CoPy +0.030 [−0.019, 0.076])。剔除争议子集后方向亦不变。
**→ G2 的零结果是关于触觉动态本身的,不是直流的假象。** 这条显著加强了原结论。
(计分 clip 数 2843/2904,61 条因过短或无预测被排除。)

**逐动作 R² 补充**:`holding`(smooth,0.276)与 `sliding`(smooth,0.293)确实略低于多数 abrupt
动作(0.32–0.37),但**差异远小于 CI**,与 ΔR² 的零结果一致。

#### 三、选择准则不一致被坐实(此前只是推测)

loss 图上 **val MSE(点线)与 val NLL(实线)明显分离**:fold0 的 val MSE **到第 8 轮仍在下降**
(约 0.53 → 0.50),而其 val NLL 从第 1 轮起就在上升;`*`(min val NLL)与 `+`(min val MSE)
**落在不同轮次**。
→ **按 val NLL 早停,确实在某些折上选走了对 harness 指标并非最优的权重。** 这不再是假设。
**处置建议**:改用 val MSE 早停(或同时保存两套权重并分别报告)。**代价很低**,因为 val MSE 已逐轮
记录;**收益**是选择准则与报告准则终于一致。

#### 四、两个"skill"不可混用(记录以免日后引用错)

驱动脚本的 `SS_vs_persistence`(逐帧池化)与报表脚本的 `skill`(逐 clip 等权,经 R² 聚合)**不是同一
估计量**:同为 ar 的 F 通道,前者 0.367、后者 0.302。**两者都对,但不可互换引用**;报表的版本是
per-clip 等权的,与 G2 的 R²/ΔR² 同源,论文正文应统一用报表口径。

#### 五、校准(D1 后)

prob_gru 的 ±2σ 覆盖率:F_R **96.9%**、CoPx_R 94.7%、CoPy_R 94.2%(名义 95.4%)。较 raw 时的
94.4/94.4/94.1% 略有上移,F 通道现在轻微**过覆盖**。重尾特征仍在但已减弱。

#### 六、下一步(更新)

**N5(零成本,最高优先)**:改用 **val MSE 早停**重跑(或从已保存的逐轮记录中重新选权重——但权重
未逐轮保存,故需重跑)。这是唯一一个**确定能提升报告指标**的改动,且它修的是方法学不一致。
**N6**:显式建模慢包络(D1b:slow/fast 分解),针对长 horizon 的 0.15–0.30 缺口。**需重开 Q1**
(目标是否仍为 RAW)。
**N3(不变)**:D1 数据上跑 map 三臂,填补与 ActionSense 的最后一个对照。
**N4(降级)**:权重衰减。天花板分析显示短 horizon 已近极限,正则化能挪动的空间有限。

### 2026-08-19续 — 下载已完成(80,821 文件),但"已完成"的证据强度不够,遂加 `--verify`

**现状**:用户 `git pull` 后跑 `--plan`,输出 `selected 80821 members (80821 already on disk)`,
`to fetch: 0`。即 **signals 全量已落到 `/users/jhao3/forcevision`**。
(`screen: command not found` —— CRC 前端没装 screen;但本次已跑完,不阻塞。**后续长任务用 `tmux`
或 `nohup ... &`**。)

**我发现自己的两个问题,均已修:**

**(1) 完成文案在撒谎(措辞级 bug)。** 旧代码**无条件**打印
"every member CRC-32 verified against the archive's central directory",
**哪怕本次 fetch 了 0 个文件**。用户看到的正是这一行——它读起来像"刚刚校验过 80,821 个文件",
**实际上本次一个字节都没读**。现改为分支:取到文件才说"each fetched member was CRC-32 checked";
0 文件时明说 `presence != integrity -- run with --verify to re-hash them from disk.`

**(2) 断点续传只证明"文件存在",不证明"内容正确"。**
`done` 的过滤条件是 `os.path.exists`,**被截断的、写了一半的文件同样满足**。
原本每个成员在**写入时**确实过了 CRC-32,但那是**上一次运行**的事;若此后磁盘写满、任务被
kill、文件被误改,续传逻辑一律视为"已完成"。**故"80821 already on disk" ≠ "80821 完好"。**

**新增 `--verify`**:从磁盘**重读**每个选中成员,核对 size 与 CRC-32(**只用缓存的 manifest,
不走网络**)。发现问题时:把坏文件从 `done.txt` 剔除 + 删除损坏文件 → **直接重跑即精准补齐**,
无需全量重下。

**实测完整闭环**(本地 101 文件切片):干净 → `all 101 files verified intact`;
人为**翻转 1 字节**(`crc 250e0f11 != 0f11bd5e`)+ **截断另一个**(`size 500 != 163820`)
→ 精确报出这 2 个;重跑只补这 2 个(95.47 KiB)→ 再验 `all 101 files verified intact`。

**待用户执行**(约 12 GiB 顺序读,几分钟):
`python3 scripts/crc/fetch_d256.py --dest ~/forcevision --verify`
**在此之前,这份数据不应被当作可信输入进入任何 D 系列决策链。**

### 2026-08-21 — SELECT_ON 作业一分钟即退:Claude 提交了自己无法执行的测试

**症状**:`SELECT_ON=mse` 的作业约 1 分钟结束(正常约 30–60 分钟)。

**原因(Claude 的错,三处)**:作业脚本 `set -euo pipefail` 且**在训练前先跑 pytest**
(见 `scripts/crc/opentouch_probgru_gpu.job:51-56` 的设计意图:"a broken fork fails in seconds
instead of after hours")。Claude 于 `d618ec3` 新增的两个测试**从未被执行过**(本机 torch 因缺
`libtorch_global_deps.dylib` 无法 import),其中有三处错误:

1. **`_cfg(tmp_path)` 根本不存在**——该文件中 `cfg` 是 **pytest fixture**(第 29 行),既有测试
   一律以参数接收。→ `NameError`,两个测试直接失败。
2. **`kept[t_in]` 是六元组 `(m, norm, fnorm, vocab, by_idx, history)`,不是 history 字典**——
   测试却按字典下标取 `["best_val_mse"]`。→ `TypeError`。已改为 `kept[t][5][...]`。
3. **`select_history` 在 `hp=None` 时会崩**:Claude 新加的 `hp.get("select_on", ...)` 直接作用于
   可选参数,而该函数签名允许 `None`。已改为先 `{**DEFAULT_HP, **(hp or {})}` 再读。

**闸门本身是对的,不该改**:它按设计在 1 分钟内拦下了一个坏 fork,而不是让它浪费数小时 GPU。
**该改的是 Claude 的做法。**

**新增工作约定(D-TEST)**:**Claude 不得在无法执行的前提下,向 gating 测试套件中新增测试。**
本机 torch 不可用,故 **凡改动 `tests/test_opentouch_prob_gru.py` 或 `tests/test_opentouch_gru_aggregate.py`,
提交作业前必须先在 CRC 上单独跑一次该测试文件**,确认通过后再 `qsub`。

**修复提交**:见下一条 commit。修复后本地仍为 skip,**故仍未验证**,须在 CRC 上先跑。

### 2026-08-21 — 【结案】d256 signals 全量落盘并通过完整性校验

**`--verify` 实测结果**(用户在 CRC 跑,355 s):
```
intact:  80821    missing: 0    corrupt: 0
all 80821 files verified intact
```
**这是逐文件重读磁盘、按归档中央目录核对 size + CRC-32 的结果**,不是"文件存在"的推断。
至此 `/users/jhao3/forcevision` 的数据可作为可信输入使用。

**最终交付物清单:**
- 路径 `/users/jhao3/forcevision/Dataset256/`,**12.34 GiB**,80,821 个成员
  = **80,819 个 `.p` clip** + `signals/ego_4d_verb.npy`(148) + `signals/ego_4d_noun.npy`(112)。
- 分组:`signals` 25,473 / `signals1` 28,426 / `signals2` 26,922;划分 train+val(**无 test**);
  受试者 S01–S05。
- 每 clip:`tactile-glove-{left,right}` (16,32,32) f32、`myo-emg-{l,r}` (16,8)、
  `myo-acc-{l,r}` (16,3)、`joint-position` (16,28,3)、`{left,right}-hand-pose` (16,24,3)、
  `label_text` + `label_idx`。
- **传输 3.49 GiB 而非 185.2 GiB**(比值 53×),靠远程 ZIP 中央目录 + Range 选取实现;
  视频侧 237.97 GiB 未取,日后可用同一脚本增量补(须先申请 `/temp180`/`/bluefs` allocation)。

**遗留的两项(均不阻塞当前使用,但需在下游动作前处理):**
1. **采样率未知。** clip 固定 16 帧且数值已预缩放到 ~[0,1],但**fps 没有出现在 pickle 里**。
   `causal_velocity(sig, fps)`(`src/opentouch/prob_gru.py:80`)要真实 fps 才有物理意义,
   **不可沿用 OpenTouch 的 fps 假设**。需从 ActionSense 原始文档或作者处确认。
2. **license 未知**,ICLR 匿名投稿页无任何条款。自用不阻塞,**对外发表/分发前须与
   `yichenl@mit.edu` 确认署名与授权**。

**用户原始请求("浏览 webpage 开始下载这个 dataset 到 CRC")到此完成。**

### 2026-08-21续2 — 【方法学分析,未改代码】baseline 的语义、与天花板的对照口径、以及可替代的衡量方式

用户提问(原文要点):OpenTouch 用 `1 − MSE_model/MSE_ymean`,ActionSense 用 `1 − MSE_model/MSE_persistence`;
这些数字该怎么解读?是不是必须跟 predictability ceiling 比才有说服力?我们这样设定 baseline 的物理动机是
什么?有没有别的衡量方式?

以下是分析结论。**本条只做分析,不改任何代码、不动任何数字。**

---

#### 一、两个 baseline 不是两个"选择",而是同一个 MSE 除以两个不同分母;它们由一条恒等式绑死

`aggregate.r2`(默认 `class_mean`)与 `aggregate.skill`/`metrics.skill` 共享同一个分子
`MSE_model`,只换分母。因此对同一模型、同一通道、同一点集恒有

    1 − R²_model = (1 − R²_persistence) · (1 − skill_vs_persistence)

用 `docs/opentouch/d1/opentouch_report_d1.csv` 的 prob_gru 逐位验证(误差在第 5 位以内):

| 通道 | R²(persistence) | skill(prob_gru) | 1−(1−R²_p)(1−skill) | 实际 R²(prob_gru) |
|---|---|---|---|---|
| F_R | 0.5032 | 0.2905 | **0.6475** | 0.6475 |
| CoPx_R | −0.3458 | 0.4517 | **0.2621** | 0.2621 |
| CoPy_R | −0.7105 | 0.4780 | **0.1071** | 0.1071 |

**推论**:三个自由度里只有两个是独立的。选 baseline 不是在选"更严格/更宽松",而是在选
**把哪一部分本领当作免费赠送**。均值基线赠送"零",persistence 基线赠送"当前水平"。

---

#### 二、D1 数字的实际解读:两个指标在 F 与 CoP 上方向相反,原因是 persistence 本身的好坏

`docs/opentouch/d1/opentouch_report_d1.csv`(overall / all,per-clip 等权口径):

| 通道 | R²(pers) | R²(ar) | R²(prob_gru) | skill(ar) | skill(prob_gru) |
|---|---|---|---|---|---|
| F_R | **0.503** | 0.653 | 0.647 | 0.302 | 0.290 |
| CoPx_R | **−0.346** | 0.265 | 0.262 | 0.454 | 0.452 |
| CoPy_R | **−0.711** | 0.115 | 0.107 | 0.483 | 0.478 |

- **F 通道:R² 好看(0.65),skill 一般(0.29)。** 因为 persistence 白拿 0.503——D1 之后 F 仍是有
  惯性的连续量,"下一帧≈这一帧"本就对。模型相对"什么都不预测"只多拿了 0.15。
  **若论文只报 R²=0.65,读者会把 persistence 免费拿到的那一半算到模型头上。**
- **CoP 通道:R² 难看(0.11–0.27),skill 好看(0.45–0.48)。** 因为 persistence 是**负的**——
  CoP 是一阶矩比值,在 16×16 栅格上按格跳变,复制上一帧比直接报均值更差。
  **此时 skill=0.48 不是"模型很强",而是"参照物很弱"。**
- **最关键的解读警告**:`opentouch_predictability_ceiling.py` 的推导给出——**纯白噪声下,任何预报器
  相对 persistence 的 skill 上限恰为 0.500**(persistence 的 MSE = 2·v_e,神谕的 MSE = v_e)。
  CoPy 的 skill 0.478 距这个"纯噪声上限"只差 0.022。
  **所以"skill 0.48"在噪声主导的通道上几乎不携带信息量**;单独引用它是误导。

**结论:两个数都不能单独报。** 必须同时给 `R²(persistence)`——它一个数就说明了参照物值不值钱。

---

#### 三、物理动机:为什么两个数据集当初选了不同的参照物

**ActionSense → persistence。** 该 harness 是**预报**装置(`src/actionsense/eval_harness/`,冻结)。
接触力由质量–弹簧–阻尼系统产生,是连续、有惯性的;牛顿力学给出的**零阶零假设**就是"这一瞬不变"。
超过 persistence ⇔ 捕捉到了 d/dt,即真的建了动力学模型。这是气象/时间序列的标准 skill score 传统。

**OpenTouch → 均值(class_mean)。** 该项目的主问题是 **G2 特质假设**(smooth vs abrupt 的 ΔR²),
不是"预报得多准"。ΔR² 需要一个**属于类别本身**的分母(该类的方差),而不是属于某个预报器的分母;
用 persistence 当分母会让 ΔR² 同时反映"两类里 persistence 谁更好用"这一无关效应。
`aggregate.py` 的 Q1 段已记录这一裁定,并注明 `class_mean` 是比 `train_mean` **更严格**的分母。

**两者的物理裂缝在 CoP 上暴露**:CoP = ∑p·x / ∑p,是**比值**,不是有惯性的状态量;低力时分母趋零
使其近乎未定义(`masking.py` 因此才存在)。对一个没有惯性的量用惯性零假设,自然得到负 R²。
**即 persistence 对 F 是有物理依据的零假设,对 CoP 不是。** 这解释了上表全部的方向差异。

---

#### 四、"是不是必须跟 ceiling 比" —— 是,但当前的对照有三个口径问题,必须先修

**必须比。** 理由是三条数字自身给的:D1 后 r(1)=0.318/0.403/0.228,噪声占比 68%/60%/77%。
在噪声主导的信号上,`skill=0.4` 到底是"接近极限"还是"差得远"**无法从 0.4 本身判断**。
唯一能把数字变成结论的量是 **FAS(fraction of attainable skill)**:

    FAS = (MSE_pers − MSE_model) / (MSE_pers − v_e) = skill_obs / skill_max

2026-08-20 记录的 78%/89%/87%(h=1)与 56%/63%/77%(h=30)正是这个量。**论文该报的是 FAS 和它随
h 的曲线,不是裸 skill。** "短 horizon 已达保守天花板的 78–89%" 是一个 convincing 的陈述;
"skill=0.4" 不是。

**但当前对照存在三处口径不一致,直接引用会出错:**

- **(1) 两个 skill 混用(SESSION_LOG 2026-08-20 已记,此处指出它污染了天花板对照)。**
  `opentouch_predictability_ceiling.py::observed_skill` 读的是 `docs/opentouch/d1/opentouch_cv4_d1.csv` 的
  `SS_vs_persistence`(**逐帧池化、逐 horizon**);而论文正文口径是 `opentouch_report_d1.csv` 的
  `skill`(**逐 clip 等权、全 horizon 合并**)。同为 ar 的 F 通道:前者 0.367,后者 0.302。
  **→ "78% of ceiling" 只对逐帧池化口径成立。** 若正文用 per-clip 口径,天花板必须按 per-clip
  等权重算,否则分子分母来自两个不同的估计量。

- **(2) 天花板脚本不施加 CoP 掩码。** `measure()` 在整段 `state[:, 0, :3]` 上算 `autocorr` 与
  `diffs`(`scripts/opentouch_predictability_ceiling.py:74-90`),**没有调用 `masking.valid_mask`**。
  而 harness 只在力超阈的目标帧上给 CoP 计分。于是 CoP 的 `v_e` 与 `MSE_persist` **都是在
  包含低力(CoP 近乎未定义)帧的population 上算的**,与观测 skill 的点集不同。
  方向上大概率高估噪声、压低天花板(与"保守"同向,故不会让结论虚假地变好),但**这是两个不同点集
  的比值,严格说不构成一个 bound**。**F 通道不受影响**(力从不被掩码)。

- **(3) 天花板目前只对 persistence 定义,对主报指标 R² 没有天花板。** 这是可以零成本补上的:
  噪声地板 v_e 是**信号的属性,与参照物无关**,故对任意参照物

      ceiling(ref) = 1 − v_e / MSE_ref

  而 `MSE_mean` 已经在 `ClipStats` 的充分统计量里(`sum_y`, `sum_y2`, `n_valid` → `constant_sse`)。
  **加一个 `R²_max = 1 − v_e/MSE_class_mean` 不需要重跑任何模型,只是表代数。**

**另需与数字一同声明的假设**:v_e ≈ (1−r(1))·var(x) 假定噪声**白且与信号独立**。本传感器是
**整数量化 + 满量程 95% + 硬顶 3072**(2026-08-16/19),量化噪声近似白,但**削顶是信号依赖的**;
且 D1 的 `max(x − (base+kσ), 0)` 是**半波整流**,会把残余噪声整流成正偏置。
**→ 白噪声模型在高力帧上偏乐观,天花板在那里可能并不保守。** 这一条目前没有写进任何文档。

---

#### 五、还有哪些衡量方式(按"对当前结论的信息增量"排序)

**P1 —— 移动平均(平滑 persistence)基线。最高优先,近乎零成本,且当前结论的可证伪性依赖它。**
2026-08-20 的自我诊断是:**"模型实际在预测过去的局部均值"**。若真如此,则
`ŷ_{t+h} = mean(y_{t−k+1..t})`(纯 numpy,`Baseline` 子类,k 在 VAL 上选)**应当拿到与 GRU/AR
相当的 skill**。
- 若 MA 基线追平 → "学到了触觉动力学"这一说法不成立,该报的是"最优预测≈局部均值,且这是噪声决定的"。
- 若 MA 基线明显落后 → 模型确实抓到了均值之外的东西,现有结论显著加强。
**两种结果都是可发表的**,而现在这个二选一**没有任何数据能判定**。这是本清单里唯一一条
"不做就无法判断现有结论是否空洞"的。

**P2 —— 一阶(漂移/线性外推)基线**:`ŷ_{t+h} = y_t + h·slope`,slope 取平滑斜率。
persistence 是零阶零假设;对有惯性的 F,物理上正确的零假设是一阶。它会**抬高 F 通道的参照线**,
使 F 的 skill 变成一个更诚实的数。

**P3 —— 对平滑后的目标计分(经验版天花板)**:用**非因果**低通(如 Savitzky–Golay)得到 s_t,
在 s 上重算 MSE 与 skill。这把 §四 的解析天花板换成**实测**的,不依赖白噪声假设,直接绕开量化/削顶/
半波整流三个问题;同时它正是 D1b(slow/fast 分解)所需的量,**一次投入两处用**。

**P4 —— 概率评分(CRPS / NLL),且它不需要 baseline。**
现状是**四个受训的方差头无人计分**:harness 只评点误差,σ 估错不付代价(2026-08-19续3 已记
`predict_with_sigma` 落盘,但计分仍未接入)。CRPS 是 proper scoring rule,在噪声主导的信号上
**恰恰是模型价值所在**——点预测被 v_e 卡死,但区间宽度是可以做对的。
±2σ 覆盖率(96.9/94.7/94.2 vs 名义 95.4)只是一个粗检,**不是评分**。
**若点预测的天花板确实是 0.44–0.72,那么诚实的 headline 也许应该是概率预报的质量,而非点误差。**

**P5 —— 事件级 / 决策级指标**:接触起始、滑移、力阈穿越在 horizon h 内的检出(lead time、ROC/PR)。
MSE 在白噪声下被硬性卡住,**事件检测不会**——它作用在慢包络上,而慢包络正是天花板分析指出
"可预测但没人在预测"的部分。且它与"触觉反馈"这一实际用途对齐,而 MSE 不。

**P6 —— 一切按 horizon 出曲线,不出标量。** 天花板随 h 从 0.44 升到 0.67–0.72,而实测 skill 基本持平;
**这个缺口(FAS 从 ~85% 掉到 ~60%)本身就是本项目最强的正面发现**——它精确定位了"慢包络的未来演化
可预测但无模型在建模它"。压成一个全 horizon 标量会把这个发现平均掉。

---

#### 六、OPEN QUESTIONS(需用户裁定,未擅自决定)

- **OQ-M1(口径)**:论文正文的 headline 用哪一个?建议 **per-clip 等权(report 口径)**,
  并把天花板按同口径重算。需用户确认后我才动 `opentouch_predictability_ceiling.py`。
- **OQ-M2(掩码)**:天花板脚本是否补上 CoP 掩码?我认为**必须补**(否则 CoP 那三行不构成 bound),
  但它会改变已写入 SESSION_LOG 2026-08-20 的天花板数值,属于**已报数字的修订**,故请用户裁定。
- **OQ-M3(P1 的优先级)**:MA 基线是否插到 N5(val MSE 早停重跑)之前?我倾向**是**——它不占 GPU,
  且 N5 之后的数字若仍要用"模型学到了动态"来解释,这个基线是绕不过去的。
- **OQ-M4(半波整流对天花板的影响)**:是否需要在高力帧上单独估一次 v_e 来检验"白噪声"假设?
  这会花一次全语料扫描(无 GPU)。

### 2026-08-21续 — d256 三件套落地(`src/d256.py` / `scripts/probe_d256.py` / `docs/d256.md`)

**请求**:在 src、docs、scripts 各建一个以数据集命名的文件;整理全部内容/格式/数量/clip 数/
地点数/entry 数。

**命名**:统一用 **`d256`**(= zip 名 `d256.zip`、顶层目录 `Dataset256`、已有的
`fetch_d256.py`)。页面本身没给数据集名。

**新增三文件**:
- `src/d256.py` —— 读取器。`iter_paths`(纯路径遍历,不 unpickle)/ `load_clip` /
  `iter_clips` / `tactile()`(堆成 (16,2,32,32),对齐既有 (T,2,H,W) 约定)/ `label_map` /
  `ego4d_vocab` / `counts`。常量含 `SIGNAL_SHAPES`、`N_CLASSES=20`、`FPS_UNKNOWN=None`。
- `scripts/probe_d256.py` —— `--fast`(默认,秒级路径遍历)/ `--full`(全量 unpickle,
  校验不变量 + schema + 值域),`--out` 导 CSV。**docs 里每个数字都能由它复算。**
- `docs/d256.md` —— 清单正文(10 节)。

**本轮新测出的、此前不知道的四件事:**

**(a) 会话目录名 == `label_idx`。** 20/20 类逐一取样验证全中(如 `.../S05/3/0.p` → class 3
"Slice a cucumber")。**故 clip 的类别可从路径直接读出**,这正是全部计数无需解 pickle 的原因。
`load_clip` 把它做成硬断言:一旦不成立立刻报错,而不是静默给出错标签。

**(b) `signals`/`signals1`/`signals2` 不是三份独立数据,而是同一录制的三种时间步长。**
用逐帧精确相等测出(`tactile-glove-left` 与 `myo-emg-left` 结果一致):
`signals[i]==signals1[3i]`、`signals2[i]==signals1[2i]`、`signals[2m]==signals2[3m]`
⇒ **步长比 signals1:signals2:signals = 1:2:3**,clip 恒 16 帧,故 `signals` 每片覆盖
**3 倍时长**;也解释了片数 25,471 < 26,922 < 28,426(步长越粗,窗口越少)。
**⇒ 三组合起来当 80,819 个独立样本用会泄漏。** 组内相邻 clip 不共享帧(clip0 vs clip1 = 0 帧相同)。

**(c) `val` 比"2%"看起来窄得多。** 每组只有 **3 个 (S05, session) 对**,即 **20 类里只覆盖 3 类**;
且三组**held-out 的类各不相同**(仅 class 2 三组共有)→ **跨组的 val 数字不可比**。
并且 **不是按受试者划分**:S05 同时是 train 里最大的受试者(18,428 片)。
held-out 单位是 (subject, session):这些 session 确实不在 train(实测 0 重叠、0 同名 clip)。
**⇒ 它测的是"没见过的 session",不是"没见过的人"。要做跨人泛化必须自建划分。**

**(d) 类别极度不平衡:13,657 / 346 = 39.5×。** 五个餐具类(0,15,16,18,19)占 **67%**,
五个清洁/倒水类(10–14)合计仅 **3.7%**。不给 per-class 分解的 accuracy 基本是餐具类的分数。

**"地点"的答案:1(且数据里无此维度)。** ActionSense 只在一间仪器化厨房录制,归档里不编码地点。
轴只有五个:group / split / subject / session(=class) / clip。**这点与 EgoTouch(5 个 scene)
和 OpenTouch 不同,不要照搬。**

**核心数字**(与 manifest 交叉核对通过):归档 187,729 members = 187,111 文件 + 618 目录项;
已取 **80,819 clip**(+2 词表)= 12.34 GiB;train 79,102 / val 1,717;20 类;5 受试者;
视频侧 106,290 npz / 237.97 GiB 未取,且 **video clip id 与 signal clip id 1:1 对应**
(`videos1/val/S05/2` 实测 303/303 全配)。

**方法学备注**:(b)(c) 都是**只看目录结构会误判**的结构性陷阱,故用逐帧比对与集合运算实测而非推断。
275-session 全采样脚本跑得太慢(远端 range,每 session 2 次请求),已中止——
`probe_d256.py --full` 在 CRC 本地磁盘上做同样的事,几分钟即可覆盖全部 80,819 片。

---

## 2026-08-21 — 【计划,待批准】把本仓库从 `Jianyi2004/TouchAnything` 的 fork 中独立出来

用户诉求:"这个 repo 里绝大部分文件都独立于 TouchAnything,我想独立出来"。以下先把事实查清,再给方案,
**未经用户在 OPEN QUESTIONS 上表态,不动任何一行代码、不推任何一个远端。**

### 一、现状实测(命令与结论)

| 项 | 实测值 |
|---|---|
| `origin` | `https://github.com/Jiayi459/TouchAnything.git`(public) |
| GitHub fork 标记 | `isFork = true`,parent = `Jianyi2004/TouchAnything` |
| 本地 remote | 只有 `origin`,**没有配置 upstream**(所以本地早已不与上游同步) |
| 提交数 | 179 = 上游 17 + 自己 162 |
| 上游最后一笔 | `d74f9ef` (Jianyi Zhou, 2026-05-21);自己第一笔 `1509fe9` (2026-06-17) |
| 追踪文件 | 668 = 自己新增 **628** + 上游遗留 **40** |
| `.git` 体积 | pack 281 MB |
| submodule | 3 个:`third_party/hamer` / `chumpy` / `EasyMocap`(**均未 init,工作树里不存在**) |

**用户的判断被证实**:94% 的文件(628/668)是本项目自己的。上游遗留的 40 个文件是:
`LICENSE`、`README.md`(上游论文主页)、`environment.yaml`、`.gitmodules`、
`assets/` 10 个 demo 图与 gif(工作树 71 MB)、`scripts/core/` 9 个、`scripts/tools/` 5 个、
`scripts/utils/` 2 个、`scripts/data_processing/` 2 个、5 个 `scripts/run_*.sh`、
`scripts/visualize_cleaned_data.py`、`scripts/batch_process_wilor_simple.py`、`src/__init__.py`。

**代码耦合只有一处是真的**(其余全是注释/文档里的路径提及,删了不影响运行):
- `src/touchanything/data/glove_augmentation.py:15-20` —— `sys.path.insert` 到
  `scripts/data_processing/`,再 `from glove_augmentation_realistic import ...`。
  **若要删除上游 `scripts/`,必须先把这个模块搬进 `src/`。**
- 纯文本提及:`touchanything_dataset.py:4,24`、`configs/.../touchanything_with_glove_aug_wilor.yaml:65`。

**仓库为什么大**:历史里最大的 blob 一半是上游 demo 资产
(`grasping_beverage...gif` 24.1 MB、`bouncing_ping_pong_ball...gif` 19.7 MB、几个中文名 `.mp4` 12–25 MB),
另一半是**我们自己入库的** `data/actionsense_states/*.npy`(402 个文件,单个最大 25.8 MB,工作树 416 MB)。
⇒ **换仓库本身不会让仓库变小**;要变小得另外做历史重写,而且主要该清的是我们自己的 `data/`。

### 二、许可证约束(与选哪条路无关,必须遵守)

上游是 MIT,且**上游代码仍留在树里**(至少 `scripts/core/`、`assets/`、`environment.yaml`)。
MIT 要求保留版权声明与许可证文本。⇒ **无论走哪条路,`LICENSE` 必须留着,并且应在新 README 里写明
"本项目基于 Jianyi2004/TouchAnything (MIT) 派生"。** 只有把上游文件全部删干净,才谈得上去掉这层义务
(即便如此,保留出处说明仍是更稳妥的做法)。

### 三、三条可选路径

**A. 新建一个非 fork 的空仓库,改 `origin` 后整体推过去(推荐)**
- 操作:GitHub 上 new repo(**不要用 fork/import**)→ `git remote set-url origin <新地址>` → `git push -u origin main`。
- 得到:GitHub 上不再显示 "forked from Jianyi2004/TouchAnything";不再属于上游 fork network;
  PR 默认目标不再指向上游;179 笔历史(含上游 17 笔)完整保留,上游作者署名也保留(符合 MIT 精神)。
- 代价:仓库地址变(旧地址的 star/watch 不跟过来,本仓库目前也没什么可丢的);体积不变。
- 风险:低,且可逆(旧 fork 先归档不删,确认无误再处理)。

**B. 在 A 的基础上另开全新历史(orphan commit)**
- 操作:`git checkout --orphan` → 一次性提交当前树 → 推到新仓库。
- 得到:历史干净、无上游 commit。
- 代价:**162 笔自己的提交历史全部丢失**(bisect/blame/"这行为什么这么写"都没了;SESSION_LOG.md 只能部分补偿)。
  且树里仍有上游文件 ⇒ MIT 义务不变,反而少了 commit 层面的署名。**不推荐。**

**C. 保留现有仓库地址,请 GitHub 断开 fork 关系**
- 先去 repo Settings → General → Danger Zone 看有没有 "Leave fork network"(GitHub 近年对部分仓库开放了自助入口);
  没有就开 support ticket 请求 detach。
- 得到:地址/star/issue 全部保留,fork 标记消失。
- 代价:**依赖 GitHub 侧处理,时间不可控**;且这条我没有把握说自助入口一定存在,需要用户去界面上确认。

**与路径正交的清理项**(可选,建议独立成一轮):
1. 删除用不到的上游遗留(先搬 `glove_augmentation_realistic.py` 进 `src/`,再删 `scripts/core` 等);
2. 重写 `README.md`(现在挂的是上游论文与作者名单,已经名不副实);
3. 删掉 `.gitmodules` 里三个从未 init 的 submodule;
4. `data/actionsense_states/` 402 个 npy 该不该在 git 里——这是仓库体积的真正大头。

### OPEN QUESTIONS(等用户回答后才动手)

- **Q1 走哪条路?** A(新空仓库+保留全历史,推荐)/ B(全新历史)/ C(请 GitHub 断开 fork)。
- **Q2 新仓库叫什么、public 还是 private?** 现名 `TouchAnything` 会继续和上游撞名,建议改一个反映本项目
  (触觉可预测性/预测研究)的名字。
- **Q3 旧 fork `Jiayi459/TouchAnything` 怎么处理?** 归档保留 / 删除 / 原样留着。
- **Q4 上游遗留文件与 `data/` 大文件,这一轮要不要一起清?** 还是先只做"独立",清理另开一轮。

### 2026-08-21续3 — 【方法学裁定草案,待用户确认】统一全数据集的 skill 估计量

用户裁定:**比那四个 OQ 更重要的是,所有数据集算 skill 的方法必须一样,要先定一种。**
以下是我的清点、我的选择与理由。**本条仍只做分析,未改任何代码。**

---

#### 一、先清点:仓库里现在有 **5 个互不相同的 skill 估计量**,在 4 个轴上都不一致

| # | 位置 | 参照物 | 聚合单元 | 掩码 | 产出 |
|---|---|---|---|---|---|
| 1 | `src/actionsense/eval_harness/{metrics,evaluate}.py`(**冻结**) | persistence / seasonal / ar | 逐帧池化 over (N,H) | ✅ `valid_mask` | `docs/actionsense/harness_baselines.csv` |
| 2 | `src/opentouch/{metrics,evaluate}.py`(#1 的逐字节 fork) | 同上 | 同上 | ✅ | `docs/opentouch_cv4*.csv` |
| 3 | `src/opentouch/aggregate.py` | class_mean / train_mean / clip_mean(R²)或另一预测器(skill) | **逐 clip 等权,ratio-of-means** | ✅ | `docs/opentouch_report*.csv` |
| 4 | `src/actionsense/state_forecast.py::skill_per_feature`;`src/actionsense/tactile_map/train.py::evaluate` | persistence(后者在**残差空间**,代数等价) | 逐帧池化 | ❌ **无掩码** | `docs/action_dynamics_results*.csv`, `docs/tactile_map_*.csv` |
| 5 | `src/tactile_pixel/tactile_utils.py::horizon_metrics` | persistence = 最后输入帧 | 逐样本池化,逐 h | ✅(有效 taxel) | `docs/RESULTS.md` |

**#4 无掩码这一条此前未被记录**:`tactile_map/train.py::evaluate`(第 110-115 行)直接
`em.mean((0,1)) / ep.mean((0,1))`,**没有调用 `valid_mask`**。故 ActionSense 的 tactile_map/
action_dynamics 那几张表里,**CoP 的 skill 把 CoP 未定义的低力帧也算进去了**,与 harness 的
`docs/actionsense/harness_baselines.csv` 不是同一个点集。这是必须统一的直接理由之一,不只是"美观问题"。

---

#### 二、"统一"要在四个轴上分别裁定,不是选一个公式

**轴 A 参照物** / **轴 B 聚合单元** / **轴 C 点集(掩码)** / **轴 D 前瞻时间的刻度**。
公式相同但任何一轴不同,数字都不可比。

##### 轴 A — 裁定:**headline 用 persistence;`class_mean` 只留给 G2**

理由是**可比性的定义级问题**:`class_mean` 是**被计分的那批 clip 的属性**,不是信号的属性。
换一批 clip、换一个数据集,分母就换了。本项目已经在这上面吃过一次亏——D1 之前 R²≈0.87–0.93,
而日志(2026-08-18 E5)自己的结论是它只能读作"对'常数+其漂移'的复现程度"。
persistence 则是**信号自身的零阶零假设**,定义与被计分的人群无关。
→ **跨数据集 headline = skill vs persistence。**
→ **G2 的 ΔR² 继续用 `class_mean`**:那是**同一数据集内**的类别对照,"每类用自己的方差当分母"
   正是 class-specific R² 的定义(`aggregate.py` Q1 已裁定)。**两者是两件事,不冲突。**

##### 轴 B — 裁定:**逐 clip 等权 ratio-of-means(即 `aggregate.py` 的算法),放弃逐帧池化**

**决定性理由:逐帧池化没有合法的重采样单元。** stride=1、H=30 时**同一帧是至多 30 个窗口的目标**,
窗口之间高度重叠,clip 内又有 r(1)≈0.3 的自相关。**在帧上做 bootstrap 得到的 CI 会严重偏窄。**
而 G2 的全部推断(ΔR² 的 CI 跨 0)都建立在 bootstrap 上 → **逐帧池化在本项目根本支撑不了推断。**
次要理由:OpenTouch clip 长度跨 87 倍(0.53 s–46 s,`aggregate.py` Q2),逐帧池化下这个数字
描述的是长 clip;而**长度分布是逐数据集不同的**,所以逐帧池化连"同一公式"都保证不了同一含义。

##### 轴 C — 裁定:**一律用 harness 的 `valid_mask`**,#4 的两处必须补上

##### 轴 D — 裁定:**前瞻时间用秒,不用帧序号;跨数据集表只在公共秒格点上取**

**这是最容易被漏掉、但确实存在的不可比。** 两边的 `horizon_s` 都是 1.0 s(好消息),但:
- ActionSense:`downsample: 3` → **10 Hz,H=10 步**,`h=1` = **100 ms**;
- OpenTouch:`downsample: 1` → **30 Hz,H=30 步**,`h=1` = **33 ms**。

→ **逐 h 的对照是错位的**(h=1 不是同一件事);
→ 更隐蔽:现在的 `horizon_step="all"` 标量是在 h=1..H 上平均,**两边平均的是不同的前瞻网格**
  (10 个点 vs 30 个点,且 OpenTouch 那 30 个点里挤满了容易的短前瞻)。**即使公式完全一致,
  这个 "all" 标量也不可比。**
**精确修法,且无需重训任何模型**:30 Hz 下 0.1 s = 3 帧,故对 OpenTouch 取
**h ∈ {3,6,9,…,30}** 即得到与 ActionSense **逐点重合**的秒格点 {0.1,0.2,…,1.0} s。
这是对已保存预测张量的**纯行选择**。
**并建议取消 "all" 这个标量**;若必须要一个数,取 **lead = 1.0 s**(config 声明的 horizon 本身),
一个定义明确的前瞻优于一个在网格上取的平均。

---

#### 三、可行性:**不需要动任何冻结文件**

`src/actionsense/` 自 2026-08-10 起从不编辑,而 `eval_harness/baselines/base.py` **没有**
OpenTouch fork 里那个 `predict_series_by_clip`,即 ActionSense 侧丢失了 clip 归属。
**但它可以被重建,不必改冻结代码**:`origins(T, cfg)` 是纯函数,且 `predict_series` 按
`sorted(data.items())` 迭代([base.py:57](src/actionsense/eval_harness/baselines/base.py#L57)),
**顺序完全确定**。故按各 recording 的长度重放 `origins` 即可还原 `clip_ids`,再喂给
`aggregate.clip_stats`。**冻结文件零改动。**

另:`src/opentouch/aggregate.py` **不是任何冻结文件的 fork**(其 docstring 明说 "NEW MODULE"),
且它的接口已经是数据集无关的——只吃 `(ytrue, mask, clip_ids, preds)` 数组。
**把它提升为共享模块是零语义风险的**,不触犯"不改 actionsense"的规矩。

---

#### 四、跨数据集统一后,一行报告长什么样

对每个 `(dataset, model, channel, lead_s)`:

| 列 | 含义 | 为什么必须在表里 |
|---|---|---|
| `MSE` | 原始误差 | 唯一不依赖参照物的量 |
| `skill` = S(model; persistence) | **headline** | 逐 clip 等权、掩码后、ratio-of-means |
| `skill_CI` | clip 级 bootstrap | 轴 B 的存在理由 |
| `R2_pers` = S(persistence; class_mean) | **参照物本身值多少钱** | 没有它,skill=0.48 无法判断是模型强还是参照物弱(2026-08-21续2 §二) |
| `FAS` = skill / (1 − v_e/MSE_pers) | 占可达上限的比例 | 唯一有希望**跨数据集真正可比**的量 |

**一条必须说清的保留意见**:即使四个轴全部统一,`skill vs persistence` 仍然**不是难度无关的**——
persistence 的好坏取决于传感器噪声与采样率(同一数据集内 F 的 R²(pers)=0.50、CoPy=−0.71,
差异已经如此之大)。**真正跨数据集可比的是 FAS**,因为它用噪声地板 v_e 归一。
但 FAS 依赖白噪声假设(OQ-M4 存疑)。
**故建议的分工:`skill` 作 headline(稳健、无附加假设),`FAS` 作解读(带假设,须与假设同时出现)。**

---

#### 五、范围上的一处反对意见

"所有 dataset 用同一种 skill" 对**估计量**成立,但对**数字**不成立:
`src/tactile_pixel` 的 skill 是 **21×21 压力图上的逐 taxel MSE**,目标空间与 F/CoP 力矩空间不同。
**可以共用同一套配方**(persistence 参照 + 逐样本(此处 = 逐 trajectory)等权 + 掩码 + 秒刻度),
**但它的数值与 moment-space 的 skill 不可放进同一列比较**。
建议:配方统一、表分开,并在文中明说这两个数不是同一个量。(EgoTouch 已 deprecated,不阻塞。)

---

#### 六、落地顺序(全部不占 GPU,全部作用于已保存的预测)

1. `aggregate.py` 提升为共享模块(纯移动 + import 修正)。
2. 新增一个非冻结的 ActionSense 事后计分脚本:重建 `clip_ids` → `clip_stats` → 同一份表。
3. 给 `#4` 的两处补掩码,或直接改为走同一条事后计分路径(更可取:少一个估计量)。
4. 秒刻度对齐:OpenTouch 取 h∈{3,…,30};取消 "all" 标量。
5. `docs/skill_comparison.md` 按新口径重写(该文件目前混用了 #1/#3/#4 的数字)。
6. 天花板脚本按 §2 轴 B/C/D 重算(即 OQ-M1/M2 的落地)。

---

#### 七、OPEN QUESTIONS(接 OQ-M1..M4)

- **OQ-M5**:同意"headline = 逐 clip 等权 skill vs persistence,`class_mean` R² 只留给 G2"吗?
  这会改变论文正文引用的数字(ar/F:0.367 → 0.302),属于已报数字的口径修订。
- **OQ-M6**:共享模块放哪?建议新建 `src/eval/`(与 `actionsense`/`opentouch`/`tactile_pixel` 平级),
  而不是让别的包 import `src.opentouch.aggregate`。
- **OQ-M7**:是否取消 `horizon_step="all"` 标量、改用 lead=1.0 s?
- **OQ-M8**:`#4`(action_dynamics / tactile_map)的历史数字要不要重算并**修订已写入文档的值**?
  它们目前无掩码,CoP 那几列与 harness 不同源。

### 2026-08-21续 — 【已执行完毕】独立完成:新仓库 `Jiayi459/tactile-forecasting`,旧 fork 已归档

用户对 OPEN QUESTIONS 的裁决:**Q1 = A**(新建非 fork 空仓库 + 保留全历史)、**Q2 = 改名 + private**、
**Q3 = 旧 fork 先归档保留**、**Q4 = 这一轮只做独立,清理另开一轮**。仓库名后续定为 `tactile-forecasting`。

**实际执行的动作(按序)与验证结果:**

1. `gh repo create Jiayi459/tactile-forecasting --private` —— 建的是**空仓库**,不是 fork/import。
   建后即验证:`isFork: false`、`visibility: PRIVATE`、`isEmpty: true`。
2. `git remote rename origin oldfork` + `git remote add origin <新地址>`。
   **刻意保留 `oldfork` 作为安全网**,而不是直接 `set-url` 覆盖。
3. `git push -u origin main` —— 281 MB,前台跑满 10 分钟未完成(超时,`git ls-remote` 确认远端**无任何 ref**,
   即未部分落地),改后台重跑,**exit 0**。
4. 简介按用户更正改为三个数据集的正确列举:**ActionSense / OpenTouch / d256**(初版误写成 EgoTouch)。
5. `gh repo archive Jiayi459/TouchAnything --yes` —— 旧 fork 现为 `isArchived: true`(只读,仍 public,未删)。

**验证(不是"应该成功",是实测):**

| 检查项 | 结果 |
|---|---|
| 本地 `main` SHA | `dc335576917e7032e5be5bbf4e15adc4a3b3926b` |
| 远端 `main` SHA | `dc335576917e7032e5be5bbf4e15adc4a3b3926b` ✅ 一致 |
| 远端 `main` 提交数 | **179**(= 本地 179,上游 17 + 自己 162 全部保留)|
| 新仓库 fork 标记 | `isFork: false` ✅ 已脱离上游 fork network |
| 上游追踪分支 | `main` → `origin/main`(指向新仓库)|
| 旧 fork | `isArchived: true`,`isFork: true`(归档保留,未删)|

**给用户的三点结论(已在对话中说明):**
- 本地目录不需要重新 clone、不需要改名;`git push` 默认已走新仓库,旧地址只有显式 `git push oldfork` 才会碰到。
- 后台大推送锁定的是启动时刻的 `dc33557`;期间新增的 commit 需要**再 push 一次**(只传增量)。
- **CRC 上光 `git remote set-url` 不够**:原仓库 public 所以无需凭据,新仓库 private,`fetch/pull` 会卡认证。
  已给出 SSH deploy key 方案(集群上不留明文 token),或 fine-grained 只读 PAT(不推荐,明文落在 `.git/config`)。

**遗留(等用户决定,未擅自改动)**:
- `scripts/crc/README.md:32` 仍硬编码旧 clone 地址 `https://github.com/Jiayi459/TouchAnything.git`,已失效。
- 三、四两节里列的清理项(README 重写、删上游遗留代码、`data/` 402 个 npy 是否出库)按 Q4 裁决**另开一轮**。
- MIT 义务未变:上游代码仍在树里 ⇒ `LICENSE` 必须保留,新 README 应写明派生自 `Jianyi2004/TouchAnything`。

## 2026-08-21 — 【SELECT_ON=mse 4折结果】方法学修正是对的,但数值上是零结果

产出:`docs/opentouch_{cv4,report,ceiling,loss,forecast}_d1_mse.*`。唯一变量:`SELECT_ON=mse`
(EPOCHS 16 vs 8 见下)。

### 一、对照组通过——只有 prob_gru 变了

| 模型 | 通道 | NLL 选 | MSE 选 | 差 |
|---|---|---|---|---|
| ar | F/CoPx/CoPy | 0.3671/0.4313/0.4762 | 同左 | **+0.0000** |
| seasonal | F/CoPx/CoPy | −0.0375/−0.0331/−0.0087 | 同左 | **+0.0000** |
| prob_gru | F_R | 0.3862 | 0.3829 | **−0.0033** |
| prob_gru | CoPx_R | 0.4271 | 0.4309 | **+0.0039** |
| prob_gru | CoPy_R | 0.4722 | 0.4724 | **+0.0002** |

ar/persistence/seasonal 不涉及早停,其数字**逐位不变**,说明改动没有波及无关部分。

### 二、结论:换准则**几乎没有影响**,方向还不一致

报表口径(逐 clip)prob_gru:F 0.2905→0.2977(+0.007)、CoPx 0.4517→0.4525、CoPy 0.4780→0.4776。
逐帧口径 F 反而**降了** 0.003。**两个口径给出相反的符号,幅度都在 ±0.007 以内。**
→ **这是一个零结果,应当如实报告。** 方法学上"用 harness 打分的准则来选权重"依然是正确的做法
(选择与报告终于一致),但**它带来的数值收益可以忽略**。此前把它列为"唯一确定能提升报告指标的
改动"是**过度乐观**,已由本轮证伪。

### 三、担心的两件事都没发生

**校准未退化**:±2σ 覆盖率 96.6/94.5/94.3%(NLL 选时 96.9/94.7/94.2%,名义 95.4%)。
按 MSE 选权重**并没有**让方差头明显变坏。此前"拿概率正确性换点误差"的顾虑**不成立**。

**16 轮足够,且比需要的多**:val MSE 的最低点落在**第 1/2/6/6 轮**(四折),此后单调上升。
上一轮"fold0 的 val MSE 到第 8 轮仍在下降"的读数**未能重现**——本轮 history sweep 改按 MSE 选,
四折选中的 t_in 变了(2/3/3/3 s),最终训练不是同一次,故曲线本就不同。
**教训:那条读数是从图上目测的,不该据以调整 epoch 预算;应当直接读 `best_val_mse_epoch`。**

### 四、Claude 的 bug:loss 图谎报了权重来源(已修)

`plot_opentouch_loss.py` 把星号硬编码为"min VAL NLL",右图纵轴硬编码为"best VAL NLL"。
本轮权重实际按 **MSE** 选、sweep 也按 MSE 打分,**于是图上星号画在了一个并没有选中权重的曲线极小
值上,纵轴标签也与所绘数据不符**。已改为从 checkpoint 的 `history["selected_on_metric"]` 读取,
星号画在**真正做出选择的那条曲线**上,标题与轴标签随之变化;旧 checkpoint 无该字段时按 NLL 处理。

### 五、过拟合依旧严重(未因 D1 或换准则而缓解)

四折 val NLL 从第 1 轮起单调上升至第 16 轮;val MSE 在第 1–6 轮见底后同样上升。
**记忆化仍然存在**,与 2026-08-19 机制判别一致。

### 六、G2 零结果第三次成立

MSE 选权重下 prob_gru 的 ΔR²:F +0.0057 [−0.070, 0.069]、CoPx −0.0357 [−0.113, 0.031]、
CoPy +0.0381 [−0.010, 0.083]——**三通道 CI 全部跨零**。
至此该零结果在 **raw / D1 / D1+MSE选** 三种设定下均成立。

### 七、下一步建议(重排)

**N6(推荐,需用户拍板)——慢/快分解,针对长 horizon 的 0.15–0.30 缺口。**
天花板测量指出:短 horizon 已达 78–89%,**剩余空间几乎全在长 horizon**,而缺口成因已定位为
"模型预测过去的局部均值,而非平滑分量在 t+h 的取值"。**这是唯一一个有明确机制、且空间够大的方向。**
**它重开 Q1(目标是否仍为 RAW)**:ActionSense 的做法是预测 FAST 分量并以 persistence-of-fast
为参照,而我们刻意保持 RAW 以便 harness 可比。**折中方案**:目标仍为 RAW(harness 不变),但**模型
内部**做慢/快分解——显式预测慢包络 + 快分量的条件分布,再合成。**这样对外可比性不受影响。**

**N3——D1 数据上跑 map 三臂(flatten/cnn/aggregate)。** 一次 GPU 作业,填补与 ActionSense
唯一还空着的对照(那边 flatten −0.042 / cnn 0.138 / aggregate 0.181)。

**N4(权重衰减)——保持降级。** 过拟合确实还在(见五),但天花板显示短 horizon 只剩 0.05–0.10,
且长 horizon 的缺口是表示问题、非过拟合问题。**收益有限且不针对主要缺口。**

**N7(新增,零成本)——把 `select_on=mse` 定为默认。** 数值上无差别,但它让选择准则与报告准则
一致,是无代价的方法学改进。**建议在下一轮实验时一并切换,而非单独跑一轮。**

## 2026-08-21续 — 【±2σ band 与 sharpness】Claude 的过度表述被用户纠正;sharpness 首次测量(含一个 bug)

### 一、band 图的结论:用户判断正确,Claude 上一条说过头了

在 `--band --sample` 下重画 `docs/opentouch_forecast_d1_band_*.png` 后,用户判断"没有特别大的差别"。
**该判断成立,Claude 此前的表述有误。**

Claude 上一条称"±2σ 的带宽足以覆盖那些振荡",语气上作为模型的优点陈述。**但带子覆盖住信号,
恰恰因为它非常宽**:图上 clip 0 的带子约从 400 铺到 2900,而真实信号仅在 500–2700 之间。
**高覆盖率是平凡可得的**——一个只输出全局均值与全局标准差的模型,覆盖率同样能到 ~95%。
**因此"覆盖率 96.6%"几乎不构成模型学到了东西的证据。**

**band 确实说明的两件事**(不推翻上述纠正):
1. 带子呈**锯齿状扇形**——每个预测起点处窄、向后 1 秒张开、随后重置,说明模型在表达"越远越不确定",
   而非输出常数 σ;
2. 覆盖率 96.6% 对名义 95.4% 仅**轻微过覆盖**,说明 σ 大致是校准的,没有明显吹大。

**综合**:带宽是数据决定的,不是模型偷懒——F 的方差 68.2% 在一帧内退相关,σ_e ≈ 386,
边际标准差约 468。**在此信噪比下,一条诚实校准的带子必然如此之宽。**

### 二、由此提出可判定的量:sharpness(校准前提下的锐度)

**覆盖率不能回答"带子是否已尽可能窄"**,故新增测量:**模型平均 σ / 噪声底 √v_e**。
- ≈1.0x → 已锐到数据允许的极限,**训练层面无事可做**;
- ≫1.0x → 模型在本可确定处仍不确定,**该部分可由训练攻取**;
- <1.0x → 过度自信(应表现为覆盖率低于名义,本例已可排除)。

合成验证:已知边际 sd=468 的数据,脚本算得噪声底 470.7;人为喂入 σ=300 读作 **0.64x**,
σ=0.05 对 0.047 读作 **1.06x**。判据工作正常。

### 三、首次真实测量失败:全部为 NaN(Claude 的 bug,已修)

`docs/opentouch_ceiling_sharpness.txt` 的 sharpness 段落**三通道五个 horizon 全部为 `nan (nanx)`**。
**原因**:部分 clip 过短、`origins` 为空,`np.nanmean` 对**空切片**返回 NaN;而跨 clip 汇总时
Claude 用的是 `np.mean` 而非 `nanmean`,**一个空 clip 即污染整列**。
**修复**:改为按 origin 数**累加求和/计数**,空 clip 自然不贡献,且各 clip 按其 origin 数加权
(此前 2 个 origin 的 clip 与 200 个 origin 的 clip 等权,本身也不对)。
已构造一个 `origins` 为空的 clip 复现并确认修复。

### 四、本次运行中**有效**的结果(sharpness 之外均正常)

天花板与自相关部分**全部有效且与 2026-08-20 完全一致**(同一份 cache,确认可复现):
r(1)=0.318/0.403/0.228,噪声占比 68.2%/59.7%/77.2%;噪声底 √v_e = **386 / 0.0660 / 0.0934**。
observed 列为 MSE 选权重版本,与 NLL 版差异 ≤0.007(见前一节)。

### 五、待办

**N8(零成本)**:重跑 sharpness(命令见下),取得该表。**这是"波动"这条线索的最后一个判据**:
若 ≈1.0x,则模型的不确定性表达已到极限,**关于"预测线太平"的追问到此结束**,应转向 N6;
若 ≫1.0x,则 σ 过大本身会把 μ 拉向全局均值,那是可训练的。

## 2026-08-21续2 — 【sharpness 定案】σ 略偏小而非偏大;高斯似然被证明设定错误

`docs/opentouch_ceiling_sharpness.txt`。两次预判(Claude 先猜 ≈1.0x、后不再预判)均未命中,结果如下。

### 一、原始判据(σ / 噪声底)被否决为**不可用**

首测 σ/floor = **1.29–2.23x**,易被误读为"多余的不确定性,可由训练消除"。**该读法错误**:
分母 √v_e 是**神谕**下界,假设慢包络在 t+h 的取值已知;真实模型不知道,**超出部分正是它对包络的无知**,
与天花板表上长 horizon 的 0.15–0.30 缺口是同一件事的两种测法。
**反证**:CoPx 的 σ 达噪声底的 2.23 倍,而其覆盖率仅 **94.5%,低于名义 95.4%**——真正被吹大的 σ
不可能出现这种组合。故改用**模型自身在同一批 origin 上的实际 RMSE** 作分母,且**只在预测文件内部
计算**,以避开本仓库两个不可互换的 MSE 口径(天花板的逐 clip 中位数 vs 驱动的逐帧池化)。

### 二、结果:σ / 自身 RMSE = **0.82–0.90**,即 σ **偏小**约 10–18%

| 通道 | h=1 | h=5 | h=10 | h=20 | h=30 |
|---|---|---|---|---|---|
| F_R | 0.90 | 0.89 | 0.90 | 0.89 | 0.90 |
| CoPx_R | 0.82 | 0.83 | 0.83 | 0.83 | 0.84 |
| CoPy_R | 0.90 | 0.90 | 0.90 | 0.89 | 0.90 |

**跨 horizon 高度稳定**(F 恒为 0.89–0.90)。→ σ 随 h 的**增长比例是对的**,只是整体一致地偏小,
是**尺度**问题而非形状问题。

**→ 决定性结论:不存在可供训练消除的过度弥散。** "σ 偏大 → NLL 中 μ 的偏差惩罚变小 → μ 被拉向
全局均值"这一机制**被排除**。**关于"预测线太平"的追问到此结束**:平的 μ 是这份数据上正确的条件均值,
剩余空间只在长 horizon 的慢包络(N6)。

### 三、新发现:高斯似然设定错误(重尾),由两个数字联立证明

σ < RMSE 与"覆盖率≈名义"看似矛盾,联立即可定量证明**残差重尾**:
| 通道 | σ/RMSE | ±2σ = k·RMSE | **高斯下应有覆盖** | **实测覆盖** | 超出 |
|---|---|---|---|---|---|
| F_R | 0.90 | 1.80 | 92.8% | **96.6%** | **+3.8** |
| CoPx_R | 0.83 | 1.66 | 90.3% | **94.5%** | **+4.2** |
| CoPy_R | 0.90 | 1.80 | 92.8% | **94.3%** | +1.5 |

若残差服从高斯,σ 必等于 RMSE 且 ±2σ 覆盖 95.45%。实测**在更窄的区间里装下了更多的质量**
(超出 +1.5 至 +4.2 个百分点),即**中心更尖、尾部更厚**——典型的重尾。
高斯 MLE 的 σ 于是落在"主体尺度"与"被尾部抬高的 RMSE"之间,恰好解释 0.82–0.90 这个比值。
**此前多次记录的"重尾特征"至此由定量论证确认,不再是印象。**

### 四、下一步(更新)

**N9(新增,plan-before-code,待用户拍板)——把高斯似然换成 Student-t。**
`nll()` 现为高斯(`prob_gru.py:229`)。改为 Student-t(ν 可学或固定)即与残差的实际形状相符。
**OPEN QUESTIONS**:
- **Q9.1 会不会损害 harness 指标?** t 似然**降低离群点权重**,而 harness 用 **MSE** 计分、
  MSE 恰恰由那些离群点主导。**故 μ 可能变得更"稳健"但 MSE 更差。这是真实的取舍,须先说定接受哪一侧。**
- **Q9.2 它改善的是概率质量,不是点预测。** 若论文主张只涉及点预测,收益有限;
  若主张涉及不确定性量化(覆盖率、校准),则它是**必要**的修正。
- **Q9.3 是否偏离 ActionSense 逐字复刻?** 高斯 NLL 是 ActionSense 的原始损失。改动会使该臂
  不再与其逐字一致,**需明确记录为有意偏离**,并考虑同时保留高斯版以供对照。

**N6(不变,仍为唯一有实质空间的方向)**——长 horizon 慢包络,缺口 0.15–0.30。
**N3(不变)**——D1 上跑 map 三臂。

## 2026-08-22 — docs/ 重组 + 【map 三臂 D1 结果】aggregate 有效,flatten/cnn 是**静默故障**

### 一、docs/ 重组(用户指示)

86 个文件原本平铺在一个目录,仅靠命名区分(`opentouch_cv4.csv` / `_df` / `_d1` / `_d1_mse` /
`_d1_map`)。现改为按传感器、再按 run 分层:
```
docs/*.md                 项目级文档(结论/计划/跨传感器对比)
docs/actionsense/     34  harness、tactile_map、action_dynamics 全部结果
docs/opentouch/{raw,df,d1,d1_mse,d1_map,exploratory}/   7/4/9/11/7/2
```
**74 个文件以 `git mv` 移动**(保留历史),**89 处引用全部改写**。两个实现要点:
(a) 替换**按路径长度从长到短**,否则 `docs/opentouch_cv4.csv` 会污染 `docs/opentouch_cv4_d1.csv`;
(b) 第一遍 glob 漏了 `src/**/*.md`,改用 `git ls-files` 覆盖全部被跟踪文本文件后重扫,
残留为 **0**。新增 `docs/README.md` 记录布局及两条最易误读之处
(绝对误差不可跨 D1 比较;两个 skill 口径不可互换)。
`d1_band` 三图归入 **`d1_mse/`**——它们由 `runs/preds_d1_mse` 绘制,不是独立训练。

### 二、map_aggregate:有效,且与既有各臂**基本打平**

| 模型 | R²(F/CoPx/CoPy) | skill(F/CoPx/CoPy) |
|---|---|---|
| **map_aggregate** | **0.6611**/0.2476/0.1012 | **0.3178**/0.4409/0.4745 |
| ar | 0.6530/**0.2654**/**0.1152** | 0.3016/**0.4541**/**0.4827** |
| prob_gru(d1_mse) | 0.6511/0.2631/0.1064 | 0.2977/0.4525/0.4776 |

**map_aggregate 在 F 上最好(R² 0.6611、skill 0.3178),在 CoP 上略差于 ar。三者差距 ≤0.02。**
→ 与此前结论一致:**换架构在这份数据上买不到东西**,天花板才是约束。

**sharpness 交叉验证(重要)**:map_aggregate 的 σ/自身 RMSE = **0.80–0.91**,
与 prob_gru 的 **0.82–0.90** 几乎相同。**两个不同模型给出同一个偏离**,
说明**残差重尾是数据的性质,不是某个模型的毛病**——8/21 的结论由此加固。

### 三、flatten 与 cnn:**静默故障,数字无效,不得引用**

**症状**:两臂 R²(−1.8159/−13.9050/−4.2122)与 skill(−4.6678/−10.0749/−2.0472)
**逐位相同**,且 **σ 恒为 0**。不同架构不可能给出相同到小数点后四位的结果。

**根因**:`aggregate` 经 `load_target` 读 `state_*.npy`;**`flatten`/`cnn` 经 `taxel_baselines`
+ `load_map` 读 `clip_*.npy`**。而 `taxel_baselines` 对缺失文件是 **`continue` 静默跳过**
(`tactile_map.py:153`),`cache_d1` 中若无 map 文件则 `bases={}` → `build_inputs` 返回**空字典**
→ **两臂在零数据上训练与预测**,落入同一 fallback。
**这正是提交前已提醒过的风险**(cache_d1 可能只有 state 文件),
**但它没有报错,而是安静地产出了一整张指标表——这比崩溃更糟。**

**修复**:`taxel_baselines` 在**一个 map 都没找到**时抛 `FileNotFoundError`,并在错误信息中给出
补救命令;`build_inputs` 在输入集为空时同样抛错。新增测试覆盖"有 state 无 map"的情形,
并断言 **aggregate 臂在无 map 时仍可用**——正是这一点使故障得以静默。
**该测试本地为 skip(torch 不可用),按 D-TEST 约定须先在 CRC 上单跑 pytest 再提交作业。**

### 四、与 ActionSense 的对照表仍有两格空缺

ActionSense:aggregate **0.181** > cnn **0.138** > flatten **−0.042**
(空间结构在该传感器上帮倒忙,除非用卷积)。
OpenTouch 现仅有 aggregate,**flatten/cnn 待重跑**。→ "空间结构有没有用"**尚未回答**。

### 五、重跑方案(推荐)

`tactile_map` **自身已实现 D1**(`taxel_baselines` 注释即 "D1's estimator",且**仅由 TRAIN 估计**,
比 cache 级 D1 更严格、不会泄漏)。故**不应**再让它读一份已校正的 map。推荐:
**把原始 map 软链接进 cache_d1** —— `ln -s ~/opentouch/cache/clip_*.npy ~/opentouch/cache_d1/`。
这样:map 输入为原始图 + 脚本内 TRAIN-only 基线扣除(**无二次校正**),
目标取自 cache_d1 的校正 state(**与 d1/d1_mse 可比**)。

### 2026-08-22续 — pytest 门有洞:tactile_map 测试从未被执行过

**发现**:在 CRC 上按 D-TEST 约定单跑 `tests/test_opentouch_tactile_map.py`,
`test_windows_are_residual_and_left_padded` **失败**(1 failed, 8 passed;Claude 新增的
无 map 护栏测试通过)。

**根因(两层)**:
1. **作业脚本的 pytest 门只列了两个文件**
   (`tests/test_opentouch_prob_gru.py tests/test_opentouch_gru_aggregate.py`),
   **`tests/test_opentouch_tactile_map.py` 从未被任何地方执行过**——本机 torch 不可用,
   CRC 又没跑它。该断言自 `33e2137` 写下起一直未验证,**整个 map 运行期间都带着它**。
2. **失败的是断言本身,不是代码**:`min_history=15` → 首个 origin 为 t=15 →
   窗口 `M[t-t_in+1 : t+1]` = `M[0:16]`,即 **16 帧真实数据、24 帧填充**;
   测试写死 `X[0, :25]`,把"origin 在 15"错算为 15 帧历史(0..15 共 16 帧,差一)。

**修复**:
- 断言改为**由配置推导**:`pad = t_in - (min_history + 1)`,并补一条
  `X[0, pad]` 非零的断言,防止将来 pad 算多了也能"通过";
- **作业脚本的门改为 `python -m pytest tests/ -q`(全量)**。
  **一个会跳过文件的门就是有洞的门**——这是本次的系统性教训,优先级高于单个断言。

**注意**:D-TEST 约定(2026-08-21)本身是有效的——正是它让这个错误在提交作业前暴露,
而不是又浪费一次 GPU 作业。

### 2026-08-22续2 — flatten/cnn 故障的**真实**根因(Claude 的第一次诊断是错的)

**先更正**:上一节把根因判为"cache_d1 缺少 clip_*.npy"。**该判断错误。**
用户在 CRC 上的检查表明:`cache_d1` 有全部 **2904** 个 `clip_*.npy`(实体文件,非软链接),
形状 (T,1,16,16) float16,且**已完成 D1 校正**(mean 6.57、**零占比 91.6%**,
对照原始 cache 的 mean 2924、零占比 0.5%)。GRID=16 也是对的。
Claude 加的"无 map 则报错"护栏本身合理,**但它并未解释这次故障**。

**真实根因(三个缺陷叠加)**:
1. `predict_with_sigma` 对"取不到输入"的 clip **静默填零**
   (`if i not in inputs or len(ors)==0: mus[i]=zeros`)。
2. `build_inputs` 丢弃**所属 shard 不在 `bases` 中**的 clip,而 `bases` 由 `base_ids` 估计。
3. 驱动传入的 `base_ids = train+val`。**按地点留出时,test 的 shard 按构造不在 train+val 中**
   → 全部 test clip 被丢 → 全部填零。

**证据(决定性)**:`mu_flatten` 与 `mu_cnn` 的 `min=max=0`、`sigma` 亦全为 0,
且 `np.array_equal(mu_flatten, mu_cnn) == True`。**两臂的"预测"是同一个全零数组**——
不是训练发散,是数组分配后从未写入。`map_aggregate` 不经此路径,故完好
(mu ∈ [−0.336, 2204],sigma ∈ [0.055, 736])。

**文档与实现长期不一致**:模块文档描述了 `--baseline-scope train` 这一**从未实现**的开关;
而 `train(base_scope="shard")` 的 "shard" 分支**实际是 `train ∪ val`**,并非文档所述的"整个
shard"。**两个分支在按地点留出下都无法为留出地点给出基线。**

**修复(三处)**:
- 新增 `scope_ids(cfg, train, val, scope)`,`shard`(默认)= 该 shard 自身全部帧、
  `trainval` = 旧的错误命名行为、`train` = 最严;
- `predict_with_sigma`:`len(ors)==0` 仍返回**空**数组(正确,该 clip 本就无预测可做);
  但 **`i not in inputs` 改为抛 `RuntimeError`**——填零不是答案,是伪造,而它被当作答案计了分;
- 驱动新增 `--baseline-scope`(默认 `shard`),作业脚本新增 `BASELINE_SCOPE`;模块文档更正。

**依据**:逐格静息电平是**硬件属性**,由该 shard 自身帧估计**只在输入上是 transductive,
目标上从不**——这是 2026-08-16 已记录的论证,本次只是让实现与之相符。

**待跑**:`docs/opentouch/d1_map/` 下 flatten/cnn 的全部数字**作废,不得引用**;
map_aggregate 的数字**有效**(其通路不受影响)。

## 2026-08-23 — 【map 三臂重跑 d1_map2】编码器排序在第二个传感器上完整复现;AR 仍胜过全部 map 臂

修复(`c0011f0`/`ab7a2f7`)后以 `--baseline-scope shard`(默认)重跑。产物:`docs/opentouch/d1_map2/`。

### 一、对照组通过——修复是外科式的

`map_aggregate` 不经过出问题的通路,其数字**应当逐位不变**:
| 通道 | d1_map(旧) | d1_map2(新) | 差 |
|---|---|---|---|
| F_R | 0.3596 | 0.3596 | **+0.00007** |
| CoPx_R | 0.4224 | 0.4223 | **−0.00015** |
| CoPy_R | 0.4686 | 0.4686 | **+0.00000** |
→ 修改**未波及**未受影响的臂。这是本轮免费的对照组,事先写定,已通过。

### 二、【主结果】编码器排序在两个传感器上完整复现

**skill vs persistence(逐帧口径,4 折均值):**
| 通道 | 排序 |
|---|---|
| F_R | **ar 0.3671 > map_aggregate 0.3596 > cnn 0.3334 > flatten 0.2727** |
| CoPx_R | **ar 0.4313 > map_aggregate 0.4223 > cnn 0.3744 > flatten 0.3308** |
| CoPy_R | **ar 0.4762 > map_aggregate 0.4686 > cnn 0.4242 > flatten 0.4066** |

**aggregate > cnn > flatten 在三个通道上无一例外,且与 ActionSense 的排序一致**
(该处 aggregate 0.181 > cnn 0.138 > flatten −0.042;**数值不可跨传感器比较,可比的是排序**)。

**排序对"给了模型多少原始空间细节"是单调递减的**:
`ar`(无 map,线性)> `map_aggregate`(无 map,GRU)> `cnn`(用 map,利用空间结构)
> `flatten`(用 map,不利用空间结构)。
→ **两条结论,现在各自有两个传感器支持**:
  1. **原始触觉图不提供 F/CoP 之外的可预测信息**——F 与 CoP 近乎是该任务的充分统计量,
     多出的 254 维只增加估计方差;
  2. **若一定要读原始图,利用空间结构(CNN)显著优于无视它(flatten)**。

**与 ActionSense 的一处差异**:OpenTouch 上 flatten **仍显著为正**(0.27–0.41),
而 ActionSense 上它**低于 persistence**(−0.042)。即"读原始图"在 OpenTouch 上**代价更小**,
但方向相同。

**最值得强调的一点**:**线性 AR 基线在全部三个通道上胜过 16×16 图上的 CNN。**

### 三、sharpness:第四次给出同一个偏离

| 模型 | σ/自身 RMSE(h=1…30) |
|---|---|
| map_aggregate | 0.80–0.91 |
| cnn | 0.79–0.89 |
| flatten | 0.78–0.92 |
| prob_gru(8/21) | 0.82–0.90 |
→ **四个架构互异的模型给出同一个 ~0.85 的偏离**。
**"残差重尾、高斯似然设定错误"由此成为数据层面的结论,与模型无关。**(N9 的依据进一步加强。)

### 四、G2 零结果第四次成立

新增的三个 map 臂,ΔR² 的 bootstrap CI **三通道全部跨零**
(cnn:F −0.033 [−0.117, 0.045]、CoPx −0.031、CoPy +0.041;flatten、map_aggregate 同样)。
至此该零结果在 **raw / D1 / D1+MSE选 / D1+map三臂** 四种设定、六个模型上均成立。

### 五、本轮的限制(须与结果一同引用)

1. **epochs=8 固定预算**。map 臂参数远多于 aggregate 臂,理论上更吃训练量。
   但已测得 val 在第 1–6 轮即见底(prob_gru),**更长训练大概率只会更糟**;尽管如此,
   "三臂在同一 8 轮预算下比较"这一前提应当明说。
2. **map 臂不保存 checkpoint**(`--save-model` 仅对 prob_gru 生效),故**无 loss 曲线**,
   无法直接验证其 epoch 预算是否充分。这是当前唯一未闭合的验证缺口。
3. 逐帧口径与报表口径**并存**:本节用逐帧(与天花板同源);报表口径下
   map_aggregate 0.3179/0.4409/0.4746、cnn 0.2935/0.3749/0.4216、flatten 0.2386/0.3382/0.4083,
   **排序一致**。

### 六、`docs/skill_comparison.md` 的两个空格现已可填

ActionSense vs OpenTouch 对照表中 flatten / cnn 两行不再为空。**待更新该文件。**

## 2026-08-23续 — 【d1_map3】复现失败暴露 GPU 非确定性,并**推翻了一条排序结论**

本轮只为取 map 臂的 loss 曲线(种子、数据、scope 全同),指标本应逐位复现。**未复现。**

### 一、不是 checkpoint 改动造成的

差异的**分布**排除了代码原因——若为代码改动,三臂应受同等影响:
| 臂 | 两次最大偏差 |
|---|---|
| cnn | **4.85e-03** |
| flatten | **4.20e-03** |
| map_aggregate | 1.52e-04 |
| ar / persistence / seasonal | **0(逐位相同)** |
经典基线**逐位不变**,证明数据与划分未变;偏差只出现在 GPU 训练的臂上,且**参数多、含卷积的臂偏离大两个数量级**。

**根因**:`configure_determinism` 用 `torch.use_deterministic_algorithms(True, warn_only=True)`
——**warn_only 意味着缺少确定性实现的算子只警告、继续用非确定性版本**;且
`CUBLAS_WORKSPACE_CONFIG` **在作业脚本中从未设置**,而 CUDA ≥10.2 上不设它 cuBLAS 即非确定性。
**已修**:作业脚本 `export CUBLAS_WORKSPACE_CONFIG=:4096:8`(注释说明这只是收窄、并未闭合)。

### 二、【结论修正】"三个通道无一例外"不成立

用两次复现的偏差作噪声量级,检验相邻名次的间距(要求间距 > 3×噪声和):
| 通道 | ar>map_agg | map_agg>cnn | cnn>flatten |
|---|---|---|---|
| F_R | 49.1x 稳 | 5.2x 稳 | 6.3x 稳 |
| CoPx_R | 58.4x 稳 | 9.6x 稳 | 4.4x 稳 |
| CoPy_R | 50.1x 稳 | 8.9x 稳 | **1.2x 不稳** |

**11/12 项稳固,但 CoPy 上的 `cnn > flatten` 不稳**:间距 0.0113,两臂噪声和 0.0091。
方向上两次复现**一致**(0.4242>0.4066、0.4194>0.4081),但**以两次运行无法确立该项**。
→ **2026-08-23 记录的"aggregate > cnn > flatten 在三个通道上无一例外"须改为:
方向在三通道两次复现中一致,但 CoPy 上的 cnn/flatten 之差落在运行间噪声量级内,尚未确立。**
其余结论(ar 胜过全部 map 臂;map_aggregate > cnn;F 与 CoPx 上 cnn > flatten)**margin 4.4x 起,不受影响**。

### 三、方法学教训

**追求逐位复现不如直接测复现性。** 本轮"浪费"的一次作业产出了**噪声底的第一手测量**,
其价值高于原定的 loss 曲线:没有它,一个 1.2x margin 的结论会被当作"无一例外"写进论文。
**建议**:此后凡涉及排序的结论,**至少两个种子**;单次运行的名次差若小于 3× 运行间偏差,记为"未确立"。

## 2026-08-24 — 【d1_map3】map 臂训练曲线:8 轮预算充分;VAL 独立复现 TEST 排序;重尾第三条证据

### 一、本轮设置(与 d1_map2 的唯一差别是保存 checkpoint)

```
CONFIG=configs/opentouch/eval_harness_d1.yaml  FOLDS=4  EPOCHS=8  MODEL=map_all
SAVE_PREDS=runs/preds_d1_map3  SAVE_MODEL=runs/models_d1_map3   (BASELINE_SCOPE 默认 shard)
OUT=docs/opentouch/d1_map3/opentouch_cv4_d1_map3.csv
```
目的**不是**取新指标(指标本应与 d1_map2 相同),而是取此前缺失的 loss 曲线。

### 二、本轮的代码改动

1. **map 三臂保存 checkpoint**(`20ed3e7`)。payload 以 `mnorm` + `baseline_scope` 替代
   prob_gru 的 `fnorm`/`vocab`;两侧均新增 `arm` 字段。
   → **闭合了 2026-08-23 记下的"唯一未闭合的验证缺口"**。
2. **loss 图重构图例**(`6033027`)。原先 3 臂 × 4 折 × (val/train) = **24 条图例**,遮住大半张图,
   且**十二条曲线各占一色使三个臂无法区分**——而区分它们正是该图的目的。
   改为:**多臂时颜色代表臂**、折数同色;图例 = 3 臂 + 线型说明,置于坐标区**之外**。单臂时行为不变。
3. **`CUBLAS_WORKSPACE_CONFIG=:4096:8`**(`76a97bb`),见 8/23 非确定性一节。
4. **主标题列出全部臂的覆盖率**——原先只写字母序第一个(`cnn`),三臂图读起来像是单臂图。

**两个 Claude 的失误(均已修)**:
- `line, = ax.plot(...)` 改为显式颜色后,**漏改三处 `line.get_color()`** → `UnboundLocalError`,
  在 CRC 上崩溃。**编译检查抓不到运行时未定义名。**
- **教训与新习惯**:凡本机无法执行的脚本(需 torch 的),**提交前跑 `pyflakes`**。
  已扫过 `scripts/` 与 `src/opentouch/`:无其他未定义名,仅余未用 import 与死变量。

### 三、分析结果

**R1 8 轮预算对 map 臂充分——d1_map2 排序结论的最后一块支撑到位。**
三臂四折的 val NLL **全部自第 1–2 轮起单调上升**,`*`(min VAL NLL)落在第 1–3 轮,
`+`(min VAL MSE)落在第 1–6 轮;train NLL 持续下降。
→ **过拟合模式与 prob_gru 完全一致**;"三臂在同一 8 轮预算下比较,是否对参数更多的 map 臂不利"
这一疑虑**排除**。

**R2 VAL 独立复现了 TEST 上的排序(此前未检验过)。**
右图 history sweep 的 best VAL NLL(越低越好),逐折均为
**map_aggregate < cnn < flatten**,与 TEST skill 的 aggregate > cnn > flatten **完全一致**。
→ 该排序**不是测试集偶然**,在从未参与选择的 VAL 上同样成立。
(注:CoPy 上 cnn vs flatten 的 1.2x margin 问题依旧,见 8/23;本条支持的是整体排序,不推翻该保留。)

**R3 history sweep 依旧近乎平坦**:1 s 与 3 s 的 best VAL NLL 差异极小,星号多落在 3 s。
与此前各轮一致——**输入历史长度不是瓶颈**(且 ≥2 s 大半是零填充)。

**R4 【重尾的第三条独立证据】**
loss 脚本打印的 `σ/median|err|` **全部高于高斯的 1.48**(F 1.85–1.90、CoP 1.52–1.60),
而 8/23 量得的 `σ/RMSE` **低于 1**(0.78–0.92)。两者相除即得纯粹的尾部统计量:

| RMSE / median&#124;err&#124; | F_R | CoPx_R | CoPy_R |
|---|---|---|---|
| cnn | **2.18** | 1.92 | 1.76 |
| flatten | **2.18** | 1.92 | 1.71 |
| map_aggregate | **2.13** | 2.00 | 1.78 |
| **高斯理论值** | **1.48** | 1.48 | 1.48 |

→ **RMSE 相对典型误差被抬高 1.7–2.2 倍,高斯只有 1.48**:多数误差比高斯预期更小,
少数远大于预期。高斯 MLE 的 σ 被夹在两者之间,故同时表现为"比典型误差宽、比 RMSE 窄"。
**至此三条互不依赖的证据**(σ/RMSE 与覆盖率联立、四架构同一偏离、本条尾部比值)**共同确立:
残差重尾,高斯似然是设定错误的那一部分。**

### 四、下一步

**N9(依据已足,待用户拍板)——Student-t 似然。** 三条证据支持,但 8/21 记下的三个 OPEN QUESTION
**尚未回答**,须先答再动:
  - **Q9.1** t 似然降低离群点权重,而 harness 用 **MSE** 计分、MSE 由离群点主导 →
    **μ 更稳健但 MSE 可能更差。取舍要先定。**
  - **Q9.2** 它改善概率质量而非点预测;论文主张若只涉点预测,收益有限。
  - **Q9.3** 偏离 ActionSense 逐字复刻,须记录为有意偏离,并考虑保留高斯版对照。

**N6(唯一有实质空间的方向)——慢包络分解**,针对长 horizon 缺口 0.15–0.30。
建议形式:**目标仍为 RAW、harness 不变**,仅在模型内部做慢/快分解。**需重开 Q1。**

**N10(新增,零成本,建议先做)——多种子复现。** 8/23 测得运行间偏差 cnn 4.9e-3、flatten 4.2e-3。
**凡涉及排序的结论应至少两个种子**;名次差小于 3× 运行间偏差者记为"未确立"
(当前唯一一例:CoPy 上的 cnn vs flatten)。跑 `SEED=1` 的一轮 map_all 即可定案。

**N11(记录用)——`docs/skill_comparison.md` 已填全**,ActionSense 与 OpenTouch 六模型三通道齐备。

## 2026-08-24续 — 【新臂族 pg_all】probGRU 骨干固定,输入表示为唯一变量 + Hausdorff 指标

### 一、用户指出的问题(成立)

map 三臂的目的是"**只换输入表示、不换模型架构**",但 `tactile_map` 家族用的是**它自己的骨干**:
一次性解码、预测残差、无动作嵌入。故:
- `map_aggregate` vs `cnn` / `flatten`:**只变输入** ✓(家族内比较干净,8/23 结论不受影响)
- `map_aggregate` vs `prob_gru`:**输入与架构同时变** ✗

且这影响实质:**probGRU 在 F 上比 GRU-aggregate 高 0.025**(0.386 vs 0.360,为噪声底的百倍)。
即我们是在一个**比手头最好模型更弱的骨干**上问"map 有没有用"。
**成因是历史的**:ActionSense 本身就把 `tactile_map/` 与 `action_dynamics.py` 分成两套,我们逐字 fork。

### 二、实现:`pg_all` = `prob_gru` + `pg_flatten` + `pg_cnn`

**`ProbGRU` 新增可选 `frame_encoder`**,逐帧作用于输入后再进编码 GRU。
`FlattenEncoder`/`CNNEncoder` 的输出恰为 `(B,t_in,d)`,与编码 GRU 的输入契约一致,**直接对接**。
**自回归解码、动作嵌入、logvar 头、高斯 NLL、早停准则全部不变。**
- `hp["input"] ∈ {raw, flatten, cnn}`;`raw` 即 ActionSense 原样五通道 → **现有 `prob_gru` 就是
  该族的 aggregate 臂**,无需新增。
- `window_set(..., maps=)` 用 map 窗口替换五维特征窗口;**目标、origins、左填充、动作 id 完全不变**。
- map 的基线扣除/log1p 压缩/尺度**复用 `tactile_map` 的实现**(不重写 D1 的逐格中位数);
  MapNorm 由 TRAIN 拟合、TEST 复用,存于 `model.input_norm` 并写入 checkpoint。
- `--baseline-scope` 一并接入该族(默认 `shard`)。

**测试**(置于 `test_opentouch_tactile_map.py`,因**只有该 fixture 生成 `clip_*.npy`**):
编码器选择与非法值报错;**raw 与 map 两路的 `A`/`Ylast`/`Y` 逐位相同、仅 `X` 形状不同**
(这条正是"只有输入不同"的字面断言);cnn 臂可训练可预测;缺 `input_norm` 时拒绝预测。
**本地为 skip(torch 不可用),按 D-TEST 须先在 CRC 跑 pytest。**

### 三、新指标:Hausdorff 距离(用户要求)

`opentouch_report.py` 新增。每个预测视为 (时间, 值) 的**点集**:时间按 h/H 归一到 [0,1],
值除以**该段真值自身的标准差**,故无量纲且逐模型同尺度。报告绝对值与**相对 persistence 的比值**。

**为何值得与 MSE 并列**:MSE 是逐点的,**一条穿过振荡中部的平直预测得分远高于其形状应得**
——而这正是本项目每个臂的行为(8/20)。Hausdorff 问的是"一条曲线最差的点离另一条整体有多远",
**平直预测会被罚到约等于振幅**。

合成验证(振幅 1 的正弦):**完美 0.0000 / 相位偏移 0.3005 / 平直 0.9945**。语义符合设计。

**局限(须与数字同时引用)**:两轴的相对权重是**约定**——时间归一到 [0,1]、值按真值标准差。
绝对数值因此不具独立意义,**可比的是模型之间的比值**。

### 四、下一步

跑 `MODEL=pg_all`,与 `d1_map2/3` 对照即可分离"输入表示"与"骨干"两个因素。

## 2026-08-24续2 — Hausdorff 接入 ActionSense;probGRU 移植的**阻塞点**;AS_forecast 三图溯源

### 一、Hausdorff:实现共享,两侧接入

新建 `src/shape_metrics.py`(**置于 `src/` 顶层,不属于任一侧**)。
`src/opentouch/metrics.py` 是 harness 的**有意分叉**;而 Hausdorff 是相反的情形——
**在两个传感器上计算它的全部意义就是互相比较,定义绝不能漂移**,故单一实现。

**关键性质(已验证)**:该度量对**共同平移不变**。ActionSense 的 tactile_map 目标是
**残差**(相对最后观测值),而绝对值 = 残差 + 同一常数(预测与真值共用),
**故在残差空间计算与在绝对空间计算完全等价,不是近似**。
另:真值在该段为常值时返回 **NaN 而非 0**——否则"无形状可比"会被记为完美匹配,美化每个模型。

合成验证(单位正弦):完美 **0.0000** / 相位偏移 0.4 rad **0.3005** / 平直(均值)**0.9945**。

**接线**:`tactile_map/train.py` 的 `evaluate()` 与 `cross_validate()` 改为**返回 dict**
(此前是 3 元组;**加第四个返回值到位置元组,正是 2026-08-19 崩掉整个作业的那种形状**),
新增 `hausdorff_ch` 与 `hausdorff_ratio_ch`;`scripts/train_tactile_map.py` 打印并写入 CSV。
OpenTouch 侧的 `opentouch_report.py` 改为 import 共享实现。
**这些改动本地无法执行(需 torch),按 D-TEST 须先在 CRC 跑 pytest。**

### 二、AS_forecast_{F,CoPx,CoPy}.png 溯源(用户提问)

**出处**:`scripts/plot_forecast_overlay.py`,用的是 **`action_dynamics.py` 的 probGRU**,
**不是** harness 打分的那套。y 轴标签即写着 "**fast** total force"。
**训练设置**:`--actions Slice,Peel`(仅两类动作)、`--input-mode raw`、`downsample=3`
(**10 Hz**)、`epochs=60`、`cut=0.4`、warmup 5 s、每个 history 长度各训一个模型。

**"为什么预测这么好"——三个原因,都不是模型更强**:
1. **目标是 FAST 分量**(`build_features` 注释原文:"target: always fast")。
   直流与慢漂移已被滤除,剩下围绕零的强振荡,自相关远高于 RAW。
   (对照:OpenTouch 校正后 r(1)=0.318。)
2. **划分是随机 clip 划分,不是按录制/被试/地点留出**。
   `split_train_test` 对 clip 索引做随机置换取 25% 作 test(seed=1)。
   **同一段录制、同一个人、同一动作的相邻片段可以一个进训练、一个进测试。**
   较 OpenTouch 的"整地点留出"宽松得多——**这是最大的差别**。
3. **10 Hz、1 秒 = 10 步自回归**,而 OpenTouch 是 30 Hz / 30 步,误差累积少三分之二。

**已确认无 clip 级泄漏**:所绘 clip 取自 `test_ids`,训练集显式排除。
**但该图不经 harness 打分**,故其观感与 harness 数字(ar 0.200 等)并不矛盾,**也不可相互引用**。

### 三、【阻塞】把 probGRU 移到 ActionSense 的 harness 路径上——需用户决定

**已定**:目标用 **RAW**(用户 2026-08-24 指示,且这是硬约束——**harness 只给 RAW 打分**,
预测 FAST 的臂填不进对照表)。

**阻塞点:动作嵌入的标签从哪来。**
- `action_dynamics.py` 的 probGRU 有 **8 维 action embedding**,其标签来自**每个 clip 一个 label**
  (clip = 一个动作片段),按前缀匹配到 `--actions` 给定的类别。
- 而 harness 路径的 `MapWindows`/`AggWindows` 是**整段录制上的滚动窗口,只产出 (X, Y),没有标签**;
  一段录制内含多个活动,**"每录制一个标签"是错的**。

**OPEN QUESTIONS(须先答再写代码)**
- **Q13.1 动作嵌入怎么办?**
  (a) **去掉嵌入** → 简单,但它就不再是"同一个 probGRU",与 OpenTouch 侧的 `pg_all` 不可比;
  (b) **从活动时间线为每个 origin 取标签** → 保持架构一致,但需确认 harness 数据里有逐帧活动标注;
  (c) 全部置为单一 id(等价于常数嵌入)→ 架构形状保留、信息为零,**最接近 (a) 但改动最小**。
  **Claude 倾向 (b);若数据不支持,则选 (c) 并明确记录该臂缺少动作信息。**
- **Q13.2 预测残差还是绝对值?** OpenTouch 的 `pg_all` 预测**绝对值**;
  ActionSense 的 tactile_map 预测**残差**。**须与 OpenTouch 一致(绝对值),否则两侧的
  pg_all 不是同一个模型。**
- **Q13.3 代码放哪?** 建议在 `src/actionsense/tactile_map/` 内新增 `backbone` 选项
  (`seq2seq` | `probgru`),复用其数据管线与 CV;**不新建平行模块**,避免第三份实现。
- **Q13.4 通道数**:harness 是 **6 通道(双手)**,而 `action_dynamics.ProbGRU` 硬编码 3(单手)。
  OpenTouch 的分叉已参数化 `n_out`,**沿用参数化版本即可**。

### 2026-08-24续3 — 两处需更正的前提;ActionSense 的 smooth/abrupt **尚未分过**

**更正 1(好消息,解除 Q13.1 的阻塞)**:ActionSense 的 harness manifest **每条录制带一个 `label`**
(`eval_harness/dataset.py:38`),`parse_label("Slice a cucumber") -> ("slice","cucumber")`。
即**一条录制 = 一个活动**,与 OpenTouch"一个 clip 一个 action"结构相同。
→ **Q13.1 选 (b) 可行且直接**:动作 id 逐窗口可得,`action_vocab` 可照 OpenTouch 的做法从 TRAIN 建。
**此前"一段录制含多个活动、每录制一个标签是错的"的判断有误,予以更正。**

**更正 2**:**ActionSense 从未做过 smooth/abrupt 分类。**
`src/opentouch/trait.py` 的**rubric 明确写为 sensor-agnostic**,但其 `TRAIT_CLASS` 表
**只覆盖 OpenTouch 的词表**(holding / sliding / wiping / cleaning / scraping / pouring …)。
仓库中 smooth/abrupt 仅出现在 opentouch 前缀的脚本里。
用户记忆中的"之前分过"应指 **OpenTouch 的 G2**,或 `docs/ACTION_CATEGORIES.md` 中的
**另一套分类**(`temporal_pattern` B1–B5 / `action_category` 23 类)——**那不是 smooth/abrupt**。

### 待办与顺序

**A. 动作清单(先跑,零成本)**:新增 `scripts/actionsense_action_inventory.py`,
从 manifest 导出真实词表(动词 × 物体 × 录制数 × 帧数)→ `docs/actionsense/action_inventory.md`。
**理由**:凭文档表格的记忆做审定,正是漏项或臆造词条的来源。

**B. trait 审定(须在任何打分之前提交)**:按 `trait.py` Layer 1 的 rubric 逐个动词判定。
**须记录的诚实限定**:这些动作上**已存在一份免训练可预测性探针结果**
(`docs/actionsense/predictability_by_category*.csv`),
故审定**必须完全立足于物理 rubric**,并且要能被看过那份结果的人认可其独立性。

**C. probGRU 移植(Q13.1=b 已解除阻塞;Q13.2/13.3 仍待确认)**
- **Q13.2 残差 vs 绝对值** → Claude 建议**绝对值**,与 OpenTouch 的 `pg_all` 一致;
  否则两侧同名而不同物,对照表又会出现"同名不可比"的行。
- **Q13.3 代码位置** → 建议在 `src/actionsense/tactile_map/` 内加 `backbone` 选项
  (`seq2seq` | `probgru`),复用其数据管线与 CV,**不建第三份实现**。
- **Q13.4 通道数** → 6(双手),沿用 OpenTouch 已参数化的 `n_out`。

## 2026-08-24 — d256 训练方案设计(PLANNING;按 CLAUDE.md 指令 5,**未实现,等裁定**)

### 请求
在 d256 上做和 OpenTouch / ActionSense 一样的事;一起设计 training 细节;**先从 data 开始,
跑通 probGRU**。

### 一、读 SESSION_LOG + 代码得到的"既有范式"(d256 要对齐的模板)
一个数据集臂 = ①`configs/<ds>/eval_harness.yaml`(冻结协议,sha256 进结果表)
②`data/<ds>_states/` 的物理态缓存 `(T,C,6)` = `[F,xbar,ybar,sxx,syy,sxy]`
(`src/actionsense/physical_state.py::frame_state`,纯解析无学习)
③`src/<ds>/{dataset,splits,baselines,evaluate}.py` ④probGRU 臂。
probGRU 本身(`src/opentouch/prob_gru.py`)= ActionSense `action_dynamics.ProbGRU` **逐字复制**:
action embedding(8)→encoder GRU→**自回归** decoder GRU(用最后一个观测目标做种子)→
mean/logvar 头(logvar clamp [-6,4])、Gaussian NLL、**早停看 VAL NLL**、hidden 48 / epochs 80 /
lr 3e-3 / batch 64。评分 = **skill vs persistence**,须 > 0。

### 二、本轮为设计而做的四项测量(**均为实测,非推断**)

**(1) clip 是步长 1 的滑窗,相邻 clip 共享 15/16 帧。**
判据 `clip c 的第 j 帧 == clip c' 的第 j' 帧 ⟺ c+j == c'+j'`:
**856 个预测相等全部成立,0 失败;1948 个"应当不等"里 0 个假阳性**(signals1/train/S01/0,clip 0–7)。
**⇒ 我在 docs 里先前写的"组内窗口不重叠"是错的**(已改正)。`signals` 看起来不重叠只是因为
它按 mod 3 的另一个余数类抽样,时间跨度其实几乎相同。

**(2) 底层连续录制可精确重建。** 275 个 cell 的 clip 编号**全部**是连续的 `0..N-1`(0 例缺号),
故 `base = clip_0 的 16 帧 + 各 clip_c 的第 15 帧`,长度 `N+15`。
**十路信号全部逐帧验证通过。**

**(3) 于是"真实数据量"是:94 段录制、约 29,836 帧。**
长度 min 36 / p10 58 / **median 201** / p90 787 / max 1068。
**"80,819 clips" 在帧口径上是 2.7× 高估**,在"独立窗口"口径上高估得多得多。
预算表:need 32 帧 → 94/94 录制、26,922 origins;**need 48 → 87/94、25,447 origins**;
need 64 → 81/94、24,080;need 128 → 60/94、19,754。

**(4) `gaze` 是第十路,且按组区分:`signals1`/`signals2` 有,`signals` 没有。**
18 个跨 cell 抽样:14/14(signals1/2)有,0/4(signals)无 —— 干净的组级差异。
**docs §3 的"schema 全同"已改正。**

### 三、由上述测量**被迫确定**(不构成 OQ)的设计
- **语料 = `signals1` 重建出的 94 段录制。** `signals`/`signals2` 不是额外数据,是同一录制的
  3 倍/2 倍抽取;用 harness 既有的 `downsample` 旋钮复现即可(与 ActionSense 的
  `fps_raw 30 / downsample 3` 同一机制)。**这同时彻底消除跨组泄漏。**
- **切分单位 = 录制(subject × class),94 个。** clip 级随机切分会因 15/16 重叠而灾难性泄漏。
  **随附的 train/val 一律弃用**(只覆盖 3/20 类、三组各不同、且 S05 同时在 train)。
- **目标 = 6 维 `[F_L,CoPx_L,CoPy_L,F_R,CoPx_R,CoPy_R]`**:d256 双手套俱全,与 ActionSense
  harness 的 target 定义完全一致,`physical_state.frame_state` 可直接吃 (T,2,32,32)。
- **16 帧不再是约束**:重建后 median 201 帧,rolling-origin 有真实历史可用。

### 四、OPEN QUESTIONS(**阻塞实现,等裁定**)

**OQ-D1 fps 未知,直接卡住 `horizon_s` 与 `causal_velocity`。**
数据无时间戳,只知相对步长 1:2:3。三条路:
 (a) **与真实 ActionSense 对齐反推**——仓库已有 `data/actionsense_states`(30 Hz,401 clip)。
     d256 源自同一录制,可用互相关把某段 d256 tactile 对到 ActionSense 的对应段,**直接测出 fps**。
     代价:数值被重缩放过,需用形状/相位而非幅度对齐。**我推荐先试这条:它是唯一能给出真值的。**
 (b) 假定 = ActionSense 原生 30 Hz,horizon 按 1 s = 30 帧(与 OpenTouch 口径一致)。
 (c) 放弃物理秒,**horizon 直接用帧**(如 16 帧),配置里明写 fps 未知。
**这个不定下来,`prob_gru` 的 `causal_velocity(sig, fps)` 就是错的量纲。**

**OQ-D2 是否做基线扣除(OpenTouch D1 的同题)。**
d256 tactile 有明显 DC 台座(实测 floor 0.13 / 0.40,远高于 0)。
`physical_state.baseline_correct(pct=5)` 现成。但 OpenTouch 那轮的结论是"D1 放弃"(2026-08-16续5),
理由与该传感器工作在满量程 95% 有关 —— **d256 是另一个传感器(32×32 手套,已被重缩放到 [0,1]),
不能照搬结论**。选项:(a) 不扣,先跑通(与 OpenTouch 最终口径一致);(b) 扣,并同时跑两臂对比。

**OQ-D3 切分协议。**
 (a) **留一受试者(LOSO,5 折)**:最干净,直接回答跨人泛化;每折 held out ~19 段录制。
 (b) 按录制分层切 6/2/2:样本更多但不测跨人。
 5 受试者 × 20 类 = 100,实有 94 段 ⇒ 每类每受试者约 1 段,**(a) 下每类训练只剩 ~4 段录制**。
 我倾向 **(a)**:与"和 OpenTouch/ActionSense 做一样的事"最贴合,且 d256 唯一充裕的轴就是受试者。

**OQ-D4 probGRU 的输入用哪些流。**
 (a) **与 ActionSense/OpenTouch 逐字一致**:`[F, CoPx, CoPy, vx, vy]`(仅 tactile 派生),
     **这样三个数据集的数字可直接对比**——我推荐先做这个,作为"跑通"的定义。
 (b) 扩展用上 d256 独有的 EMG/pose/gaze。这是新科学问题,**建议作为第二臂**,不要混进第一次跑通。

**OQ-D5 action embedding 的 id 用什么。**
 OpenTouch 用 action 字段(长尾,<min_group_size 并入 "other")。d256 的天然候选是 **20 个 class**,
 但 **class 同时是 LOSO 里唯一的类别轴**,把它喂进 embedding 等于把标签给模型。
 选项:(a) 全部同一 id(等于关掉 embedding);(b) 用 20 类 id,并明确这是 "label-conditioned forecasting"。
 **这个必须先想清楚,否则 skill 数字的含义会被悄悄改变。**

### 五、拟定执行顺序(裁定后)
1. `scripts/extract_d256_states.py`:94 段重建 → `frame_state` → `data/d256_states/state_*.npy` + manifest。
2. `configs/d256/eval_harness.yaml`(冻结,含 OQ-D1/D2/D3 的裁定值)。
3. `src/d256/splits.py`(按录制/LOSO)+ `dataset.py`(Norm、origins、eligible)。
4. persistence / seasonal / AR 基线 → 拿到 skill 的分母。
5. probGRU 臂,复用 `src/opentouch/prob_gru.py` 的结构。
**第 4 步不先做完,probGRU 的 skill 无意义 —— 这是 OpenTouch 那条链已经踩实的顺序。**

### 2026-08-24续 — 【OQ-D1 实测结案:fps = 6 Hz】【OQ-D3 裁定:LOSO】

**用户反馈**:(i) 没看懂 fps 那个问题;(ii) 问 20 个 class 分别是什么;(iii) **切分选 LOSO(5 折)**。

#### OQ-D1 —— 我没有让用户选,而是把它测出来了(走的是选项 (a))
**方法**:`data/actionsense_states/manifest.jsonl` 里每段真实 ActionSense 录制带
`label`(与 d256 的 `label_text` 逐字相同)、`fps: 30`、长度 `T`。d256 每段录制长度
`L = N_clips + 15`(由滑窗重建式给出)。若 d256 是 ActionSense 的时间抽取,则 `T/L = k` 应为常数。

**第一轮按类求和,比值 4.89~26.08,看似否定假设。诊断出这是我的配对错误**:
多数类 ActionSense 有 15 段而 d256 只有 5 段,"降序取前 k 段"配错了对象。

**只取录制条数 1:1 无歧义的三类**(Load / Stack / Unload dishwasher,共 **15 段独立录制**):
```
ratios 4.92 4.81 4.83 4.98 4.95 | 4.95 5.13 4.97 4.88 4.94 | 5.07 4.89 5.02 4.86 5.00
median 4.948   mean 4.948 ± 0.085   range [4.81, 5.13]
```
**⇒ k = 5,`fps(signals1) = 30/5 = 6 Hz`。** 派生:`signals2` = 3 Hz,`signals` = 2 Hz。
自洽性:16 帧 @6 Hz = **2.67 s**,与配对的 16 帧视频同跨度,对一个动作识别数据集是合理的 clip 长度。

**这条把量纲问题全部解开**:`causal_velocity(sig, fps=6)` 量纲正确;
horizon 1 s = **6 帧**,与 OpenTouch(1 s=30 帧@30Hz)、ActionSense(1 s=10 帧@10Hz)
**是同一物理口径,三个数据集的 skill 可直接对比**。

**重新表述的数据规模(6 Hz 口径)**:94 段录制,median **34 s**,p10 10 s,max 178 s,
合计 **83 分钟**。预算 `min_history 24(=4 s,与 ActionSense 同秒数)+ horizon 6(=1 s)= 30 帧`
⇒ **94/94 段录制全部存活**。16 帧的表观限制至此完全消失。

**证据强度声明**:这是**由长度比值反推**,不是逐帧对齐的直接测量。15 段独立录制、±1.7% 一致,
已足以据此冻结配置;若日后需要更硬的证据,可用 d256 的 F(t) 与 ActionSense state 的 F(t)
做互相关(数值被重缩放过,须按形状/相位而非幅度对齐)。**配置里会记下这是推断值及其依据。**

#### OQ-D3 裁定:**LOSO 留一受试者,5 折**。每折 held out ~19 段录制,每类每折约 4 段训练录制。

#### 仍未决:OQ-D2(基线扣除)、OQ-D4(输入流)、OQ-D5(action embedding)
用户问 20 类内容即为决 OQ-D5 做准备 —— 类名见 `docs/d256.md` §5(ActionSense 厨房活动原文)。

### 2026-08-24续2 — 【裁定齐】【data 步落地:`scripts/extract_d256_states.py`】【发现 cell ≠ 录制】

#### 裁定汇总
- **OQ-D1 fps = 6 Hz**(实测,见上)。**OQ-D3 = LOSO 5 折**。
- **OQ-D2 = 不做基线扣除**,与 OpenTouch 最终口径一致。
- **OQ-D5 = 两臂消融**:arm A `action_id ≡ 0`(信号自预测,可与另两个数据集横比),
  arm B `action_id = label_idx`(label-conditioned)。**差值即"知道活动标签值多少 skill"。**
- **OQ-D4(未单独问,取推荐值)= 与 ActionSense/OpenTouch 逐字一致的 `[F, CoPx, CoPy, vx, vy]`**,
  仅 tactile 派生。EMG/pose/gaze 作为后续第三臂;`--aux` 已把这些流一并落盘,**避免日后再扫一遍语料**。

#### 新增 `scripts/extract_d256_states.py`
cell → 滑窗折回 → `frame_state` → `(T,2,6)` = `[F,xbar,ybar,sxx,syy,sxy]`/手,
输出与 `data/actionsense_states/` **同布局**(frozen harness 可直接吃),**并多一个 `subject` 字段**
—— ActionSense manifest 恰恰缺这个(见其 config 注释 "the manifest has no subject field"),
**这正是 d256 能做 LOSO 而 ActionSense 做不了的原因**。

#### 【重要修正】cell **不是**一段录制,而是多段拼接
写完第一版后本地实跑,**验证器立刻在 `signals1/train/S04/13` 的 clip 15 处报错**。
诊断:逐对求位移,`c→c+1` 的位移**全是 −1,唯独 14→15 断开**。
⇒ 段 A = clip 0–14(30 帧)、段 B = clip 15–20(21 帧),**30+21 = 51,恰等于该 cell 实测的
51 个不同帧**。

**成因**:ActionSense 每个受试者对同一活动录了**多次**(如 "Clean a pan with a towel" 有 15 段,
而 d256 只有 5 个 cell)⇒ **d256 把多段的窗口并进了同一个连续编号的目录**。

**⇒ 录制的单位是 SEGMENT,不是 cell。判据 `clip_c[j+1] == clip_{c+1}[j]`。**
**这个如果搞错,不是少拿数据,而是把不相干的时刻拼成一条时间序列,之后所有 forecast 都在拟合假象。**
已改为按段切分;段内仍做**全流、全帧**的窗口核验(`_continues` 只看一路一帧的重叠,不足以排除部分失步)。

**⇒ 先前"94 段录制"作废**:那是 cell 数。真实段数更多、单段更短。
**真值须等 CRC 全量抽取后才知道**,而 `min_history`/`horizon` 预算依赖它,
**故 `configs/d256/eval_harness.yaml` 现在不写** —— 先拿到长度分布再冻结,顺序不能倒。

**本地实跑**(2 个 cell,均为最小的那几个,不代表全局):3 段、90 帧;
预算表已由脚本自动打印(16/24/30/40/64 帧各保留多少段与 origins)。

#### 下一步(待用户在 CRC 执行)
```
python scripts/extract_d256_states.py --root ~/forcevision --out data/d256_states --aux
```
产出真实的段数与长度分布 → 据此冻结 harness config → splits(LOSO)→ 基线 → probGRU。
**基线不先跑完,probGRU 的 skill 没有分母。**

### 2026-08-24续3 — scripts/ 按数据集分组(落地了 REPO_ORGANIZATION.md 里搁置已久的那一项)

**请求**:把 scripts/ 按 actionsense / opentouch / d256 三个数据集整理;不属于这三个的、或多数据集
共用的,留在外面并问用户。

**结果**:`scripts/{actionsense(18), opentouch(15), d256(2)}/`,顶层留 15 个待裁定。

**注意**:`docs/REPO_ORGANIZATION.md` 149–153 与 170–172 行**早就把这次重构列为 "NOT moved
(intentional)"**,并精确点名了两个障碍。两个都确实存在,都已处理:
1. **`sys.path` 深度**:23 个脚本用 `dirname(dirname(abspath(__file__)))` 取仓库根,下沉一层后
   会指到 `scripts/`。已全部补一层。**验证方式不是 grep**(3 层写法包含 2 层子串,grep 会给假阳性),
   而是**逐个跑 `--help`:35/35 全部 import 通过**。
2. **调用引用**:重写 **101 处 / 50 个文件**(docs、CRC `.job`、`src/`、`tests/`)。
   另有两处**跨脚本 import**(`scripts.train_action_dynamics`、`scripts.plot_opentouch_fcop`)
   也必须改,这类 grep 路径字符串是抓不到的。

**两处刻意不动:**
- **`SESSION_LOG.md` 不改写。** 它记录的是"当时实际跑了什么命令",改路径等于伪造记录。
  译码规则与完整清单写进新建的 `scripts/README.md`。
- **`configs/opentouch/eval_harness_d1.yaml` 已回滚。** 批量重写把它第一行注释里的脚本路径改了
  —— 而 `config_hash = sha256(整个文件字节)` 会被**盖进结果表用于溯源**,改注释就等于换了协议实例、
  让既有结果行对不上配置。**改完才意识到,已 `git checkout` 撤回并复核三个 harness config 的哈希未变**
  (`001dcee8e81efda3` / `916820c096c7666a` / `947e650076742574`)。
  **教训:批量文本重写必须先排除按字节哈希的冻结文件。**

**核验**:`pytest` **66 passed / 4 skipped**;35/35 脚本 import 通过;
移动**未产生任何新的悬空引用**(残留的全在 SESSION_LOG,外加 4 个本就不存在的旧引用:
`download_data.sh`、`predictability_by_category.py`、`run_inference_mano.sh`、`scripts/X.py` 占位符)。

**不属于我的改动**:`docs/actionsense/forecaster_comparison.png` 在工作区有修改(有效 PNG,
1080×660),非本次重写所致(重写只碰 .md/.yaml/.job/.sh/.py),**未 stage、未提交**。

**顶层剩下的 15 个,分三类,待用户裁定**(见 `scripts/README.md`):
EgoTouch(第四个数据集,5 个)、真正跨数据集(`build_skill_comparison.py`、`aggregate_results.py`、
`check_leakage.py`)、上游 TouchAnything 仓库继承的(7 个)。

### 2026-08-24续4 — ActionSense 的 trait 表(预注册,提交于任何按类打分之前)

**词表实测**(`scripts/actionsense/actionsense_action_inventory.py`):**299 条录制、14 个动词**,
**无长尾**(对照 OpenTouch 的 66 个动作串)。全部 14 个已审定,`unaudited()` 为空。

**新建 `src/actionsense/trait.py`**,**不重述 rubric**——rubric 只有一份,在
`src/opentouch/trait.py`(Layer 1 + R1/R2),那里已写明其为 sensor-agnostic;
本文件只承载 ActionSense 的**词表**。**重述一遍 rubric 等于制造第二份定义,而跨传感器比较恰恰经不起两份定义。**
从 opentouch 侧只引入 `SMOOTH`/`ABRUPT`/`UnauditedAction`/`normalize_action`。

**判定结果**:
- **SMOOTH(218 条)**:clean 60、slice 45、peel 30、spread 30、clear 28、pour 25
- **ABRUPT(81 条)**:get 30、get/replace 15、open/close 9、open 6、set 6、stack 5、load 5、unload 5

**用户裁定 [U](2026-08-24)**:`peel` / `slice` / `clear` 判为 **SMOOTH**。
其余 11 项为 [R](由 rubric 推出)。

**【必须与任何 G2 结果同时引用的跨传感器分歧】**
`slice` 与 `clear` 在此为 SMOOTH,而 OpenTouch 表中结构对应的 `cutting` 与 `scooping` 为 **ABRUPT**。
**rubric 的 R1 恰以切菜为示例**:"a knife striking the cutting board … **is what forces `cutting`
-> abrupt**"。故该分歧**真实存在,且不是 R1 的字面应用**;它成立是因为它是用户的明确裁定,
本文件按惯例**逐字记录用户裁定而不再推导**。
**处置**:二者均置入 **CONTENTIOUS**,由 Layer 3 的敏感性分析覆盖——
主结果照裁定报告,同时报告剔除后重算的结果;**方向不变则结论不依赖二者如何归类**。
**在此分歧存续期间,smooth/abrupt 对比在两个传感器上含义不同,任何跨传感器引用都必须声明这一点。**

**CONTENTIOUS 集(5 项,118 条录制)**:`slice`、`clear`、`peel`、`open`、`open/close`。
- `peel`:每刀的"接触面突变"与连续力调制两可,且部分被试用离散刀法;
- `open` / `open/close`:旋开可为持续旋转(rubric 已将 `twisting`/`unscrewing` 列为争议)。

**剔除 CONTENTIOUS 后仍为 smooth 115 / abrupt 66**,两类均非空,**敏感性分析可算**。
(对照 OpenTouch:299 vs 2544,极不平衡;ActionSense 这边**统计功效好得多**。)

**测试**(`tests/test_actionsense_trait.py`,6 项,本地可跑、已通过):覆盖完整性、未知动词报错而非默认、
两侧共用同一对类名常量、跨传感器分歧被 CONTENTIOUS 覆盖、`partition` 的丢弃可计数、
剔除争议后两类仍非空。

### 2026-08-24续5 — ActionSense 的 probGRU 臂:实现(Q13.2/13.3 已定)

**用户裁定**:Q13.2 = **绝对值**;Q13.3 = **放进 `src/actionsense/tactile_map/`,加 `backbone` 选项**。

**实现要点(骨干与 OpenTouch 逐条一致,见 `docs/model_comparability.md` §1)**
- `models.py` 新增 **`ProbGRU`**:8 维动作嵌入 → 编码 GRU → **自回归解码**(以最后观测值播种、均值喂回)
  → mu/logvar(截断 [-6,4])→ 高斯 NLL。`n_out` **参数化**(6 通道),`frame_encoder` 可插。
  `build_model(..., backbone=)` 二选一;非法值抛错。**未新建第四份实现。**
- `data.py` 新增 `verbs_of` / `action_vocab` / `aid_of`:动词取自 manifest label 的首词
  (`parse_label`),**词表仅由该折的 TRAIN 建**,TRAIN 中过稀或测试期未见者折入 `other`(id 0)
  ——与 `src/opentouch/prob_gru.py` 同一纪律。
- 两个 Dataset **恒定返回四元组**(window, action id, last observed, target)。
  **元素个数随参数变化正是 2026-08-19 崩掉整个作业的形状**;Seq2Seq 直接忽略中间两个。
  新增 `residual` 开关:Seq2Seq 保持残差目标,probGRU 用**绝对值**。
- `train.py` 的两骨干差异**收敛到一个函数 `_call`**,训练循环/验证/预测各保持单份。
- **`_predict` 现在返回 persistence 参照,而不是假定为 0**:残差空间下它是 0,绝对空间下是
  最后观测值沿 horizon 重复。**硬编码 0 会让 probGRU 臂被拿错参照打分**——这是本次最易犯且最隐蔽的错。
- `evaluate` 的 skill、覆盖率、Hausdorff 全部改用该参照,故对两种骨干都成立。
- 驱动 `scripts/actionsense/train_tactile_map.py` 新增 `--backbone`。

**测试**:更新既有 4 个按 2 元组解包的用例;新增
`test_probgru_backbone_matches_the_opentouch_one`(三编码器 × 形状/clamp/嵌入维度/解码器种子维度)、
`test_action_vocab_is_built_from_train_only`、以及残差与绝对值目标相差恰为最后观测值的断言。
**本地全部 skip(需 torch),按 D-TEST 须先在 CRC 跑 pytest。**

**尚未做**:该臂的 Hausdorff/skill 已可算,但**尚未跑**;按用户指示**先只跑 aggregate(probGRU),
再加 cnn 与 flatten**。

### 2026-08-25 — scripts/ 分组第二轮:egotouch / shared 落地,顶层清空至 8 项

**裁定**:(1) 建 `scripts/egotouch/`(5 个);(2) 建 `scripts/shared/`,但
**`check_leakage.py` 按实现归 `actionsense/`**;(3) 上游 TouchAnything 的 7 个留顶层不动。

**执行**:`egotouch 5` / `shared 2`(`build_skill_comparison.py`、`aggregate_results.py`)/
`actionsense 20`。5 个脚本补 `sys.path` 深度;重写 **23 处引用 / 16 个文件**。
校验:**42/43 脚本 import 通过**,唯一失败是 `download_egotouch.py` 缺 `huggingface_hub`
——**该脚本不 import `src`,故是纯环境缺依赖,与移动无关**。`pytest 72 passed / 4 skipped`。

**`check_leakage.py` 归属的理由记录在案**:它意图通用,但断言跑的是 `src.actionsense` 与
`data/actionsense_states`,**只能对一个臂运行**。放进 `shared/` 会对外宣称一种它并不具备的通用性。
日后若真做成数据集无关的,再移。

**并发情况**:本轮开始前树里已有他处推入的 4 个 commit(`0c837df`/`3559577`/`9f4ff7d`/`5c623e1`,
ActionSense probGRU backbone 与 trait 表),测试数 66→72 即由此而来。
本地与 `origin/main` 同步无分叉;**引用重写在这些 commit 之后执行,故已覆盖它们**。

**【必须保持"错误"的一条路径】**
`configs/opentouch/eval_harness_d1.yaml` 第 1 行注释仍写着
`scripts/opentouch_apply_baseline.py`(现已移动)。**刻意不改**:该文件的 `config_hash`
= sha256(全文件字节),会盖进结果表做溯源,**改注释即让既有结果行与其配置脱钩**。
本轮的批量重写脚本已把三个 frozen config **写进硬编码排除集**,不再依赖"我记得别碰"。
理由与三个哈希值已写入 `scripts/README.md`,防止后人"顺手修好"而静默破坏溯源。

**最终布局**:`scripts/{actionsense 20, opentouch 15, d256 2, egotouch 5, shared 2}`
+ `crc/`(按集群而非数据集组织,不动)+ 上游的 `core/ data_processing/ tools/ utils/` 与顶层 8 项。

## 2026-08-25 — 【ActionSense probGRU aggregate】骨干比较**未跨传感器复现**;但存在一处未堵的归因漏洞

运行:`BACKBONE=probgru ENCODERS=aggregate EPOCHS=60 FOLDS=5`,5 折按录制交叉验证。
产出:`docs/actionsense/probgru_agg/`。

### 一、结果:probGRU 在 ActionSense 上**更差**,方向与 OpenTouch 相反

| history | Seq2Seq(残差) | probGRU(绝对值) | 差 |
|---|---|---|---|
| 1 s | 0.1202 | 0.0576 | **−0.063** |
| 3 s | 0.1379 | 0.0701 | **−0.068** |
| 10 s | 0.1422 | 0.0702 | **−0.072** |

**六通道中五个更差**(唯一例外:hist=1 的 F_L,+0.012),三个 history 一致。
对照 **OpenTouch:probGRU 比 Seq2Seq 骨干高 +0.025(F)**。→ **骨干优劣未跨传感器复现。**

### 二、【必须先堵的漏洞】Seq2Seq 那一列来自旧代码

上表的 Seq2Seq 数字取自 `tactile_map_cv_results_aggregate.csv`,**写于 `_predict` 被修改之前**。
本次改动让 `_predict` **返回** persistence 参照,而非假定为 0;残差空间下该参照**应当**仍为 0,
**但这一点从未被验证**。
→ **−0.07 中有多少来自骨干、多少来自 Claude 的改动,目前无法区分。**
**处置**:已提交 `seq2seq_agg_recheck`(旧骨干 + 新代码,其余全同)。
**它必须复现 0.1202 / 0.1379 / 0.1422;不复现则先查改动,上表的比较作废。**

### 三、模型未坏,另有两条线索

**覆盖率 93.8–94.1%**(名义 95.4%)、**Hausdorff 比 persistence 好 10%(0.90x)**
——**产出的是校准合理的预测,只是 MSE skill 低**。不像实现错误。

**训练不足是真实可能**:绝对值头**没有 persistence 先验**(残差头预测 0 即等于 persistence)。
冒烟 2 轮时 meanSkill 为 **−0.003**,60 轮为 **0.058–0.070**,**可能仍在爬**。
**已提交 `probgru_agg_e150`(150 轮)作为判据**:
明显更高 → 60 轮对该骨干不公平;基本不动 → 骨干在 ActionSense 上确实更差,那才是结论。

### 四、Hausdorff 目前只有这一个模型有

Seq2Seq 的旧 CSV 与 OpenTouch 的报表均**早于** Hausdorff 的加入。
`seq2seq_agg_recheck` 会顺带补上 ActionSense 侧;**OpenTouch 侧只需重跑报表(不占 GPU)**:
`opentouch_report.py --preds runs/preds_d1_map2 ...`。**在补齐之前,Hausdorff 不可跨模型比较。**

### 五、docs/actionsense/ 改为按 run 分目录(约定已写入 docs/README.md)

`probgru_agg/`、`seq2seq_agg_recheck/`、`probgru_agg_e150/`。
**原有 34 个文件保持原位不动**——它们来自哪一次运行已不可考,
**按猜测归档等于给数字安上错误的出处,比没有出处更糟**。

### 六、附带发现:仓库里跟踪了约 2.6 GB 数据

`data/actionsense_states/` 下 **399 个 `.npy` 被 git 跟踪**(本机每个约 6.6 MB)。
CRC 上它们是指向真实数据的软链接,故每次 `git status` 都显示为 **typechange (`T`)**,
并反复阻塞 `pull --rebase`。**这是长期存在的仓库卫生问题,非本轮造成。**
**处置**:`git pull --rebase --autostash` 可绕过;
**根治需 `git rm --cached` + `.gitignore`(如 `data/opentouch_states` 已做),
但那会在提交里混入 400 个删除,应单独成一次清理提交,不与结果提交混在一起。**

### 2026-08-25续 — 【CRC 抽取结果】166 段 / 30,916 帧;`min_history=24` 经数据检验站得住

**用户实跑输出**:`166 recordings from 94 cells, 30916 frames (85.9 min)`,
长度 min 16 / p10 22 / **median 78** / p90 657 / max 1068。

**用户问"一共 163 条?" —— 是 166**(索引 0–165)。且它们是**录制段**,不是 clip;
clip 有 80,819 个,166 是折回来的段数。

**一个很硬的自洽检验(段切分模型成立的独立证据)**:
先前按 cell 估计 29,836 帧;166 段比 94 个 cell 多 72 段,每段重建多带 15 帧
⇒ 29,836 + 15×72 = **30,916**,与实际输出**完全相等**。
另:94 个 cell 出 166 段 ⇒ **72 个 cell 含多段**,与"ActionSense 每受试者对同一活动录多次"一致。

**`eval.min_history` 由 provisional 转为已检验**:预算表
`12→160/166`、`18→148`、**`24→138/166(83%)、26,310 origins`**、`30→128`、`40→117`。
24 帧 = 4 s = 与 ActionSense 同物理历史,**代价仅 17% 的段**,故**维持 24,不启用 2 s 回退**。
config 不改 ⇒ `config_hash = 2ec2ba37fa8cbb0b` 保持。

#### 本轮落地的代码
- **`src/d256.py` → `src/d256/` 包**:`raw.py`(clip 级读取)+ `__init__.py` 重导出,
  故 `from src import d256; d256.load_clip(...)` **保持可用**,两个 scripts 无需改。
- **`configs/d256/eval_harness.yaml`**(hash `2ec2ba37fa8cbb0b`):6 维双手目标、
  `fps_raw 6.0 / downsample 1 / horizon_s 1.0 → 6 步`、mask pct 5、
  `fit_scope: label_idx`、`ar_orders [2,3,6,9,12,18]`(= ActionSense 的 0.2–3.0 s 在 6 Hz 下的**同物理秒数**)、
  `protocol: loso`。**每个字段都写明是被传感器强制、由 d256 自测、还是为跨臂可比而保持同物理量。**
- **`src/d256/dataset.py`**:`load_target`/`Norm`/`force_thresholds` **直接复用 ActionSense harness**
  ——d256 的 state 与它**同形 `(T,2,6)`**,复用而非分叉(与 `opentouch/dataset.py` 同策略)。
  自有的只有两处:`group_keys` 直读 `label_idx`(无需 ActionSense 的 `parse_label` 字符串解析,
  因为**目录名即类别**);`eligible_recordings` 作用在**段**上。另加 `budget_table()`。
- **`src/d256/splits.py`**:LOSO。**VAL 只从 TRAIN 受试者里取** —— 早停读 VAL,
  若 VAL 来自 held-out 受试者,等于**用被测者本人挑 checkpoint**。
  VAL 按类分层且**永不取走某类最后一段**(否则该类 AR 组无 fit)。
  折内直接断言 `missing_groups` 为空,让失败发生在建折时并指名是哪一折,
  而不是深埋在 `baselines/ar.py` 的 KeyError。
  `load()` 校验 `config_hash`,协议一变就拒绝加载旧 split。
- **`tests/test_d256_splits.py`**:fixture 用**真实的 166 段几何**(受试者/类别/长度全部照抄实跑输出),
  故真实数据上退化的折在测试里同样退化。7 项断言:几何自洽(166 段/30,916 帧)、
  138 段存活、三分区互不相交、**held-out 受试者不出现在 train 或 val**、
  每折 TRAIN 覆盖 20 类、AR 组全被 fit、split 文件在 config 变更后拒绝加载。

**实测 LOSO 分布**(真实几何):5 折 TRAIN 均覆盖 **20/20 类**;
TEST 类数 15–20(S03 仅 15,因其本就缺 1/3/4/5/6 类)。`pytest 79 passed`。

**下一步**:probGRU 两臂(A `action_id≡0` / B `action_id=label_idx`)+ 基线。
**基线先行**——没有 persistence/seasonal/AR 就没有 skill 的分母。

## 2026-08-25续 — 【d1_pg】固定 probGRU 骨干、只换输入;并**推翻 Claude 关于骨干优劣的说法**

### 一、对照组逐位通过

`prob_gru`(raw 输入)与 `d1` 那轮**差 0.0000**(三通道)。→ 为加 `frame_encoder` 而改动 `ProbGRU`
**未波及 raw 路径**,本轮数字可信。

### 二、固定骨干、只换输入(用户要求的干净比较)——排序与 Seq2Seq 骨干下一致

| 通道 | raw(聚合) | pg_cnn | pg_flatten |
|---|---|---|---|
| F_R | **0.3862** | 0.3561 | 0.3219 |
| CoPx_R | **0.4271** | 0.4143 | 0.3925 |
| CoPy_R | **0.4722** | 0.4304 | 0.4464 |

**聚合 > cnn > flatten,三通道无一例外**,与 `d1_map2`(Seq2Seq 骨干)下的排序相同。
→ **"原始触觉图不带来 F/CoP 之外的可预测信息"现已在两种骨干上各自成立**,
不再依赖单一架构——这正是用户提出该实验时的疑虑。

### 三、【更正】Claude 此前"probGRU 在 OpenTouch 三臂全胜"的说法**不成立**

该说法只在**逐帧池化**口径下为真。**逐 clip 等权口径下,F 通道三臂全部反号**:

| 输入 | 通道 | 逐帧池化 | 逐clip等权 | 一致? |
|---|---|---|---|---|
| aggregate | F_R | +0.0266 | **−0.0274** | **否** |
| cnn | F_R | +0.0227 | **−0.0256** | **否** |
| flatten | F_R | +0.0492 | **−0.0146** | **否** |
| (三臂的 CoPx / CoPy) | | 全部为正 | 全部为正 | 是 |

**成因**:驱动**按帧池化**,长 clip 与高方差 clip 主导;报表**按 clip 等权**。
即 **probGRU 在"帧多的地方"更好,在"典型 clip"上更差**。
→ **任何关于"哪个骨干在 F 上更好"的陈述必须指明口径**;不指明的陈述(包括 Claude 之前那条)
**不被数据支持**。已写入 `docs/skill_comparison.md` 的专门一节。

### 四、【最有价值】Hausdorff 把排序整个翻转:`seasonal` 由最差变最好

| 模型 | MSE skill(F,逐clip) | **Hausdorff(F)** |
|---|---|---|
| **seasonal** | **−0.087(唯一输给 persistence)** | **2.699 / 0.82x(全场最好)** |
| ar | 0.302(最好) | 2.766 / 0.84x |
| prob_gru | 0.291 | 2.883 / 0.87x |
| pg_cnn | 0.268 | 2.975 / 0.90x |
| pg_flatten | 0.224 | 3.013 / 0.91x |
| persistence | 0 | 3.301 / 1.00x |

**三个通道均如此。** 成因:seasonal 输出**周期波形而非平线**;相位大概率错(故 MSE 差),
但**取值范围与形状与真值相当**,而 Hausdorff 是集合间距离,对相位错配远比对"整条压平"宽容。
→ **这是自 2026-08-20 起反复观察到的"所有神经臂都在预测局部均值"的正交证据**:
**唯一产生正确形状的模型,恰是点误差最差的那个。**
**稳健性**:神经三臂之间的排序在两个口径下**一致**(`prob_gru > pg_cnn > pg_flatten`),
故"输入表示"那条结论不依赖指标选择。

### 五、pg_flatten 是最过度自信的一臂

覆盖率 92.4–95.0%(三臂最低),σ/RMSE 0.76–0.88(最低),skill 亦最低。
→ **输入表示越差,模型越不知道自己不知道。**

### 六、报表末尾那句 D1 警告是错的(已修)

原为硬编码:"F is ~99.3% DC (**D1 declined**) … read every number as how well the constant was
reproduced"。**但 `d1` / `d1_mse` / `d1_map2` / `d1_pg` 四份报表跑的正是扣除直流后的数据**,
该句把读者引向相反结论。
**改为实测**:D1 校正后**实测 F 仍有 94.9% 是直流**(未校正时 ~99.3%),
即**变异份额由 0.7% 升至 5.1%,约 7 倍,但直流并未被"去掉"**——接触力本身恒为正。
新文案**不设阈值**(94.9% 处两种说法同时为真,二选一必然误导),同时说明:
绝对误差仍由复现水平主导、故 MSE/R² 跨目标不可比,而 **skill 因分子分母共享该水平而可比**。

### 七、docs/skill_comparison.md 已扩充

新增 `d1_pg` 一列、`probGRU + CNN` / `probGRU + flatten` 两行、**Hausdorff 专节**、
以及**两种 skill 口径分歧**专节。生成器路径已随 `scripts/` 重组移至
`scripts/shared/build_skill_comparison.py`。
(途中 Claude 用循环变量 `a` 遮蔽了 argparse 的 `a`,pyflakes 无法察觉——两个名字都有定义;
由运行时报错暴露,已改名并注明。)

### 2026-08-25续2 — 【概念澄清】clip vs 录制段;【代码】evaluate + probGRU 两臂落地

#### clip 与"录制段"的区别(用户提问,写死在此以免后续再混)
- **clip** = d256 **实际发行的文件**,16 帧,**80,819 个**;是**步长 1 的滑窗**,
  clip c = 第 c..c+15 帧,**相邻共享 15/16 帧**。
- **录制段(segment / recording)** = 底层真实连续录制,由 clip 折回,**166 段 / 30,916 帧**,
  彼此**时间不重叠**。
- **cell** = 目录 `<group>/<split>/<subject>/<session>`,94 个;**一个 cell 可含多段**
  (72/94 确实如此),因为 ActionSense 每人对同一活动录了多次。
- 类比:把 30,916 帧的录像**每一个可能的 16 帧窗口**各存一份 → 8 万个文件、仍是 3 万帧内容。
- **用途分工**:clip = 发行格式(**不可当独立样本**);段 = `state_N.npy` 的单位、splits 的单位、
  rolling origin 的滚动范围。**按 clip 随机切分 = 两侧是共享 15/16 帧的近重复 ⇒ skill 虚高。**

#### 新增 `src/d256/evaluate.py`
复用 ActionSense harness 的 baselines / masking / metrics / rolling-origin batcher
(d256 目标与之同形,**无一处是数据集特有的**),本文件只加协议:
- **5 折而非单一切分**:每个受试者轮流 held out,每个基线拟合 5 次,数字报**折间均值±标准差**。
  单切分会掩盖"折 2(S03)只在 15 类上测,其余在 20 类上测"这一事实。
- **Norm / force threshold / AR 系数 / seasonal 周期全部按折从 TRAIN 重算**。
  **不是整洁问题**:全局 Norm 会把 held-out 受试者的尺度泄进它自己的测试分。
- `score_external()`:让 probGRU 在**与基线完全相同的 origins / mask / metrics** 上被打分。
  **形状即契约**——预测张量必须是 `(n_origins, H, 6)` 且按 `origins()` 顺序,
  否则当场报错。已用负例测试验证:错形状与缺录制**都被拒绝**。

#### 新增 `src/d256/prob_gru.py` + `scripts/d256/train_prob_gru.py`
架构与损失**逐字复制** ActionSense(embedding 8 → encoder GRU → 自回归 decoder GRU
以最后观测目标为种子 → mu/logvar 头、logvar clamp[-6,4]、Gaussian NLL、
**早停只看 VAL NLL**、hidden 48/epochs 80/lr 3e-3/batch 64)。被迫不同的四处:
1. **6 通道双手**(d256 双手套),输入 = 6 raw + 双手 CoP 因果速度 = **10 维**;
   `raw+df` 消融加双手 dF/dt = 12 维(沿用 OpenTouch 补的那个不对称性消融)。
2. **fps = 6 进入 `causal_velocity`**,非装饰性:速度按 fps 缩放。
3. **两臂 = 本实验本体**(OQ-D5):`none`(全部 id=0,embedding 退化为常量,
   **这才是可与另两个数据集横比的数**)vs `class`(id=label_idx,**回答的是"已知活动时的可预测性"**,
   **不得与另两臂并列报告**)。两臂**其余逐字节相同**,故差值可归因。
4. **窗口来自 harness `origins()`**,训练与打分同一批窗口;`t_in` 默认 = `min_history` = 24。

#### 校验
- fixture(真实 166 段几何 + 合成信号)上**端到端跑通**:两臂均产出、
  `score_external` 接受、metrics.csv 与 history.json 落盘。
- **⚠️ fixture 的 skill 数字(AR/probGRU ≈ +0.99)是纯正弦波的产物,不是结果**,
  只证明管道通。真实数字须在 CRC 上用真 states 跑。
- 新增 3 项测试(共 10 项):harness origins 契约的**负例**、两臂只在词表上不同、
  **特征因果性**(截断未来不改变过去的特征值)。
- `pytest 80 passed / 6 skipped`。

**本地 torch 环境**:`/opt/anaconda3` 的 torch 装坏(缺 dylib,**预先存在**);
可用的是 `/opt/anaconda3/envs/trajectron++`(torch 1.13.1)。
测试的跳过守卫**不用 `pytest.importorskip`**——坏装抛的是 `OSError` 而非 `ImportError`,
importorskip 会放行并让测试因环境原因假失败。已改为捕获任意异常再 skip;
两个环境下分别验证:坏环境 **skip**,好环境 **run**。

### 2026-08-25续2 — Hausdorff 补齐 Seq2Seq 侧:骨干在形状上的差异是**全面且单向**的

`d1_map2` 以当前代码重跑报表(`_hd`),补上此前缺失的 Hausdorff。

**结果:同输入换骨干,probGRU 的 Hausdorff 在 9/9 个格子上全部更差(+0.111 ~ +0.227)。**
按形状排名,**每一个 Seq2Seq 臂都胜过每一个 probGRU 臂**;`seasonal` 仍居首。
→ **用户 2026-08-25 的观察("GRU-aggregate 波动更大、细节更多")由此量化证实。**

**骨干效应 ≥ 输入效应**:形状上骨干差 0.11–0.23,而**同一骨干内三种输入的跨度**
仅 0.08(Seq2Seq)与 0.13(probGRU)。→ **在这份数据上,解码方式比喂给它什么更重要。**

**两个指标的分歧是有结构的,不是噪声**:
Δ Hausdorff **9/9 为正**(probGRU 形状更差);Δ 逐clip skill **3/9 为负,且恰为 3 个 F 通道**,
6 个 CoP 通道为正。→ **F 正是两种 skill 口径互相矛盾的那个通道**,须特别小心。

**机制(已写入 `docs/skill_comparison.md` 专节)**:
- probGRU **自我平滑**:每步喂回上一步的均值,而均值是去噪后的部分,故逐步剥离变化;
  第 30 步时解码器完全运行在自身光滑输出上——这是**朝光滑路径的收缩**。
- Seq2Seq **各步之间无耦合**:H 个输出是同一隐向量的 H 个独立线性读出,无物平滑之。
- **部分"细节"来自锚点而非模型**:Seq2Seq 预测残差再加到最后观测值上,
  该锚点在每个起点处像 persistence 一样跳变。

**Claude 的一处错误(已修)**:该节初稿断言 "Δ skill 为负者六于九",而表中实为 **三于九**。
已改为**由数据计数生成**而非写死——这正是"用生成器出文档"本要防止的漂移,
断言与表格同处一页却互相矛盾。

**文档完善**:`docs/skill_comparison.md` 现含
(a) 每轮的 Hausdorff 表;(b) **按输入配对的骨干并排表**(两指标同处一表);
(c) 两种 skill 口径的分歧表;(d) 两个骨干的架构差异与机制专节。
`REPORTS` 中 `d1_map2` 指向 `_hd` 重跑版——同一批预测,只是在 Hausdorff 存在之后重算过。

## 2026-08-25续3 — 【决策】两个骨干各自证明了什么、该留哪个、故事怎么讲

### 一、结论上的区别(全部有数字支撑)

| 判据 | 赢家 | 幅度 |
|---|---|---|
| 逐帧池化点误差 | **probGRU** | 9/9 为正,+0.005 ~ +0.062 |
| 逐 clip 点误差(CoP) | **probGRU** | 6/6 为正 |
| 逐 clip 点误差(**F**) | **Seq2Seq** | 3/3 为负 |
| **曲线形状(Hausdorff)** | **Seq2Seq** | **9/9,+0.111 ~ +0.227** |
| 训练成本 | **Seq2Seq** | 一次性输出 vs 30 步串行 rollout |
| 起步稳健性 | **Seq2Seq** | 残差头预测 0 即等于 persistence |
| 可采样生成轨迹 | **probGRU** | 一次性头**结构上做不到** |
| 与 ActionSense 原始实现一致 | **probGRU** | `action_dynamics.py` 即此架构 |
| 携带动作信息 | **probGRU** | 8 维 embedding |

### 二、【核心判断】"哪个能预测出我们想要的未来"——**都不能**

Hausdorff 的绝对值说明了这一点:persistence 3.301,最好的 `seasonal` 2.699,
Seq2Seq 2.738,probGRU 2.883。**两个骨干彼此的距离(0.14),远小于它们各自离"真正跟踪振荡"的距离。**
它们都在预测局部水平;Seq2Seq 只是**没那么平**,不是**跟上了波动**。

**唯一形状最接近真值的是 `seasonal`**——而它是**唯一点误差输给 persistence 的模型**。
它靠的是投射一个周期模板,**不是预测**。

→ 与天花板测量一致(2026-08-23):短 horizon 已达保守上限的 78–89%,
剩余空间几乎全在长 horizon,且成因是"预测过去的均值而非未来的包络"。
**"想要的未来"目前无人能给,这本身是本项目的结论,不是待补的空缺。**

### 三、【建议】两个都保留,但**理由不是冗余,而是对照本身就是结果**

**保留双骨干的理由**:两者**恰好在能暴露指标问题的那个维度上相反**——
probGRU 逐帧点误差更好**却**形状更差。**只有一个骨干时,这个矛盾无法展示。**
`seasonal`(点误差最差、形状最好)是同一论证的第三条腿。
**若只留一个,项目最有力的方法学发现(MSE 奖励"压平")就退化为一句无证据的断言。**

**分工建议**:
- **probGRU 为主臂**:它是 ActionSense 原始实现的架构、两个传感器现已共用、携带动作信息,
  且**唯一在结构上支持采样生成轨迹**(自回归;一次性头无法在自身抽样上继续条件化)。
  跨数据集比较与后续 N9(Student-t)/N6(慢包络)都建立在它之上。
- **Seq2Seq 为形状对照**:凡主张"MSE 误排序"处引用它。它同时是**更便宜的稳健基线**
  (无 rollout、自带 persistence 先验)。

**不建议**:以"哪个 skill 高"为由二选一。**F 通道上两种 skill 口径符号相反**,
以此做取舍等于让结论取决于一个未声明的聚合约定。

### 四、故事线(建议)

1. **两个触觉传感器、一套冻结的 harness**、经典基线 + 神经臂,一切按滚动起点打分。
2. **D1 揭示目标本身有 99.3% 是常数**;校正后 skill 近乎翻倍——**同时也让模型真正在做什么变得可见**。
3. **测出可预测性天花板**:校正后 68–77% 的方差在一帧内退相关;
   短 horizon 已耗尽 78–89%。**这是关于数据的结论,不是关于模型的。**
4. **每一个模型——线性 AR、一次性 GRU、自回归 probGRU,聚合输入或原始图——都收敛到预测局部均值。**
   三条独立证据:预测曲线图、skill 随 horizon 不降反升、Hausdorff。
5. **原始触觉图不带来 F/CoP 之外的信息**,在**两个传感器 × 两种骨干**上各自成立。
6. **MSE 奖励这种"压平"**。Hausdorff 把排序翻转:唯一点误差输给 persistence 的 `seasonal`,
   形状上全场最好。**指标选择改变结论。**
7. **贡献 = 对极限的刻画 + 使该极限可见的测量协议**(harness + 天花板 + 形状指标),
   而非"我们的模型更好"。

**这条故事线的诚实之处**:它不需要任何一个模型赢。它需要的恰恰是**模型们都输在同一个地方**,
而我们能说清那个地方在哪、有多远、以及为什么常用指标看不见它。

**唯一有实质空间的后续是 N6(慢包络分解)**,针对长 horizon 那 0.15–0.30 的缺口——
若做成,故事第 7 点可加一句"并给出一条越过它的路径";若不成,前 7 点仍然成立。
