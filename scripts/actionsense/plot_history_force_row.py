"""Paper figure: ActionSense total-force forecast under 1/2/3/5/10 s of input history.

FIVE panels stacked VERTICALLY -- one per history length -- so the only thing that changes top to
bottom is how much past the model was given. Force only (channel 0); CoP is not drawn.

WHAT IS ON THE y AXIS (read this before citing the figure). This pipeline's target is the
CAUSAL HIGH-PASS ("fast") component of total force, not the absolute force:
`action_dynamics.TARGETS = ("F_fast","x_fast","y_fast")` and `build_features` sets
`F_fast = F - causal_lowpass(F)` (action_dynamics.py:28,68-72). The axis is labelled as such.
Reconstructing absolute force would require extrapolating the slow component, which no model
here predicts, so it is deliberately NOT done.

SHARED ANCHORS. `plot_forecast_overlay.forecast_all` starts its rolling forecast at `a = t_in`,
so a 10 s-history panel would begin 9 s later than the 1 s panel. Here every history forecasts at
the SAME origins, `a >= max(t_in)`, stepping by the horizon: all five panels show an IDENTICAL
time span and ground truth; only the depth of past differs (SESSION_LOG 2026-07-07 TODO).

WHY THE FORECAST IS DRAWN IN SEGMENTS. Each forecast is one 1 s rollout from one origin; at the
next origin the model re-anchors on the MEASURED value, so consecutive segments do not join --
the jump at a segment boundary is the model's 1 s-ahead error, and hiding it would misrepresent
the method. Each segment is drawn starting from its origin's last observed sample, so the only
remaining discontinuities are those real re-anchoring jumps.

CHECKPOINTS are cached in `--ckpt-dir` (one per history) and reused on redraw when their meta
matches the requested config; a mismatch is refused, `--retrain` forces a fresh fit.

SKILL IS PRINTED on each panel by default (`--no-skill` removes it): this figure exists to show
skill differences across histories (user, 2026-09-14), which overrides the Fig. 1 no-numbers ruling.
The default clip is SELECTED for a large spread; pooled over all test clips the five histories are
flat (+.536..+.541) -- a caption must say so.

Usage (needs numpy/torch/scipy/matplotlib):
    python scripts/actionsense/plot_history_force_row.py            # clip 16, 20 s, 3.5 in
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.actionsense import action_dynamics as AD  # noqa: E402

C_TXT = "#1a1a1a"
C_GREY = "#6a6a6a"
C_TRUTH = "#141414"
C_PERS = "#8a8a8a"
C_MODEL = "#74c476"          # light green (user, 2026-09-14): distinct from the black truth


def forecast_at(model, norm, clip, t_in, t_out, origins):
    """Forecast channel-0 (F_fast) at the GIVEN origins. Input = frames [a-t_in, a), strictly
    before the predicted frames [a, a+t_out). -> (t_idx (n,t_out), mu (n,t_out), pers (n,t_out))."""
    import torch
    feat, targ, aid = clip
    fn = norm.nx(feat)
    model.eval()
    mus, pers, idx = [], [], []
    with torch.no_grad():
        for a in origins:
            x = torch.tensor(fn[a - t_in:a][None])
            yl = torch.tensor(norm.ny(targ[a - 1])[None])
            mu, _ = model(x, torch.tensor([aid]), yl, t_out)
            mus.append(norm.dy(mu[0].numpy())[:, 0])
            pers.append(np.full(t_out, targ[a - 1, 0]))
            idx.append(np.arange(a, a + t_out))
    return np.array(idx), np.array(mus), np.array(pers)


def get_model(a, p, t_in, t_out, n_act, din, trn, val):
    """Load the cached checkpoint for history p if its meta matches, else train + cache it."""
    meta = dict(actions=a.actions, hand=a.hand, input_mode=a.input_mode, t_in=t_in, t_out=t_out,
                cut=a.cut, downsample=a.downsample, hidden=a.hidden, epochs=a.epochs,
                seed=a.seed, split_seed=1, n_act=n_act, din=din)
    meta.update(a.provenance)
    path = os.path.join(a.ckpt_dir, f"{a.hand}_{a.input_mode}_e{a.epochs}_s{a.seed}_p{p:g}s.pt")
    if os.path.exists(path) and not a.retrain:
        m, norm, got = AD.load(path)
        bad = {k: (got.get(k), v) for k, v in meta.items() if got.get(k) != v}
        if bad:
            sys.exit(f"{path}: cached meta differs from request {bad}; pass --retrain")
        print(f"  history {p:>4g}s  loaded {path}")
        return m, norm
    m, norm = AD.train(trn, n_act, t_in, t_out, hidden=a.hidden, epochs=a.epochs,
                       seed=a.seed, val_clips=val)                     # early-stop on VAL
    AD.save(path, m, norm, meta)
    print(f"  history {p:>4g}s  trained + saved {path}")
    return m, norm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "actionsense_states"))
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--hand", default="right", help="left | right | active")
    ap.add_argument("--input-mode", default="raw", help="raw | highpass (raw ~= highpass, proven)")
    ap.add_argument("--pasts", default="1,2,3,5,10", help="history lengths (s), one panel each")
    ap.add_argument("--future-sec", type=float, default=1.0)
    ap.add_argument("--downsample", type=int, default=3, help="30 Hz / 3 -> 10 Hz")
    ap.add_argument("--cut", type=float, default=0.4, help="slow/fast split cutoff (Hz)")
    ap.add_argument("--clip", type=int, default=16,
                    help="TEST-split clip index to draw; 16 = largest skill spread across histories "
                         "(a SELECTED example, SESSION_LOG 2026-09-14; 52 = near-median); "
                         "-1 = longest eligible test clip")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="draw only this many seconds from the first shared origin (default: all)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "runs", "history_force_row"))
    ap.add_argument("--retrain", action="store_true", help="ignore cached checkpoints")
    ap.add_argument("--train-only", action="store_true", help="fit/cache the models, draw nothing")
    ap.add_argument("--width", type=float, default=3.5, help="inches (3.5 = IEEE \\columnwidth)")
    ap.add_argument("--panel-height", type=float, default=1.05, help="inches per history panel")
    ap.add_argument("--no-skill", dest="show_skill", action="store_false",
                    help="do not print each panel's skill (printed by default)")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "history_force_row"))
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()

    subs = [s.strip() for s in a.actions.split(",")]
    fps = 30.0 / a.downsample
    t_out = int(round(a.future_sec * fps))
    pasts = [float(p) for p in a.pasts.split(",")]
    tins = {p: int(round(p * fps)) for p in pasts}
    max_tin = max(tins.values())

    recs = AD.pooled_ids(a.root, subs, a.downsample)
    train_ids, test_ids = AD.split_train_test(len(recs), seed=1)
    r = np.random.default_rng(a.seed * 100 + 7)
    order = r.permutation(train_ids); nv = max(2, len(train_ids) // 6)
    split = {"train": sorted(recs[i] for i in order[nv:]),
             "val": sorted(recs[i] for i in order[:nv]),
             "test": sorted(recs[i] for i in test_ids)}
    parts, cfg = AD.fold_data(a.root, subs, split, a.downsample, a.cut, a.input_mode, a.hand)
    from src.calibration import checkpoint_provenance
    a.provenance = checkpoint_provenance(cfg)
    data = AD.load_pooled(cfg.abspath("states_root"), subs, a.downsample, a.cut,
                          input_mode=a.input_mode, hand=a.hand)
    trn, val = parts["train"], parts["val"]
    din = data[0][0].shape[1]
    models = {p: get_model(a, p, tins[p], t_out, len(subs), din, trn, val) for p in pasts}
    if a.train_only:
        return

    # the drawn recording must hold the LONGEST history plus at least a few horizons
    need = max_tin + 3 * t_out
    eligible = [i for i in test_ids if data[i][0].shape[0] >= need]
    if not eligible:
        sys.exit(f"no test clip has >= {need} frames for a {max(pasts):g}s history")
    ci = a.clip if a.clip >= 0 else max(eligible, key=lambda i: data[i][0].shape[0])
    if ci not in test_ids:
        sys.exit(f"clip {ci} is in TRAIN, not TEST -- refusing to draw a fitted recording")
    if data[ci][0].shape[0] < need:
        sys.exit(f"clip {ci} has {data[ci][0].shape[0]} frames, needs >= {need}")

    T = data[ci][0].shape[0]
    last = T - t_out
    if a.seconds is not None:
        last = min(last, max_tin + int(round(a.seconds * fps)) - t_out)
    origins = list(range(max_tin, last + 1, t_out))            # SHARED across every history
    print(f"clips={len(data)} train={len(train_ids)} test={len(test_ids)} | draw clip {ci} "
          f"({T} frames = {T/fps:.1f}s) | {len(origins)} shared origins, "
          f"{origins[0]/fps:g}-{(origins[-1]+t_out)/fps:g}s")

    truth = data[ci][1][:, 0]
    panels, skills = {}, {}
    for p in pasts:
        m, norm = models[p]
        ti, mu, pe = forecast_at(m, norm, data[ci], tins[p], t_out, origins)
        y = truth[ti]
        skills[p] = 1.0 - ((mu - y) ** 2).mean() / ((pe - y) ** 2).mean()
        panels[p] = (ti, mu, pe)
        print(f"  history {p:>4g}s  skill vs persistence (clip {ci}, drawn span) = {skills[p]:+.3f}")

    # ------------------------------------------------------------------ draw --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "axes.edgecolor": C_GREY, "axes.linewidth": .7,
                         "text.color": C_TXT, "axes.labelcolor": C_TXT,
                         "xtick.color": C_GREY, "ytick.color": C_GREY})
    n = len(pasts)
    head, foot = .44, .42                                     # inches: legend strip, x label
    height = n * a.panel_height + head + foot
    fig, axes = plt.subplots(n, 1, figsize=(a.width, height), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[:, 0]
    t0, t1 = (origins[0] - 1) / fps, (origins[-1] + t_out) / fps
    span = np.arange(origins[0] - 1, origins[-1] + t_out)

    for k, p in enumerate(pasts):
        ax = axes[k]
        ti, mu, pe = panels[p]
        ax.plot(span / fps, truth[span], color=C_TRUTH, lw=.9, zorder=4)
        for j in range(len(ti)):
            a0 = ti[j][0] - 1                                   # origin: last observed sample
            tt = np.r_[a0, ti[j]] / fps
            ax.plot(tt, np.r_[truth[a0], pe[j]], color=C_PERS, lw=.7, ls=(0, (2.2, 1.6)),
                    alpha=.8, zorder=3)
            ax.plot(tt, np.r_[truth[a0], mu[j]], color=C_MODEL, lw=1.4, zorder=5)
        ax.set_title(f"{p:g} s history", loc="left", fontsize=7.5, pad=2.0, color=C_TXT)
        ax.set_xlim(t0, t1)
        ax.tick_params(labelsize=7, length=2.5, pad=1.5)
        ax.grid(alpha=.18, lw=.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if a.show_skill:
            ax.set_title(f"$S$ = {skills[p]:+.2f}", loc="right", fontsize=7, pad=2.0,
                         color=C_GREY)

    axes[-1].set_xlabel("time (s)", fontsize=8.5, labelpad=2)
    fig.supylabel("fast (high-pass) total force (a.u.)", fontsize=8.5, x=.012)

    h = [plt.Line2D([], [], color=C_TRUTH, lw=1.1),
         plt.Line2D([], [], color=C_MODEL, lw=1.6),
         plt.Line2D([], [], color=C_PERS, lw=1.0, ls=(0, (2.2, 1.6)))]
    fig.legend(h, ["measured", "forecast", "persistence"], fontsize=7.2,
               loc="upper center", bbox_to_anchor=(.5 + .31 / a.width, 1.0), ncol=3,
               frameon=False, handlelength=1.8, columnspacing=1.2, handletextpad=.4)

    fig.subplots_adjust(left=.62 / a.width, right=1 - .12 / a.width, top=1 - head / height,
                        bottom=foot / height, hspace=.34)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".png", dpi=400, facecolor="white")
    print(f"[done] {a.out}.png")
    if not a.no_pdf:
        fig.savefig(a.out + ".pdf", facecolor="white")
        print(f"[done] {a.out}.pdf")


if __name__ == "__main__":
    main()
