"""Training curves for the probGRU, plus a calibration check on its variance head.

Reads the checkpoints --save-model wrote: each carries the final training's per-epoch NLL,
the history sweep's best VAL score per input length, and which epoch was kept. The star
marks the minimum of whichever curve hp["select_on"] used, read from the checkpoint, so a
run selected on MSE is not drawn as if NLL had chosen its weights.

WHAT THE CURVE IS FOR. Early stopping keeps the lowest-VAL-NLL weights, so a run cannot be
hurt by training too long -- but where the minimum sits says something. On the 2026-08-17
run VAL bottomed at epoch 2 and rose monotonically after, while ActionSense's own code
notes its loss "overfits badly after ~epoch 10". Overfitting four times sooner under a
location-held-out split is not a nuisance to be tuned away; it says what the extra epochs
were learning did not survive a change of environment.

Train NLL is sampled every 5th epoch (log_train_every), so it is drawn with markers and
gaps rather than interpolated through points that were never measured.

CALIBRATION, from the saved forecasts rather than the checkpoint. The Gaussian head is
trained but never scored -- the harness measures point error only -- so a model whose
spread is wrong pays nothing and no MSE table shows it. Coverage does: the fraction of
truths falling inside +-2 sigma should be about 95%. Materially above that is a model
hedging with intervals wider than its errors, which is what the forecast plot suggests.

    python scripts/plot_opentouch_loss.py --models runs/models --preds runs/preds
"""
from __future__ import annotations

import argparse
import collections
import glob
import os

import numpy as np


def load_ckpts(d):
    import torch
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.pt"))):
        try:
            out.append((os.path.basename(p), torch.load(p, map_location="cpu",
                                                        weights_only=False)))
        except Exception as exc:                                   # pragma: no cover
            print(f"  skipped {p}: {type(exc).__name__}: {exc}")
    return out


def coverage(preds_dir, k_sigma=2.0):
    """Coverage and scale for EVERY probabilistic arm, per channel.

    Per model, because OQ-G was overturned globally (2026-08-19) and all of them now emit a
    variance; a head that is trained and never checked against the errors it claims to
    describe is unfalsifiable decoration.

    Per channel, because that is the point rather than a refinement: F sits near 750,000
    while CoP lives in [-1,1], so a pooled median sigma is whatever the CoP channels say and
    carries nothing about F, and a pooled coverage can read 95% while one channel hedges and
    another is overconfident, cancelling.

    Both statistics are kept: coverage alone cannot separate a well-sized interval from a
    wide one that happens to sit in the right place, and the sigma-to-error ratio alone
    ignores where the mean is. For Gaussian errors the ratio is 1/0.6745 = 1.48; materially
    below it means the tails are heavier than the interval admits.

    -> {model: {channel: (coverage, median sigma, median |error|)}}
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(
        lambda: [0, 0, [], []]))                       # model -> channel -> in,tot,sig,err
    chans = None
    for p in sorted(glob.glob(os.path.join(preds_dir, "clip_*.npz"))):
        z = np.load(p, allow_pickle=True)
        ors = z["origins"]
        if len(ors) == 0:        # too short to yield an origin; a file was still written
            continue
        if chans is None and "channels" in z.files:
            chans = [str(c) for c in z["channels"]]
        for key in z.files:
            if not key.startswith("sigma_") or f"mu_{key[6:]}" not in z.files:
                continue
            name = key[6:]
            mu, sd = z[f"mu_{name}"], z[key]
            if mu.shape[0] == 0:
                continue
            H = mu.shape[1]
            truth = np.stack([z["y"][t + 1:t + 1 + H] for t in ors])
            d = np.abs(truth - mu)
            if chans is None:
                chans = [f"c{j}" for j in range(d.shape[-1])]
            for j, c in enumerate(chans):
                e = acc[name][c]
                e[0] += int((d[..., j] <= k_sigma * sd[..., j]).sum())
                e[1] += d[..., j].size
                e[2].append(sd[..., j].ravel()); e[3].append(d[..., j].ravel())
    return {m: {c: (v[0] / max(v[1], 1),
                    float(np.median(np.concatenate(v[2]))),
                    float(np.median(np.concatenate(v[3]))))
                for c, v in chs.items()}
            for m, chs in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="runs/models")
    ap.add_argument("--preds", default="runs/preds")
    ap.add_argument("--out", default="docs/opentouch/raw/opentouch_loss.png")
    a = ap.parse_args()

    cks = load_ckpts(a.models)
    if not cks:
        raise SystemExit(f"no *.pt under {a.models} -- rerun with --save-model")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Room on the right for the legend: with three arms over four folds the in-axes legend
    # was 24 entries and covered most of the curves it was labelling.
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.4))
    ax, axs = axes
    ax2 = ax.twinx()
    ax2.set_ylabel("VAL MSE (dotted, + = its own best)", fontsize=8)

    arms_present = {ck.get("arm") or os.path.basename(n).split("_location")[0]
                    for n, ck in cks}
    colours: dict[str, str] = {}
    labelled: set[str] = set()
    crits = set()
    for name, ck in cks:
        h = ck.get("history", {})
        # Which VAL curve actually kept the weights. Older checkpoints predate the knob and
        # were all NLL-selected; newer ones say so themselves. The star has to sit on the
        # curve that DID the selecting, or the figure claims weights came from somewhere
        # they did not -- which is what the 2026-08-21 MSE run's first plot did.
        crit = str(h.get("selected_on_metric", "nll")).split()[0].lower()
        crits.add(crit)
        va = np.asarray(h.get("val_nll", []), dtype=float)
        tr = np.asarray(h.get("train_nll", []), dtype=float)
        if va.size == 0:
            continue
        fold = ck.get("split", name).replace("location-k4-", "").replace("-seed0", "")
        # One directory can now hold several arms (the map family saves checkpoints since
        # 2026-08-23), and four folds of three arms all labelled "fold0..fold3" would be
        # twelve indistinguishable curves. Older checkpoints predate the field and fall back
        # to the filename, which already begins with the arm.
        arm = ck.get("arm") or os.path.basename(name).split("_location")[0]
        # With more than one arm, COLOUR CARRIES THE ARM and folds repeat it. Giving all
        # twelve series their own colour made the legend longer than the plot was tall and
        # left the arms indistinguishable anyway, which is the comparison the figure is for.
        # With one arm, colour goes back to the fold, as every earlier figure has it.
        multi = len(arms_present) >= 2
        key = arm if multi else fold
        col = colours.setdefault(key, f"C{len(colours) % 10}")
        labelled.add(key)
        ep = np.arange(1, va.size + 1)
        ax.plot(ep, va, lw=1.4, color=col)
        m = np.isfinite(tr)
        if m.any():
            ax.plot(ep[m], tr[m], "o--", ms=3, lw=0.8, alpha=0.55, color=col)
        if crit == "nll":
            b = int(np.nanargmin(va))
            ax.plot(ep[b], va[b], "*", ms=12, color=col)
        # VAL MSE on a twin axis: if it stays flat while NLL climbs, the mean is fine and
        # only the variance head is degrading -- and early stopping on NLL is then picking
        # weights by a criterion the harness never scores.
        vm = np.asarray(h.get("val_mse", []), dtype=float)
        if vm.size == va.size and np.isfinite(vm).any():
            ax2.plot(ep, vm, lw=1.0, ls=":", color=col, alpha=0.8)
            j = int(np.nanargmin(vm))
            ax2.plot(ep[j], vm[j], "*" if crit == "mse" else "P",
                     ms=12 if crit == "mse" else 7, color=col)

        sw = ck.get("sweep") or {}
        if sw:
            xs = sorted(int(x) for x in sw)
            axs.plot([x / 30.0 for x in xs], [sw[x] if x in sw else sw[str(x)] for x in xs],
                     "o-", lw=1.4, color=col)
            axs.plot(ck["t_in"] / 30.0, min(sw.values()), "*", ms=12, color=col)

    kept = "/".join(sorted(c.upper() for c in crits)) or "NLL"
    other = "MSE" if kept == "NLL" else "NLL"
    who = ", ".join(sorted(arms_present))
    ax.set_xlabel("epoch"); ax.set_ylabel("Gaussian NLL")
    ax.set_title(f"{who} training curves — solid/dashed = NLL, dotted = VAL MSE\n"
                 f"star = weights kept (min VAL {kept});  + = min VAL {other}",
                 fontsize=10)
    ax.grid(alpha=0.25)
    axs.set_xlabel("input history (s)"); axs.set_ylabel(f"best VAL {kept}")
    axs.set_title(f"history sweep, chosen on VAL {kept}\n"
                  "(≥2 s is mostly zero-padding: <50% of clips are long enough)",
                  fontsize=10)
    axs.grid(alpha=0.25)

    # One legend for the figure, outside both axes. Colour keys the arm when several are
    # present and the fold when only one is, and the line styles are spelled out here rather
    # than multiplied through every series.
    from matplotlib.lines import Line2D
    per = "one line per fold" if len(arms_present) >= 2 else "VAL"
    ents = [Line2D([], [], color=c, lw=1.6, label=k) for k, c in sorted(colours.items())]
    ents += [Line2D([], [], color="none", label=" "),
             Line2D([], [], color="0.35", lw=1.6, label=f"VAL NLL ({per})"),
             Line2D([], [], color="0.35", lw=0.9, ls="--", marker="o", ms=3,
                    label="TRAIN NLL"),
             Line2D([], [], color="0.35", lw=1.0, ls=":", label="VAL MSE (right axis)")]
    fig.legend(handles=ents, loc="center left", bbox_to_anchor=(0.845, 0.5),
               fontsize=8, frameon=False)

    cov = coverage(a.preds)
    if cov:
        print(f"\n{'model':14s} {'channel':10s} {'±2σ cov':>9s} {'median σ':>12s} "
              f"{'median |err|':>13s} {'σ/|err|':>8s}")
        for m in sorted(cov):
            for c, (f_, s_, e_) in cov[m].items():
                print(f"{m:14s} {c:10s} {f_:9.2%} {s_:12.5g} {e_:13.5g} "
                      f"{s_ / max(e_, 1e-12):8.2f}")
        print("Gaussian errors give σ/|err| = 1.48 and 95.4% coverage. Above -> the head "
              "hedges with intervals wider than its mistakes; below -> it is overconfident. "
              "The harness scores point error only, so neither shows up anywhere else.")
        # Every arm, not just the alphabetically first: a three-arm figure titled with one
        # arm's coverage reads as if that were the figure's subject, the same way the
        # forecast overlay once did (2026-08-20).
        chans = list(cov[sorted(cov)[0]])
        bits = ["  ".join(f"{m} " + "/".join(f"{cov[m][c][0]:.1%}" for c in chans)
                          for m in sorted(cov))]
        fig.suptitle(f"±2σ coverage (nominal 95.4%), {'/'.join(chans)} — {bits[0]}",
                     fontsize=9)
    # The right bound leaves the legend its column; tight_layout would otherwise expand the
    # axes back over it.
    fig.tight_layout(rect=(0, 0, 0.84, 0.93 if cov else 1.0))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"wrote {a.out} ({len(cks)} checkpoints)")


if __name__ == "__main__":
    raise SystemExit(main())
