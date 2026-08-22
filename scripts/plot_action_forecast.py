"""Plot the v2 action-dynamics forecast (thin CLI over src/actionsense/action_dynamics.py).

This script ONLY plots — all model/training/forecast logic lives in the library. Two modes:
  * default: sweep past-context lengths (1/2/3/5/10 s), training one model per length via the
    library on a held-out split, and plot each model's honest multi-step forecast of the next
    `--future-sec` on a TEST clip (vs persistence).
  * --ckpt PATH: load a saved checkpoint (from train_action_dynamics.py) and plot that one model.

    python scripts/plot_action_forecast.py --actions Slice,Peel --viz-action Slice --target 0
    python scripts/plot_action_forecast.py --ckpt runs/ad_slice-peel.pt --viz-action Slice
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense import action_dynamics as AD  # noqa: E402


def pick_viz(data, subs, viz_action):
    return next(i for i, d in enumerate(data)
                if subs[d[2]].lower().startswith(viz_action.lower())
                or viz_action.lower().startswith(subs[d[2]].lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--viz-action", default="Slice")
    ap.add_argument("--target", type=int, default=0, help="0=F_fast 1=x_fast 2=y_fast")
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--input-mode", default="highpass", help="raw | highpass (sweep mode)")
    ap.add_argument("--hand", default="active", help="left | right | active (sweep mode)")
    ap.add_argument("--pasts", default="1,2,3,5,10", help="past-context lengths (s) to sweep")
    ap.add_argument("--future-sec", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--ckpt", default=None, help="plot a saved checkpoint instead of sweeping")
    ap.add_argument("--out", default="docs/actionsense/action_forecast_density.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TGT = AD.TARGETS[args.target]
    k = args.target
    panels = []          # (label, t_in, forecast_dict, test_skill)

    if args.ckpt:        # ---- MODE 1: load one checkpoint and plot it ----
        model, norm, meta = AD.load(args.ckpt)
        subs = meta["subs"]
        data = AD.load_pooled(args.root, subs, meta["downsample"], meta["cut"],
                              input_mode=meta.get("input_mode", "highpass"), hand=meta.get("hand", "active"))
        viz_i = pick_viz(data, subs, args.viz_action)
        _, test_ids = AD.split_train_test(len(data), force_test=[viz_i])
        vtarg = data[viz_i][1]
        fc = AD.forecast_clip(model, norm, data[viz_i], meta["t_in"], meta["t_out"], k,
                              sigma_scale=meta.get("sigma_scale", 1.0))
        _, _, mean_sk, _ = AD.evaluate(model, norm, [data[i] for i in test_ids], meta["t_in"],
                                       meta["t_out"], sigma_scale=meta.get("sigma_scale", 1.0))
        panels.append((f"checkpoint {os.path.basename(args.ckpt)}", meta["t_in"], fc, mean_sk))
    else:                # ---- MODE 2: sweep past-context, training via the library ----
        subs = [s.strip() for s in args.actions.split(",")]
        data = AD.load_pooled(args.root, subs, args.downsample, args.cut,
                              input_mode=args.input_mode, hand=args.hand)
        fps = 30.0 / args.downsample
        t_out = int(round(args.future_sec * fps))
        viz_i = pick_viz(data, subs, args.viz_action)
        tr_ids, te_ids = AD.split_train_test(len(data), force_test=[viz_i])
        train = [data[i] for i in tr_ids]; test = [data[i] for i in te_ids]
        vtarg = data[viz_i][1]
        print(f"train {len(train)} / test {len(test)} clips; viz clip #{viz_i}; future={t_out} frames")
        for p in [float(x) for x in args.pasts.split(",")]:
            t_in = int(round(p * fps))
            model, norm = AD.train(train, len(subs), t_in, t_out,
                                   hidden=args.hidden, epochs=args.epochs)
            _, _, mean_sk, _ = AD.evaluate(model, norm, test, t_in, t_out)
            fc = AD.forecast_clip(model, norm, data[viz_i], t_in, t_out, k)
            panels.append((f"past {p:.0f}s -> next {args.future_sec:.0f}s", t_in, fc, mean_sk))
            print(f"  {panels[-1][0]:<22} test-set skill={mean_sk:+.3f}  this-clip={fc['skill']:+.3f}")

    # ---- draw one row per panel ----
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.1 * len(panels)),
                             sharex=True, squeeze=False)
    tt = np.arange(min(pn[1] for pn in panels), vtarg.shape[0])
    for ax, (lab, t_in, fc, sk_test) in zip(axes[:, 0], panels):
        ax.plot(tt, vtarg[tt, k], "k-", lw=1.4, label="true")
        for i, (t, mu, sd, _pe) in enumerate(fc["segments"]):
            ax.plot(t, mu, "C0-", lw=1.5, label="forecast (autoregressive)" if i == 0 else None)
            ax.fill_between(t, mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.20,
                            label="+/-2 sigma" if i == 0 else None)
        ax.plot(fc["ts"], fc["pers"], color="0.6", ls="--", lw=0.9, label="persistence-of-fast")
        for a in [s[0][0] for s in fc["segments"]]:
            ax.axvline(a, color="0.9", lw=0.5)
        ax.set_ylabel(TGT)
        ax.set_title(f"{lab}   |   test-set skill {sk_test:+.2f}, this-clip {fc['skill']:+.2f}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1, 0].set_xlabel("time step (~10 Hz)  |  grey verticals = anchors (forecast restarts from truth)")
    fig.suptitle(f"v2 forecast — {args.viz_action}, {TGT} (held-out test clip)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
