"""Train + evaluate the v1 physical-state forecaster for one action (pour or slice).

Compares a learned GRU seq2seq against structured baselines (persistence, linear-velocity,
local-linear-fit = the 'ramp' model), reporting skill vs persistence PER physical variable.

    python scripts/actionsense/train_state_forecaster.py --action Pour
    python scripts/actionsense/train_state_forecaster.py --action "Slice" --t-in 30 --t-out 30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense import state_forecast as SF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--action", default="Pour", help="label substring (Pour, Slice, Peel, Clean)")
    ap.add_argument("--downsample", type=int, default=5,
                    help="subsample trajectories by this factor (30 Hz -> ~6 Hz native at 5)")
    ap.add_argument("--t-in", type=int, default=6)
    ap.add_argument("--t-out", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trajs = SF.load_trajectories(args.root, args.action)
    if args.downsample > 1:
        trajs = [t[:: args.downsample] for t in trajs]
    if len(trajs) < 3:
        raise SystemExit(f"only {len(trajs)} trajectories for {args.action!r}")
    D = trajs[0].shape[1]
    n_hands = D // len(SF.FEATS)
    names = SF.feature_names(n_hands)
    tr_tj, va_tj = SF.split_trajectories(trajs, val_frac=0.25, seed=args.seed)
    print(f"[{args.action}] {len(trajs)} trajectories (train {len(tr_tj)} / val {len(va_tj)}), "
          f"D={D} ({n_hands} hands), t_in={args.t_in} t_out={args.t_out}")

    norm = SF.Normalizer(tr_tj)
    Xtr, Ytr = SF.make_windows(tr_tj, args.t_in, args.t_out, args.stride)
    Xva, Yva = SF.make_windows(va_tj, args.t_in, args.t_out, max(1, args.stride))
    print(f"windows: train={len(Xtr)} val={len(Xva)}")
    if len(Xtr) < 8 or len(Xva) < 1:
        raise SystemExit("too few windows — lower --t-in/--t-out/--stride")

    # --- baselines (raw units) ---
    persist = SF.bl_persistence(Xva, args.t_out)
    base = {
        "persistence": persist,
        "velocity": SF.bl_velocity(Xva, args.t_out),
        "linfit(ramp)": SF.bl_linfit(Xva, args.t_out),
    }

    # --- GRU (train on normalized) ---
    Xtr_n = norm.fwd(Xtr); Ytr_n = norm.fwd(Ytr)
    Xva_n = norm.fwd(Xva)
    xt = torch.tensor(Xtr_n, dtype=torch.float32)
    yt = torch.tensor(Ytr_n, dtype=torch.float32)
    model = SF.build_gru(D, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.MSELoss()
    bs = 64
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = model(xt[b], args.t_out)
            loss = lossf(pred, yt[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  train_mse(norm)={tot/len(xt):.4f}")

    model.eval()
    with torch.no_grad():
        gru_pred = norm.inv(model(torch.tensor(Xva_n, dtype=torch.float32), args.t_out).numpy())

    # --- skill per feature (raw units) vs persistence ---
    def report(name, pred):
        sk, mse, _ = SF.skill_per_feature(pred, Yva, persist)
        print(f"\n== {name} ==  mean-skill={sk.mean():+.3f}")
        for i, nm in enumerate(names):
            print(f"   {nm:<10} skill={sk[i]:+.3f}  mse={mse[i]:.4g}")
        return sk.mean()

    print("\n" + "=" * 60)
    results = {}
    for name, pred in base.items():
        results[name] = report(name, pred)
    results["GRU"] = report("GRU", gru_pred)

    # core feedback variables = F, xbar, ybar per hand (indices 0,1,2 within each 6-block)
    core = [i for i in range(D) if (i % len(SF.FEATS)) in (0, 1, 2)]

    def core_skill(pred):
        sk, _, _ = SF.skill_per_feature(pred, Yva, persist)
        return sk[core].mean()

    print("\n" + "=" * 60)
    print(f"SUMMARY [{args.action}]  mean skill vs persistence (all 12 feats | core F+CoP):")
    for name, pred in list(base.items()) + [("GRU", gru_pred)]:
        print(f"  {name:<14} all={results[name]:+.3f}   core={core_skill(pred):+.3f}")


if __name__ == "__main__":
    main()
