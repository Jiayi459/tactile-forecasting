"""Feasibility probe: how predictable is future tactile from past tactile?

Computes, on the Grasp/Hold/Lift subset (NaN sensor-mask -> 0):
  - persistence baseline (y_hat[t+h] = y[t]) normalized MSE vs horizon h
  - temporal autocorrelation of total force and of taxel maps vs lag
  - frame-to-frame change magnitude (signal smoothness)
This sets the bar any learned model must beat. Analysis only (no training).
"""
import os, glob
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "datasets", "grasp_hold_lift_tactile")


def load_seq(pz):
    z = np.load(pz)
    L = np.nan_to_num(z["left_pressure_grid"], nan=0.0)
    R = np.nan_to_num(z["right_pressure_grid"], nan=0.0)
    # stack hands as channels -> (T, 2, 21, 21)
    return np.stack([L, R], axis=1).astype(np.float32)


def main():
    seqs = [load_seq(p) for p in glob.glob(os.path.join(OUT, "*", "*", "pressure_grids.npz"))]
    seqs = [s for s in seqs if s.shape[0] >= 35]
    print(f"Probe on {len(seqs)} sequences (T>=35).")

    horizons = [1, 3, 5, 10, 15, 30]
    # persistence nMSE = MSE(y[t+h], y[t]) / Var(y)  (per sequence, then averaged)
    for h in horizons:
        nmses = []
        for s in seqs:
            if s.shape[0] <= h:
                continue
            a, b = s[:-h], s[h:]
            mse = np.mean((b - a) ** 2)
            var = np.var(s) + 1e-8
            nmses.append(mse / var)
        print(f"  persistence h={h:>2} frames ({h/30*1000:>4.0f} ms): nMSE={np.mean(nmses):.4f}")

    # autocorrelation of total force (sum over taxels) vs lag
    print("Total-force autocorrelation vs lag:")
    for lag in [1, 3, 5, 10, 15, 30]:
        cors = []
        for s in seqs:
            f = s.reshape(s.shape[0], -1).sum(1)
            if f.std() < 1e-6 or s.shape[0] <= lag:
                continue
            a, b = f[:-lag], f[lag:]
            c = np.corrcoef(a, b)[0, 1]
            if np.isfinite(c):
                cors.append(c)
        print(f"  lag={lag:>2} ({lag/30*1000:>4.0f} ms): r={np.mean(cors):.3f}")

    # mean per-frame change vs signal energy (smoothness)
    d1 = np.mean([np.mean(np.abs(np.diff(s, axis=0))) for s in seqs])
    energy = np.mean([np.mean(np.abs(s)) for s in seqs])
    print(f"Smoothness: mean |Δframe|={d1:.4f} vs mean |signal|={energy:.4f} "
          f"(ratio={d1/energy:.3f})")


if __name__ == "__main__":
    main()
