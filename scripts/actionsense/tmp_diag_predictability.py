"""One-off diagnostic (no torch): is F_fast intrinsically more predictable than CoP_fast?

For each Slice/Peel clip (both hands): build the same causal fast targets as action_dynamics
(ds=3 -> 10 Hz, cut=0.4 Hz, warmup 5 s), then per channel report
  * autocorrelation at lag 0.1 s and 1.0 s
  * spectral centroid (Hz) -- how slow/smooth the fast component is
  * skill vs persistence at the 1.0 s step of a RIDGE-regression linear predictor
    (train/test split by clip, seed=1, same protocol shape as the GRU) -- a floor for the GRU.
"""
import json, os, sys
import numpy as np
from scipy.signal import butter, sosfilt

ROOT = "data/actionsense_states"
SUBS = ["slice", "peel"]
DS, CUT, WARM = 3, 0.4, 5.0
FPS = 30.0 / DS
T_IN, T_OUT = 10, 10  # 1s -> 1s

def slow_fast(sig):
    sos = butter(2, CUT / (FPS / 2.0), "low", output="sos")
    s = sosfilt(sos, sig, axis=0)
    return s, sig - s

rows = [json.loads(l) for l in open(os.path.join(ROOT, "manifest.jsonl"))]
clips = []  # (targ (T,3), hand)
for r in rows:
    if not any(r["label"].lower().startswith(s) for s in SUBS):
        continue
    st = np.load(os.path.join(ROOT, f"state_{r['idx']}.npy"))[::DS]
    for h in (0, 1):
        F, x, y = st[:, h, 0], st[:, h, 1], st[:, h, 2]
        t = np.stack([slow_fast(F)[1], slow_fast(x)[1], slow_fast(y)[1]], 1)
        w = int(round(WARM * FPS))
        t = t[w:]
        if t.shape[0] >= 40:
            clips.append((t.astype(np.float64), h))

names = ["F_fast", "x_fast", "y_fast"]
print(f"{len(clips)} clip-hands")

# --- autocorrelation + spectral centroid, pooled over clips (variance-weighted) ---
for k, nm in enumerate(names):
    ac1, ac10, cent, wts = [], [], [], []
    for t, _ in clips:
        s = t[:, k] - t[:, k].mean()
        v = s.var()
        if v < 1e-12 or len(s) < 60:
            continue
        ac1.append(np.corrcoef(s[:-1], s[1:])[0, 1])
        ac10.append(np.corrcoef(s[:-10], s[10:])[0, 1])
        f = np.fft.rfftfreq(len(s), 1 / FPS)
        P = np.abs(np.fft.rfft(s)) ** 2
        cent.append((f * P).sum() / P.sum())
        wts.append(v * len(s))
    wts = np.array(wts); wts /= wts.sum()
    print(f"{nm}: AC(0.1s)={np.average(ac1, weights=wts):+.3f}  "
          f"AC(1.0s)={np.average(ac10, weights=wts):+.3f}  "
          f"spectral centroid={np.average(cent, weights=wts):.2f} Hz")

# --- linear ridge floor: predict step +1.0s from past 1s of the 3 fast channels ---
rng = np.random.default_rng(1)
order = rng.permutation(len(clips))
n_test = max(2, round(0.25 * len(clips)))
te = set(order[:n_test].tolist())

def windows(idx_set):
    X, Y, LAST = [], [], []
    for i, (t, _) in enumerate(clips):
        if (i in te) != (idx_set == "te"):
            continue
        for s in range(0, t.shape[0] - T_IN - T_OUT + 1, 2):
            X.append(t[s:s + T_IN].ravel())          # past 1s, all 3 channels
            Y.append(t[s + T_IN + T_OUT - 1])        # value at +1.0s
            LAST.append(t[s + T_IN - 1])             # persistence source
    return np.array(X), np.array(Y), np.array(LAST)

Xtr, Ytr, _ = windows("tr")
Xte, Yte, Lte = windows("te")
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
lam = 1e-2 * len(Xtr)
W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
b = Ytr.mean(0) - 0  # X already centered
pred = Xte @ W + Ytr.mean(0) - (Xtr @ W).mean(0)
for k, nm in enumerate(names):
    em = np.mean((pred[:, k] - Yte[:, k]) ** 2)
    ep = np.mean((Lte[:, k] - Yte[:, k]) ** 2)
    print(f"{nm}: linear-ridge skill@1.0s = {1 - em / ep:+.3f}   "
          f"(persistence MSE={ep:.4g}, ridge MSE={em:.4g})")
