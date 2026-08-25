"""Prepare + profile the Grasp/Hold/Lift tactile subset for tactile->tactile forecasting.

- Collects pressure_grids.npz for the 8 core-grasp tasks into datasets/grasp_hold_lift_tactile/
- Writes a manifest CSV (one row per trajectory) with signal statistics
- Prints aggregate EDA used to ground preprocessing/method decisions
"""
import os, json, shutil, csv
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "datasets", "EgoTouch")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "datasets", "grasp_hold_lift_tactile")

GRASP_TASKS = [  # category: Grasp/Hold/Lift (core grasp verbs)
    "grasp_body_lotion", "grasp_cola", "grasp_floral_water", "grasp_power_adapter",
    "grasp_sunscreen", "grip_hand_dynamometer", "hold_teapot", "lift_towel",
]
SCENE = "Home"  # all 8 are under Home/


def hand_stats(arr):
    """arr: (T,21,21). Return dict of per-hand stats handling NaNs."""
    T = arr.shape[0]
    flat = arr.reshape(T, -1)
    nan_frac = float(np.isnan(flat).mean())
    finite = flat[~np.isnan(flat)]
    if finite.size == 0:
        return dict(T=T, nan_frac=1.0, vmin=None, vmax=None, mean=None,
                    active_frac=0.0, sparsity=None, peak_t=None)
    # active cell = pressure > small epsilon
    eps = 1e-3
    per_frame_active = np.nansum(flat > eps, axis=1)
    per_frame_sum = np.nansum(np.where(np.isnan(flat), 0.0, flat), axis=1)
    active_frames = int((per_frame_sum > eps).sum())
    return dict(
        T=T,
        nan_frac=round(nan_frac, 4),
        vmin=round(float(finite.min()), 4),
        vmax=round(float(finite.max()), 4),
        mean=round(float(finite.mean()), 4),
        active_frac=round(active_frames / T, 4),
        sparsity=round(float((finite > eps).mean()), 4),
        peak_t=int(np.argmax(per_frame_sum)),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    Ts = []
    for task in GRASP_TASKS:
        tp = os.path.join(ROOT, SCENE, task)
        for traj in sorted(os.listdir(tp)):
            d = os.path.join(tp, traj)
            if not os.path.isdir(d):
                continue
            pz = os.path.join(d, "pressure_grids.npz")
            if not os.path.exists(pz):
                continue
            # copy into subset dir
            od = os.path.join(OUT, task, traj)
            os.makedirs(od, exist_ok=True)
            shutil.copy2(pz, os.path.join(od, "pressure_grids.npz"))
            z = np.load(pz)
            L, R = z["left_pressure_grid"], z["right_pressure_grid"]
            ls, rs = hand_stats(L), hand_stats(R)
            Ts.append(ls["T"])
            rows.append(dict(
                task=task, traj=traj, T=ls["T"],
                tactile_max=float(z["tactile_max"]) if "tactile_max" in z.files else None,
                L_nan=ls["nan_frac"], L_active=ls["active_frac"], L_max=ls["vmax"],
                L_mean=ls["mean"], L_sparsity=ls["sparsity"],
                R_nan=rs["nan_frac"], R_active=rs["active_frac"], R_max=rs["vmax"],
                R_mean=rs["mean"], R_sparsity=rs["sparsity"],
            ))

    # write manifest
    man = os.path.join(OUT, "manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # aggregate EDA
    Ts = np.array(Ts)
    print(f"Subset: {len(rows)} trajectories from {len(GRASP_TASKS)} tasks -> {OUT}")
    print(f"Sequence length T (frames @30fps): min={Ts.min()} max={Ts.max()} "
          f"mean={Ts.mean():.0f} median={np.median(Ts):.0f} "
          f"(~{Ts.mean()/30:.1f}s mean, total {Ts.sum()} frames)")
    for hand, pfx in (("LEFT", "L"), ("RIGHT", "R")):
        nan = np.array([r[f"{pfx}_nan"] for r in rows])
        act = np.array([r[f"{pfx}_active"] for r in rows])
        spars = np.array([r[f"{pfx}_sparsity"] for r in rows if r[f"{pfx}_sparsity"] is not None])
        print(f"{hand:>5}: all-NaN traj={int((nan==1.0).sum())}/{len(rows)} | "
              f"mean nan_frac={nan.mean():.3f} | mean active_frame_frac={act.mean():.3f} | "
              f"mean spatial sparsity(active cells)={spars.mean():.3f}")
    # how many traj have a usable (non-all-nan, some activity) hand
    usable = sum(1 for r in rows if (r["L_active"] > 0.1 or r["R_active"] > 0.1))
    print(f"Trajectories with >=10% active frames on at least one hand: {usable}/{len(rows)}")
    print(f"Manifest: {man}")


if __name__ == "__main__":
    main()
