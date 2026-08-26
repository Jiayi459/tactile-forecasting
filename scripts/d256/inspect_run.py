#!/usr/bin/env python3
"""Did that probGRU run actually train? Reads a run dir and says which.

A short wall clock is not by itself evidence of a broken run: at ~16,300 windows/fold, a
6-step horizon and 5 folds, a GPU epoch is fast enough that a *complete* run can finish in
minutes. What distinguishes the cases is the epoch trace, not the clock:

  * stopped at ~16-17 epochs in every fold  -> early stopping fired at its floor. With
    patience 15 that means VAL NLL never improved after epoch 1-2, i.e. the model overfits
    almost immediately. This is the OpenTouch pattern (overfit from epoch 2, SESSION_LOG
    2026-08-17). A real finding, not a bug -- but it means the 80-epoch budget is fiction.
  * ran all 80 epochs                       -> the budget was used; the clock was just fast.
  * best_epoch == 1                         -> nothing was learned after initialisation.
  * n_train_windows far below ~16,300       -> it trained on the wrong (or empty) data.

Usage:
    python scripts/d256/inspect_run.py runs/d256_probgru_none
"""
from __future__ import annotations

import json
import os
import sys

EXPECTED_WINDOWS = 16_300      # per fold, at min_history 24 / horizon 6 (SESSION_LOG 2026-08-25)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/d256_probgru_none"
    hpath = os.path.join(run, "history.json")
    if not os.path.exists(hpath):
        sys.exit(f"no history.json in {run} -- the run did not reach the end of training")
    with open(hpath) as fh:
        d = json.load(fh)

    hp = d["hp"]
    print(f"run:        {run}")
    print(f"arm:        {d['arm']}    config_hash {d.get('config_hash')}")
    print(f"budget:     epochs {hp['epochs']}  patience {hp['patience']}  "
          f"batch {hp['batch']}  t_in {hp['t_in']}  hidden {hp['hidden']}  lr {hp['lr']}")
    print(f"features:   {hp['features']}")
    print()
    print(f"  {'fold':>4} {'epochs':>7} {'best':>5} {'val NLL':>10} {'train win':>10} "
          f"{'val win':>8} {'secs':>7}")
    verdicts = []
    for k in sorted(d["folds"], key=lambda x: int(x)):
        h = d["folds"][k]
        n_ep = len(h["epoch"])
        print(f"  {k:>4} {n_ep:>7} {h['best_epoch']:>5} {h['best_val_nll']:>10.4f} "
              f"{h['n_train_windows']:>10,} {h['n_val_windows']:>8,} {h['seconds']:>7.1f}")
        verdicts.append((int(k), n_ep, h["best_epoch"], h["n_train_windows"]))

    print()
    n_folds = len(verdicts)
    problems = []
    if n_folds < 5:
        problems.append(f"only {n_folds}/5 folds present -- was --folds passed?")
    if any(w < EXPECTED_WINDOWS * 0.5 for _, _, _, w in verdicts):
        problems.append("a fold trained on far fewer windows than expected (~16,300) -- "
                        "check eval.min_history and that states_root is the real cache")
    if all(b <= 2 for _, _, b, _ in verdicts):
        problems.append("best epoch <= 2 in EVERY fold: VAL NLL never improved past the first "
                        "epochs. The model is overfitting immediately or not learning; the "
                        "80-epoch budget is not being used and hidden/lr/regularisation is "
                        "the thing to look at, not the runtime")
    floor = hp["patience"] + 1
    if all(n <= floor + 2 for _, n, _, _ in verdicts) and hp["epochs"] > floor + 2:
        problems.append(f"every fold stopped at ~{floor} epochs, the early-stopping floor -- "
                        f"so the run IS complete, it just gave up early. Short wall clock "
                        f"explained; the finding is that VAL NLL stops improving almost at once")
    if all(n >= hp["epochs"] for _, n, _, _ in verdicts):
        print("VERDICT: every fold ran its full epoch budget. The run is complete; a short "
              "wall clock just means the GPU is fast for this model size (17k params, "
              "6-step horizon).")

    if problems:
        print("VERDICT: look at these --")
        for p in problems:
            print(f"  * {p}")
    elif not all(n >= hp["epochs"] for _, n, _, _ in verdicts):
        print("VERDICT: folds stopped between the floor and the budget -- normal early "
              "stopping. Compare best_val_nll across folds for stability.")


if __name__ == "__main__":
    main()
