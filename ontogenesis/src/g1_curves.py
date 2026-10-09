#!/usr/bin/env python3
"""Plot G1 go/no-go curves: W/R (surface sensitivity) and T (structure decoding).
Usage: python g1_curves.py --model bert_base
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIGURES, RESULTS, ensure_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ensure_dirs()

    rows = list(csv.DictReader(open(RESULTS / f"{args.model}_curves.csv")))
    summ = json.load(open(RESULTS / f"{args.model}_summary.json"))
    L = np.array([int(r["layer"]) for r in rows])
    W = np.array([float(r["W_l"]) for r in rows])
    lo = np.array([float(r["W_lo"]) for r in rows])
    hi = np.array([float(r["W_hi"]) for r in rows])
    B = np.array([float(r["B_l"]) for r in rows])
    R = np.array([float(r["R_l"]) for r in rows])
    a_in = np.array([float(r["acc_in"]) for r in rows])
    a_cr = np.array([float(r["acc_cross"]) for r in rows])
    a_sh = np.array([float(r["acc_shuf"]) for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax0 = ax[0]
    ax0.fill_between(L, lo, hi, alpha=0.22, color="tab:blue")
    ax0.plot(L, W, "o-", color="tab:blue", lw=2, label=r"$W_l$ within-family (absolute)")
    ax0.plot(L, B, "s--", color="tab:gray", lw=1.6, label=r"$B_l$ between-family, form-matched")
    ax0.set_xlabel("layer")
    ax0.set_ylabel("cosine distance")
    ax0.set_title("Surface sensitivity: absolute vs between")
    ax0.legend(loc="upper left", fontsize=8.5)
    ax0.grid(alpha=0.3)
    axR = ax0.twinx()
    axR.plot(L, R, "D-", color="tab:orange", lw=2, label=r"$R_l = W_l/B_l$ (relative)")
    axR.set_ylabel("R (relative surface sensitivity)", color="tab:orange")
    axR.tick_params(axis="y", colors="tab:orange")
    axR.legend(loc="upper right", fontsize=8.5)

    ax[1].plot(L, a_in, "s--", color="tab:green", lw=1.8, label="in-form (P2)")
    ax[1].plot(L, a_cr, "o-", color="tab:red", lw=2.4,
               label=f"cross-form (P2)  peak={a_cr.max():.2f}@L{int(a_cr.argmax())}")
    ax[1].plot(L, a_sh, "^:", color="gray", lw=1.5, label="shuffled-label control")
    ax[1].axhline(0.5, color="k", lw=1, ls=":", label="chance")
    if summ.get("l_struct") is not None:
        ls = summ["l_struct"]
        ax[1].axvline(ls, color="tab:red", lw=1, alpha=0.5)
        ax[1].annotate(f"l* = {ls}", xy=(ls, 0.55), fontsize=9, color="tab:red")
    ax[1].set_xlabel("layer")
    ax[1].set_ylabel("accuracy")
    ax[1].set_ylim(0.3, 1.02)
    ax[1].set_title("Structure decoding: form-invariant readout ($T_l$)")
    ax[1].legend(loc="lower right", fontsize=8.5)
    ax[1].grid(alpha=0.3)

    verdict = "GO" if summ.get("go_no_go") else "NO-GO"
    fig.suptitle(f"G1 — {args.model} — {verdict}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / f"g1_{args.model}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
