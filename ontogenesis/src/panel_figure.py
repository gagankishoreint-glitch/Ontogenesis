#!/usr/bin/env python3
"""
Paper panel figures — G1 grid and RQ5 grid.

  python src/panel_figure.py --kind g1
  python src/panel_figure.py --kind g1  --models distilbert,gpt2,pythia_410m,gpt2_medium
  python src/panel_figure.py --kind rq5
  python src/panel_figure.py --kind rq5 --models distilbert,gpt2,bert_random

g1  : one row per model, two columns:
      (a) W_l / B_l (CI band) with R_l twin axis
      (b) T_l curves (in-form / cross-form / shuffled) + span probe overlay
          (if present) + l_struct / l_struct_span markers + chance line
rq5 : one row per model, one column:
      pooled rho_l (red) + within-form mean rho_l (blue, form-partialed)
      + l*_span marker.  Pooled peaks at L0 = form confound (expected);
      the within-form curve is the interpretable one.

Models auto-discovered from results/*_curves.csv (g1) or results/*_rq5.json
(rq5) unless given via --models.  Output: figures/panel_{kind}.png
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


def discover(kind):
    if kind == "g1":
        return [p.name[: -len("_curves.csv")] for p in sorted(RESULTS.glob("*_curves.csv"))]
    return [p.name[: -len("_rq5.json")] for p in sorted(RESULTS.glob("*_rq5.json"))]


def plot_g1(ax_row, model):
    rows = list(csv.DictReader(open(RESULTS / f"{model}_curves.csv")))
    summ = json.load(open(RESULTS / f"{model}_summary.json"))
    L = np.array([int(r["layer"]) for r in rows])
    g = lambda k: np.array([float(r[k]) for r in rows])
    has_span = "acc_span_cross" in rows[0]

    a0, a1 = ax_row
    a0.fill_between(L, g("W_lo"), g("W_hi"), alpha=0.22, color="tab:blue")
    a0.plot(L, g("W_l"), "o-", color="tab:blue", lw=1.8, ms=3, label=r"$W_l$")
    a0.plot(L, g("B_l"), "s--", color="tab:gray", lw=1.4, ms=3, label=r"$B_l$")
    a0.set_xlabel("layer"); a0.set_ylabel("cosine distance")
    a0.grid(alpha=0.3); a0.legend(fontsize=7.5, loc="upper left")
    aR = a0.twinx()
    aR.plot(L, g("R_l"), "D-", color="tab:orange", lw=1.4, ms=3, label=r"$R_l=W_l/B_l$")
    aR.set_ylabel("R", color="tab:orange"); aR.tick_params(colors="tab:orange")
    aR.legend(fontsize=7.5, loc="upper right")
    verdict = "GO" if summ.get("go_no_go") else "NO-GO"
    span_tag = " | span GO" if summ.get("go_no_go_span") else ""
    a0.set_title(f"{model}  [{verdict}{span_tag}]", fontsize=10)

    a1.plot(L, g("acc_in"), "s--", color="tab:green", lw=1.4, ms=3, label="in-form")
    a1.plot(L, g("acc_cross"), "o-", color="tab:red", lw=2, ms=3,
            label=f"cross-form  peak={g('acc_cross').max():.2f}@L{int(g('acc_cross').argmax())}")
    a1.plot(L, g("acc_shuf"), "^:", color="gray", lw=1.2, ms=3, label="shuffled")
    if has_span:
        a1.plot(L, g("acc_span_cross"), "o-", color="tab:purple", lw=1.6, ms=3,
                alpha=0.85, label=f"span cross  peak={g('acc_span_cross').max():.2f}")
        a1.plot(L, g("acc_span_shuf"), "^:", color="tab:purple", lw=0.9, ms=2, alpha=0.4)
    a1.axhline(0.5, color="k", lw=1, ls=":")
    if summ.get("l_struct") is not None:
        a1.axvline(summ["l_struct"], color="tab:red", ls="--", lw=1, alpha=0.5)
        a1.annotate(f"l*={summ['l_struct']}", xy=(summ["l_struct"], 0.33),
                    fontsize=8, color="tab:red")
    if has_span and summ.get("l_struct_span") is not None:
        a1.axvline(summ["l_struct_span"], color="tab:purple", ls=":", lw=1, alpha=0.6)
    a1.set_xlabel("layer"); a1.set_ylabel("probe accuracy")
    # shared, data-driven floor: below-chance points must stay visible
    acc_min = min(g("acc_in").min(), g("acc_cross").min(), g("acc_shuf").min(),
                  g("acc_span_cross").min() if has_span else 1.0)
    a1.set_ylim(min(0.3, acc_min - 0.03), 1.03)
    a1.grid(alpha=0.3); a1.legend(fontsize=7.5, loc="lower right")
    return a1, min(0.3, acc_min - 0.03)


def plot_rq5(ax, model):
    d = json.load(open(RESULTS / f"{model}_rq5.json"))
    rho = np.array(d["rho_by_layer"], dtype=float)
    L = np.arange(len(rho))
    series = [rho]
    ax.plot(L, rho, "o-", color="tab:red", lw=2, ms=3, label="pooled " + r"$\rho_l$")
    if d.get("rho_within_by_layer"):
        rw = np.array(d["rho_within_by_layer"], dtype=float)
        series.append(rw)
        ax.plot(L, rw, "s--", color="tab:blue", lw=2, ms=3,
                label="within-form " + r"$\rho_l$")
        ax.annotate(f"max={rw.max():.2f}@L{int(rw.argmax())}",
                    xy=(int(rw.argmax()), rw.max()), fontsize=8, color="tab:blue",
                    xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ls = d.get("l_struct_span")
    if ls is not None:
        ax.axvline(ls, color="tab:purple", ls=":", lw=1.2, alpha=0.7)
        ax.annotate(f"l*span={ls}", xy=(ls, 0.03), fontsize=8, color="tab:purple")
    ax.set_xlabel("layer"); ax.set_ylabel("Spearman rho")
    ax.set_title(f"{model}   " + rf"$\rho$(l*)={d['rho_at_lstar']:.2f}, "
                 rf"$\Delta R^2$={d['delta_R2_margin_over_form']:.3f}", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center right")
    return min(v.min() for v in series), max(v.max() for v in series)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["g1", "rq5"], default="g1")
    ap.add_argument("--models", default=None, help="comma-separated; default = discover all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ensure_dirs()

    models = (args.models.split(",") if args.models else discover(args.kind))
    models = [m.strip() for m in models if m.strip()]
    if not models:
        raise SystemExit(f"no results/*_{ 'curves.csv' if args.kind=='g1' else 'rq5.json' } found")
    missing = [m for m in models
               if not (RESULTS / (f"{m}_curves.csv" if args.kind == "g1" else f"{m}_rq5.json")).exists()]
    if missing:
        raise SystemExit(f"missing results for: {missing}")

    if args.kind == "g1":
        n = len(models)
        fig, axes = plt.subplots(n, 2, figsize=(10.5, 3.1 * n), squeeze=False)
        floors = []
        for i, m in enumerate(models):
            _, fl = plot_g1(axes[i], m)
            floors.append(fl)
        for i in range(n):          # shared probe-accuracy floor across rows
            axes[i][1].set_ylim(min(floors), 1.03)
        fig.suptitle("G1 — surface sensitivity & structure decoding (T_l + T_span)",
                     fontsize=13, fontweight="bold", y=1.0 - 0.012 / n)
    else:
        n = len(models)
        fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.0 * n), squeeze=False)
        lims = [plot_rq5(axes[i][0], m) for i, m in enumerate(models)]
        # shared y-limits across panels, data-driven so no curve is clipped
        lo = min(l for l, _ in lims); hi = max(h for _, h in lims)
        lo = min(lo - 0.06, -0.05); hi = max(hi + 0.06, 0.85)
        for i in range(n):
            axes[i][0].set_ylim(lo, hi)
        fig.suptitle("RQ5 — margin-behavior specificity (pooled vs form-partialed)",
                     fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = Path(args.out) if args.out else FIGURES / f"panel_{args.kind}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}  ({len(models)} models: {', '.join(models)})")


if __name__ == "__main__":
    main()
