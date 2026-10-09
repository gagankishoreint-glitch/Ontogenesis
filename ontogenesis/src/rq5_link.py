#!/usr/bin/env python3
"""
RQ5 link v3.1: does the per-item SPAN margin at l*_span predict behavioral
generalization under surface change?

  margin_s = (2*y - 1) * [ g_l(h_u) - g_l(h_v) ]
      g_l   = span-agent probe (frozen features at layer l, trained on P1
              {active,passive}, BOTH slots -> label balanced within form)
      y     = anchored label from behavior v3 (1 iff queried u is the agent)
  The margin is IDENTICAL for the two query orders of a sentence (algebra:
  both reduce to g(h_n2) - g(h_n1)), so analyses are AGGREGATED to sentence
  level (n = 600 cross-form sentences; p_correct = mean over the 2 orders).
  Within-sentence differencing removes sentence-level form information from
  the margin's definition; the per-form curves separate what remains.

Analyses:
  1. rho_l        = Spearman(margin, p_correct) per layer, POOLED (registered)
  2. rho_within_l = mean of per-form Spearman per layer (form-partialed
                    specificity curve — adjudicates pre-registered P2')
  3. at l*: pooled + per-form rho with p-values; span-probe pairwise acc
                    per form (instrument diagnostic)
  4. delta-R2: rank-OLS [form dummies] vs [+ margin]   (sentence level)
  5. margin-tercile bars per form
  Output: results/{model}_rq5.json + figures/rq5_{model}.png

Requires: results/{model}_behavior.csv from behavior.py v3 (anchored) and
results/{model}_summary.json containing l_struct_span (invariance.py v3).

Usage: python rq5_link.py --model distilbert
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
from config import (FEATURES, FIGURES, ITEMS_CSV, PROBE_CROSS_FORMS,
                    PROBE_TRAIN_FORMS, RESULTS, SPLIT_TEST, SPLIT_TRAIN, ensure_dirs)


def bh_fdr(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); m = len(p); prev = 1.0
    for rank, idx in enumerate(reversed(o)):
        prev = min(prev, p[idx] * m / (m - rank)); q[idx] = min(prev, 1.0)
    return q


def fit_probe(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Xtr, ytr)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=None,
                    help="override analysis layer (needed when l_struct_span is null, e.g. bert_random)")
    args = ap.parse_args()
    ensure_dirs()

    beh = list(csv.DictReader(open(RESULTS / f"{args.model}_behavior.csv")))
    need = {"id", "form", "slot_u", "pair", "y", "p_correct"}
    if not need.issubset(beh[0].keys()):
        raise SystemExit(f"{args.model}_behavior.csv is v1 (columns {list(beh[0])}); "
                         "rerun behavior.py v3 (anchored) first.")
    meta = list(csv.DictReader(open(ITEMS_CSV, newline="")))
    summ = json.load(open(RESULTS / f"{args.model}_summary.json"))
    lstar = args.layer if args.layer is not None else summ.get("l_struct_span")
    if lstar is None:
        raise SystemExit("summary has no l_struct_span and no --layer given — "
                         "rerun invariance.py v3 or pass --layer N (control models).")
    mid = {m["id"]: i for i, m in enumerate(meta)}

    z = np.load(FEATURES / f"{args.model}.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)
    feats = np.concatenate([X[:, :, 0, :], X[:, :, 1, :]], axis=-1).transpose(1, 0, 2)
    H2 = feats.shape[2] // 2

    # span-probe training set (P1 active+passive, both slots)
    sp_i, sp_k, sp_y = [], [], []
    for i, m in enumerate(meta):
        if m["structure"] != "trans":
            continue
        if not (m["form"] in PROBE_TRAIN_FORMS and m["split"] == SPLIT_TRAIN):
            continue
        lab = int(m["label_patient_first"])
        for k in (0, 1):
            sp_i.append(i); sp_k.append(k)
            sp_y.append(int(k == lab))             # k is the agent slot (see invariance)
    sp_i = np.array(sp_i); sp_k = np.array(sp_k); sp_y = np.array(sp_y)

    def span_mat(l):
        F = feats[l][sp_i]
        return np.where(sp_k[:, None] == 0, F[:, :H2], F[:, H2:])

    # ---- behavior rows -> CROSS-FORM, then aggregate to sentences --------
    ids_all = np.array([r["id"] for r in beh])
    idx_all = np.array([mid[r["id"]] for r in beh])
    su_all = np.array([int(r["slot_u"]) for r in beh])
    yb_all = np.array([int(r["y"]) for r in beh])
    p_all = np.array([float(r["p_correct"]) for r in beh])
    form_all = np.array([r["form"] for r in beh])
    cr = np.isin(form_all, PROBE_CROSS_FORMS)
    ids, idx, su, yb, p_all, form_b = (a[cr] for a in
                                       (ids_all, idx_all, su_all, yb_all, p_all, form_all))
    assert len(set(r["pair"] for r in beh)) == 2, "expect both query orders per item"

    # sentence aggregation: 2 rows per id; y complementary
    uniq, inv = np.unique(ids, return_inverse=True)
    n_s = len(uniq)
    cnt = np.bincount(inv)
    assert (cnt == 2).all(), "each cross sentence must have both query orders"
    y_m = np.bincount(inv, weights=yb) / cnt
    assert np.abs(y_m - 0.5).max() < 1e-9, "y must be complementary per sentence"
    p_s = np.bincount(inv, weights=p_all) / cnt
    s_row = np.array([np.flatnonzero(inv == j)[0] for j in range(n_s)])   # one row per sentence
    idx_s, form_s = idx[s_row], form_b[s_row]
    n_row = len(ids)

    from scipy.stats import spearmanr

    def margins_at(l, clf=None):
        if clf is None:
            clf = fit_probe(span_mat(l), sp_y)
        F = feats[l][idx]                       # rows (2 per sentence)
        hu = np.where(su[:, None] == 0, F[:, :H2], F[:, H2:])
        hv = np.where(su[:, None] == 1, F[:, :H2], F[:, H2:])
        d = (2 * yb - 1) * (clf.decision_function(hu) - clf.decision_function(hv))
        # INVARIANT: margin must be identical for both query orders of a sentence
        dbar = np.bincount(inv, weights=d) / np.bincount(inv)
        assert np.abs(d - dbar[inv]).max() < 1e-5, "margin differs across query orders"
        return d[s_row], d, clf

    forms_sorted = sorted(set(form_b))

    def rho_within(mg):
        vals = [spearmanr(mg[form_s == f], p_s[form_s == f])[0] for f in forms_sorted]
        return float(np.nanmean(vals))

    rho_l, p_l, rho_w_l = [], [], []
    for l in range(feats.shape[0]):
        ds, _, _ = margins_at(l)
        r, p = spearmanr(ds, p_s)
        rho_l.append(r); p_l.append(p)
        rho_w_l.append(rho_within(ds))
    rho_l = np.array(rho_l); p_l = np.array(p_l); rho_w_l = np.array(rho_w_l)
    q_l = bh_fdr(p_l)

    margin_s, _, clf_star = margins_at(lstar)
    rho_star, p_star = spearmanr(margin_s, p_s)
    rho_form = {f: spearmanr(margin_s[form_s == f], p_s[form_s == f])
                for f in forms_sorted}

    # span-probe pairwise accuracy PER FORM at l* (instrument diagnostic)
    acc_form = {}
    for f in forms_sorted:
        rows_f = [mid[i] for i in uniq[form_s == f]]
        F = feats[lstar][rows_f]
        s0 = clf_star.decision_function(F[:, :H2])
        s1 = clf_star.decision_function(F[:, H2:])
        n1_agent = np.array([int(meta[i]["label_patient_first"]) == 0 for i in rows_f])
        acc_form[f] = float(np.mean((s0 > s1) == n1_agent))
    print("span-probe pairwise acc by form @l*:", acc_form)

    # rank-OLS delta-R2 (sentence level): form dummies vs + margin
    from scipy.stats import rankdata
    yv = rankdata(p_s)
    Fd = np.zeros((n_s, len(forms_sorted)))
    for j, f in enumerate(forms_sorted):
        Fd[form_s == f, j] = 1

    def r2(Xd):
        A = np.c_[np.ones(n_s), Xd]
        resid = yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]
        return 1 - resid.var() / yv.var()
    d_r2 = r2(np.c_[Fd, rankdata(margin_s)]) - r2(Fd)

    out = dict(model=args.model, l_struct_span=lstar, layer_used=int(lstar),
               n_sentences_cross=int(n_s), n_rows_cross=int(n_row),
               rho_at_lstar=float(rho_star), p_at_lstar=float(p_star),
               rho_within_at_lstar=rho_within(margin_s),
               rho_by_form={k: dict(rho=float(v[0]), p=float(v[1]))
                            for k, v in rho_form.items()},
               span_pairwise_acc_by_form_at_lstar=acc_form,
               delta_R2_margin_over_form=float(d_r2),
               rho_by_layer=[float(x) for x in rho_l],
               p_by_layer=[float(x) for x in p_l],
               q_by_layer=[float(x) for x in q_l],
               rho_within_by_layer=[float(x) for x in rho_w_l],
               rho_max_layer=int(np.nanargmax(rho_l)), rho_max=float(np.nanmax(rho_l)),
               rho_within_max_layer=int(np.nanargmax(rho_w_l)),
               rho_within_max=float(np.nanmax(rho_w_l)))
    with open(RESULTS / f"{args.model}_rq5.json", "w") as f:
        json.dump(out, f, indent=2)
    show = {k: v for k, v in out.items()
            if k not in ("rho_by_layer", "p_by_layer", "q_by_layer", "rho_within_by_layer")}
    print(json.dumps(show, indent=2))

    # ---- figure: specificity curves + margin-tercile bars ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    L = np.arange(len(rho_l))
    ax[0].plot(L, rho_l, "o-", color="tab:red", lw=2, label=r"pooled $\rho_l$")
    ax[0].plot(L, rho_w_l, "s--", color="tab:blue", lw=2,
               label=r"within-form mean $\rho_l$ (form-partialed)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axvline(lstar, color="tab:blue", ls="--", alpha=0.4)
    ax[0].annotate(f"l*span = {lstar}", xy=(lstar, 0.03), fontsize=9, color="tab:blue")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("Spearman rho")
    ax[0].set_title(f"RQ5 specificity — {args.model}")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)

    import pandas as pd
    terc = np.asarray(pd.qcut(margin_s, 3, labels=["low", "mid", "high"], duplicates="drop"))
    means = [[np.mean(p_s[(form_s == f) & (terc == t)]) for t in ("low", "mid", "high")]
             for f in forms_sorted]
    x = np.arange(len(forms_sorted)); w = 0.26
    for k, t in enumerate(("low", "mid", "high")):
        ax[1].bar(x + (k - 1) * w, [m[k] for m in means], w, label=f"margin {t}")
    ax[1].set_xticks(x); ax[1].set_xticklabels(forms_sorted)
    ax[1].set_ylabel("mean P(correct) behavior"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Behavioral generalization by probe-margin tercile")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    outpng = FIGURES / f"rq5_{args.model}.png"
    fig.savefig(outpng, dpi=150)
    print(f"saved {outpng}")


if __name__ == "__main__":
    main()
