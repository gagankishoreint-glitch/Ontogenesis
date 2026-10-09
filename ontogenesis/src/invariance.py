#!/usr/bin/env python3
"""
G1 analysis: invariance-transition curves for one model.

  W_l (=S_l)  surface sensitivity, ABSOLUTE
       mean within-family cosine distance across the 4 surface forms
       (same lexicon, same roles, different form), cluster-bootstrap 95% CI.
       NOTE: may RISE with depth (contextual divergence) — informative either way.
  B_l  between-family, FORM-MATCHED distance (different families, same form)
  R_l  relative surface sensitivity = W_l / B_l
       <1: family members cluster against form-matched strangers;
       FALLING R_l = structural organization strengthening despite depth.
  T_l  structure decoding = logistic probe on [noun1; noun2] predicting
       label_patient_first:
         train:   {active, passive}, split P1  (fillers AND roles randomized)
         in-form: {active, passive}, split P2
         cross:   {cleft, topical},  split P2   ← the invariance metric
       + shuffled-label selectivity; BH-FDR across layers.

Outputs:
  results/{model}_curves.csv
  results/{model}_summary.json   (l_struct, l_surf, go_no_go)

Usage: python invariance.py --model bert_base
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (FEATURES, ITEMS_CSV, PROBE_CROSS_FORMS, PROBE_TRAIN_FORMS,
                    RESULTS, SPLIT_TEST, SPLIT_TRAIN, TRANS_FORMS_S, ensure_dirs)

RNG = np.random.default_rng(20261007)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        k = m - rank
        prev = min(prev, p[idx] * m / k)
        q[idx] = min(prev, 1.0)
    return q


def cos_dist(X):
    X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
    G = X @ np.swapaxes(X, -1, -2)
    return 1.0 - G


def within_between(pooled, meta):
    """pooled: [L, n, H] -> W[L], CI[L,L], B[L], family tensor access."""
    n_layers = pooled.shape[0]
    fam2rows = defaultdict(dict)
    for idx, m in enumerate(meta):
        if m["structure"] == "trans" and m["form"] in TRANS_FORMS_S:
            fam2rows[m["family_id"]][m["form"]] = idx
    fams = sorted(f for f, d in fam2rows.items() if len(d) == len(TRANS_FORMS_S))
    F = len(fams)
    # tensor V[l, f, k, H]
    V = np.empty((n_layers, F, len(TRANS_FORMS_S), pooled.shape[2]), dtype=np.float32)
    for fi, f in enumerate(fams):
        for k, form in enumerate(TRANS_FORMS_S):
            V[:, fi, k, :] = pooled[:, fam2rows[f][form], :]

    W = np.zeros(n_layers); CI = np.zeros((n_layers, 2)); B = np.zeros(n_layers)
    for l in range(n_layers):
        D = cos_dist(V[l])  # [F, 4, 4]
        tri = np.triu_indices(len(TRANS_FORMS_S), 1)
        per_fam = D[:, tri[0], tri[1]].mean(axis=1)
        W[l] = per_fam.mean()
        boots = [per_fam[RNG.integers(0, F, F)].mean() for _ in range(200)]
        CI[l] = np.percentile(boots, [2.5, 97.5])
        # between: different families, same form
        Db = []
        for k in range(len(TRANS_FORMS_S)):
            Dk = cos_dist(V[l][:, k, :])
            Db.append(Dk[np.triu_indices(F, 1)].mean())
        B[l] = float(np.mean(Db))
    return W, CI, B, V


def probe_acc(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ensure_dirs()

    with open(ITEMS_CSV, newline="") as f:
        meta = list(csv.DictReader(f))
    z = np.load(FEATURES / f"{args.model}.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)  # [n, L, 4, H]
    n, n_layers, slots, H = X.shape
    assert n == len(meta), "items/features mismatch"

    pooled = X[:, :, :3, :].mean(axis=2).transpose(1, 0, 2)  # [L, n, H]

    # ---------------- W_l / B_l / R_l -------------------------------------
    W, CI, B, _ = within_between(pooled, meta)
    R = W / (B + 1e-9)

    # ---------------- T_l --------------------------------------------------
    forms = [m["form"] for m in meta]
    splits = [m["split"] for m in meta]
    structs = [m["structure"] for m in meta]
    y = np.array([int(m["label_patient_first"]) for m in meta])

    tr = np.array([(s in PROBE_TRAIN_FORMS) and (sp == SPLIT_TRAIN) and (st == "trans")
                   for s, sp, st in zip(forms, splits, structs)])
    te_in = np.array([(s in PROBE_TRAIN_FORMS) and (sp == SPLIT_TEST) and (st == "trans")
                      for s, sp, st in zip(forms, splits, structs)])
    te_cr = np.array([(s in PROBE_CROSS_FORMS) and (sp == SPLIT_TEST) and (st == "trans")
                      for s, sp, st in zip(forms, splits, structs)])
    print(f"train={tr.sum()} in-test={te_in.sum()} cross-test={te_cr.sum()}")

    feats = np.concatenate([X[:, :, 0, :], X[:, :, 1, :]], axis=-1).transpose(1, 0, 2)

    acc_in = np.zeros(n_layers)
    acc_cr = np.zeros(n_layers)
    acc_shuf = np.zeros((n_layers, 3))
    for l in range(n_layers):
        acc_in[l] = probe_acc(feats[l][tr], y[tr], feats[l][te_in], y[te_in])
        acc_cr[l] = probe_acc(feats[l][tr], y[tr], feats[l][te_cr], y[te_cr])
        for s in range(3):
            ysh = RNG.permutation(y[tr])
            acc_shuf[l, s] = probe_acc(feats[l][tr], ysh, feats[l][te_cr], y[te_cr])
    acc_shuf_m = acc_shuf.mean(axis=1)

    from scipy.stats import binomtest
    pvals = [binomtest(int(round(acc_cr[l] * te_cr.sum())), int(te_cr.sum()), 0.5).pvalue
             for l in range(1, n_layers)]  # layer 0 = embeddings, excluded
    q = np.ones(n_layers)
    q[1:] = bh_fdr(pvals)
    sig_layers = [l for l in range(1, n_layers) if q[l] < 0.05 and acc_cr[l] > 0.5]
    l_struct = sig_layers[0] if sig_layers else None

    # surface half-drop layer for W (informational)
    base, mn = W[1], W[1:].min()
    half = base - 0.5 * (base - mn)
    l_surf = next((l for l in range(1, n_layers) if W[l] <= half), None)

    # ---------------- T_span,l : span-level AGENT probe (v3) ----------------
    # x = one span's hidden state; y = 1 iff that span is the AGENT.
    # Unlike label_patient_first (= f(form) instrument-wide), this label is
    # balanced WITHIN every construction, so cross-form accuracy requires role
    # marking rather than construction recognition. Train on individual spans
    # (P1 {active,passive}); evaluate PAIRWISE per sentence: correct iff the
    # agent span's logit > the patient span's logit. One Bernoulli per
    # sentence (chance 0.5) -> binomtest + BH-FDR exactly as for T_l.
    sp_i, sp_k, sp_y, sp_f, sp_s = [], [], [], [], []
    for i, m in enumerate(meta):
        if m["structure"] != "trans":
            continue
        lab = int(m["label_patient_first"])       # 1 => noun1 is patient
        agent_k = lab                             # 0 => n1 is agent; 1 => n2 is agent
        for k in (0, 1):
            sp_i.append(i); sp_k.append(k)
            sp_y.append(int(k == agent_k))
            sp_f.append(m["form"]); sp_s.append(m["split"])
    sp_i = np.array(sp_i); sp_k = np.array(sp_k); sp_y = np.array(sp_y)
    sp_tr = np.array([(f in PROBE_TRAIN_FORMS) and (s == SPLIT_TRAIN)
                      for f, s in zip(sp_f, sp_s)])
    # trans sentences in meta order; both slots appended consecutively -> pos
    trans_rank, _r = {}, 0
    for i, m in enumerate(meta):
        if m["structure"] == "trans":
            trans_rank[i] = _r; _r += 1
    pos = {(i, k): 2 * trans_rank[i] + k for i in trans_rank for k in (0, 1)}
    sents_in = [i for i, m in enumerate(meta) if m["structure"] == "trans"
                and m["form"] in PROBE_TRAIN_FORMS and m["split"] == SPLIT_TEST]
    sents_cr = [i for i, m in enumerate(meta) if m["structure"] == "trans"
                and m["form"] in PROBE_CROSS_FORMS and m["split"] == SPLIT_TEST]
    H2 = feats.shape[2] // 2

    def span_mat(l):
        F = feats[l][sp_i]                              # [m, 2H]
        return np.where(sp_k[:, None] == 0, F[:, :H2], F[:, H2:])

    def fit_probe(Xtr, ytr):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xtr, ytr)
        return clf

    def pair_acc(clf, X, sents):
        a = np.array([pos[(i, 0)] for i in sents])
        b = np.array([pos[(i, 1)] for i in sents])
        d0, d1 = clf.decision_function(X[a]), clf.decision_function(X[b])
        n1_agent = np.array([int(meta[i]["label_patient_first"]) == 0 for i in sents])
        return float(np.mean((d0 > d1) == n1_agent))

    acc_span_in = np.zeros(n_layers)
    acc_span_cr = np.zeros(n_layers)
    acc_span_shuf = np.zeros((n_layers, 3))
    for l in range(n_layers):
        Xs = span_mat(l)
        clf = fit_probe(Xs[sp_tr], sp_y[sp_tr])
        acc_span_in[l] = pair_acc(clf, Xs, sents_in)
        acc_span_cr[l] = pair_acc(clf, Xs, sents_cr)
        for s in range(3):
            ysh = RNG.permutation(sp_y[sp_tr])
            acc_span_shuf[l, s] = pair_acc(fit_probe(Xs[sp_tr], ysh), Xs, sents_cr)
    acc_span_shuf_m = acc_span_shuf.mean(axis=1)
    pvals_sp = [binomtest(int(round(acc_span_cr[l] * len(sents_cr))), len(sents_cr), 0.5).pvalue
                for l in range(1, n_layers)]
    q_sp = np.ones(n_layers)
    q_sp[1:] = bh_fdr(pvals_sp)
    sig_sp = [l for l in range(1, n_layers) if q_sp[l] < 0.05 and acc_span_cr[l] > 0.5]
    l_struct_span = sig_sp[0] if sig_sp else None

    # ---------------- write outputs --------------------------------------
    rows = []
    for l in range(n_layers):
        rows.append(dict(layer=l, W_l=W[l], W_lo=CI[l, 0], W_hi=CI[l, 1],
                         B_l=B[l], R_l=R[l],
                         acc_in=acc_in[l], acc_cross=acc_cr[l],
                         acc_shuf=acc_shuf_m[l], q_cross=q[l],
                         acc_span_in=acc_span_in[l], acc_span_cross=acc_span_cr[l],
                         acc_span_shuf=acc_span_shuf_m[l], q_span=q_sp[l]))
    out_csv = RESULTS / f"{args.model}_curves.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = dict(
        model=args.model, n_items=n, n_layers=n_layers,
        W_start=float(W[1]), W_final=float(W[-1]), W_max=float(W.max()),
        R_start=float(R[1]), R_final=float(R[-1]), R_min=float(R.min()),
        acc_cross_max=float(acc_cr.max()),
        acc_cross_at_max_layer=int(acc_cr.argmax()),
        acc_cross_final=float(acc_cr[-1]),
        acc_in_final=float(acc_in[-1]),
        l_struct=l_struct, l_surf=l_surf,
        go_no_go=bool(acc_cr.max() > 0.7 and l_struct is not None),
        # v3 span probe (role marking; construction-balanced labels)
        l_struct_span=l_struct_span,
        acc_span_cross_max=float(acc_span_cr.max()),
        acc_span_cross_max_layer=int(acc_span_cr.argmax()),
        acc_span_cross_final=float(acc_span_cr[-1]),
        go_no_go_span=bool(acc_span_cr.max() > 0.7 and l_struct_span is not None),
    )
    with open(RESULTS / f"{args.model}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
