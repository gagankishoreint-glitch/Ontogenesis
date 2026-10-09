#!/usr/bin/env python3
"""
P1 surface controls + nonlinear probe — span-agent task (v3 instrument).

Question: is cross-form role decoding (T_span) about representations, or
could SURFACE statistics do it? Three probes, same masks/eval as invariance:

  position : x = span slot index only (layer-free). A pure position rule
             cannot fit {active, passive} (they contradict) -> pooled
             pairwise ~0.5 (cleft/topical mirror).
  bow      : x = sentence TF-IDF bag-of-words + span slot (layer-free).
             Reads passive morphology + position: fits train, transfers to
             agent-fronted cleft, fails patient-fronted-without-morphology
             topical -> pooled cross-form ~0.5.
  mlp      : nonlinear readout on frozen span features at l*_span.
             If the linear probe is already ~1.0, no nonlinear headroom.

Pre-registered expectation: position ~0.5, bow ~0.5 cross-form, mlp ~ linear.

Output: results/{model}_controls.json
Usage: python controls.py --model distilbert
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (FEATURES, ITEMS_CSV, PROBE_CROSS_FORMS, PROBE_TRAIN_FORMS,
                    RESULTS, SPLIT_TEST, SPLIT_TRAIN, ensure_dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ensure_dirs()

    with open(ITEMS_CSV, newline="") as f:
        meta = list(csv.DictReader(f))
    z = np.load(FEATURES / f"{args.model}.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)
    feats = np.concatenate([X[:, :, 0, :], X[:, :, 1, :]], axis=-1).transpose(1, 0, 2)
    H2 = feats.shape[2] // 2
    summ = json.load(open(RESULTS / f"{args.model}_summary.json"))
    lstar = summ.get("l_struct_span")

    # ---- shared span-example index (both slots of every trans sentence) --
    sp_i, sp_k, sp_y, sp_f, sp_s = [], [], [], [], []
    for i, m in enumerate(meta):
        if m["structure"] != "trans":
            continue
        lab = int(m["label_patient_first"])
        for k in (0, 1):
            sp_i.append(i); sp_k.append(k)
            sp_y.append(int(k == lab))            # k is the agent slot
            sp_f.append(m["form"]); sp_s.append(m["split"])
    sp_i = np.array(sp_i); sp_k = np.array(sp_k); sp_y = np.array(sp_y)
    tr_m = np.array([(f in PROBE_TRAIN_FORMS) and (s == SPLIT_TRAIN)
                     for f, s in zip(sp_f, sp_s)])
    te_in = np.array([(f in PROBE_TRAIN_FORMS) and (s == SPLIT_TEST)
                      for f, s in zip(sp_f, sp_s)])
    te_cr = np.array([(f in PROBE_CROSS_FORMS) and (s == SPLIT_TEST)
                      for f, s in zip(sp_f, sp_s)])
    sents_in = [i for i, m in enumerate(meta) if m["structure"] == "trans"
                and m["form"] in PROBE_TRAIN_FORMS and m["split"] == SPLIT_TEST]
    sents_cr = [i for i, m in enumerate(meta) if m["structure"] == "trans"
                and m["form"] in PROBE_CROSS_FORMS and m["split"] == SPLIT_TEST]
    trans_rank, r = {}, 0
    for i, m in enumerate(meta):
        if m["structure"] == "trans":
            trans_rank[i] = r; r += 1
    pos = {(i, k): 2 * trans_rank[i] + k for i in trans_rank for k in (0, 1)}
    n1_agent = {i: int(meta[i]["label_patient_first"]) == 0 for i in trans_rank}

    def pair_acc(scores, sents):
        """Pairwise eval: correct iff score(n1) > score(n2) matches truth."""
        ok = 0
        for i in sents:
            ok += int((scores[pos[(i, 0)]] > scores[pos[(i, 1)]]) == n1_agent[i])
        return ok / len(sents)

    out = {"model": args.model, "l_struct_span": lstar}
    tr_sent = sorted(set(sp_i[tr_m]))

    # ---- 1. position-only baseline (layer-free) --------------------------
    from sklearn.linear_model import LogisticRegression
    slot_x = sp_k[:, None].astype(float)
    clf = LogisticRegression(max_iter=2000).fit(slot_x[tr_m], sp_y[tr_m])
    sc = clf.decision_function(slot_x)
    out["position"] = dict(
        acc_in=pair_acc(sc, sents_in),
        acc_cross=pair_acc(sc, sents_cr),
        coef=float(clf.coef_.ravel()[0]),
        note="coef~0 (train balanced by slot) -> pairwise ties break to n2-agent",
    )

    # ---- 2. bag-of-words over a context WINDOW around each span ----------
    # (sentence-level BoW would cancel in the pairwise diff -> strawman)
    def word_spans(text):
        out, ws = [], 0
        for w in text.split():
            s = text.find(w, ws); out.append((s, s + len(w))); ws = s + len(w)
        return out
    wcache = {i: word_spans(meta[i]["text"]) for i in set(sp_i.tolist())}
    PAD = 25  # ~2-3 words either side of the span
    win_texts = []
    for i, k in zip(sp_i, sp_k):
        m = meta[i]
        s = int(m[f"noun{k+1}_span_start"]); e = int(m[f"noun{k+1}_span_end"])
        toks = [meta[i]["text"][a:b] for a, b in wcache[i]
                if b > s - PAD and a < e + PAD]
        win_texts.append(" ".join(toks))
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(min_df=2, max_features=20000, lowercase=True)
    vec.fit([win_texts[j] for j in range(len(sp_i)) if tr_m[j]])   # train rows only
    W = vec.transform(win_texts)
    slot_col = sp.csr_matrix(sp_k.reshape(-1, 1).astype(float))
    Xbow = sp.hstack([W, slot_col], format="csr")
    bow = LogisticRegression(max_iter=3000, C=1.0)
    bow.fit(Xbow[tr_m], sp_y[tr_m])
    sc = bow.decision_function(Xbow)
    out["bow"] = dict(
        acc_in=pair_acc(sc, sents_in),
        acc_cross=pair_acc(sc, sents_cr),
        n_features=int(Xbow.shape[1]),
    )

    # ---- 3. nonlinear (MLP) readout at l* --------------------------------
    if lstar is not None:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        F = feats[lstar]
        Xm = np.where(sp_k[:, None] == 0, F[sp_i, :H2], F[sp_i, H2:])
        Xs = StandardScaler().fit_transform(Xm)
        mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3,
                            early_stopping=True, validation_fraction=0.15,
                            max_iter=400, random_state=20261007)
        mlp.fit(Xs[tr_m], sp_y[tr_m])
        proba = mlp.predict_proba(Xs)[:, 1]
        out["mlp_at_lstar"] = dict(
            acc_train_pairwise=pair_acc(proba, tr_sent),
            acc_in=pair_acc(proba, sents_in),
            acc_cross=pair_acc(proba, sents_cr),
            layer=int(lstar),
        )
    else:
        out["mlp_at_lstar"] = None

    with open(RESULTS / f"{args.model}_controls.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
