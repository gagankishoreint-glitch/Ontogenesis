#!/usr/bin/env python3
"""
RQ5 behavioral assay v3 — ANCHORED pair-query task (fixes label = f(form)).

v1 flaw (found 2026-10-08): label "is noun1 the patient" is a deterministic
function of the construction (active/swap/cleft -> 0, passive/topical -> 1), so
end-to-end training fit the form-level rule "passive morphology <=> patient
first": per-form accuracy saturated (topical 0.0204 = systematic shortcut
failure), and cross-form behavior carried no role information.

v3: each sentence yields TWO examples, one per query order:
    model input : the sentence (unchanged)
    head input  : [h_u ; h_v]      (query order = head order)
    label y     : 1 iff u is the AGENT of the sentence
Every construction now contains BOTH labels (query (n1,n2) and (n2,n1)),
so construction identity is uninformative; position-of-u alone contradicts
across {active, passive} and is suppressed. The model must read contextual
role cues from the span representations.

Train   : P1 {active, passive} x 2 query orders = 1,200 (balanced per form)
Select  : best epoch on in-form P2 {active, passive} x 2 (no test leakage)
Evaluate: P2 {active, passive, cleft, topical} x 2 = 2,400 graded P(correct)

Output: results/{model}_behavior.csv
  columns: id, form, split, slot_u, pair, y, p_correct, pred
    slot_u = sentence slot (0/1) holding the queried u
    pair   = query order label, e.g. "q01" (u in slot 0) / "q10"
  Per-form mean AND std of p_correct are printed: within-form std is the
  variance RQ5 needs — if a form's std collapses near 0, apply the
  pre-registered content-pressure fallback (reduced-relative train arm).

Usage: python behavior.py --model distilbert   (distilbert|bert_base|gpt2|roberta_base|bert_random)
"""
import argparse
import copy
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ITEMS_CSV, MAX_LEN, MODELS, RESULTS, SPLIT_TEST, SPLIT_TRAIN, device, ensure_dirs

SEED = 20261007
EPOCHS = 6
BATCH = 8
LR = 2e-5

TRAIN_FORMS = ["active", "passive"]      # P1 only
VAL_FORMS = ["active", "passive"]        # P2 in-form (epoch selection)
CROSS_FORMS = ["cleft", "topical"]       # P2 cross-form (final eval only)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def span_to_token(offsets, start, end):
    if start < 0:
        return None
    for t, (s, e) in enumerate(offsets):
        if e > s and s < end and e > start:
            return t
    return None


def anchor_rows(meta_rows):
    """Expand each sentence into its two query orders (anchored labels)."""
    out = []
    for r in meta_rows:
        lab = int(r["label_patient_first"])   # 1 => noun1 is patient
        agent_slot = lab                      # 0 => n1 is agent; 1 => n2 is agent
        for su in (0, 1):                     # u occupies sentence slot su
            out.append(dict(
                id=r["id"], form=r["form"], split=r["split"],
                text=r["text"],
                n1s=int(r["noun1_span_start"]), n1e=int(r["noun1_span_end"]),
                n2s=int(r["noun2_span_start"]), n2e=int(r["noun2_span_end"]),
                slot_u=su, pair=f"q{su}{1 - su}",
                y=int(su == agent_slot),
            ))
    return out


def load_rows():
    with open(ITEMS_CSV, newline="") as f:
        meta = list(csv.DictReader(f))
    tr = [r for r in meta if r["structure"] == "trans" and r["split"] == SPLIT_TRAIN
          and r["form"] in TRAIN_FORMS]
    ev = [r for r in meta if r["structure"] == "trans" and r["split"] == SPLIT_TEST
          and r["form"] in VAL_FORMS + CROSS_FORMS]
    return anchor_rows(tr), anchor_rows(ev)


def balance_report(rows):
    """Label balance per form — must be 0.5 everywhere (the v3 guarantee)."""
    seen = {}
    for r in rows:
        seen.setdefault(r["form"], []).append(r["y"])
    return {f: float(np.mean(v)) for f, v in sorted(seen.items())}


class PairDS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(tok):
    def collate(batch):
        enc = tok([r["text"] for r in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        i1, i2, y = [], [], []
        for j, r in enumerate(batch):
            a = span_to_token(offs[j].tolist(), r["n1s"], r["n1e"])
            b = span_to_token(offs[j].tolist(), r["n2s"], r["n2e"])
            assert a is not None and b is not None, r["id"]
            su = int(r["slot_u"])              # head order = query order
            i1.append(a if su == 0 else b)     # slot for u
            i2.append(b if su == 0 else a)     # slot for v
            y.append(int(r["y"]))
        return enc, torch.tensor(i1), torch.tensor(i2), torch.tensor(y), batch
    return collate


class PairClassifier(nn.Module):
    def __init__(self, base, H, dropout=0.1):
        super().__init__()
        self.base = base
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * H, 2))

    def forward(self, enc, i1, i2):
        h = self.base(**enc).last_hidden_state
        a = h[torch.arange(h.size(0)), i1]
        b = h[torch.arange(h.size(0)), i2]
        return self.head(torch.cat([a, b], dim=-1))


@torch.no_grad()
def evaluate(model, loader, device, forms=None):
    """Per-item records + accuracy (restricted to `forms` if given)."""
    model.eval()
    recs, correct, n = [], 0, 0
    for enc, i1, i2, y, batch in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(enc, i1.to(device), i2.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        yy = y.numpy()
        for j, r in enumerate(batch):
            if forms is not None and r["form"] not in forms:
                continue
            p = probs[j, yy[j]]
            pred = int(probs[j].argmax())
            recs.append(dict(id=r["id"], form=r["form"], split=r["split"],
                             slot_u=int(r["slot_u"]), pair=r["pair"],
                             y=int(yy[j]), p_correct=float(p), pred=pred))
            correct += int(pred == yy[j]); n += 1
    return recs, (correct / n if n else 0.0)


def build_model(key):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    name = MODELS.get(key)
    if name is None:
        raise SystemExit(f"behavior.py: unknown model '{key}' (see config.MODELS)")
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if key == "bert_random":                 # control: random init, no pretraining
        cfg = AutoConfig.from_pretrained(name)
        base = AutoModel.from_config(cfg)
    else:
        base = AutoModel.from_pretrained(name)
    return tok, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ensure_dirs()
    set_seed(SEED)
    dev = device()

    tok, base = build_model(args.model)
    H = base.config.hidden_size
    model = PairClassifier(base, H).to(dev)

    tr_rows, ev_rows = load_rows()
    bal = balance_report(tr_rows)
    assert all(abs(v - 0.5) < 1e-9 for v in bal.values()), f"labels not balanced: {bal}"
    tr_loader = DataLoader(PairDS(tr_rows, tok), batch_size=BATCH, shuffle=True,
                           collate_fn=make_collate(tok))
    ev_loader = DataLoader(PairDS(ev_rows, tok), batch_size=BATCH, shuffle=False,
                           collate_fn=make_collate(tok))
    print(f"train={len(tr_rows)} (P1 active+passive x2 queries, label balance/form={bal}) "
          f"eval={len(ev_rows)} (P2 4 forms x2 queries)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_acc, best_state, best_ep = 0.0, None, -1
    for ep in range(EPOCHS):
        model.train()
        total, nb = 0.0, 0
        for enc, i1, i2, y, _ in tr_loader:
            enc = {k: v.to(dev) for k, v in enc.items()}
            loss = nn.functional.cross_entropy(model(enc, i1.to(dev), i2.to(dev)),
                                               y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        _, acc_in = evaluate(model, ev_loader, dev, forms=VAL_FORMS)  # IN-FORM only
        print(f"  epoch {ep+1}/{EPOCHS}  loss={total/nb:.4f}  in-form P2 acc={acc_in:.4f}")
        if acc_in > best_acc:
            best_acc, best_ep, best_state = acc_in, ep, copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    recs, _ = evaluate(model, ev_loader, dev)                       # all forms
    _, acc_in = evaluate(model, ev_loader, dev, forms=VAL_FORMS)
    _, acc_cr = evaluate(model, ev_loader, dev, forms=CROSS_FORMS)
    print(f"best epoch {best_ep+1} (selected on in-form P2 only)  "
          f"in-form acc={acc_in:.4f}  cross-form acc={acc_cr:.4f}")
    print("per-form graded outcome (mean +/- std of P(correct)):")
    by_form = {}
    for r in recs:
        by_form.setdefault(r["form"], []).append(r["p_correct"])
    for f, v in sorted(by_form.items()):
        flag = ("  <-- VARIANCE COLLAPSE (below ceiling, see header fallback)"
                if np.std(v) < 0.02 and np.mean(v) < 0.97 else "")
        print(f"  form={f:8s} n={len(v)}  mean={np.mean(v):.4f}  std={np.std(v):.4f}{flag}")

    out = RESULTS / f"{args.model}_behavior.csv"
    with open(out, "w", newline="") as w:
        wr = csv.DictWriter(w, fieldnames=list(recs[0].keys()))
        wr.writeheader(); wr.writerows(recs)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
