#!/usr/bin/env python3
"""
Ontogenesis instrument (C1): item-family dataset.

Design (blueprint D1):
  * TRANS families: one argument structure (agent PRED patient),
    realized in 4 surface forms: active, passive, cleft, patient-topicalization
    + a camouflaged 'swap' item (same active frame, roles reversed).
  * DAT families: dative alternation (NP-PP vs double-object).
  * AMBIG: PP-attachment controls.
All lexical fillers (nouns AND verbs) are split disjointly across P1 (probe train)
and P2 (probe test), so any cross-form decoding transfer must abstract over fillers.

Output: data/items.csv  (+ integrity report: span text must equal the word)

Columns:
  id, family_id, split, structure, form, text,
  agent_span_start/end      # the DOER's word
  patient_span_start/end    # the UNDERGOER's word (theme for dative)
  predicate_span_start/end
  noun1_span_start/end      # first content noun in surface order
  noun2_span_start/end
  label_patient_first       # 1 iff noun1 == patient span (probe label)
  lexical columns...
"""
import csv
import random
from itertools import product
from pathlib import Path

SEED = 20261007
rng = random.Random(SEED)

HERE = Path(__file__).resolve().parent
OUT = HERE / "items.csv"

# ---------------------------------------------------------------- lexicon
A_POOL = [
    "chef", "waiter", "nurse", "teacher", "student", "lawyer", "doctor", "pilot",
    "singer", "farmer", "editor", "guard", "actor", "writer", "manager", "officer",
    "baker", "coach", "dancer", "engineer", "plumber", "carpenter", "tailor",
    "banker", "broker", "curator", "librarian", "mechanic", "pharmacist",
    "physicist", "poet", "reporter", "sailor", "soldier", "surgeon", "captain",
]
B_POOL = [
    "uncle", "aunt", "cousin", "neighbor", "colleague", "partner", "client",
    "customer", "patient", "visitor", "tourist", "passenger", "driver", "rider",
    "athlete", "musician", "artist", "painter", "sculptor", "inventor", "scientist",
    "historian", "philosopher", "economist", "senator", "mayor", "judge", "priest",
    "monk", "king", "queen", "prince", "princess", "duke", "baron", "knight",
]
R_POOL = [
    "boy", "girl", "child", "infant", "man", "woman", "friend", "pupil", "intern",
    "volunteer", "founder", "owner", "tenant", "landlord", "vendor", "buyer",
    "seller", "supplier", "dealer", "host", "guest", "apprentice", "nephew",
    "niece", "widow", "widower", "hero", "villain", "spy", "detective",
    "grandson", "granddaughter", "coachman", "butler", "gardener", "chauffeur",
]
TRANS_VERBS = [
    "praise", "help", "invite", "visit", "call", "follow", "admire", "assist",
    "trust", "challenge", "notice", "greet", "support", "thank", "honor",
    "respect", "confirm", "describe", "examine", "guide", "inform", "mark",
    "test", "welcome",
]
DAT_VERBS = [
    "mail", "pass", "hand", "forward", "deliver", "assign", "award", "grant",
    "offer", "present",
]
THEMES = [
    "book", "letter", "parcel", "gift", "trophy", "medal", "prize", "report",
    "message", "package", "ticket", "coupon", "certificate", "diploma",
    "contract", "invitation", "photograph", "painting", "map", "diagram",
    "recipe", "poem", "novel", "magazine", "newspaper", "key", "watch", "radio",
    "camera", "laptop", "bottle", "basket", "bowl", "plate", "cup", "umbrella",
]
TOOLS = [
    "telescope", "binoculars", "ladder", "hammer", "camera", "knife", "brush",
    "rope", "whistle", "fishing-rod",
]


def past(v: str) -> str:
    return v + "d" if v.endswith("e") else v + "ed"


def split_pool(pool):
    p = pool[:]
    rng.shuffle(p)
    h = len(p) // 2
    return p[:h], p[h:]


def build(segments):
    """segments: list of (text, tag|None) -> (text, spans{tag:(s,e)}).
    First occurrence of each tag wins; every tagged word is verified later."""
    out, pos, spans = [], 0, {}
    for txt, tag in segments:
        start = pos
        out.append(txt)
        pos += len(txt)
        if tag and tag not in spans:
            spans[tag] = (start, pos)
    return "".join(out), spans


def emit(rows, family_id, split, structure, form, segments, expected, extra=None):
    """expected: {tag: word} for span-integrity verification."""
    text, spans = build(segments)
    for tag, word in expected.items():
        s, e = spans[tag]
        assert text[s:e] == word, (
            f"SPAN MISMATCH {form}: tag={tag} got={text[s:e]!r} want={word!r} text={text!r}")
    a = spans.get("agent", (-1, -1))
    p = spans.get("patient", (-1, -1))
    pred = spans.get("predicate", (-1, -1))
    cands = sorted(s for s in (a, p) if s[0] >= 0)
    noun1 = cands[0]
    noun2 = cands[1] if len(cands) > 1 else (-1, -1)
    label = 1 if (p[0] >= 0 and p[0] == noun1[0]) else 0
    row = {
        "id": f"item_{len(rows):05d}",
        "family_id": family_id,
        "split": split,
        "structure": structure,
        "form": form,
        "text": text,
        "agent_span_start": a[0], "agent_span_end": a[1],
        "patient_span_start": p[0], "patient_span_end": p[1],
        "predicate_span_start": pred[0], "predicate_span_end": pred[1],
        "noun1_span_start": noun1[0], "noun1_span_end": noun1[1],
        "noun2_span_start": noun2[0], "noun2_span_end": noun2[1],
        "label_patient_first": label,
    }
    if extra:
        row.update(extra)
    rows.append(row)


def main():
    rows = []

    # G1 FIX (pool confound): ONE noun pool, roles randomized per family.
    # Both splits are random halves of the SAME distribution, and every word
    # appears as both doer and undergoer across families -> word identity
    # carries no label information at any layer.
    npool = A_POOL + B_POOL
    rng.shuffle(npool)
    h = len(npool) // 2
    n1, n2 = npool[:h], npool[h:]
    assert len(set(n1) & set(n2)) == 0 and len(n1) + len(n2) == 72

    r1, r2 = split_pool(R_POOL)
    v1, v2 = split_pool(TRANS_VERBS)
    dv1, dv2 = split_pool(DAT_VERBS)
    t1, t2 = split_pool(THEMES)
    halves = {
        "P1": dict(N=n1, R=r1, V=v1, DV=dv1, T=t1),
        "P2": dict(N=n2, R=r2, V=v2, DV=dv2, T=t2),
    }
    n_trans, n_dat = 300, 200

    for split, H in halves.items():
        # ------------------------------------------------ TRANS families
        # ordered pairs (doer, undergoer), doer != undergoer, both directions
        pairs = [(u, v) for u in H["N"] for v in H["N"] if u != v]
        rng.shuffle(pairs)
        combos = [(ag, pt, rng.choice(H["V"])) for ag, pt in pairs[:n_trans]]
        # role-coverage check: every word must appear on BOTH sides
        as_doer = {c[0] for c in combos}
        as_under = {c[1] for c in combos}
        missing = (set(H["N"]) - as_doer) | (set(H["N"]) - as_under)
        assert not missing, f"role coverage failure in {split}: {sorted(missing)}"
        for i, (ag, pt, vb) in enumerate(combos):
            fid = f"trans_{split}_{i:04d}"
            P = past(vb)
            lex = dict(agent_word=ag, patient_word=pt, verb=vb)
            emit(rows, fid, split, "trans", "active",
                 [("The ", None), (ag, "agent"), (" ", None), (P, "predicate"),
                  (" the ", None), (pt, "patient"), (".", None)],
                 dict(agent=ag, predicate=P, patient=pt), lex)
            emit(rows, fid, split, "trans", "passive",
                 [("The ", None), (pt, "patient"), (" was ", None), (P, "predicate"),
                  (" by the ", None), (ag, "agent"), (".", None)],
                 dict(patient=pt, predicate=P, agent=ag), lex)
            emit(rows, fid, split, "trans", "cleft",
                 [("It was the ", None), (ag, "agent"), (" who ", None),
                  (P, "predicate"), (" the ", None), (pt, "patient"), (".", None)],
                 dict(agent=ag, predicate=P, patient=pt), lex)
            emit(rows, fid, split, "trans", "topical",
                 [("The ", None), (pt, "patient"), (", the ", None), (ag, "agent"),
                  (" ", None), (P, "predicate"), (".", None)],
                 dict(patient=pt, agent=ag, predicate=P), lex)
            emit(rows, fid, split, "trans", "swap",
                 [("The ", None), (pt, "agent"), (" ", None), (P, "predicate"),
                  (" the ", None), (ag, "patient"), (".", None)],
                 dict(agent=pt, predicate=P, patient=ag), lex)

        # ------------------------------------------------ DAT families
        dcombos = list(product(H["N"], H["DV"], H["T"], H["R"]))
        rng.shuffle(dcombos)
        for i, (ag, dv, th, rc) in enumerate(dcombos[:n_dat]):
            fid = f"dat_{split}_{i:04d}"
            P = past(dv)
            lex = dict(agent_word=ag, theme_word=th, recipient_word=rc, verb=dv)
            emit(rows, fid, split, "dat", "dat_pp",
                 [("The ", None), (ag, "agent"), (" ", None), (P, "predicate"),
                  (" the ", None), (th, "patient"), (" to the ", None),
                  (rc, "recipient"), (".", None)],
                 dict(agent=ag, predicate=P, patient=th, recipient=rc), lex)
            emit(rows, fid, split, "dat", "dat_do",
                 [("The ", None), (ag, "agent"), (" ", None), (P, "predicate"),
                  (" the ", None), (rc, "recipient"), (" the ", None),
                  (th, "patient"), (".", None)],
                 dict(agent=ag, predicate=P, recipient=rc, patient=th), lex)

    # ------------------------------------------- attachment-ambiguity aux
    n = 0
    for ag, pt, tool in product(npool[:10], npool[10:20], TOOLS):
        if n >= 60:
            break
        emit(rows, f"ambig_aux_{n:03d}", "aux", "ambig", "attach",
             [("The ", None), (ag, "agent"), (" ", None), ("saw", "predicate"),
              (" the ", None), (pt, "patient"), (" with the ", None),
              (tool, "theme"), (".", None)],
             dict(agent=ag, predicate="saw", patient=pt, theme=tool),
             dict(agent_word=ag, patient_word=pt, tool=tool, verb="see"))
        n += 1

    # ---------------------------------------------------------------- write
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter((r["structure"], r["form"], r["split"]) for r in rows)
    print(f"Wrote {len(rows)} items -> {OUT}  (all span assertions passed)")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")
    lab = {}
    for r in rows:
        lab.setdefault(r["form"], set()).add(r["label_patient_first"])
    print("label sets by form:", {k: sorted(v) for k, v in lab.items()})


if __name__ == "__main__":
    main()
