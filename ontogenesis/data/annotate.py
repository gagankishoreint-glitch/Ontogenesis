#!/usr/bin/env python3
"""
T2 of the naturalistic validation protocol — terminal annotator (§4).

  python data/annotate.py               # main pass (resumable: Ctrl-C anytime)
  python data/annotate.py --list        # stats only
  python data/annotate.py --recheck 20  # blind re-annotation of 20 keeps (QC)

Per candidate (screen shows numbered words + retrieval guess as a hint):
  form  : 1=cleft  2=topical  3=canonical  o=other->reject  r=reject
          u=uncertain->review queue  s=skip (rotate)  q=quit
  spans : n1 / n2 word ranges (surface order enforced), predicate word,
          agent index (1/2)  ->  [k]eep [e]dit [r]eject [b]ack

Append-only journal: data/natural_annotations.csv (crash-safe resume).
Export: python data/annotate.py --export
  -> data/naturalistic_items.csv   (items.csv schema + source,lexicon_overlap)
  Gates: >= --min-per-form keeps/form (default 40), offsets valid, spans map
  under MAX_LEN for the distilbert tokenizer (skip: --no-token-check),
  no duplicate texts.  Prints reject-reason histogram, per-form label balance,
  lexicon/source splits.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # data/
ROOT = HERE.parent
try:                                               # config needs torch; the
    sys.path.insert(0, str(ROOT / "src"))          # annotator itself does not
    from config import ITEMS_CSV, MAX_LEN
except Exception:
    ITEMS_CSV = HERE / "items.csv"
    MAX_LEN = 64
sys.path.insert(0, str(HERE))
from retrieve_candidates import PRE_VERB, MODALS   # verb lexicons (no torch)

AUXP = {"was", "were", "is", "are", "been", "being", "has", "have", "had"}
IRREG_VEN = {"shown", "given", "taken", "made", "done", "sent", "kept",
             "held", "told", "sold", "built", "bought", "caught", "taught",
             "sought", "won", "born", "worn", "torn", "drawn", "grown",
             "known", "thrown", "broken", "spoken", "chosen", "frozen",
             "gotten", "hidden", "ridden", "risen", "stolen", "sworn",
             "woven", "written", "eaten", "fallen", "forgiven", "sung",
             "sunk", "swum", "begun", "run", "come", "become", "set",
             "cut", "hit", "hurt", "let", "put", "read", "shut", "split",
             "spread", "fed", "led", "met", "paid", "said", "thought",
             "brought", "fought", "lost", "found", "heard", "seen", "felt",
             "left", "slept", "swept", "wept", "stood", "understood", "got",
             "forgotten"}


def _ven(w):
    return w in IRREG_VEN or (len(w) > 3 and (w.endswith("ed") or w.endswith("en")))


def _np_end(words, low, start, maxlen=4):
    """end (1-indexed incl.) of an NP starting at `start` (0-indexed)."""
    e = start
    while e + 1 < len(words) and (e - start) < maxlen - 1:
        nx = low[e + 1]
        if words[e + 1].endswith(",") or nx.endswith(",") or \
                nx in {"and", "or", "nor", ";", ":", "who", "that", "which"} or \
                nx in PRE_VERB or nx in MODALS:
            break
        e += 1
    return e + 1


def suggest(words, form):
    """Heuristic span prefill; the annotator always lets the human confirm."""
    low = [w.lower().strip(",.;:!?\"'“”‘’([]") for w in words]
    n = len(words)
    if form == "passive":
        for i in range(1, n - 1):
            if low[i] in AUXP and i + 1 < n and _ven(low[i + 1]):
                b = next((k for k in range(i + 2, n) if low[k] == "by"), None)
                if b is None or b + 1 >= n:
                    return None
                return dict(n1=(1, i), pred=i + 2, n2=(b + 2, _np_end(words, low, b + 1)),
                            agent=2)
        return None
    if form == "cleft":
        if low[0] != "it":
            return None
        if low[1] in {"is", "was", "were"}:
            f0 = 2
        elif low[1] == "has" and low[2] == "been":
            f0 = 3
        else:
            return None
        r = next((k for k in range(f0, n) if low[k] in {"who", "that", "which"}), None)
        if r is None or r == f0:
            return None
        v = next((k for k in range(r + 1, min(n, r + 5))
                  if _ven(low[k]) or low[k] in PRE_VERB), None)
        if v is None or v + 1 >= n:
            return None
        if low[v + 1] in {"it", "him", "her", "them", "me", "us", "you", "i"}:
            return None
        return dict(n1=(f0 + 1, r), pred=v + 1,
                    n2=(v + 2, _np_end(words, low, v + 1)), agent=1)
    if form == "canonical":
        v = next((k for k in range(1, min(n, 6))
                  if _ven(low[k]) or low[k] in PRE_VERB), None)
        if v is None or v + 1 >= n:
            return None
        if low[v + 1] in {"it", "him", "her", "them", "me", "us", "you", "i"}:
            return None
        return dict(n1=(1, v), pred=v + 1,
                    n2=(v + 2, _np_end(words, low, v + 1)), agent=1)
    return None


def fmt_span(words, rng_):
    return f"{rng_[0]}-{rng_[1]} '{' '.join(words[rng_[0]-1:rng_[1]])}'"

CANDS = HERE / "natural_candidates.csv"
ANNOT = HERE / "natural_annotations.csv"
EXPORT = HERE / "naturalistic_items.csv"
DISAGREE = HERE / "qc_disagreements.csv"
SEED = 20261007

ANNOT_COLS = ["cand_id", "status", "reason", "form", "n1a", "n1b", "n2a",
              "n2b", "pred", "agent", "method", "pass2"]
REASONS = {"w": "wrong_form", "a": "ambiguous_roles",
           "p": "pronoun_or_coord", "x": "other"}


def load_cands():
    if not CANDS.exists():
        raise SystemExit(f"{CANDS} missing — run data/retrieve_candidates.py first")
    rows = list(csv.DictReader(open(CANDS, newline="")))
    for r in rows:
        r["words"] = json.loads(r["words_json"])
    return rows


VALID_STATUS = {"kept", "rejected", "skipped", "pending"}
_warned = False


def load_annots():
    global _warned
    if not ANNOT.exists():
        return []
    rows, bad = [], 0
    for r in csv.DictReader(open(ANNOT, newline="")):
        if r.get("status", "") in VALID_STATUS:
            rows.append(r)
        else:
            bad += 1
    if bad and not _warned:
        _warned = True
        print(f"WARNING: {bad} journal row(s) have no valid status token "
              f"(column misalignment?) — ignored. Run "
              f"`python data/repair_journal.py`.", file=sys.stderr)
    return rows


def append_row(row, method="human"):
    new = not ANNOT.exists()
    row = dict(row)
    row.setdefault("method", method)
    with open(ANNOT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOT_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in ANNOT_COLS})


def parse_range(s, n, what):
    s = s.strip()
    try:
        if "-" in s:
            a, b = s.split("-", 1)
            a, b = int(a), int(b)
        else:
            a = b = int(s)
    except ValueError:
        return None, f"  ! '{s}' is not a range like 3 or 3-5"
    if not (1 <= a <= b <= n):
        return None, f"  ! {what} out of range 1..{n}"
    return (a, b), None


def overlaps(r1, r2):
    return not (r1[1] < r2[0] or r2[1] < r1[0])


def screen(cid, c, stats, recheck=False):
    words = c["words"]
    print("\033[2J\033[H", end="")             # clear
    tag = " [RECHECK]" if recheck else ""
    print(f"=== {cid}{tag}  guess={c['form_guess']}  src={c['source']}  "
          f"overlap={c['lexicon_overlap']} ===")
    print(f"kept: cleft={stats['cleft']} passive={stats.get('passive', 0)} "
          f"topical={stats['topical']} canonical={stats['canonical']} | "
          f"rej={stats['rejected']} review={stats['review']} | "
          f"remaining={stats['remaining']}")
    line = ""
    for i, w in enumerate(words, 1):
        piece = f"{i}:{w} "
        if len(line) + len(piece) > 96:
            print("  " + line.rstrip())
            line = ""
        line += piece
    if line:
        print("  " + line.rstrip())
    print(f"  (text: {' '.join(words)})")


def ask_form():
    print("  form: [1]cleft [2]topical [3]canonical [4]passive [o]ther-reject "
          "[r]eject [u]ncertain [s]kip [q]uit")
    while True:
        a = input("  > ").strip().lower()
        if a in {"1", "2", "3", "4", "o", "r", "u", "s", "q", "b", "e", "k"}:
            return a
        print("  ! enter one of 1 2 3 4 o r u s q")


def ask_spans(words, prefill=None):
    n = len(words)
    st = prefill or {}
    while True:
        r, err = parse_range(input("  n1 words (first-occurring NP): "), n, "n1")
        if err:
            print(err); continue
        n1 = r
        r, err = parse_range(input("  n2 words (second NP): "), n, "n2")
        if err:
            print(err); continue
        n2 = r
        if overlaps(n1, n2):
            print("  ! spans overlap — retry"); continue
        if n1[0] > n2[0]:
            n1, n2 = n2, n1                                  # surface order
            print(f"  (swapped: n1={n1[0]}-{n1[1]} n2={n2[0]}-{n2[1]})")
        while True:
            pr, err = parse_range(input("  predicate word: "), n, "predicate")
            if err:
                print(err); continue
            if overlaps(pr, n1) or overlaps(pr, n2):
                print("  ! predicate overlaps a noun span — retry"); continue
            break
        while True:
            ag = input("  agent of predicate [1/2]: ").strip()
            if ag in {"1", "2"}:
                break
            print("  ! 1 or 2")
        return n1, n2, pr, int(ag)


def run_main(args):
    cands = load_cands()
    by_id = {c["cand_id"]: c for c in cands}
    ann = load_annots()
    decided = {r["cand_id"]: r for r in ann
               if r.get("pass2", "") == "" and r["status"] in ("kept", "rejected")}
    reviews = {r["cand_id"] for r in ann if r["status"] == "review"}
    pending = [c["cand_id"] for c in cands
               if c["cand_id"] not in decided and c["cand_id"] not in reviews]
    if args.only_form:
        pending = [cid for cid in pending
                   if by_id[cid]["form_guess"] == args.only_form]
    if args.order_llm and (HERE / "llm_prescreen.csv").exists():
        llmk = {r["cand_id"]: int(r["keep"]) for r in
                csv.DictReader(open(HERE / "llm_prescreen.csv", newline=""))}
        pending.sort(key=lambda cid: 0 if llmk.get(cid, 0) else 1)  # stable
    print(f"candidates={len(cands)} decided={len(decided)} "
          f"review-queue={len(reviews)} pending={len(pending)}")

    rng = random.Random(SEED)
    while pending:
        cid = pending[0]
        c = by_id[cid]
        kept = [r for r in decided.values() if r["status"] == "kept"]
        stats = {
            "cleft": sum(r["form"] == "cleft" for r in kept),
            "passive": sum(r["form"] == "passive" for r in kept),
            "topical": sum(r["form"] == "topical" for r in kept),
            "canonical": sum(r["form"] == "canonical" for r in kept),
            "rejected": sum(r["status"] == "rejected" for r in decided.values()),
            "review": len(reviews),
            "remaining": len(pending),
        }
        screen(cid, c, stats)
        a = ask_form()
        if a == "q":
            print("saved — resume by re-running"); return
        if a == "s":
            pending.append(pending.pop(0)); continue
        if a in {"o", "r"}:
            print("  reason: [w]rong_form [a]mbiguous [p]ronoun/coord [x]other")
            rr = input("  > ").strip().lower()
            rr = rr if rr in REASONS else "x"
            append_row(dict(cand_id=cid, status="rejected", reason=REASONS[rr]))
            decided[cid] = dict(cand_id=cid, status="rejected", reason=REASONS[rr])
            pending.pop(0); continue
        if a == "u":
            append_row(dict(cand_id=cid, status="review"))
            reviews.add(cid); pending.pop(0); continue

        form = {"1": "cleft", "2": "topical", "3": "canonical",
                "4": "passive"}[a]
        w = c["words"]
        sg = suggest(w, form)
        if sg:
            print(f"  suggested: n1={fmt_span(w, sg['n1'])}  "
                  f"n2={fmt_span(w, sg['n2'])}  pred={sg['pred']} "
                  f"('{w[sg['pred']-1]}')  agent={sg['agent']}")
        while True:
            if sg:
                k = input("  [k]eep as suggested [e]dit spans [r]eject "
                          "[b]ack to form: ").strip().lower()
                if k == "k":
                    n1, n2, pr, ag = sg["n1"], sg["n2"], (sg["pred"],) and \
                        (sg["pred"], sg["pred"]), sg["agent"]
                elif k == "e":
                    n1, n2, pr, ag = ask_spans(w)
                elif k == "b":
                    break
                elif k == "r":
                    n1 = None
                else:
                    print("  ! enter k e r or b"); continue
            else:
                k = input("  [k]eep [e]dit spans [r]eject [b]ack to form: "
                          ).strip().lower()
                if k in ("k", "e"):
                    n1, n2, pr, ag = ask_spans(w)
                elif k == "b":
                    break
                elif k == "r":
                    n1 = None
                else:
                    print("  ! enter k e r or b"); continue
            if n1 is None:                        # reject
                print("  reason: [w]rong_form [a]mbiguous [p]ronoun/coord "
                      "[x]other (Enter=x)")
                rr = input("  > ").strip().lower()
                rr = rr if rr in REASONS else "x"
                append_row(dict(cand_id=cid, status="rejected", reason=REASONS[rr]))
                decided[cid] = dict(cand_id=cid, status="rejected")
                pending.pop(0); break
            lab = 0 if ag == 1 else 1
            print(f"  n1='{ ' '.join(w[n1[0]-1:n1[1]]) }'  "
                  f"n2='{ ' '.join(w[n2[0]-1:n2[1]]) }'  "
                  f"pred='{ ' '.join(w[pr[0]-1:pr[1]]) }'  "
                  f"agent={ag} -> label_patient_first={lab}")
            conf = input("  [k]eep [e]dit [r]eject [b]ack: ").strip().lower() \
                if not (sg and k == "k") else "k"
            if conf == "k":
                append_row(dict(cand_id=cid, status="kept", form=form,
                                n1a=n1[0], n1b=n1[1], n2a=n2[0], n2b=n2[1],
                                pred=pr[0], agent=ag))
                decided[cid] = dict(cand_id=cid, status="kept", form=form)
                pending.pop(0)
                break
            if conf == "r":
                print("  reason: [w]rong_form [a]mbiguous [p]ronoun/coord "
                      "[x]other (Enter=x)")
                rr = input("  > ").strip().lower()
                rr = rr if rr in REASONS else "x"
                append_row(dict(cand_id=cid, status="rejected", reason=REASONS[rr]))
                decided[cid] = dict(cand_id=cid, status="rejected")
                pending.pop(0); break
            if conf == "b":
                break
            sg = None                              # edit -> manual loop
    print("all candidates decided — run with --export")


# ------------------------------------------------------- LLM review mode
def live_stats(decided_rows, remaining):
    kept = [r for r in decided_rows if r["status"] == "kept"]
    return {
        "cleft": sum(r["form"] == "cleft" for r in kept),
        "topical": sum(r["form"] == "topical" for r in kept),
        "canonical": sum(r["form"] == "canonical" for r in kept),
        "passive": sum(r["form"] == "passive" for r in kept),
        "rejected": sum(r["status"] == "rejected" for r in decided_rows),
        "review": sum(r["status"] == "review" for r in decided_rows),
        "remaining": remaining,
    }


def run_review_llm(args):
    pres = HERE / "llm_prescreen.csv"
    if not pres.exists():
        raise SystemExit("data/llm_prescreen.csv missing — run "
                         "python data/llm_prescreen.py first")
    llm = {r["cand_id"]: r for r in csv.DictReader(open(pres, newline=""))}
    cands = {c["cand_id"]: c for c in load_cands()}
    decided = {r["cand_id"]: r for r in load_annots()}
    order = {"passive": 0, "canonical": 1, "cleft": 2, "topical": 3}
    queue = sorted((cid for cid, r in llm.items()
                    if int(r["keep"]) == 1 and cid not in decided
                    and cid in cands),
                   key=lambda cid: (order.get(llm[cid]["form"], 9), cid))
    print(f"LLM proposed keeps={sum(int(r['keep']) == 1 for r in llm.values())}"
          f"  to review={len(queue)}  (already decided={len(decided)})")
    while queue:
        cid = queue[0]
        c, L = cands[cid], llm[cid]
        words = c["words"]
        screen(cid, c, live_stats(list(decided.values()), len(queue)))
        sg = dict(form=L["form"], n1=(int(L["n1a"]), int(L["n1b"])),
                  n2=(int(L["n2a"]), int(L["n2b"])), pred=int(L["pred"]),
                  agent=int(L["agent"]))
        print(f"  LLM proposes KEEP as {sg['form']}: n1={fmt_span(words, sg['n1'])}"
              f"  n2={fmt_span(words, sg['n2'])}  pred={sg['pred']} "
              f"('{words[sg['pred']-1]}')  agent={sg['agent']}")
        print("  [k]eep as proposed [e]dit spans [r]eject [s]kip [q]uit")
        a = input("  > ").strip().lower()
        if a == "q":
            print("saved — resume by re-running"); return
        if a == "s":
            queue.append(queue.pop(0)); continue
        if a == "e":
            n1, n2, pr, ag = ask_spans(words)
            form = sg["form"]
        elif a == "k":
            n1, n2, pr, ag = sg["n1"], sg["n2"], (sg["pred"], sg["pred"]), \
                sg["agent"]
            form = sg["form"]
        elif a == "r":
            print("  reason: [w]rong_form [a]mbiguous [p]ronoun/coord "
                  "[x]other (Enter=x)")
            rr = input("  > ").strip().lower()
            rr = rr if rr in REASONS else "x"
            append_row(dict(cand_id=cid, status="rejected", reason=REASONS[rr]),
                       method="llm+human")
            decided[cid] = dict(cand_id=cid, status="rejected")
            queue.pop(0); continue
        else:
            print("  ! enter k e r s or q"); continue
        lab = 0 if ag == 1 else 1
        print(f"  n1='{ ' '.join(words[n1[0]-1:n1[1]]) }'  "
              f"n2='{ ' '.join(words[n2[0]-1:n2[1]]) }'  "
              f"pred='{ ' '.join(words[pr[0]-1:pr[1]]) }'  "
              f"agent={ag} -> label_patient_first={lab}")
        conf = "k" if a == "k" else input(
            "  [k]eep [r]eject [b]ack: ").strip().lower()
        if conf == "k":
            append_row(dict(cand_id=cid, status="kept", form=form,
                            n1a=n1[0], n1b=n1[1], n2a=n2[0], n2b=n2[1],
                            pred=pr[0], agent=ag), method="llm+human")
            decided[cid] = dict(cand_id=cid, status="kept", form=form)
            queue.pop(0)
        elif conf == "r":
            print("  reason: [w]rong_form [a]mbiguous [p]ronoun/coord "
                  "[x]other (Enter=x)")
            rr = input("  > ").strip().lower()
            rr = rr if rr in REASONS else "x"
            append_row(dict(cand_id=cid, status="rejected", reason=REASONS[rr]),
                       method="llm+human")
            decided[cid] = dict(cand_id=cid, status="rejected")
            queue.pop(0)
        # b -> loop back to the same candidate
    print("review queue empty — run --export")


def run_agree():
    pres = HERE / "llm_prescreen.csv"
    if not pres.exists():
        raise SystemExit("no llm_prescreen.csv")
    llm = {r["cand_id"]: r for r in csv.DictReader(open(pres, newline=""))}
    # agreement = human decisions on LLM-proposed keeps (mode-agnostic,
    # so it works even for journals written by older file versions)
    ann = [r for r in load_annots()
           if r["cand_id"] in llm and int(llm[r["cand_id"]]["keep"]) == 1
           and r.get("pass2", "") == ""]
    kept = [r for r in ann if r["status"] == "kept"]
    rej = [r for r in ann if r["status"] == "rejected"]
    exact = sum(1 for r in kept
                if llm[r["cand_id"]]["form"] == r["form"]
                and llm[r["cand_id"]]["n1a"] == r["n1a"]
                and llm[r["cand_id"]]["n1b"] == r["n1b"]
                and llm[r["cand_id"]]["n2a"] == r["n2a"]
                and llm[r["cand_id"]]["n2b"] == r["n2b"]
                and llm[r["cand_id"]]["pred"] == r["pred"]
                and llm[r["cand_id"]]["agent"] == r["agent"])
    print(f"reviewed={len(ann)}  kept={len(kept)}  rejected={len(rej)}")
    if ann:
        print(f"human keep-rate of LLM proposals: {len(kept)/len(ann):.1%}")
    if kept:
        print(f"span+form EXACT agreement among kept: {exact}/{len(kept)} "
              f"= {exact/len(kept):.1%}")
    meta = json.load(open(HERE / "llm_prescreen.meta.json")) \
        if (HERE / "llm_prescreen.meta.json").exists() else {}
    print(f"co-annotator: {meta.get('backend','?')}/{meta.get('model','?')} "
          f"temp={meta.get('temperature','?')}")


# ------------------------------------------------------------------ export
def word_offsets(words):
    offs, pos = [], 0
    for w in words:
        offs.append(pos)
        pos += len(w) + 1
    return offs


def span_chars(words, offs, rng_):
    a, b = rng_
    w = words[a - 1]
    lead = len(w) - len(w.lstrip("\"'“‘(["))
    core = w.strip("\"'“‘([").rstrip(",.;:!?\"'”’)]")
    if not core:
        return None
    s = offs[a - 1] + lead
    wl = words[b - 1]
    leadl = len(wl) - len(wl.lstrip("\"'“‘(["))
    corel = wl.strip("\"'“‘([").rstrip(",.;:!?\"'”’)]")
    if not corel:
        return None
    e = offs[b - 1] + leadl + len(corel)
    return s, e


def span_to_token(offsets, start, end):
    if start < 0:
        return None
    for t, (s, e) in enumerate(offsets):
        if e > s and s < end and e > start:
            return t
    return None


def run_export(args):
    cands = {c["cand_id"]: c for c in load_cands()}
    ann = load_annots()
    kept, demoted = {}, []
    for r in ann:
        if r.get("pass2", "") or r["status"] != "kept":
            continue
        cid = r["cand_id"]
        if cid in kept:
            continue
        c = cands.get(cid)
        if c is None:
            raise SystemExit(f"annotation references unknown {cid}")
        words, offs = c["words"], word_offsets(c["words"])
        n1 = span_chars(words, offs, (int(r["n1a"]), int(r["n1b"])))
        n2 = span_chars(words, offs, (int(r["n2a"]), int(r["n2b"])))
        pr = span_chars(words, offs, (int(r["pred"]), int(r["pred"])))
        if not (n1 and n2 and pr) or n1[0] >= n2[0]:
            demoted.append((cid, "bad_offsets")); continue
        text = c["text"]
        for s, e in (n1, n2, pr):
            if not text[s:e].strip("\"'“‘([,.;:") :
                demoted.append((cid, "bad_offsets")); break
        else:
            kept[cid] = dict(r, text=text, n1=n1, n2=n2, pr=pr)

    # duplicate texts
    seen = set()
    for cid in list(kept):
        key = kept[cid]["text"].lower()
        if key in seen:
            demoted.append((cid, "dup")); kept.pop(cid)
        seen.add(key)

    # token check (mirrors extract.py truncation behaviour)
    if not args.no_token_check:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("distilbert-base-uncased",
                                                use_fast=True)
            for cid in list(kept):
                k = kept[cid]
                enc = tok(k["text"], return_offsets_mapping=True,
                          truncation=True, max_length=MAX_LEN)
                offs = enc["offset_mapping"]
                ok = all(span_to_token(offs, s, e) is not None
                         for s, e in (k["n1"], k["n2"], k["pr"]))
                if not ok:
                    kept.pop(cid); demoted.append((cid, "too_long"))
        except Exception as e:
            print(f"WARNING: token check skipped ({e})")

    by_form = {}
    for cid, k in kept.items():
        by_form.setdefault(k["form"], []).append((cid, k))
    print("\n== QC report ==")
    print("kept per form: " + "  ".join(
        f"{f}={len(v)}" for f, v in sorted(by_form.items())))
    if demoted:
        from collections import Counter
        print("demoted: " + str(dict(Counter(r for _, r in demoted))))
    rej = [r for r in ann if r.get("pass2", "") == "" and r["status"] == "rejected"]
    from collections import Counter
    print(f"rejects ({len(rej)}): " +
          str(dict(Counter(r["reason"] for r in rej))))
    short = [f for f in ("cleft", "passive", "canonical")
             if len(by_form.get(f, [])) < args.min_per_form]
    if short:
        raise SystemExit(
            f"GATE FAIL: below {args.min_per_form} for {short} "
            f"({', '.join(f'{f}={len(by_form.get(f, []))}' for f in short)}). "
            "Add candidates (more shelf books / --source wikipedia), adjudicate "
            "the review queue, or pass --min-per-form N to override.")

    with open(ITEMS_CSV, newline="") as f:
        hdr = next(csv.reader(f))
    out_hdr = hdr + ["source", "lexicon_overlap"]
    with open(EXPORT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_hdr)
        w.writeheader()
        for cid in sorted(kept):
            k = kept[cid]
            c = cands[cid]
            words = c["words"]
            lab = int(k["agent"]) - 1                     # 1->0, 2->1
            n1, n2, pr = k["n1"], k["n2"], k["pr"]
            agent_sp, pat_sp = (n1, n2) if lab == 0 else (n2, n1)
            core = lambda rg: c["text"][rg[0]:rg[1]].strip("\"'“‘([,.;:")
            row = {h: "" for h in out_hdr}
            row.update(
                id=f"nat_{cid}", family_id=f"nat_{cid}", split="PN",
                structure="trans", form=k["form"], text=k["text"],
                agent_span_start=agent_sp[0], agent_span_end=agent_sp[1],
                patient_span_start=pat_sp[0], patient_span_end=pat_sp[1],
                predicate_span_start=pr[0], predicate_span_end=pr[1],
                noun1_span_start=n1[0], noun1_span_end=n1[1],
                noun2_span_start=n2[0], noun2_span_end=n2[1],
                label_patient_first=lab,
                agent_word=core(agent_sp), patient_word=core(pat_sp),
                verb=core(pr),
                source=c["source"], lexicon_overlap=c["lexicon_overlap"])
            w.writerow(row)
    print(f"\nwrote {EXPORT}  (n={len(kept)})")
    for f in sorted(by_form):
        v = [k for _, k in by_form[f]]
        if v:
            bal = sum(int(k["agent"]) - 1 for _, k in by_form[f]) / len(v)
            ov = sum(int(cands[cid]["lexicon_overlap"]) for cid, _ in by_form[f])
            print(f"  {f:9s} n={len(v):3d}  label_patient_first={bal:.3f}  "
                  f"lexicon_overlap={ov}")


def run_recheck(args):
    cands = {c["cand_id"]: c for c in load_cands()}
    ann = load_annots()
    orig = {r["cand_id"]: r for r in ann
            if r.get("pass2", "") == "" and r["status"] == "kept"}
    done2 = {r["cand_id"] for r in ann if r.get("pass2", "") == "recheck"}
    pool = [cid for cid in orig if cid not in done2]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    sample = pool[: args.recheck]
    print(f"blind re-annotation of {len(sample)} keeps "
          f"(previous answers hidden)")
    dis, agree = [], 0
    for cid in sample:
        c = cands[cid]
        screen(cid, c, {"cleft": "-", "topical": "-", "canonical": "-",
                        "rejected": "-", "review": "-", "remaining": "-"},
               recheck=True)
        a = ask_form()
        if a in {"o", "r"}:
            r2 = dict(cand_id=cid, status="rejected", reason="x")
        elif a == "u":
            r2 = dict(cand_id=cid, status="review")
        else:
            form = {"1": "cleft", "2": "topical", "3": "canonical",
                    "4": "passive"}[a]
            while True:
                n1, n2, pr, ag = ask_spans(c["words"])
                k = input("  [k]eep [r]eject [b]ack: ").strip().lower()
                if k == "b":
                    a = ask_form(); form = None; break
                if k == "r":
                    r2 = dict(cand_id=cid, status="rejected", reason="x")
                    break
                r2 = dict(cand_id=cid, status="kept", form=form, n1a=n1[0],
                          n1b=n1[1], n2a=n2[0], n2b=n2[1], pred=pr[0],
                          agent=ag)
                break
            if form is None:
                continue
        r2["pass2"] = "recheck"
        append_row(r2)
        o = orig[cid]
        same = all(str(o.get(k, "")) == str(r2.get(k, ""))
                   for k in ("status", "form", "n1a", "n1b", "n2a", "n2b",
                             "pred", "agent") if k in o or k in r2)
        if same:
            agree += 1
        else:
            dis.append(cid)
            print(f"  DISAGREE {cid}: pass1={ {k: o.get(k) for k in ('status','form','n1a','n1b','n2a','n2b','pred','agent')} } "
                  f"pass2={ {k: r2.get(k) for k in ('status','form','n1a','n1b','n2a','n2b','pred','agent')} }")
    rate = agree / len(sample) if sample else 1.0
    print(f"\nrecheck agreement: {agree}/{len(sample)} = {rate:.1%}")
    if rate < 0.95:
        print("WARNING: >5% disagreement — protocol requires full re-review "
              "of the affected form(s): " + ", ".join(sorted(
                  orig[c]["form"] for c in dis if orig[c]["status"] == "kept")))
    if dis and not DISAGREE.exists():
        with open(DISAGREE, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["cand_id"]); w.writerows([[c] for c in dis])
        print(f"wrote {DISAGREE}")


def show_list():
    ann = load_annots()
    orig = [r for r in ann if r.get("pass2", "") == ""]
    kept = [r for r in orig if r["status"] == "kept"]
    from collections import Counter
    print(f"decided={len(orig)} kept={len(kept)} "
          f"forms={dict(Counter(r['form'] for r in kept))} "
          f"rejects={dict(Counter(r['reason'] for r in orig if r['status'] == 'rejected'))} "
          f"review={sum(r['status'] == 'review' for r in orig)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--review-llm", action="store_true")
    ap.add_argument("--agree", action="store_true")
    ap.add_argument("--recheck", type=int, default=0, metavar="N")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--min-per-form", type=int, default=55)
    ap.add_argument("--only-form", choices=["cleft", "passive", "canonical",
                                            "topical"], default=None,
                    help="manual mode: only show candidates with this "
                         "retrieval guess")
    ap.add_argument("--order-llm", action="store_true",
                    help="manual mode: float LLM-proposed keeps to the front "
                         "(shortlist only; every item still human-decided)")
    ap.add_argument("--no-token-check", action="store_true")
    args = ap.parse_args()
    if args.list:
        show_list()
    elif args.agree:
        run_agree()
    elif args.export:
        run_export(args)
    elif args.review_llm:
        run_review_llm(args)
    elif args.recheck:
        run_recheck(args)
    else:
        try:
            run_main(args)
        except (EOFError, KeyboardInterrupt):
            print("\nsaved — resume by re-running")


if __name__ == "__main__":
    main()
