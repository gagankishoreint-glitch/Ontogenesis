#!/usr/bin/env python3
"""Supplementary strict-cleft retriever (insurance for the cleft arm).

The main retriever's cleft prescreen (CLEFT_RE + cleft_ok) is permissive.
This one accepts only high-precision it-clefts:

    It is/was  [full-NP focus]  who/that/whom/which  [full-NP subject]
               [transitive verb] ... [NP object] ... .

with pronoun/indefinite/adverbial foci rejected ("It was he who...",
"It was in 1946 that...", "It was raining that day") and pronoun clause
subjects rejected ("...that he told").  Appends NEW cand_ids to
data/natural_candidates.csv (never rewrites existing rows), skipping any
sentence already present in the pool.

  python data/retrieve_clefts.py --dry-run          # count + sample only
  python data/retrieve_clefts.py                    # brown + gutenberg, append
  python data/retrieve_clefts.py --source brown
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieve_candidates import (OUT, DET, REL_PRON, INTRANS, PRE_VERB,
                                 MODALS, LEX_RE, clean, read_brown,
                                 gutenberg_sentences)

BE = {"is", "was", "'s"}
REL = {"who", "that", "whom", "which"}
# foci that are not full NPs (pronouns, indefinites, adverbials)
FOCUS_EXCL = REL_PRON | {
    "it", "this", "that", "these", "those", "one", "ones", "all", "some",
    "any", "none", "something", "anything", "nothing", "everything",
    "somebody", "anybody", "nobody", "everybody", "someone", "anyone",
    "everyone", "much", "more", "most", "less", "least", "here", "there",
    "then", "now", "not", "so", "in", "on", "at", "for", "from", "with",
    "by", "to", "of", "under", "over", "after", "before", "during",
    "because", "although", "though", "while", "when", "where", "how",
    "only", "just", "even", "still", "already", "perhaps", "probably",
    "raining", "snowing", "likely", "possible", "clear", "obvious", "true",
    "said", "believed", "thought", "reported", "estimated", "found"}


def _np_start(low, words, i):
    """Full-NP-ish start at i: DET+noun or proper noun."""
    if i >= len(low):
        return False
    if low[i] in DET and i + 1 < len(words) and words[i + 1].isalpha():
        return True
    w = words[i].strip("\"'“‘([")
    return bool(w[:1].isupper() and w.isalpha() and len(w) > 1)


def strict_cleft(words):
    low = [w.lower().strip(",.;:") for w in words]
    if low[0] != "it" or low[1] not in BE:
        return False
    rel = next((r for r in range(3, len(low)) if low[r] in REL), None)
    if rel is None or rel - 2 > 5:                 # focus NP: 1..5 words
        return False
    focus = low[2:rel]
    if not focus or focus[0] in FOCUS_EXCL:
        return False
    if not _np_start(low, words, 2):               # focus must be full NP
        return False
    clause = low[rel + 1:]
    if not clause or clause[0] in FOCUS_EXCL or not clause[0].isalpha():
        return False                               # clause subject: full NP
    if clause[0] in DET:                           # det+noun subject
        vi = 2
    else:
        vi = 1
    if vi >= len(clause):
        return False
    verb = clause[vi]
    if not verb.isalpha() or verb in INTRANS or verb in PRE_VERB \
            or verb in MODALS:
        return False                               # transitive content verb
    obj = clause[vi + 1:]
    if not any(low2 in DET for low2 in obj):       # NP object downstream
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["brown", "gutenberg", "all"],
                    default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pool = []
    if args.source in ("brown", "all"):
        pool += [(w, "brown") for w in read_brown()]
    if args.source in ("gutenberg", "all"):
        pool += [(w, "gutenberg") for w in gutenberg_sentences()]

    # existing pool texts (dedupe) and next cand_id
    have, last = set(), 0
    if OUT.exists():
        for r in csv.DictReader(open(OUT, newline="")):
            have.add(" ".join(json.loads(r["words_json"])).lower())
            last = max(last, int(r["cand_id"][1:]))

    hits, seen = [], set()
    for w, src in pool:
        c = clean(w)
        if not c or not strict_cleft(c):
            continue
        key = " ".join(c).lower()
        if key in have or key in seen:
            continue
        seen.add(key)
        hits.append((c, src))
    print(f"strict clefts found: {len(hits)} "
          f"(brown={sum(s == 'brown' for _, s in hits)} "
          f"gutenberg={sum(s == 'gutenberg' for _, s in hits)})")
    for c, _ in hits[:10]:
        print("  " + " ".join(c))

    if args.dry_run:
        print("dry-run: nothing appended"); return
    if not hits:
        print("nothing to append"); return
    with open(OUT, "a", newline="") as f:
        wr = csv.writer(f)
        for i, (c, src) in enumerate(hits, last + 1):
            wr.writerow([f"c{i:04d}", src, "cleft",
                         int(bool(LEX_RE.search(" ".join(c)))),
                         " ".join(c), json.dumps(c)])
    print(f"appended {len(hits)} clefts as c{last + 1:04d}..c{last + len(hits):04d}"
          f" -> {OUT}\nannotate with: python data/annotate.py --only-form cleft")


if __name__ == "__main__":
    main()
