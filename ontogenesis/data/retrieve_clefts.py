#!/usr/bin/env python3
"""Supplementary cleft retriever (insurance for the cleft arm).

Two explicit, logged retrieval modes over an explicit, versioned corpus:

  --pattern strict   (DEFAULT — unchanged original behaviour)
      High-precision it-cleft filter:  It is/was [full-NP focus] who/that/
      whom/which [full-NP subject] [transitive verb] ... [NP object] ...
      with pronoun/indefinite/adverbial foci rejected ("It was he who...",
      "It was in 1946 that...", "It was raining that day") and pronoun
      clause subjects rejected ("...that he told").

  --pattern lenient  (Phase-1 retrieval repair, 2026-10-09)
      The ESTABLISHED T1 prescreen from retrieve_candidates.py (protocol §3):
      CLEFT_RE (it-cleft with NP focus, extraposed-clause and pronoun-focus
      foci excluded) + cleft_ok (relative clause must have a full-NP subject
      and a transitive verb).  This is the filter that produced the 132-cleft
      pool; it is reused here rather than introducing a new NLP stack.
      Lenient candidates still require human annotation — nothing is
      auto-accepted.

  --shelf v1         (DEFAULT) the original curated 24-book Gutenberg shelf
                     (GUTENBERG_SHELF_V1 in retrieve_candidates.py).
  --shelf v2         v1 + 8 explicitly listed, versioned additions
                     (GUTENBERG_SHELF_V2, 2026-10-09) — bounded escalation
                     for the Phase-1 cleft deficit.  No candidate counts are
                     claimed for any shelf; dry runs report actuals.

Appends NEW cand_ids to the candidate pool (default
data/natural_candidates.csv) — never rewrites existing rows, never touches
data/natural_annotations.csv, and skips any sentence already present in the
pool.  Dedup uses the documented norm_key (retrieve_candidates.norm_key:
tokens joined with single spaces, lowercased).  New ids continue from the
max existing numeric cand_id suffix and are checked against the full
existing id set (no collisions).  A backup copy of the pool
(natural_candidates.backup.csv) is made before any append.

Dry runs write nothing and report, per source: raw matches, post-filter
matches, already-present matches, duplicate matches and net-new candidates.

Usage:
  python data/retrieve_clefts.py --dry-run                       # strict, v1 (as before)
  python data/retrieve_clefts.py --source brown --dry-run
  python data/retrieve_clefts.py --pattern lenient --shelf v2 --dry-run   # report only
  python data/retrieve_clefts.py --pattern lenient --shelf v2             # append net-new
  python data/retrieve_clefts.py --pattern lenient --shelf v2 --limit 100 # capped append
  python data/retrieve_clefts.py --pattern lenient --out /tmp/new_pool.csv # separate pool
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieve_candidates import (OUT, DET, REL_PRON, INTRANS, PRE_VERB,
                                 MODALS, LEX_RE, CLEFT_RE, cleft_ok, clean,
                                 read_brown, gutenberg_sentences,
                                 GUTENBERG_SHELVES, norm_key)

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

EXPECTED_HEADER = ["cand_id", "source", "form_guess", "lexicon_overlap",
                   "text", "words_json"]


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


def lenient_cleft(words):
    """Lenient mode = the established T1 prescreen (protocol §3):
    CLEFT_RE (it-cleft with NP focus; extraposed clausal foci and pronoun
    foci excluded) AND cleft_ok (relative clause needs a full-NP subject
    and a transitive verb).  Identical to retrieve_candidates.classify's
    cleft branch — reuse, not a new filter."""
    return bool(CLEFT_RE.search(" ".join(words))) and cleft_ok(words)


PATTERNS = {"strict": strict_cleft, "lenient": lenient_cleft}


def load_existing(out_path):
    """Read an existing candidate pool.  Returns (norm_keys, cand_ids,
    max_numeric_suffix).  Aborts (refuses to touch the file) if the header
    does not match the candidate schema."""
    keys, ids, maxnum = set(), set(), 0
    if not out_path.exists() or out_path.stat().st_size == 0:
        return keys, ids, maxnum
    with open(out_path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header != EXPECTED_HEADER:
            raise SystemExit(
                f"refusing to touch {out_path}: header {header} does not "
                f"match the candidate schema {EXPECTED_HEADER}")
        for r in rdr:
            if not r or not r[0]:
                continue
            ids.add(r[0])
            try:
                maxnum = max(maxnum, int(r[0][1:]))
            except (ValueError, IndexError):
                print(f"WARNING: unrecognized cand_id {r[0]!r} in "
                      f"{out_path} — ignored for id allocation",
                      file=sys.stderr)
            try:
                keys.add(norm_key(json.loads(r[5])))
            except (ValueError, IndexError):
                keys.add(norm_key(r[4].split()))
    return keys, ids, maxnum


def collect(pool, existing_keys, pattern):
    """Run one retrieval pattern over pool = iterable of (words, source).

    Returns (hits, stats).  hits = net-new (words, source) pairs in pool
    order.  stats = {source: {raw, already_present, duplicates, net_new}};
    post-filter = raw - already_present (everything passing clean() + the
    pattern that is not already in the pool); net_new = post-filter minus
    within-run duplicates (norm_key comparison, documented in
    retrieve_candidates.norm_key)."""
    pred = PATTERNS[pattern]
    stats, seen, hits = {}, set(), []
    for words, src in pool:
        c = clean(words)
        if not c or not pred(c):
            continue
        st = stats.setdefault(src, dict(raw=0, already_present=0,
                                        duplicates=0, net_new=0))
        st["raw"] += 1
        key = norm_key(c)
        if key in existing_keys:
            st["already_present"] += 1
            continue
        if key in seen:
            st["duplicates"] += 1
            continue
        seen.add(key)
        st["net_new"] += 1
        hits.append((c, src))
    return hits, stats


def append_hits(out_path, hits, start_num):
    """Append net-new cleft candidates.  Append-only: existing rows are
    never rewritten.  Rows use the exact candidate-pool schema and the
    csv module's native CRLF line terminator, matching the committed file.
    The caller creates the backup first."""
    new_file = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(EXPECTED_HEADER)
        for i, (c, src) in enumerate(hits, start_num):
            wr.writerow([f"c{i:04d}", src, "cleft",
                         int(bool(LEX_RE.search(" ".join(c)))),
                         " ".join(c), json.dumps(c)])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Supplementary cleft retriever: strict (original) or "
                    "lenient (established T1 prescreen) over a versioned "
                    "Brown + Gutenberg shelf; appends net-new cleft "
                    "candidates, never rewrites existing rows.")
    ap.add_argument("--source", choices=["brown", "gutenberg", "all"],
                    default="all")
    ap.add_argument("--pattern", choices=["strict", "lenient"],
                    default="strict",
                    help="strict = original high-precision filter "
                         "(unchanged); lenient = established T1 prescreen "
                         "CLEFT_RE + cleft_ok (protocol §3)")
    ap.add_argument("--shelf", choices=sorted(GUTENBERG_SHELVES), default="v1",
                    help="versioned Gutenberg shelf (v1 = original 24 books; "
                         "v2 = v1 + 8 documented additions, 2026-10-09)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on appended candidates (0 = no cap)")
    ap.add_argument("--out", default=str(OUT),
                    help="candidate pool to append to (default: "
                         "data/natural_candidates.csv; append-only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; write nothing, back up nothing")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    existing_keys, existing_ids, maxnum = load_existing(out_path)

    pool = []
    if args.source in ("brown", "all"):
        pool += [(w, "brown") for w in read_brown()]
    if args.source in ("gutenberg", "all"):
        shelf = GUTENBERG_SHELVES[args.shelf]
        pool += [(w, "gutenberg")
                 for w in gutenberg_sentences(shelf)]

    hits, stats = collect(pool, existing_keys, args.pattern)
    avail = sum(st["net_new"] for st in stats.values())   # total available
    if args.limit and args.limit < len(hits):
        hits = hits[:args.limit]          # planned for append (< available)

    print(f"[mode={args.pattern}] [shelf={args.shelf}] "
          f"[source={args.source}] [out={out_path}]")
    tot = dict(raw=0, already_present=0, duplicates=0, net_new=0)
    for src in ("brown", "gutenberg"):
        if src not in stats:
            continue
        st = stats[src]
        post_filter = st["raw"] - st["already_present"]
        print(f"[source={src}] raw={st['raw']} post-filter={post_filter} "
              f"already-present={st['already_present']} "
              f"duplicates={st['duplicates']} net-new={st['net_new']}")
        for k in tot:
            tot[k] += st[k]
    post_total = tot["raw"] - tot["already_present"]
    print(f"[total] raw={tot['raw']} post-filter={post_total} "
          f"already-present={tot['already_present']} "
          f"duplicates={tot['duplicates']} net-new={tot['net_new']}")
    if args.limit:
        note = (f"[limit] --limit={args.limit}: {avail} net-new available, "
                f"{len(hits)} planned for append")
        if len(hits) < avail:
            note += f" ({avail - len(hits)} held back)"
        print(note)
    for c, _ in hits[:10]:
        print("  " + " ".join(c))

    next_num = maxnum + 1
    next_id = f"c{next_num:04d}"
    if next_id in existing_ids:
        raise SystemExit(f"ID collision: {next_id} already exists in "
                         f"{out_path} — aborting")
    print(f"existing pool: {len(existing_ids)} cand_ids "
          f"(max suffix c{maxnum:04d}) -> next id would be {next_id}")

    if args.dry_run:
        print("dry-run: nothing appended; candidate pool and "
              "natural_annotations.csv untouched")
        return
    if not hits:
        print("nothing to append")
        return

    last_id = f"c{next_num + len(hits) - 1:04d}"
    print(f"WRITE PLAN: append {len(hits)} rows to {out_path} as "
          f"{next_id}..{last_id} (form_guess=cleft, source=brown|gutenberg, "
          f"candidate-pool schema unchanged). Existing rows are never "
          f"rewritten; natural_annotations.csv is not touched.")
    if out_path.exists() and out_path.stat().st_size > 0:
        backup = out_path.with_name(out_path.stem + ".backup.csv")
        shutil.copy(out_path, backup)
        print(f"backup of pre-append pool: {backup}")
    append_hits(out_path, hits, next_num)
    print(f"appended {len(hits)} {args.pattern} clefts as "
          f"{next_id}..{last_id} -> {out_path}")
    print("annotate with: python data/annotate.py --only-form cleft")


if __name__ == "__main__":
    main()
