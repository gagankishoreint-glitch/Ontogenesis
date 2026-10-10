#!/usr/bin/env python3
"""Repair data/natural_annotations.csv after mixed-version writes.

Symptoms this fixes: duplicate cand_ids, rows whose columns don't line up
with the header (written by a divergent annotate.py), so the annotator
re-shows already-decided candidates and --list/--agree undercount.

Strategy: parse every row position-agnostically (find the status token,
the form token, the six span integers, the source token anywhere in the
row), keep the LAST decision per cand_id, and rewrite a clean journal
under the current column order. A backup is made first.  Rows whose source
column is empty (written by older annotate.py versions) are backfilled
from data/natural_candidates.csv by cand_id; existing source values,
decisions, labels and span boundaries are never altered.
"""
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = HERE / "natural_annotations.csv"
STATUSES = {"kept", "rejected", "skipped", "pending"}
FORMS = {"cleft", "passive", "canonical", "topical"}
COLS = ["cand_id", "status", "form", "n1a", "n1b", "n2a", "n2b", "pred",
        "agent", "source", "pass2", "reason", "method"]


def is_int(x):
    return x.lstrip("-").isdigit()


def load_candidate_sources():
    """cand_id -> source from data/natural_candidates.csv (provenance
    backfill: journal rows written by older annotate.py versions have an
    empty source column; the candidate pool is the authoritative record)."""
    src = {}
    c = HERE / "natural_candidates.csv"
    if c.exists():
        with open(c, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("cand_id") and r.get("source"):
                    src[r["cand_id"]] = r["source"]
    return src


def parse_row(vals, lineno):
    v = [x.strip() for x in vals]
    if not v or not (v[0].startswith("c") and v[0][1:].isdigit()):
        return None, f"line {lineno}: bad cand_id {v[:1]!r}"
    cid, rest = v[0], v[1:]
    status = next((x for x in rest if x in STATUSES), None)
    if status is None:
        return None, f"line {lineno}: no status token in {rest[:3]!r}"
    form = next((x for x in rest if x in FORMS), "")
    si = rest.index(status)
    nums = []
    for x in rest[si + 1:]:
        if is_int(x) and len(nums) < 6:
            nums.append(x)
        elif len(nums) == 6:
            break
    nums += [""] * (6 - len(nums))
    source = next((x for x in rest if x in {"brown", "gutenberg"}), "")
    method = "llm+human" if "llm+human" in rest else "human"
    known = STATUSES | FORMS | {"brown", "gutenberg", "llm+human", "human", ""}
    others = [x for x in rest if x not in known and not is_int(x)]
    reason = others[-1] if others else ""
    return dict(zip(COLS, [cid, status, form] + nums +
                    [source, "", reason, method])), None


def main():
    if not J.exists():
        print("no journal to repair"); return
    raw = list(csv.reader(open(J, newline="")))
    hdr, body = raw[0], raw[1:]
    ok, bad, order = {}, [], []
    for i, vals in enumerate(body, start=2):
        row, err = parse_row(vals, i)
        if err:
            bad.append(err); continue
        if row["cand_id"] not in ok:
            order.append(row["cand_id"])
        ok[row["cand_id"]] = row           # last decision wins
    # source backfill: restore empty source values from the candidate pool
    # (by cand_id).  Existing non-empty source values are never overwritten.
    cand_src = load_candidate_sources()
    backfilled = 0
    for row in ok.values():
        if not row["source"] and row["cand_id"] in cand_src:
            row["source"] = cand_src[row["cand_id"]]
            backfilled += 1
    shutil.copy(J, HERE / "natural_annotations.backup.csv")
    with open(J, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for cid in order:
            w.writerow(ok[cid])
    kept = [r for r in ok.values() if r["status"] == "kept"]
    by = {k: sum(r["form"] == k for r in kept) for k in
          ("cleft", "passive", "canonical")}
    print(f"rows in file: {len(body)}  unique candidates: {len(ok)}")
    print(f"unparseable rows dropped: {len(bad)}")
    for b in bad[:10]:
        print("  " + b)
    print(f"kept={len(kept)} (cleft={by['cleft']} passive={by['passive']} "
          f"canonical={by['canonical']})  "
          f"rejected={sum(r['status'] == 'rejected' for r in ok.values())}")
    print(f"source backfilled from natural_candidates.csv: {backfilled} "
          f"row(s)")
    print("backup: data/natural_annotations.backup.csv")
    if len(bad):
        print("inspect dropped rows in the backup before continuing")


if __name__ == "__main__":
    main()
