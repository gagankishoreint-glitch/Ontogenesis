#!/usr/bin/env python3
"""Tests for the Phase-1 cleft retrieval repair.

Covers (per the repair constraints):
  duplicate suppression, ID uniqueness, source provenance, output schema,
  repeat-run idempotency, preservation of existing data, dry-run safety,
  strict-mode behaviour preservation, lenient == established T1 prescreen,
  the documented norm_key dedup, and the repair_journal.py source backfill
  (including a read-only verification against the committed journal).

Stdlib unittest only — no pytest, no network, no corpora needed:

  python data/test_retrieve_clefts.py            # from ontogenesis/
  python -m unittest data.test_retrieve_clefts   # from ontogenesis/
"""
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent            # ontogenesis/data
sys.path.insert(0, str(HERE))
import retrieve_clefts as rc                          # noqa: E402
from retrieve_candidates import (CLEFT_RE, cleft_ok,  # noqa: E402
                                 GUTENBERG_SHELVES, norm_key)
import repair_journal as rj                           # noqa: E402

REPO_DATA = HERE                                     # committed CSVs live here
HEADER = ["cand_id", "source", "form_guess", "lexicon_overlap",
          "text", "words_json"]

S_ARBTYPE = "It was the doctor who called the nurse ."       # subject-relative
S_OBJREL = "It was the man who the king named the heir ."    # object-relative
S_KING = "It was the king who wrote the letter ."
S_QUEEN = "It was the queen who sang the song ."
S_PRON = "It was the man that he told ."                     # pronoun clause subj
S_EXTRA = "It was clear that he left the room ."             # extraposed clause
S_NOCLEFT = "The dog barked at the cat ."


def write_pool(path, rows):
    """rows: list of (cand_id, source, form_guess, text, words)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for cid, src, form, text, words in rows:
            w.writerow([cid, src, form, 1, text, json.dumps(words)])


def read_pool(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class CleftPatternTests(unittest.TestCase):
    """Strict keeps its exact current behaviour; lenient == T1 prescreen."""

    def test_strict_object_relative_passes(self):
        self.assertTrue(rc.strict_cleft(S_OBJREL.split()))

    def test_strict_subject_relative_archetype_fails(self):
        # Characterization of CURRENT strict behaviour (frozen): the
        # subject-relative archetype fails because the clause parser
        # mis-indexes the verb/object for "who VERB ..." clauses.
        self.assertFalse(rc.strict_cleft(S_ARBTYPE.split()))

    def test_strict_subject_relative_with_late_det_passes(self):
        w = "It was the doctor who called the nurse at the hospital .".split()
        self.assertTrue(rc.strict_cleft(w))

    def test_strict_rejects_pronoun_clause_subject(self):
        self.assertFalse(rc.strict_cleft(S_PRON.split()))

    def test_strict_rejects_adverbial_focus(self):
        self.assertFalse(rc.strict_cleft(
            "It was in 1946 that the war ended .".split()))

    def test_lenient_accepts_subject_relative_archetype(self):
        self.assertTrue(rc.lenient_cleft(S_ARBTYPE.split()))

    def test_lenient_rejects_extraposed_and_pronoun(self):
        self.assertFalse(rc.lenient_cleft(S_EXTRA.split()))
        self.assertFalse(rc.lenient_cleft(S_PRON.split()))

    def test_lenient_is_exactly_the_established_prescreen(self):
        for s in (S_ARBTYPE, S_OBJREL, S_KING, S_QUEEN, S_PRON, S_EXTRA,
                  S_NOCLEFT):
            w = s.split()
            expect = bool(CLEFT_RE.search(" ".join(w))) and cleft_ok(w)
            self.assertEqual(rc.lenient_cleft(w), expect, s)

    def test_norm_key_is_case_insensitive_token_join(self):
        self.assertEqual(norm_key("It was the cat .".split()),
                         norm_key("it was the cat .".split()))
        self.assertNotEqual(norm_key("It was the cat .".split()),
                            norm_key("It was the dog .".split()))

    def test_shelves_are_versioned_and_v1_unchanged(self):
        self.assertEqual(len(GUTENBERG_SHELVES["v1"]), 24)
        self.assertEqual(GUTENBERG_SHELVES["v2"][:24],
                         GUTENBERG_SHELVES["v1"])
        self.assertEqual(len(GUTENBERG_SHELVES["v2"]), 32)
        # v2 additions are the 8 documented books (ids only)
        self.assertEqual(GUTENBERG_SHELVES["v2"][24:],
                         [100, 43, 1023, 1184, 1400, 2542, 2600, 5200])


class CollectStatsTests(unittest.TestCase):
    """collect(): duplicate suppression + per-source accounting."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ont-test-"))
        self.pool = self.tmp / "pool.csv"
        write_pool(self.pool, [("c0001", "brown", "cleft", S_ARBTYPE,
                                S_ARBTYPE.split())])
        self.keys, self.ids, self.maxnum = rc.load_existing(self.pool)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_stats_and_dedup(self):
        pool = [
            (S_ARBTYPE.split(), "brown"),      # already present (pool row)
            (S_ARBTYPE.split(), "gutenberg"),  # same text, other source
            (S_KING.split(), "brown"),         # net-new
            (S_KING.split(), "gutenberg"),     # within-run duplicate
            (S_NOCLEFT.split(), "brown"),      # fails the pattern
            (S_QUEEN.split(), "gutenberg"),    # net-new
        ]
        hits, stats = rc.collect(pool, self.keys, "lenient")
        self.assertEqual(stats["brown"],
                         dict(raw=2, already_present=1, duplicates=0,
                              net_new=1))
        self.assertEqual(stats["gutenberg"],
                         dict(raw=3, already_present=1, duplicates=1,
                              net_new=1))
        self.assertEqual([src for _, src in hits], ["brown", "gutenberg"])
        self.assertEqual([" ".join(w) for w, _ in hits], [S_KING, S_QUEEN])

    def test_load_existing_reads_keys_ids_max(self):
        self.assertEqual(self.ids, {"c0001"})
        self.assertEqual(self.maxnum, 1)
        self.assertIn(norm_key(S_ARBTYPE.split()), self.keys)

    def test_load_existing_rejects_wrong_header(self):
        bad = self.tmp / "bad.csv"
        bad.write_text("a,b,c\n1,2,3\n")
        with self.assertRaises(SystemExit):
            rc.load_existing(bad)

    def test_load_existing_missing_or_empty(self):
        keys, ids, mx = rc.load_existing(self.tmp / "nope.csv")
        self.assertEqual((keys, ids, mx), (set(), set(), 0))
        empty = self.tmp / "empty.csv"
        empty.write_text("")
        keys, ids, mx = rc.load_existing(empty)
        self.assertEqual((keys, ids, mx), (set(), set(), 0))


class AppendSafetyTests(unittest.TestCase):
    """ID uniqueness, provenance, schema, preservation, idempotency."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ont-test-"))
        self.pool = self.tmp / "pool.csv"
        # gapped ids on purpose: max suffix is 7, next must be c0008
        write_pool(self.pool, [
            ("c0001", "brown", "cleft", S_ARBTYPE, S_ARBTYPE.split()),
            ("c0003", "gutenberg", "passive", S_KING, S_KING.split()),
            ("c0007", "brown", "canonical", S_QUEEN, S_QUEEN.split()),
        ])
        self.before_bytes = self.pool.read_bytes()
        self.before_rows = read_pool(self.pool)
        self.hits = [(S_KING.split(), "brown"),
                     ("It was the queen who sang the song .".split(),
                      "gutenberg")]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_append_allocates_non_colliding_ids(self):
        rc.append_hits(self.pool, self.hits, 8)
        rows = read_pool(self.pool)
        self.assertEqual([r["cand_id"] for r in rows],
                         ["c0001", "c0003", "c0007", "c0008", "c0009"])

    def test_append_preserves_existing_bytes_and_rows(self):
        rc.append_hits(self.pool, self.hits, 8)
        after = self.pool.read_bytes()
        self.assertTrue(after.startswith(self.before_bytes),
                        "existing rows must be byte-identical prefix")
        rows = read_pool(self.pool)
        for old, new in zip(self.before_rows, rows):
            self.assertEqual(old, new)
        self.assertEqual(len(rows), len(self.before_rows) + 2)

    def test_append_schema_and_provenance(self):
        rc.append_hits(self.pool, self.hits, 8)
        rows = read_pool(self.pool)[-2:]
        for row, (words, src) in zip(rows, self.hits):
            self.assertEqual(list(row.keys()), HEADER)
            self.assertEqual(row["source"], src)          # provenance
            self.assertEqual(row["form_guess"], "cleft")
            self.assertIn(row["lexicon_overlap"], {"0", "1"})
            self.assertEqual(json.loads(row["words_json"]), words)
            self.assertEqual(row["text"], " ".join(words))

    def test_append_creates_header_for_fresh_pool(self):
        fresh = self.tmp / "fresh.csv"
        rc.append_hits(fresh, self.hits, 1)
        rows = read_pool(fresh)
        self.assertEqual([r["cand_id"] for r in rows], ["c0001", "c0002"])


class CliEndToEndTests(unittest.TestCase):
    """main() with injected fixture corpora (no network, no real data)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ont-test-"))
        self.pool = self.tmp / "pool.csv"
        write_pool(self.pool, [("c0001", "brown", "cleft", S_ARBTYPE,
                                S_ARBTYPE.split())])
        self._rb, self._gs = rc.read_brown, rc.gutenberg_sentences
        rc.read_brown = lambda: [S_ARBTYPE.split(), S_KING.split(),
                                 S_NOCLEFT.split()]
        rc.gutenberg_sentences = lambda shelf=None: [S_KING.split(),
                                                     S_QUEEN.split()]

    def tearDown(self):
        rc.read_brown, rc.gutenberg_sentences = self._rb, self._gs
        shutil.rmtree(self.tmp)

    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc.main(list(argv))
        return buf.getvalue()

    def test_dry_run_reports_and_writes_nothing(self):
        before = self.pool.read_bytes()
        out = self.run_cli("--pattern", "lenient", "--shelf", "v2",
                           "--out", str(self.pool), "--dry-run")
        self.assertEqual(self.pool.read_bytes(), before)
        self.assertFalse((self.tmp / "pool.backup.csv").exists())
        self.assertIn("[mode=lenient] [shelf=v2]", out)
        self.assertIn("[source=brown] raw=2 post-filter=1 "
                      "already-present=1 duplicates=0 net-new=1", out)
        self.assertIn("[source=gutenberg] raw=2 post-filter=2 "
                      "already-present=0 duplicates=1 net-new=1", out)
        self.assertIn("[total] raw=4 post-filter=3 already-present=1 "
                      "duplicates=1 net-new=2", out)
        self.assertIn("next id would be c0002", out)
        self.assertIn("dry-run: nothing appended", out)

    def test_append_then_repeat_run_is_idempotent(self):
        out1 = self.run_cli("--pattern", "lenient", "--out", str(self.pool))
        self.assertIn("appended 2 lenient clefts as c0002..c0003", out1)
        self.assertTrue((self.tmp / "pool.backup.csv").exists())
        mid = self.pool.read_bytes()
        rows_mid = read_pool(self.pool)
        self.assertEqual(len(rows_mid), 3)
        out2 = self.run_cli("--pattern", "lenient", "--out", str(self.pool))
        self.assertIn("net-new=0", out2)
        self.assertIn("nothing to append", out2)
        self.assertEqual(self.pool.read_bytes(), mid)   # byte-identical
        self.assertEqual(read_pool(self.pool), rows_mid)

    def test_default_cli_is_strict_v1(self):
        out = self.run_cli("--out", str(self.pool), "--dry-run")
        self.assertIn("[mode=strict] [shelf=v1]", out)
        # strict finds nothing in the fixtures (archetype fails strict)
        self.assertIn("[total] raw=0", out)

    def test_limit_caps_appends(self):
        rc.gutenberg_sentences = lambda shelf=None: [S_KING.split(),
                                                     S_QUEEN.split(),
                                                     S_ARBTYPE.split()]
        out = self.run_cli("--pattern", "lenient", "--shelf", "v2",
                           "--limit", "1", "--out", str(self.pool))
        # the report must distinguish total available from planned append
        self.assertIn("[limit] --limit=1: 2 net-new available, "
                      "1 planned for append (1 held back)", out)
        self.assertIn("appended 1 lenient clefts", out)
        self.assertEqual(len(read_pool(self.pool)), 2)


class RepairJournalBackfillTests(unittest.TestCase):
    """repair_journal.py source backfill, on temp copies only."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ont-test-"))
        write_pool(self.tmp / "natural_candidates.csv", [
            ("c0001", "brown", "cleft", S_ARBTYPE, S_ARBTYPE.split()),
            ("c0002", "gutenberg", "passive", S_KING, S_KING.split()),
        ])
        cols = rj.COLS
        with open(self.tmp / "natural_annotations.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            # c0001: kept, source empty -> must be backfilled 'brown'
            w.writerow(["c0001", "kept", "passive", "1", "2", "3", "4",
                        "5", "2", "", "", "", "human"])
            # c0002: rejected, source already present -> never overwritten
            w.writerow(["c0002", "rejected", "", "", "", "", "", "",
                        "", "gutenberg", "", "other", "human"])
            # c0003: rejected, source empty, NOT in candidates -> stays empty
            w.writerow(["c0003", "rejected", "", "", "", "", "", "",
                        "", "", "", "other", "human"])
        self._J, self._HERE = rj.J, rj.HERE
        rj.J = self.tmp / "natural_annotations.csv"
        rj.HERE = self.tmp

    def tearDown(self):
        rj.J, rj.HERE = self._J, self._HERE
        shutil.rmtree(self.tmp)

    def read_journal(self):
        with open(rj.J, newline="") as f:
            return list(csv.DictReader(f))

    def test_backfill_restores_source_and_preserves_decisions(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rj.main()
        out = buf.getvalue()
        self.assertIn("source backfilled from natural_candidates.csv: "
                      "1 row(s)", out)
        self.assertTrue((self.tmp / "natural_annotations.backup.csv")
                        .exists())
        rows = {r["cand_id"]: r for r in self.read_journal()}
        self.assertEqual(rows["c0001"]["source"], "brown")
        self.assertEqual(rows["c0002"]["source"], "gutenberg")
        self.assertEqual(rows["c0003"]["source"], "")
        # decisions, labels, spans untouched
        self.assertEqual(rows["c0001"]["status"], "kept")
        self.assertEqual(rows["c0001"]["form"], "passive")
        for k in ("n1a", "n1b", "n2a", "n2b", "pred", "agent"):
            self.assertEqual(rows["c0001"][k],
                             {"n1a": "1", "n1b": "2", "n2a": "3",
                              "n2b": "4", "pred": "5", "agent": "2"}[k])
        self.assertEqual(rows["c0002"]["reason"], "other")

    def test_repair_is_idempotent(self):
        with redirect_stdout(io.StringIO()):
            rj.main()
        first = rj.J.read_bytes()
        with redirect_stdout(io.StringIO()):
            rj.main()
        self.assertEqual(rj.J.read_bytes(), first)


class CommittedJournalSafetyTests(unittest.TestCase):
    """Read-only verification against the COMMITTED journal + candidates:
    the repair (with backfill) must preserve every human decision and
    reproduce the reported counts.  Writes happen in a temp dir only."""

    def test_committed_journal_repair_preserves_all_decisions(self):
        tmp = Path(tempfile.mkdtemp(prefix="ont-test-"))
        self.addCleanup(shutil.rmtree, tmp)
        for name in ("natural_annotations.csv", "natural_candidates.csv"):
            shutil.copy(REPO_DATA / name, tmp / name)
        j, here = rj.J, rj.HERE
        rj.J, rj.HERE = tmp / "natural_annotations.csv", tmp
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rj.main()
            out = buf.getvalue()
        finally:
            rj.J, rj.HERE = j, here
        # the counts reported in the project brief, reproduced:
        self.assertIn("rows in file: 100  unique candidates: 100", out)
        self.assertIn("unparseable rows dropped: 0", out)
        self.assertIn("kept=29 (cleft=7 passive=22 canonical=0)  "
                      "rejected=71", out)
        self.assertIn("source backfilled from natural_candidates.csv: "
                      "100 row(s)", out)
        # every human decision, label and span preserved:
        orig = {r["cand_id"]: r for r in
                csv.DictReader(open(REPO_DATA / "natural_annotations.csv",
                                    newline=""))}
        with open(tmp / "natural_annotations.csv", newline="") as f:
            new = {r["cand_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(set(orig), set(new))
        self.assertEqual(len(new), 100)
        for cid, o in orig.items():
            n = new[cid]
            for k in ("status", "form", "n1a", "n1b", "n2a", "n2b", "pred",
                      "agent", "pass2", "reason", "method"):
                self.assertEqual(o[k], n[k], f"{cid}.{k}")
        # source provenance now populated and correct:
        csrc = {r["cand_id"]: r["source"] for r in
                csv.DictReader(open(REPO_DATA / "natural_candidates.csv",
                                    newline=""))}
        for cid, n in new.items():
            self.assertEqual(n["source"], csrc[cid])
        # committed files untouched by the test (read-only usage):
        self.assertTrue((REPO_DATA / "natural_annotations.csv").exists())
        self.assertEqual(len(orig), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
