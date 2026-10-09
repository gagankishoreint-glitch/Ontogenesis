# Project Execution Board — Computational Ontogenesis

**Date:** 2026-10-09 · **Branch:** `arena/297f220d-ontogenesis`
**Frozen measurement commit:** `24166f57ab7e5fe5cc9417a72a9165e058231016` (kept
distinct from all subsequent retrieval-repair commits — no history rewrites)
**Companions:** `MEASUREMENT_SPEC.md`, `CLAIM_LEDGER.md`,
`naturalistic_validation_protocol.md`, `paper_blueprint.md`, `venue_strategy.md`

---

## 1. Current phase and headline status

**Current phase: Phase 1 — naturalistic annotation (in progress).**

Core paper questions: **RQ1** (where the structural transition emerges),
**RQ6** (recurrence across the tested architectures), **RQ5** (margin→behavior
link). RQ3/RQ4 deferred. The naturalistic arm (N1) validates these three
claims; it is not a fourth research question.

### Annotation scoreboard (verified 2026-10-09, `python data/annotate.py --list`)

| form | accepted | target | preregistered floor | remaining to target |
|---|---:|---:|---:|---:|
| passive | 22 | 60 | 55 | 38 |
| cleft | 7 | 60 | 55 | 53 |
| canonical | 0 | 60 | 55 | 60 |
| **total** | **29** | **180** | **165** | **151** |

Decided 100 of 563 candidates (rejects: other 63, pronoun_or_coord 8;
review queue 0). Candidate pool: passive 200, canonical 200, cleft 135,
topical 28 (topical is exploratory only, not in the form set).

**Floor rule (preregistered):** the 55-per-form / 165-total floor may be
invoked only if corpus availability genuinely prevents reaching 60/180, and
the shortfall must be documented **before** computing any naturalistic
margin–behavior correlation.

---

## 2. Phase 0–9 schedule

| phase | name | content | status |
|---|---|---|---|
| **0** | Design lock | Blueprint, venue strategy, literature map; dataset generator; G1 go/no-go gate (instrument v2) | ✅ done — G1: 8/8 pretrained GO, bert_random NO-GO |
| **1** | Naturalistic annotation | T1 retrieval, T2b LLM prescreen + T2 human annotation to 60/form (floor 55); retrieval repair (2026-10-09) | 🔄 **current** — 29/180 accepted |
| **2** | Export + QC | `annotate.py --export` gates; blind recheck of 20 % of keeps ≥ 48 h later (agreement ≥ 95 %); `--agree` co-annotator report | ⏳ pending Phase 1 gates |
| **3** | Naturalistic extraction (T3a) | `src/extract.py --items data/naturalistic_items.csv --out features/{model}_natural.npz` | ⏳ |
| **4** | Naturalistic behavior (T3b) | `src/behavior.py` template-trained head, `--eval-external data/naturalistic_items.csv` | ⏳ |
| **5** | Naturalistic link (T3c / N1) | `src/rq5_link.py --natural`; N1 decision rules R1–R5; N1 clause pre-registered in `paper_blueprint.md` **before** the first naturalistic pipeline run | ⏳ |
| **6** | Paper integration | RQ5 naturalistic section + side-by-side figure; limitation sentences (topicalization scarcity, cleft n); appendix (annotation protocol, reject histogram, label balance, agreement, source split) | ⏳ |
| **7** | Internal review | Reviewer-preemption pass (blueprint §9); FILL placeholders resolved only from verified sources | ⏳ |
| **8** | Internal submission | **Submit internally by 3 January 2027** — one day ahead of the deadline | ⏳ |
| **9** | ARR submission | **ACL 2027 via ARR, 4 January 2027** (Kyoto; per `venue_strategy.md`); revision management, no dual submission | ⏳ |

*Note for review: the Phase 0–9 numbering is this board's canonical
consolidation of blueprint §7 (W1–W6), protocol §3–§5 (T1/T2/T3), and
`venue_strategy.md`; only "Phase 1 = naturalistic annotation" is stated in the
project brief. Confirm the numbering at the next project review.*

## 3. Annotation gates (Phase 1 → 2 exit criteria)

1. **Target:** 60 accepted per form (cleft, passive, canonical); 180 total.
2. **Floor:** 55 per form; 165 total — invocation conditions in §1.
3. **Export gates** (`python data/annotate.py --export`): offsets resolve; all
   three spans tokenize under `MAX_LEN = 64` (distilbert fast tokenizer);
   no duplicate texts; ≥ `--min-per-form` keeps per form (default 55).
   Export refuses to write and states the shortfall if gates fail.
4. **QC:** blind re-annotation of 20 % of keeps after ≥ 48 h
   (`--recheck 36`); agreement < 95 % ⇒ full re-review of the affected form.
5. **Agreement:** `--agree` output reported with the pinned co-annotator id
   (`data/llm_prescreen.meta.json`); if exact agreement < 70 %, the LLM
   proposals are unreliable → manual mode for the whole pool and the
   prescreen reported as a failed pilot (protocol §6 R5).
6. **Human authority:** every analyzed sentence is human-verified; no
   auto-acceptance at any stage.

## 4. Retrieval-repair record (2026-10-09, Phase 1)

- **Root cause:** the `--pattern lenient --full-shelf` command could not run
  because those flags never existed in the committed `retrieve_clefts.py`
  (only `--source`, `--dry-run`); the only shelf in the repository is the
  curated 24-book list in `retrieve_candidates.py`. No 50-book list and no
  211-candidate result exist anywhere in the repository — the earlier report
  could not be reproduced and is treated as unverified. Additionally, the
  committed strict filter is structurally narrow for subject-relative
  it-clefts (its clause parser mis-indexes the verb/object for "who VERB …"
  clauses), which explains `strict clefts found: 0` over Brown + the 24-book
  shelf independent of cleft rarity.
- **Deficit (measured):** 135 cleft candidates in the pool, 28 decided, 7
  accepted (25 % observed precision; 107 undecided). Reaching 60 needs ≈ 240
  candidates at observed precision (≈ 220 for the 55 floor) → deficit ≈ 85–105
  cleft candidates. Canonical (0/60) and passive (22/60) shortfalls are
  annotation-throughput, not retrieval, problems (200 candidates each).
- **Fix (committed after tests):** `retrieve_clefts.py` gains an explicit,
  documented CLI — `--pattern {strict,lenient}` (strict = unchanged original;
  lenient = the established T1 prescreen `CLEFT_RE + cleft_ok`, protocol §3),
  `--shelf {v1,v2}` (v1 = original 24 books; v2 = v1 + 8 explicitly listed,
  versioned additions), `--limit`, `--out`, `--dry-run`. Appends are
  append-only with pre-append backup (`natural_candidates.backup.csv`),
  norm_key dedup, collision-checked new `cand_id`s, per-source dry-run
  accounting (raw / post-filter / already-present / duplicates / net-new).
  `retrieve_candidates.py` gains the versioned shelf constants and the
  documented `norm_key` (default behavior unchanged).
  `repair_journal.py` gains source backfill from `natural_candidates.csv`
  (existing source values never overwritten).
- **Tests:** `ontogenesis/data/test_retrieve_clefts.py` — 25 tests, all
  passing (duplicate suppression, ID uniqueness, provenance, schema,
  idempotency, preservation, dry-run safety, strict preservation, lenient ==
  prescreen, backfill, committed-journal decision preservation).
- **Data safety:** `natural_candidates.csv`, `natural_annotations.csv` and
  `natural_annotations.backup.csv` are byte-identical to `24166f5`; no
  annotation or candidate record was lost, duplicated, or rewritten.

## 5. Submission timeline

- **Internal submission: 3 January 2027** (hard internal gate, one day of
  buffer).
- **ARR / ACL 2027 deadline: 4 January 2027** (Kyoto, Aug 2027) — primary
  venue per `venue_strategy.md`.
- Optional fast pass: EACL 2027 workshop (calls ~13 Oct 2026). Backups:
  CoNLL 2027 (~Feb 2027), EMNLP 2027 (later ARR cycle). Never dual-submit.

## 6. Open items (marked for review)

- [ ] Confirm the Phase 0–9 numbering (§2 note).
- [ ] Cleft candidate yield from shelf v2 is **unknown until a dry run on a
      machine with corpus access** — no counts are claimed. Escalation if v2
      is insufficient: extend the versioned shelf further (documented
      additions only) or re-open the protocol §3 Wikipedia escalation
      (previously tested and rejected); do not lower annotation standards.
- [ ] Run `annotate.py --agree` and replace the draft's FILL agreement
      figures with verified numbers.
- [ ] Pre-register the N1 clause in `paper_blueprint.md` before the first
      naturalistic pipeline run (Phase 2→3 gate).
- [ ] Optional local git hygiene: `git config core.whitespace cr-at-eol` so
      `git diff --check` stops flagging the CSVs' native CRLF line endings
      (see §7). Do **not** renormalize the CSVs.

## 7. Data-file conventions (git hygiene)

- The three CSVs (`natural_candidates.csv`, `natural_annotations.csv`,
  `natural_annotations.backup.csv`) natively use **CRLF** line endings (the
  `csv` module's default terminator), are 100 % CRLF, and contain **zero**
  trailing-whitespace lines (verified 2026-10-09). `git diff --check` flags
  `^M` on modified lines because a carriage return at end-of-line is reported
  as trailing whitespace unless `core.whitespace` includes `cr-at-eol`; the
  CRs are the files' native format, not stray whitespace. Do not strip them
  (that would rewrite every line and create mixed endings).
- `natural_annotations.backup.csv` is the **pre-repair** journal snapshot
  (mixed column order, rows 74–101 in the old `annotate.py` column order);
  `natural_annotations.csv` is the **repaired** journal (uniform column
  order). Keep both; do not re-run `repair_journal.py` over the live journal
  except deliberately, with its automatic backup.
- New candidate rows appended by `retrieve_clefts.py` use the same CRLF
  convention and the same schema, so the annotator picks them up unchanged.
