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

## 2. Phase 0–9 schedule (previously adopted numbering, restored 2026-10-09)

| phase | name | content | status |
|---|---|---|---|
| **0** | Freeze specification | Design lock: blueprint, venue strategy, literature map; dataset generator (`data/generate_dataset.py` → `items.csv`); G1 go/no-go gate (instrument v2); **measurement specification frozen at `24166f5`** (`MEASUREMENT_SPEC.md`) | ✅ done — G1: 8/8 pretrained GO, bert_random NO-GO; spec frozen |
| **1** | Naturalistic annotation | T1 retrieval, T2b LLM prescreen + T2 human annotation to 60/form (floor 55); retrieval repair (2026-10-09). **Exit gates:** annotation export (`--export`), co-annotator agreement (`--agree`), blind recheck of 20 % of keeps ≥ 48 h later (`--recheck 36`, agreement ≥ 95 %) — see §3 | 🔄 **current** — 29/180 accepted |
| **2** | Naturalistic pipeline T3a–c | T3a `src/extract.py --items data/naturalistic_items.csv`; T3b `src/behavior.py --eval-external`; T3c `src/rq5_link.py --natural` → `results/{model}_rq5_natural.json` + `figures/rq5_{model}_natural.png`; N1 decision rules R1–R5; N1 clause pre-registered in `paper_blueprint.md` **before** the first naturalistic pipeline run | ⏳ gated on Phase 1 exit gates |
| **3** | Evidence audit and uncertainty | Resolve every `[FILL: …]` placeholder in `paper_draft.md` only from verified sources (results JSONs/CSVs, annotation export); verify each `CLAIM_LEDGER.md` entry against its evidence; document uncertainties: cleft candidate deficit / floor invocation if any, attested-topicalization scarcity, probe-relative \(l^*\), non-causal margin–behavior association, single primary annotator + blind QC, pending co-annotator agreement | ⏳ |
| **4** | Methods and Results | `paper_draft.md` §3–§5 (instrument, measures, results RQ1/RQ6/RQ5 + naturalistic N1 section); figures F1–F6 + `panel_*` + `rq5_*_natural.png`; tables T1–T4 | ⏳ |
| **5** | Introduction and Related Work | `paper_draft.md` §1–§2 from `computational_ontogenesis_literature_map.md` (four clusters) | ⏳ |
| **6** | Adversarial review | Reviewer-preemption pass (`paper_blueprint.md` §9): Tenney-decadability attack, probing≠mechanism, templateese/ecological validity, laptop-scale scope; stress-test each claim-ledger entry against its stated limitation | ⏳ |
| **7** | Revision and manuscript lock | Incorporate review feedback; freeze the manuscript text (no further results changes without a new spec version) | ⏳ |
| **8** | Reproducibility and submission-compliance audit | Re-run the frozen pipeline end-to-end from `24166f5` on a clean checkout; confirm every reported number regenerates; confirm venue formatting and submission-compliance requirements (ARR / ACL 2027) | ⏳ |
| **9** | Final buffer; submission | **Internal submission by 3 January 2027**; **official ARR deadline 4 January 2027** (ACL 2027, Kyoto; per `venue_strategy.md`); revision management, no dual submission | ⏳ |

*Numbering note (2026-10-09): an earlier draft of this board renumbered the
phases; the previously adopted schedule above is restored. Annotation export,
agreement and blind recheck are Phase 1 exit gates (§3), not a replacement
numbering scheme.*

## 3. Phase 1 exit gates (annotation → pipeline)

1. **Target:** 60 accepted per form (cleft, passive, canonical); 180 total.
2. **Floor:** 55 per form; 165 total — invocation conditions in §1.
3. **Export gate** (`python data/annotate.py --export`): offsets resolve; all
   three spans tokenize under `MAX_LEN = 64` (distilbert fast tokenizer);
   no duplicate texts; ≥ `--min-per-form` keeps per form (default 55).
   Export refuses to write and states the shortfall if gates fail.
4. **Agreement gate:** `--agree` output reported with the pinned co-annotator id
   (`data/llm_prescreen.meta.json`); if exact agreement < 70 %, the LLM
   proposals are unreliable → manual mode for the whole pool and the
   prescreen reported as a failed pilot (protocol §6 R5).
5. **Blind recheck gate:** blind re-annotation of 20 % of keeps after ≥ 48 h
   (`--recheck 36`); agreement < 95 % ⇒ full re-review of the affected form.
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

## 5. Submission timeline (Phase 9)

- **Internal submission: 3 January 2027** (hard internal gate, one day of
  buffer).
- **ARR / ACL 2027 deadline: 4 January 2027** (Kyoto, Aug 2027) — primary
  venue per `venue_strategy.md`.
- Optional fast pass: EACL 2027 workshop (calls ~13 Oct 2026). Backups:
  CoNLL 2027 (~Feb 2027), EMNLP 2027 (later ARR cycle). Never dual-submit.

## 6. Open items (marked for review)

- [ ] Cleft candidate yield from shelf v2 is **unknown until a dry run on a
      machine with corpus access** — no counts are claimed. Escalation if v2
      is insufficient: extend the versioned shelf further (documented
      additions only) or re-open the protocol §3 Wikipedia escalation
      (previously tested and rejected); do not lower annotation standards.
- [ ] Run `annotate.py --agree` and replace the draft's FILL agreement
      figures with verified numbers (Phase 1 exit gate).
- [ ] Pre-register the N1 clause in `paper_blueprint.md` before the first
      naturalistic pipeline run (Phase 1→2 gate).
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
