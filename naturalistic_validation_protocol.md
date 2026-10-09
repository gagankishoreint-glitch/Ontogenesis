# Naturalistic Validation Protocol — RQ5 replication on attested sentences

Status: **v3 (2026-10-08)** — LLM co-annotator pre-screen added per user
decision (machine triage + human verification of every analyzed sentence).
Form-set per v2 user decision (swap_passive). T1 retriever, T2 annotator, T2b
LLM pre-screen implemented in `data/retrieve_candidates.py`,
`data/annotate.py`, `data/llm_prescreen.py`. Pre-register the N1 clause (§6)
in `paper_blueprint.md` **before** the first naturalistic pipeline run.

---

## 1. Purpose and claim under test

**Threat addressed.** RQ5's margin→behavior specificity (P2′/P4″) is measured
entirely on template sentences. A reviewer can object: *the link may be a
property of templateese — templates are unnaturally uniform, and both
instruments (span probe, behavior head) were trained on the same templates.*

**Claim to replicate (N1).** On **attested, human-verified sentences**, the
per-sentence span-probe margin at l\* still predicts the model's behavioral
generalization *within form*, with both instruments applied **zero-shot**
(nothing retrained on natural data):

> N1: Spearman(margin_s, p_correct_s) > 0 within form on natural sentences,
> for pretrained models, with magnitude comparable to the template result.

**Form set (v2).** Natural sentences of three constructions:

| form | construction | probe/behavior status (templates) | role in N1 |
|---|---|---|---|
| cleft (it-cleft, NP focus) | role-reversing | cross-form (unseen) | headline cross-form replication |
| passive (by-phrase) | role-reversing | train-form (seen) | role reversal under style shift |
| canonical (unmarked SVO) | baseline | train-form | instrument-sanity anchor (R3) |

Topicalization is **not** in the form set: attested NP-fronting is <0.2 % of
real prose (28 candidates survived prescreen over 47 306 sentences from Brown
+ 24 Gutenberg novels; reaching n = 60 would need ≳10 000 human screenings).
Report this scarcity as a limitation (§8) — the template-side topical results
stand unchallenged by, and are not extended to, natural data.

---

## 2. Sample design

| target form | kept n | candidates (observed 2026-10-08, v2 filters) |
|---|---|---|
| passive | 60 | 375 pool, quota 200 shown first (≈50–60 % precision) |
| canonical | 60 | quota 200 (≈75 % precision) |
| cleft | 60 | 132 pool (≈50 % precision) |
| topical | — | 28 (exploratory only, not in form set) |
| **kept total** | **180** | 560 candidates written, precision-ordered |

**Power** (unchanged from v1 approval): within-form n = 60 → Fisher-z SE =
1/√57 ≈ 0.133; template ρ ≈ 0.34 ⇒ |t| ≈ 2.7 per form; pooled over the two
role-forms (n = 120) ⇒ |t| ≈ 3.8. Detects the registered effect; no
equivalence test promised.

**Sentence constraints.** ≤ 22 words (safe under MAX_LEN = 64 subwords);
exactly two core participants of the marked predicate, both full NPs;
interpretable out of context.

---

## 3. Candidate retrieval — `data/retrieve_candidates.py` (T1, done)

```
mkdir -p data/corpora && curl -sL -o data/corpora/brown.zip \
  https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip \
  && unzip -oq data/corpora/brown.zip -d data/corpora/
python data/retrieve_candidates.py --source all     # brown + gutenberg shelf
```

- **Sources:** Brown (written register, raw `word/POS` reader, zero deps) +
  curated 24-book Gutenberg fiction shelf (dialogue-rich; clefts/fronting
  live here). Wikipedia escalation was **tested and rejected**: random
  articles are mostly stubs and the API rate-limits (429) per-title fetches.
- **Filters:** 6–22 words; ≥2 NP hints; no quoted-speech/trace tokens; must
  end in terminal punctuation (kills fragments/headings); dedup.
- **Prescreens (hints only — the human decides):**
  - cleft: `It (is|was|were|has been|had been) [NP focus] … (who|which|that)`
    with NP-focus requirement, plus `cleft_ok` (relative clause must have a
    full-NP subject and a transitive verb — kills "who replied", "that he …");
  - passive: `(was|were|…been|being) VEN … by` plus `passive_ok` (agentive
    by-phrase — kills "by other means/the aid of"; no ", and"/";"/":"
    multi-clause; no pronoun/it/there subject);
  - topical (exploratory, not in form set): fronted-NP comma patterns with an
    object-gap check;
  - canonical: the rest (quota 200).
- Output order is precision-ordered (passive → canonical → cleft → topical),
  so keeps arrive early and junk late.
- `lexicon_overlap ∈ {0,1}` flags sentences containing a surface form of a
  TRANS_VERBS lemma (dative verbs excluded) — the clean template-vs-natural
  contrast split (§6 R4).
- Output `data/natural_candidates.csv`: cand_id, source, form_guess,
  lexicon_overlap, text, words_json. SEED 20261007, deterministic.

---

## 4. Annotation — LLM pre-screen + human review (T2/T2b, done)

```
brew install ollama && ollama serve &          # once
ollama pull qwen2.5:7b-instruct-q4_K_M         # ~4.7 GB, pinned co-annotator
python data/llm_prescreen.py                   # ~5 min, temp 0, writes
                                               # llm_prescreen.csv + meta.json
rm data/natural_annotations.csv   # only if --list shows kept=0
python data/annotate.py --review-llm           # human reviews ONLY proposed
                                               # keeps; ~20 min
python data/annotate.py --agree                # human-LLM agreement numbers
python data/annotate.py --export
python data/annotate.py --recheck 36           # blind human QC (>=48 h later)
python data/annotate.py                        # manual fallback mode (full pool)
```

**Division of labour (the v3 claim):** the pinned local LLM (temperature 0,
model id + date recorded in `llm_prescreen.meta.json`) triages the 560
candidates and proposes keeps with spans; the human confirms, edits, or
rejects every proposal. Every sentence in the analyzed set is therefore
human-verified, and the protocol additionally yields a human–LLM
inter-annotator agreement figure (§6 R5) that a solo-human design cannot
report. API backends (OPENAI_API_KEY / ANTHROPIC_API_KEY env) are supported
if a stronger co-annotator is desired; the manual full-pool mode remains the
fallback if any form falls short of its gate.

In `--review-llm` mode each screen shows the sentence plus the LLM's
proposal (`KEEP as passive: n1=… n2=… pred=… agent=…`); keys: `k` keep as
proposed (one keypress), `e` edit spans manually, `r` reject (bare Enter =
other), `s` skip, `q` quit. When the stats line shows ≥60/60/60 for
cleft/passive/canonical, `q` and export. In manual mode the keys are
`1` cleft `2` topical `3` canonical `4` passive plus heuristic span
suggestions with the same k/e/r flow.

**Decision rules (in the tool's flow; also printed here for the paper):**
- **cleft**: it-cleft with NP focus and who/which/that relative; clausal or
  because-focus ⇒ `o`.
- **passive**: keep **only if both the patient subject and the by-phrase
  agent are full NPs**; passive without a by-phrase ⇒ `p` (one participant).
  agent = the by-phrase NP.
- **canonical**: unmarked SVO with two full-NP participants.
- **participants**: full NPs only — pronoun or coordinated NP ⇒ `p`;
  cross-sentential coreference ⇒ `p`; genuinely ambiguous roles ⇒ `a`.
- **predicate**: the main-clause predicate whose arguments are n1/n2
  (for passives: the passive participle).
- n1 = earlier surface offset enforced by the tool (swaps with a notice);
  `label_patient_first` derived from the agent index exactly as in items.csv.
- export gates: ≥ 55 keeps per form (cleft/passive/canonical), offsets
  resolve, all 3 spans tokenize under MAX_LEN (distilbert fast tokenizer),
  no duplicate texts; prints reject-reason histogram, per-form
  label_patient_first, lexicon/source splits.

**QC (single annotator):** blind re-annotation of 20 % after ≥ 48 h
(`--recheck 36`); agreement < 95 % ⇒ full re-review of the affected form.
Export refuses to write if gates fail and states the shortfall.

**Human time (v2):** ≈300 reviews to reach 60/60/60 (keeps are one
keypress) ≈ 45–60 min, in 1–2 sessions.

---

## 5. Pipeline integration (T3a–c — next code deliverable)

- **T3a** `src/extract.py --items data/naturalistic_items.csv
  --out features/{model}_natural.npz` — same slots/layers path.
- **T3b** `src/behavior.py` saves the template-trained head
  (`results/{model}_behavior_head.pt`, seed-fixed retrain) and gains
  `--eval-external data/naturalistic_items.csv` →
  `results/{model}_behavior_natural.csv` (same columns).
- **T3c** `src/rq5_link.py --natural` — span probe fitted on template P1
  features exactly as today; margins from the natural npz; forms =
  {cleft, passive, canonical}; sentence aggregation over the two query
  orders; outputs `results/{model}_rq5_natural.json` (ρ per form, mean
  within-form ρ over the two role-forms = primary, pooled reported but
  never headline, ΔR², tercile bars, Fisher-z CIs, template ρ side-by-side)
  + `figures/rq5_{model}_natural.png`.

Compute: < 30 min for distilbert + gpt2 on the RTX 3050.

---

## 6. Decision rules (pre-register as N1 before running)

- **R1 (primary):** mean-within-form ρ (cleft + passive) > 0 with q < 0.05
  (BH over forms) for distilbert **and** gpt2 ⇒ N1 supported.
- **R2 (magnitude):** Fisher-z 95 % CI of natural ρ next to template ρ;
  CI containing the template value ⇒ "magnitude preserved"; excluding it ⇒
  report the gap as an effect size. No re-tuning allowed either way.
- **R3 (instrument gate for R1):** on natural canonical sentences, behavior
  p_s std ≥ 0.02 and span-probe pairwise accuracy ≥ 0.65. Failure ⇒ report
  as a template→natural transfer limitation, no ρ claim.
- **R4 (vocabulary split):** report the `lexicon_overlap` split; if ρ
  replicates only on the overlap subset, vocabulary shift — not template
  style — is the boundary condition.
- **R5 (co-annotator agreement):** report `--agree` output — human keep-rate
  of LLM proposals and span+form exact agreement among keeps — with the
  pinned model id from `llm_prescreen.meta.json`. If exact agreement < 70 %,
  the LLM proposals are unreliable: fall back to manual mode for the whole
  pool and report the pre-screen as a failed pilot.
- No hyperparameter adjustments on natural data, ever.

---

## 7. User checklist

```
1. [x] form-set amendment signed off (swap_passive, 2026-10-08)
2. [ ] Brown download one-liner (§3) — sandbox already has it; you need it too
3. [ ] python data/retrieve_candidates.py --source all
4. [ ] python data/annotate.py              ← ~2 h human, sessions
5. [ ] export gates pass (python data/annotate.py --export)
6. [ ] python data/annotate.py --recheck 36 (≥48 h later)
7. [ ] I deliver T3a–c; append N1 clause to paper_blueprint.md
8. [ ] extract/behavior/rq5 --natural for distilbert + gpt2; paste JSONs
```

---

## 8. Paper wiring

- **Methods sentence (co-annotation):** "Candidate sentences were pre-
  annotated by a pinned local LLM (qwen2.5-7b-instruct, temperature 0; model
  and date recorded); every proposed keep was then verified, corrected, or
  rejected by a human annotator, so all analyzed sentences are human-
  verified. Human–LLM exact agreement on the kept set was X % (keep-rate of
  proposals Y %), reported as an inter-annotator figure."
- **RQ5 section:** paragraph + side-by-side figure (template vs natural ρ).
- **Limitation sentence (topicalization):** "Attested patient-topicalization
  proved too rare in our corpora (<0.2 % of screened sentences) to support a
  naturalistic arm; the naturalistic replication therefore covers clefts
  (cross-form) and passives (role-reversed train-form), and the template-side
  topicalization results remain unvalidated on attested sentences."
- **Appendix:** annotation protocol (§4), reject-reason histogram, observed
  label_patient_first per form, inter-pass agreement, source split.
