# Claim Ledger — Computational Ontogenesis

**Date:** 2026-10-09 · **Measurement spec frozen at:** `24166f57ab7e5fe5cc9417a72a9165e058231016`
**Companion documents:** `MEASUREMENT_SPEC.md` (frozen measurements),
`PROJECT_EXECUTION_BOARD.md` (phase schedule and gates),
`naturalistic_validation_protocol.md` (observer/annotation protocol v3).

Every claim below carries: statement, status, evidence (verified file
references), scope and limitations, and what would falsify it. Nothing marked
PENDING may be cited as a result.

---

## CL-1 (RQ1) — A measurable structural transition emerges in transformer layers

**Statement.** In frozen pretrained transformers, cross-form structure decoding
becomes statistically significant at a layer-localized depth: the span-level
AGENT probe, trained on {active, passive} (P1 fillers), transfers above chance
to unseen forms {cleft, topical} (P2 fillers) from a first significant layer
\(l^*\) (`l_struct_span`) onward. The transition is a property of pretraining,
not of architecture: the random-init control shows no crossover at any layer.

**Status: SUPPORTED (template instrument).**

**Evidence (verified 2026-10-09 against `ontogenesis/results/`):**
- 8/8 pretrained models pass the G1 gate (`go_no_go: true`; span gate
  `go_no_go_span: true` where reported): bert_base \(l_{\text{struct}}=6\),
  roberta_base 5, distilbert 3 (\(l^*=3\)), gpt2 5 (\(l^*=4\)),
  gpt2_medium 6 (\(l^*=6\)), opt_125m 6 (\(l^*=3\)), pythia_160m 5
  (\(l^*=1\)), pythia_410m 6 (\(l^*=1\)).
- Control `bert_random`: `l_struct = null`, `l_struct_span = null`,
  max cross-form accuracy 0.493 (L4), span cross-form max 0.505 (L6),
  `go_no_go = false` on both levels — the transition does not appear without
  pretraining.
- `ontogenesis/G1_REPORT.md` (2026-10-07, instrument v2): curve narrative,
  shuffled-label ≈ chance, layer-0 excluded from FDR.
- Curves: `ontogenesis/results/{model}_curves.csv`; figures:
  `ontogenesis/figures/g1_*.png`, `panel_g1.png`.

**Scope and limitations (must travel with the claim):**
- **Probe-relative.** \(l^*\) is defined by a *linear span-agent probe* with a
  fixed training form set, filler-disjoint splits, binomial + BH-FDR
  inference. It localizes where structure becomes *readable by this
  instrument*; it is not a claim about raw representational geometry
  (\(W_l\) actually *rises* with depth for BERT — no naive convergence) and not
  a circuit-level mechanism claim (blueprint §6 guardrail: phenomenological
  localization; mechanism is follow-up work).
- \(l^*\) (`l_struct_span`) and \(l_{\text{struct}}\) are distinct instruments
  (span-level AGENT probe vs sentence-level `label_patient_first` probe);
  both are reported, never conflated (see `MEASUREMENT_SPEC.md` §3).
- Scope: the 9-model panel in `MEASUREMENT_SPEC.md` §2 (66M–355M parameters).

**Falsified if:** a pretrained model fails the G1 gate on rerun with the frozen
pipeline, or the random-init control develops a significant cross-form layer.

---

## CL-2 (RQ6) — The phenomenon recurs across the tested architectures

**Statement.** The structural transition recurs across architectures: all
eight pretrained models in the panel pass the crossover gate, and the
relative position of the transition is conserved (\(l_{\text{struct}}/(L-1)\)
between 0.25 for both 24-layer models and 0.50 for the 12-layer models and
distilbert), while the untrained control fails it.

**Status: SUPPORTED (template instrument).**

**Evidence (verified 2026-10-09):** `paper_draft.md` §5.2 table transcribed
from the 9 per-model summary JSONs (2026-10-08): four model families (BERT;
RoBERTa/DistilBERT encoder line; GPT-2, Pythia, OPT decoders), two
architecture classes, 66M–355M parameters; `ontogenesis/results/*_summary.json`.

**Scope and limitations (must travel with the claim):**
- **Scoped to the tested architectures.** The claim is profile conservation
  across the tested panel — it is *not* a claim about frontier-scale models,
  all architectures, or endpoint behavior. The random-init control failing
  the gate is part of the claim (the transition is learned, not
  architecture-given).
- Relative-depth normalization \(l_{\text{struct}}/(L-1)\) is the comparison
  metric; absolute layer numbers differ across depths.

**Falsified if:** a tested pretrained architecture fails the gate on rerun, or
the normalized transition position is not conserved within the panel.

---

## CL-3 (RQ5) — Per-item structural margin predicts behavioral generalization (association, non-causal)

**Statement.** On the template instrument, the per-sentence span-probe margin
at \(l^*\) is positively associated with the model's behavioral
generalization *within form*: Spearman \(\rho(\text{margin}_s, p_{correct,s})\)
computed within form on cross-form sentences (cleft, topical), aggregated as
the mean of the per-form \(\rho\) values.

**Status: SUPPORTED (template instrument), associational.**

**Evidence (verified 2026-10-09 against `ontogenesis/results/`):**
- distilbert (\(l^*=3\)): \(\rho_{within} = 0.338\) (cleft 0.349,
  p = 5.4 × 10⁻¹⁰; topical 0.327, p = 6.7 × 10⁻⁹), ΔR² = 0.0343 over
  surface controls (`distilbert_rq5.json`).
- gpt2 (\(l^*=4\)): \(\rho_{within} = 0.228\) (cleft 0.318, p = 1.8 × 10⁻⁸;
  topical 0.138, p = 0.017), ΔR² = 0.0135 (`gpt2_rq5.json`).
- Controls (`{model}_controls.json`): position-only baseline at chance
  (0.5/0.5); window bag-of-words probe collapses from 0.977 to 0.500 under the
  filler-disjoint split (the margin is not lexical); MLP probe does not
  exceed the linear span probe (0.817 vs 0.988 distilbert; 0.598 vs 1.000
  gpt2) — no linear-probe artifact.
- Transparency note (carried from the draft): the random-init control retains
  a weak cleft margin→behavior correlation (\(\rho = 0.205\), p = 3.4 × 10⁻⁴),
  halved versus pretrained and absent for topical (p = 0.055); since its
  invariance instrument fails the gate (CL-1), this is read as residual
  item-difficulty covariance, not counter-evidence, and is flagged for
  follow-up.

**Scope and limitations (must travel with the claim):**
- **Non-causal interpretation.** The margin is a probe-derived *measurement*
  correlated with behavior; the design does not manipulate the margin and
  supports no causal claim. The association is item-level and within-form.
- **Pooled \(\rho\) is never the estimand.** Pooled correlations are
  form-confounded; the random-init control's pooled \(\rho = 0.777\) with a
  broken span instrument (topical pairwise accuracy 0.013, NO-GO) demonstrates
  that pooled "signal" can be entirely artifactual.
- The estimand is the mean within-form \(\rho\) over the two role-forms;
  per-form values are reported alongside.

**Falsified if:** within-form \(\rho \leq 0\) (or not significant) for both
assay models on rerun with the frozen pipeline.

---

## CL-4 (N1) — Naturalistic validation on attested, human-verified sentences

**Statement (pre-registered as N1, protocol §6 R1–R5).** On attested,
human-verified sentences, the per-sentence span-probe margin at \(l^*\) still
predicts the model's behavioral generalization *within form*, with both
instruments applied zero-shot (nothing retrained on natural data):
Spearman(margin, \(p_{correct}\)) > 0 within form on natural sentences, for
pretrained models, with magnitude comparable to the template result.

**Status: PENDING — not yet a result.** Naturalistic validation is a
**validation of CL-1/CL-2/CL-3, not a fourth research question.** N1 may be
claimed only after **all** of the following pass:
1. **Annotation gates:** 60 accepted sentences per form (cleft, passive,
   canonical; 180 total), preregistered floor 55 per form (165 total). Floor
   invocation only if corpus availability genuinely prevents the target, with
   the shortfall documented *before* computing any naturalistic
   margin–behavior correlation.
2. **Export gates** (`data/annotate.py --export`): offsets resolve, all spans
   tokenize under `MAX_LEN`, no duplicate texts, ≥ 55 keeps per form.
3. **QC:** blind re-annotation of 20 % of keeps ≥ 48 h later, agreement
   ≥ 95 %; co-annotator agreement reported (`--agree`).
4. **T3a–c runs:** `src/extract.py --items data/naturalistic_items.csv`;
   `src/behavior.py --eval-external`; `src/rq5_link.py --natural` for
   distilbert + gpt2 → `results/{model}_rq5_natural.json`.
5. **N1 decision rules** (protocol §6): R1 primary, R2 magnitude, R3
   canonical instrument gate, R4 lexicon split, R5 agreement — with the N1
   clause pre-registered in `paper_blueprint.md` before the first
   naturalistic pipeline run.

**Current annotation status (verified 2026-10-09, `data/annotate.py --list`):**
decided = 100, kept = 29 — passive 22, cleft 7, canonical 0 (target 60 each;
floor 55 each). Rejects: other 63, pronoun_or_coord 8; review queue 0.
Candidate pool: 563 candidates (passive 200, canonical 200, cleft 135,
topical 28; brown 244, gutenberg 319). `data/naturalistic_items.csv` does not
exist yet (export gate not reached).

**Evidence so far (process evidence only, not N1 results):** retrieval T1 done
(`data/retrieve_candidates.py`, SEED 20261007, deterministic); LLM prescreen
pilot done (`data/llm_prescreen.csv`, pinned `qwen2.5:7b-instruct-q4_K_M`,
temperature 0, 2026-10-08, n = 560, 90 proposed keeps; human review of every
analyzed sentence remains authoritative — the draft's placeholder agreement
figures are FILL and unverified until `annotate.py --agree` is run);
journal repaired and preserved (`data/repair_journal.py`; backup
`data/natural_annotations.backup.csv` holds the pre-repair snapshot).

**Scope and limitations:** forms are cleft (cross-form headline), passive
(role-reversed train-form), canonical (instrument-sanity anchor). Attested
topicalization is too rare (< 0.2 % of screened sentences) for a naturalistic
arm and is reported as a limitation; template-side topicalization results
stand unchallenged by, and are not extended to, natural data.

**Falsified if:** R1 fails (mean within-form \(\rho \leq 0\), q ≥ 0.05) for
either assay model, or R3 fails (reported as a template→natural transfer
limitation, no \(\rho\) claim).

---

## Governance claims (process)

- **G-1.** The measurement specification is frozen at `24166f5`; retrieval
  repair changes candidate *provenance tooling* only and touches no frozen
  measurement (see `MEASUREMENT_SPEC.md`).
- **G-2.** Human annotation is authoritative; no sentence is auto-accepted;
  the annotation schema and observer protocol are unchanged
  (`naturalistic_validation_protocol.md` v3).
- **G-3.** Existing human decisions, labels, span boundaries and source
  provenance are preserved across the retrieval repair (verified by
  `ontogenesis/data/test_retrieve_clefts.py`, 25 tests, 2026-10-09).
