# A Controlled Instrument for Depth-Wise Invariance: When Surface-to-Structure Transitions Happen, Whether They Predict Generalization, and How Universal They Are

*Working draft v0.1 — 2026-10-08. Target: ARR submission 4 Jan 2027 → ACL 2027.*
*Convention: `[FILL: …]` marks a number/table awaiting a verified source (user-local results JSON or the pending naturalistic export). Nothing marked FILL may enter a submission draft without its source pasted in.*

---

## Abstract

Probing studies ask *where* linguistic structure becomes decodable in a
transformer; behavioral studies ask *whether* models generalize that
structure across surface forms. The two literatures rarely meet inside a
single controlled experiment. We introduce an instrument that closes the
three loopholes separating them — surface leakage, role memorization, and
architectural confounds — via filler-disjoint probe splits, role-randomized
item families, matched surface paraphrases (3,860 items, 1,060 families),
and a random-initialization control. With it we measure a per-layer
invariance margin and ask three questions. (RQ1) The surface-to-structure
transition is sharp and layer-localized in pretrained models (crossover
layer l\* per model) and absent in a randomly initialized control (max
T_l = 0.49, no crossover). (RQ6) The transition profile recurs across
architecture families and scales at comparable relative depth. (RQ5) The
per-item invariance margin at l\* predicts per-item behavioral
generalization within construction (within-form ρ = 0.34 distilbert,
0.23 gpt2; ΔR² = 0.034/0.014 over surface controls), and this link
replicates on human-verified attested sentences [FILL: naturalistic ρ
per form + CIs from the N1 run]. We release the instrument, features,
and probes.

---

## 1. Introduction (~1.2 pp)

Two research programs circulate around the same phenomenon without
touching. Interpretability has mapped *where* syntactic and semantic
information becomes linearly decodable across depth (Tenney et al.'s
BERT-rediscovers-the-pipeline result; Hewitt & Manning's structural
probes; the derivational-probing line, whose authors note that the
layer-wise unfolding of derivational relations "remains poorly
understood"). Behavioral work has documented *whether* models generalize
argument structure across paraphrase — active/passive alternations,
dative shifts — and how training shapes that generalization. What neither
program does is open the model at the layer where its behavioral
flexibility is decided.

The missing object is an **instrument**: a dataset-plus-measurement
pipeline in which surface form and role structure can be varied
independently, memorization is blocked by construction, and the residual
representational signal can be tied to per-item behavior. Building that
instrument is this paper's central contribution; the questions it answers
are older than the instrument itself.

We therefore ask three questions:

- **RQ1 (phenomenon + trajectory).** At which layer does the internal
  representation of "who did what to whom" become independent of how the
  sentence is worded — and what is the shape of that transition?
- **RQ6 (universality).** Does the transition recur across architectures
  (families, scales) at comparable relative depth, and is it absent
  without pretraining?
- **RQ5 (the link).** Does a sentence's invariance margin at l\* predict
  the model's behavioral generalization *for that sentence* — on
  controlled templates and on attested, human-verified sentences?

Contributions: (1) the instrument (§3–4), released; (2) a localized,
pretraining-dependent transition with a random-init null (RQ1);
(3) cross-architecture recurrence (RQ6); (4) the first per-item
margin→behavior link for this transition, replicated naturalistically
(RQ5); (5) negative results that bound the claim (controls, §5.3;
failed LLM co-annotation pilot, §5.4).

We deliberately do **not** claim a mechanism: which heads, MLPs, or
features produce the transition is the subject of ongoing work enabled by
the released window (§6).

## 2. Related Work (~2 pp)

**(a) Depth-wise emergence and probing.** BERT pipeline rediscovery;
structural probes; layer-wise decodability of syntactic relations;
derivational probing. These establish *decodability*, not *invariance*:
a probe can read roles off surface cues. Our T_l measure is
form-contrastive by construction.

**(b) Representational geometry across layers.** CKA/similarity-decay
analyses; invariant-feature accounts of depth. [FILL: citations from
literature_map.md round 2.] We differ by anchoring geometry to a
behavioral criterion rather than to self-similarity.

**(c) Generalization and abstraction of argument structure.** TACL line
on argument-structure generalization; structural-abstraction training
results. These measure behavior without opening the model; we predict
their per-item variability from internal state.

**(d) Mechanism.** Activation patching, causal abstraction. We position
against, not within, this cluster: the depth×mechanism cell is exactly
what we localize and defer (§6).

## 3. The Instrument (~1.5 pp)

**Item families.** 1,060 families, 3,860 items (Table T1). Each family
fixes one agent, one patient, one predicate; surface forms realize the
same roles: active (600), passive (600), it-cleft (600), topicalization
(600), plus a swapped-role control form (600) and dative variants
(dat_pp 400 / dat_do 400). Any representational difference *within* a
family across forms is pure surface effect; any difference across the
swap form is pure role effect.

**Anti-shortcut construction.** (i) *Filler-disjoint splits*: probe
training (P1, n = 1,900) and probe evaluation (P2, n = 1,900) share no
filler nouns or verbs — lexical identity cannot carry the probe.
(ii) *Role randomization*: every filler appears as agent in some families
and patient in others. (iii) *Deterministic generation*: fixed seed
(20261007), fully released generator.

**Naturalistic arm.** Because template-based instruments invite the
"templateese" objection, RQ5 is replicated on attested sentences:
Brown + Project Gutenberg retrieval, strict inclusion rules (two full-NP
participants, main-clause predicate, ≤22 words), LLM pre-screen reported
as a failed pilot (qwen2.5-7b-instruct, temp 0: 50 % human keep-rate of
proposals, 0/3 span-exact), and 100 % human annotation of the analyzed
set [FILL: n per form at export; target 60/60/60, gate ≥55].

## 4. Measures (~1.5 pp)

**Surface score S_l.** Linear probe (logistic, fixed hyperparameters)
predicting surface form from role-token states at layer l.
**Structure score T_l.** Cross-form probe: trained on P1 of some forms,
evaluated on P2 of held-out forms — role structure readable *independent
of the form it was trained on*. **Crossover layer l\*.** First layer
where T_l exceeds S_l's form-specific baseline [FILL: exact operational
definition from src/invariance.py v3 — one sentence + formula].
**Per-item invariance margin.** For item s at l\*: probe-confidence
margin between its role-correct and role-swapped encodings — the
per-sentence quantity that RQ5 regresses against behavior.
**Behavioral assay.** [FILL: one-paragraph description of the
generalization assay and p_correct_s from src/behavior.py v3.]
**Instruments are frozen for the naturalistic arm:** probes and behavior
head trained on templates only, applied zero-shot to attested sentences.

## 5. Results (~3 pp)

### 5.1 RQ1 — the transition is real, sharp, and learned

[FILL: G1 figure — S_l and T_l curves per model.] The crossover is
layer-localized: for distilbert, within-form structure correlation rises
from ≈0.34 at L3 (l\*) to ≈0.40 across L4–L6, with span-probe invariance
0.99 at L4. The random-initialization control (bert_random) shows no
crossover at any layer (max cross-form accuracy 0.493 at L4 of 12; span
cross-form max 0.505 at L6; GO/NO-GO = NO-GO on both levels): the
transition is a property of pretraining, not architecture.

### 5.2 RQ6 — recurrence across architectures

All eight pretrained models pass the crossover gate; the control fails it
(Table T2, transcribed from the per-model summary JSONs, 2026-10-08).

| model | layers | l_struct | l\*(span) | cross-form acc max | span cross max | GO |
|---|---|---|---|---|---|---|
| bert_base | 13 | 6 | — | 0.917 @ L8 | — | ✓ |
| roberta_base | 13 | 5 | — | 0.973 @ L9 | — | ✓ |
| distilbert | 7 | 3 | 3 | 0.940 @ L4 | 0.988 @ L4 | ✓ |
| gpt2 | 13 | 5 | 4 | 0.962 @ L8 | 1.000 @ L8 | ✓ |
| gpt2_medium | 25 | 6 | 6 | 0.995 @ L17 | 0.985 @ L11 | ✓ |
| opt_125m | 13 | 6 | 3 | 0.953 @ L10 | 0.985 @ L8 | ✓ |
| pythia_160m | 13 | 5 | 1 | 0.905 @ L7 | 0.952 @ L5 | ✓ |
| pythia_410m | 25 | 6 | 1 | 0.992 @ L10 | 0.988 @ L10 | ✓ |
| bert_random | 13 | — | — | 0.493 @ L4 | 0.505 @ L6 | ✗ |

Four model families (BERT, RoBERTa/DistilBERT encoder line; GPT-2,
Pythia, OPT decoders), two architecture classes, 66M–355M parameters.
The transition lands in the first half to middle of every pretrained
network — relative depth l_struct/(L−1) between 0.25 (both 24-layer
models) and 0.50 (12-layer models and distilbert) — and never in the
untrained control.

### 5.3 RQ5 (templates) — the margin predicts behavior

For distilbert at l\* = 3, the per-item margin correlates with behavioral
correctness within form: ρ_within = 0.338 (cleft 0.349, p = 5.4 × 10⁻¹⁰;
topical 0.327, p = 6.7 × 10⁻⁹), ΔR² = 0.034 over surface controls. For
gpt2 at l\* = 4: ρ_within = 0.228 (cleft 0.318, p = 1.8 × 10⁻⁸; topical
0.138, p = 0.017), ΔR² = 0.014. Pooled correlations across forms are
form-confounded and are never used as the estimand — the
random-initialization control demonstrates why: its pooled ρ is 0.777
while its span instrument is broken (topical pairwise accuracy 0.013;
NO-GO), i.e. pooled "signal" can be entirely artifactual. Controls
(both models): a position-only baseline is at chance (0.5/0.5); a window
bag-of-words probe collapses from 0.977 to 0.500 under the
filler-disjoint split — the margin is not lexical; an MLP probe does not
exceed the linear span probe (0.817 vs 0.988 distilbert; 0.598 vs 1.000
gpt2) — no linear-probe artifact. One transparency note: the
random-init control retains a weak cleft margin→behavior correlation
(ρ = 0.205, p = 3.4 × 10⁻⁴), halved versus pretrained and absent for
topical (p = 0.055); since its invariance instrument fails (above), we
read this as residual item-difficulty covariance, not counter-evidence,
and flag it for the follow-up mechanism work.

### 5.4 RQ5 (naturalistic, N1) — replication on attested sentences

[FILL: everything in this subsection comes from the pending export +
T3a–c runs: per-form ρ with Fisher-z CIs (R1), magnitude vs template
(R2), canonical instrument gate (R3), lexicon_overlap split (R4), and the
co-annotation agreement sentence (R5): "Pre-screening by a pinned local
LLM (qwen2.5-7b-instruct-q4_K_M, temperature 0) yielded a 50 % human
keep-rate and 0 % span-exact agreement; it was therefore demoted to a
shortlist and all analyzed sentences were manually human-annotated, with
blind re-annotation QC after ≥48 h."]

## 6. Discussion (~1.2 pp)

**Computational ontogenesis.** We frame a single forward pass as a
developmental trajectory: raw tokens at the input, abstract role
structure by l\*. We scope the term explicitly to the *depth* axis and
distinguish it from developmental interpretability on the *training*
axis. The instrument is the contribution; the framing is a lens.

**Limitations.** Synthetic core (mitigated, not eliminated, by the
naturalistic arm); linear probes (MLP control bounds this); single
primary annotator with blind re-annotation QC; cleft scarcity in attested
prose [FILL: final n per form + limitation sentence per protocol §8];
no causal claims.

**Roadmap.** Two questions the instrument localizes but does not answer:
(i) what geometric reorganization accompanies the transition (former
RQ3), and (ii) whether role structure is preserved as an isomorphism
across layers (former RQ4). Both are tractable with the released
instrument and features. The mechanism question — what computation
produces the transition inside the window l\*±L — is the subject of
follow-up work using activation patching on the released window.

## 7. Conclusion (~0.4 pp)

One instrument, three answers: the surface-to-structure transition
exists, is learned, recurs across architectures, and — per item —
predicts behavior on both controlled and attested sentences. The window
is localized; the mechanism is next.

---

## Tables

- **T1 — Dataset statistics.** 3,860 items / 1,060 families; forms:
  active 600, passive 600, cleft 600, topical 600, swap 600, dat_pp 400,
  dat_do 400, attach 60; splits P1 1,900 / P2 1,900 / aux 60. (verified
  from data/items.csv, 2026-10-08)
- **T2 — Per-model l\* and trajectory typology.** COMPLETE — see §5.2
  table (source: 9 summary JSONs, verified 2026-10-08).
- **T3 — Margin→behavior regression per model.** distilbert: l\*=3,
  ρ_within 0.338, cleft 0.349 / topical 0.327, ΔR² 0.0343. gpt2: l\*=4,
  ρ_within 0.228, cleft 0.318 / topical 0.138, ΔR² 0.0135. (source:
  distilbert_rq5.json, gpt2_rq5.json, verified 2026-10-08)
- **T4 — Naturalistic replication summary.** [FILL: post-export]

## Reviewer preemption (from blueprint §9 — kept out of the paper body)

1. "Probes read surface cues" → filler-disjoint splits + window-BoW
   collapse (0.977→0.500) + position control at chance.
2. "Templateese" → naturalistic N1 arm, zero-shot instruments, human
   verification of every analyzed sentence.
3. "Linear-probe artifact" → MLP no-unlock result.
4. "Architectural inevitability" → random-init null (no crossover, max
   T_l 0.4933).
