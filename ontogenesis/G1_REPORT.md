# G1 Go/No-Go Report — Invariance Transition Pilot

**Date:** 2026-10-07 · **Instrument v2** (pool-confound fix applied: single shuffled noun pool, roles randomized per family → word identity carries zero label information) · **Gate:** blueprint §7 W1

---

## Verdict: **GO** ✅ (4/5 models complete; pythia deferred to user hardware)

| Model | Cross-form peak | @Layer | l* (first significant) | In-form (final) | GO? |
|---|---|---|---|---|---|
| **bert-base** | 0.917 | L8/12 | **6** | 0.992 | ✅ |
| **roberta-base** | 0.973 | L9/12 | **5** | ~0.99 | ✅ |
| **distilbert** | 0.940 | L4/6 | **3** | ~0.99 | ✅ |
| **gpt2** (decoder) | 0.962 | L8/12 | **5** | 0.990 | ✅ |
| pythia-410m | deferred (sandbox too small; run on RTX 3050/M5) | | | | ⏳ |

Decoder-specific notes (gpt2): W profile much flatter than BERT (0.0046→0.0023, final-layer *drop*); R reaches 0.56 (family coherence retained better than BERT's 0.31); cross-form accuracy **collapses at the final layer** (0.96@L8 → 0.63@L12) — a sharper end-of-stack specialization than BERT's gentle ease-off. W2 item: decoder probes should get a predicate/last-token feature variant (left-to-right context means noun1 lacks right context by construction).

Controls: shuffled-label probe ≈ chance at all layers; layer-0 excluded from FDR (embeddings-only baseline); BH-FDR q<0.05 across layers.

---

## What the curves say (bert-base, `figures/g1_bert_base.png`)

1. **T_l — form-invariant structure readout develops across depth.**
   Training a probe on {active, passive} (P1 fillers) and testing on *unseen forms* {cleft, topical} with *unseen fillers* (P2): accuracy starts at/below chance (0.47@L0, anti-phase through L2), crosses significance at **L6**, rises steeply 0.52→0.70→0.82, **peaks 0.917@L8**, then eases to 0.88–0.90 at top layers. In-form control saturates immediately (~0.99+ from L0), isolating the gap: *knowing the roles* is easy from the start; *knowing them in a form you weren't trained on* is the developmental achievement.

2. **W_l vs B_l — geometry dissociates from readout.**
   Absolute within-family surface divergence **rises** (0.028→0.109): the four forms of one structure drift apart with depth (contextual processing differentiates them — naive "representations converge" is false here). Meanwhile form-matched between-family distance **falls** (0.65→0.25): deep layers homogenize across different sentences. Relative index R = W/B rises (0.04→0.31) but stays <1 — family coherence persists, weakened.

3. **The dissociation is the story:** *cross-form structural decoding improves precisely while raw representational form-invariance does not increase.* Invariance, for BERT, is a property of the **readout** (a linear projection recovers roles despite form divergence), not of the **geometry** (points don't converge). This is a publishable nuance against the geometry cluster (Valeriani et al., invariant-features 2026) and directly serves RQ1–RQ3.

## Cross-model first look (RQ6 teaser)

Normalized timing is consistent across four models and two architecture families:

| | l*/depth | peak/depth |
|---|---|---|
| bert-base | 0.50 | 0.67 |
| roberta-base | 0.42 | 0.75 |
| distilbert | 0.50 | 0.67 |
| gpt2 | 0.42 | 0.67 |

The transition isn't a BERT quirk, and its relative position is strikingly conserved. Random-init control still needed to show it comes from pretraining, not architecture.

## Instrument notes (for the paper's §3 integrity)

- v1 of the instrument failed G1's credibility check: profession-pool vs relative-pool made label lexically readable (apparent 0.98 cross-form from L2). v2 removes the confound; *this failed pilot is worth one honest paragraph in the paper* (it operationalizes exactly the Hewitt–Liang selectivity worry).
- All 3,860 items: span assertions passed at generation; 0 span→token mapping failures at extraction (all models so far).
- Layer-0 anomaly: below-chance cross-form transfer (anti-phase probe) — expected; embeddings-only layer excluded from inference.

## Next (blueprint W2)

1. Rerun gpt2 (pad token fixed) + finish pythia → complete 5-model panel.
2. Random-init BERT control (architecture without pretraining — expect l* to vanish or move to top).
3. Lock F1/F2 → begin geometry module (RQ3: ID, neighborhood overlap, CKA conditioned on form/structure splits).
