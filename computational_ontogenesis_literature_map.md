# Literature Reconnaissance: "Computational Ontogenesis" & Adjacent Research

**Purpose:** Map the existing research space before committing to framing, questions, or methods.
**Structure:** existing work → what it measures → what it explains → what it doesn't → gap → where the six RQs fit.

---

## TL;DR

1. **The term "Computational Ontogenesis" is effectively unused in ML/interpretability.** Nothing prevents coining it — but two adjacent communities use "ontogenesis/ontogeny" differently (theoretical biology of self-replication; Simondonian philosophy of AI images), and — more importantly — **"developmental interpretability" already claims the *development* metaphor for the training-time axis.** Your usage must be explicitly scoped to the *depth axis* (construction across forward-pass computation) to avoid collision.

2. **The literature splits cleanly along two axes:**
   - **Axis A — where you look:** across **depth** (one forward pass) vs. across **training** (learning over time).
   - **Axis B — what kind of question:** **phenomenology** (what is measurable/decodable, where) vs. **mechanism** (what computation causally produces the change).

3. **The crowded cells are:** depth × phenomenology (probing, geometry, lenses) and behavior-endpoint × mechanism (circuits, patching). **The sparse cell — your actual target — is depth × mechanism:** *what computation in layers ℓ..ℓ′ causally transforms representation X into representation Y?* Very little work asks this directly about a *representational transition* rather than about a *behavior*.

4. **Your six RQs all sit in the depth × phenomenology cell.** They are a legitimate scaffold, but RQ1–RQ4 have substantial near-neighbors (Tenney 2019, Hewitt & Manning 2019, Derivational Probing 2025, Valeriani 2023). Novelty must come from the *controlled surface/structure design* and the *explicit bridge to mechanism*, not from "layerwise analysis" itself.

5. **The closest competitor to read first:** *Derivational Probing* (2025) — it already asks "how are syntactic structures constructed across layers" and answers with probes only, leaving the causal mechanism question open. That paper both validates your question and marks the gap.

---

## Part I — The Term Check (Questions 1–4)

### 1.1 Is "Computational Ontogenesis" already used?

**As an exact phrase, essentially no — in the sense you mean it.** Searches surface:

| Work | What they mean by it | Relation to your framing |
|---|---|---|
| [Computational Ontogeny (Buckley, *Biological Theory*)](https://www.researchgate.net/publication/235432119_Computational_Ontogeny) | L-system models of machine **self-replication** — how a constructor constructs a copy of itself | Different problem entirely (replication, not representation-formation) |
| [Ontogenesis (Synthesis Center)](https://synthesiscenter.net/projects/ontogenesis/) | Process-philosophy research program on open-ended living processes "beyond complexity" | Humanities/art; no ML usage |
| [Beyond the Black Box: Mapping the Ontogenesis of AI-Generated Images (2026)](https://link.springer.com/article/10.1007/s13347-026-01141-1) | Simondonian individuation: diffusion latent space as "pre-individual field," denoising as transduction | **Closest philosophical usage**: "ontogenesis" of an artifact's production — but about *image generation dynamics*, not layer-wise representation construction in a fixed model; philosophy-of-technology framing, not an interpretability method |
| [Ontogenesis of AGI Consciousness in ChatGPT (2026)](https://link.springer.com/chapter/10.1007/978-3-032-20749-4_5) | Gradual layering of *consciousness* functions around an LLM via external modules | Cognitive architecture; unrelated to representation analysis |
| Developmental biology ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10341769)) | Ontogenesis of organisms | Established meaning in biology; a source of the metaphor, not competition |

**Verdict:** The term is coinable. No established field owns it. But note the two "adjacent owners" of ontogenesis language: self-replication theory (Buckley) and Simondonian philosophy (Springer 2026) — a related-work footnote should acknowledge both to preempt "the term is already taken" reviews from interdisciplinary reviewers.

### 1.2 What adjacent terms carry essentially the same idea?

These are the terms your framing must position against — they are where the same *idea* already lives:

| Adjacent term | Core meaning | Axis it claims |
|---|---|---|
| **Developmental interpretability** ([review, 2025](https://pith.science/paper/2508.15841); [overview + Lehalleur et al. position paper](https://danmackinlay.name/notebook/ai_dev_interp.html)) | How capabilities/mechanisms *form during training* — "consistent ontogeny" is literally the review's word for circuit formation across checkpoints | **Training time × mechanism** |
| **Mechanistic interpretability** ([Bereska & Gavves review](https://leonardbereska.github.io/blog/2024/mechinterpreview/); [2026 survey](https://arxiv.org/html/2602.11180v1)) | Reverse-engineering features/circuits that implement behaviors | **Behavior-endpoint × mechanism** |
| **Probing / representation analysis** ([Belinkov & Søgaard survey](https://ar5iv.labs.arxiv.org/html/2102.12452)) | What is decodable from hidden states, layer by layer | **Depth × phenomenology** |
| **Iterative inference / prediction trajectories** ([logit & tuned lens](https://learnmechinterp.com/topics/logit-lens-and-tuned-lens/); [Tuned Lens paper](https://www.researchgate.net/publication/369233674_Eliciting_Latent_Predictions_from_Transformers_with_the_Tuned_Lens)) | Layer-wise refinement of the model's own predicted output | **Depth × phenomenology (output-side)** |
| **Universality hypothesis** ([three dimensions of universality](https://learnmechinterp.com/topics/universality/)) | Similar features/circuits across models/architectures | **Cross-model comparison** |
| **Representation evolution / layerwise analysis** ([Voita et al.](https://lena-voita.github.io/posts/emnlp19_evolution.html); [Valeriani et al.](https://arxiv.org/pdf/2302.00294)) | How representations change across layers | **Depth × phenomenology** |

**Key strategic point:** "Developmental interpretability" = ontogenesis **along the training axis**. Your framing = ontogenesis **along the depth axis** (and possibly a two-axis synthesis). This is a real, defensible distinction — and probably the sentence your eventual paper needs most: *"Developmental interpretability studies how mechanisms form over training; we study how a single forward pass progressively constructs a representation."*

---

## Part II — The 2×2 Map of the Territory

|  | **Phenomenology** (measure/decode) | **Mechanism** (causally explain) |
|---|---|---|
| **Across DEPTH** (forward pass) | 🟩 **CROWDED** — probing, structural probes, expected layers, geometry/ID, lenses, invariance metrics | 🟨 **SPARSE — YOUR GAP** — a handful of layer-skip/intervention studies; almost no work causally explaining a *representational transition* |
| **Across TRAINING** (time) | 🟩 Crowded — emergence, grokking, phase transitions | 🟨 Rising fast — developmental interpretability, circuit formation over checkpoints |
| **Behavior endpoint** (whole model) | 🟩 Crowded — benchmarks, behavioral evals | 🟩 Crowded — IOI, induction, ACDC, patching, SAE circuits |

**Why the sparse cell is genuinely sparse:** mechanistic work normally anchors on a *behavior* ("which heads implement indirect-object identification?") and asks which components cause *output changes*. Nobody routinely anchors on a *representational transition* ("surface-invariant → structure-sensitive between layers 5 and 8") and asks which components cause *that*. The tools exist (patching, interchange interventions, causal abstraction); the targeting has not been done this way at scale.

---

## Part III — The Detailed Map (existing work → measures → explains → doesn't → gap)

### Cluster A — Layer-wise emergence & probing ("when does what appear?")

| Work | Measures | Explains | Doesn't |
|---|---|---|---|
| [Tenney et al., BERT Rediscovers the Classical NLP Pipeline (ACL 2019)](https://aclanthology.org/P19-1452/) | Expected layer / center-of-gravity per linguistic task via edge probing; finds POS → parse → NER → SRL → coref in expected order | *Where* linguistic information becomes decodable; ordering | Whether the model *uses* it; how one layer computes the next stage; dynamics of transition |
| [Hewitt & Manning structural probe (2019)](https://arxiv.org/html/2402.16168) (via [non-linear probe extension](https://arxiv.org/html/2402.16168)) | Linear transform making Euclidean distance ≈ dependency-tree distance (UUAS); layer-wise syntax recovery | That full syntactic trees are geometrically encoded, deepening with layer | Causal role; mechanism of construction |
| [Derivational Probing (2025)](https://arxiv.org/html/2506.21861v1) — **closest competitor** | Per-layer UUAS + expected-layer metric for subgraphs; finds **bottom-up** construction (micro-phrase structure before macro root structure) in BERT, parallel results in GPT-2 | A *trajectory of construction order* for syntax across depth | Still purely correlational probing; explicitly leaves "how" unasked |
| [Voita et al., Evolution of Representations (EMNLP 2019)](https://lena-voita.github.io/posts/emnlp19_evolution.html) | Layer-to-layer change (PWCCA), information-bottleneck view for LM/MLM/MT objectives | Different training objectives produce different depth-dynamics (encoding vs reconstruction phases) | No causal decomposition; task-agnostic measures |
| [Evolution of Syntactic Information under fine-tuning (EACL 2021)](https://aclanthology.org/2021.eacl-main.191.pdf) | Structural-probe performance across fine-tuning checkpoints | Syntactic info is forgotten/reinforced/preserved depending on task | Mechanism; within-forward-pass computation |
| [Evidence of Generative Syntax (CoNLL 2025)](https://aclanthology.org/2025.conll-1.25.pdf) | Repurposes structural probe for constructions identical in dependency but differing in generative syntax | Whether sub-surface theoretical structures are reflected | Still probing; no mechanism |

**Gap:** Everyone here answers *when/where*, none answer *what computation produces the transition*. Derivational Probing's own framing ("derivations… remain poorly understood") confirms the gap from inside the closest competitor.

### Cluster B — Geometry across depth

| Work | Measures | Explains | Doesn't |
|---|---|---|---|
| [Valeriani et al., Geometry of Hidden Representations (NeurIPS 2023)](https://arxiv.org/pdf/2302.00294) | Intrinsic dimension + neighborhood overlap across layers; three-phase pattern (expansion → plateau → re-expansion) | That semantic content peaks at the ID trough; unsupervised layer selection | Why the phases occur; causality; linguistic-structure specificity |
| [Layer-wise similarity analysis (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/03d113a060c0ac93a5859517a0f07271-Paper-Conference.pdf) | Per-sample cosine/CKA across layers; theoretical justification via geodesic assumption; saturation events | Why predictions stabilize across depth; early-exit justification | Computation causing similarity growth |
| [Layer by Layer: Uncovering Hidden Representations (ICML 2025)](https://arxiv.org/abs/2502.02013) | Unified framework: information-theoretic + geometric + **perturbation-invariance** metrics per layer | Intermediate layers often beat final layers; a metric toolkit | No causal account of transitions; no linguistic structure focus |
| [Invariant Features in Language Models (2026)](https://www.alphaxiv.org/fr/abs/2605.06458) | Local geometric eigen-decomposition separating nuisance (paraphrase) directions from invariant semantic subspaces across depth; **representation-level interventions** | Where invariance to surface variation lives (mid-late layers); causal role of invariant components via intervention | Intervention targets the invariant *subspace*, not the per-layer *computation that builds it*; not structure/syntax-specific |
| [Abstract representational geometry supports inference (2026)](https://pith.science/paper/2606.23345) | Layer-organized manifolds (stimulus identity low, abstract context high); geometric regularization interventions | Geometry hierarchically organizes abstraction and causally supports generalizable inference | Task-specific reversal learning; not general depth mechanism |

**Gap:** Geometry tells you *the shape of the transition*, not *the computation implementing it*. The 2026 invariant-features paper is the first in this cluster to add interventions — it is the geometry cluster's beachhead into your gap cell, and must be cited.

### Cluster C — Prediction trajectories (lenses)

| Work | Measures | Explains | Doesn't |
|---|---|---|---|
| [Logit lens / Tuned lens (Belrose et al. 2023)](https://www.researchgate.net/publication/369233674_Eliciting_Latent_Predictions_from_Transformers_with_the_Tuned_Lens); [synthesis](https://learnmechinterp.com/topics/logit-lens-and-tuned-lens/) | Vocabulary-decodable "prediction" at every layer; trajectory converging to final output | That inference is *iterative refinement*; where predictions crystallize | Explicitly observational: "neither shows that the model **uses** it"; no causal decomposition |
| [QLens (2025)](https://arxiv.org/html/2510.11963v1) | Formalizes layer transitions as state evolutions on top of tuned lens | A mathematical characterization attempt of per-layer contribution | Toy model; still not causal in the intervention sense |

**Gap:** Lenses are the phenomenology of the *output side* of depth. They document trajectories without explaining the machinery producing each step.

### Cluster D — Abstraction, invariance, generalization

| Work | Measures | Explains | Doesn't |
|---|---|---|---|
| [How Abstract Is Linguistic Generalization? (TACL 2023)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00608/118116/How-Abstract-Is-Linguistic-Generalization-in-Large) | Behavioral generalization of argument structure with novel verbs; Type 1 vs Type 2 knowledge | That apparent abstraction often rests on **surface linear-order heuristics** | Internal representations and layer dynamics behind the failure |
| [Structural Abstraction as Inductive Bias (2026)](https://arxiv.org/html/2603.17198v2) | Training-time loss shaping toward surface-invariant representations; structural generalization benchmarks | That biasing toward invariance improves structural generalization | Pretrained-model internals; depth mechanism |

**Gap:** The behavioral literature (does generalization survive surface change?) and the representation literature (does invariance emerge with depth?) barely talk to each other. **RQ5's actual room: linking measured depth-wise invariance to behavioral generalization, within the same controlled design.**

### Cluster E — Mechanism & causal intervention (the toolkits you'll eventually use)

| Work | Measures | Explains | Doesn't |
|---|---|---|---|
| [Activation patching guide (Heimersheim & Nanda 2024)](https://learnmechinterp.com/topics/activation-patching/) + [ACDC (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Paper-Conference.pdf) | Necessity/sufficiency of components via activation swaps; automated circuit discovery | Which components causally carry a *behavior* | Anchored on outputs, not on representational transitions across depth |
| [Causal Abstraction / interchange interventions (Geiger et al., NeurIPS 2021)](https://arxiv.org/abs/2106.02997); [DAS alignment (2023)](https://arxiv.org/html/2303.02536v4) | Align high-level causal variables with subspaces; test with interchange interventions (IIA) | Whether a representation *plays a causal role* matching an interpretable variable | Rarely applied to *depth-staged construction* (it tests presence/role, not the construction process) |
| [Transformer Layers as Painters (2024)](https://arxiv.org/html/2407.09298v2) | Layer skipping/swapping/parallelization in frozen models | Middle layers share a representation space; order matters less than expected | Coarse granularity; no task-specific representational transition targeted |
| [Hidden Heroes & Gradient Bloats (2026)](https://pith.science/paper/2602.01442) | Ablation vs gradient attribution across layers | Gradient attribution *inverts* causal importance across depth — a methodological warning | Small algorithmic tasks; warns rather than explains transitions |
| [Mechanistic interpretability surveys](https://arxiv.org/html/2602.11180v1), [Bereska & Gavves](https://leonardbereska.github.io/blog/2024/mechinterpreview/) | Taxonomies of circuit discovery, patching, SAEs, steering | Mature toolkit; acknowledged scaling and validation gaps | Toolkit exists; transition-targeted application does not |

### Cluster F — Training-axis development (your nearest framing neighbor)

| Work | Measures | Explains | Doesn't |
|---|---|---|---|
| [Developmental interpretability review (2025)](https://pith.science/paper/2508.15841) | Circuit formation, biphasic knowledge, phase transitions, transient ICL across training | The *training-time* ontogeny of mechanisms; explicitly parallels human development | Says nothing about within-forward-pass construction across depth |
| [Circuits consistent across training & scale (2024)](https://arxiv.org/html/2407.10827v1) | Circuit emergence over 300B Pythia tokens, 70M–2.8B params | Algorithm stable while components swap; emergence at consistent token counts | Behavior-anchored circuits; no depth-transition account |
| [Induction heads / phase transitions (Olsson et al. 2022)](https://mbrenndoerfer.com/writing/mechanistic-interpretability) (summary) | Formation of induction circuit coinciding with loss "bump" | A concrete mechanism's birth during training | One circuit; training axis only |

**Strategic implication:** this cluster owns the word "development/ontogeny." Your framing must either (a) restrict itself to depth, or (b) explicitly propose *two axes of ontogenesis* (construction across depth within a pass; construction across training across passes) and claim the understudied one.

### Cluster G — Universality / cross-model (RQ6's neighborhood)

| Work | Measures | Explains | Doesn't |
|---|---|---|---|
| [Universality across models — synthesis](https://learnmechinterp.com/topics/universality/) | CKA/SVCCA, SAE feature matching across seeds/scales/architectures (Transformer↔Mamba) | Three dimensions of universality; architecture-level feature overlap (~0.68 mean pairwise correlation, near seed skyline) | Endpoint comparisons; not depth-profile or trajectory comparisons |
| [Contrastive-Difference CKA (2026)](https://arxiv.org/html/2606.16897) | Concept-specific geometric alignment across five architecture families | Moderate geometric convergence + ≥94% functional transfer — a *dissociation* | Static, not depth-staged |
| [Mechanistic universality of comparison circuits (ICLR 2026 workshop)](https://openreview.net/forum?id=79igg0kRtd) | Circuit structures across model families/sizes with causal interventions | Whether circuits generalize across families | Single task; behavior-anchored |

**RQ6 room:** universality is studied at *endpoints* (final representations, features, circuits). **Universality of the depth-trajectory itself** (do all architectures pass through the same phenomenological stages in the same order?) is much less charted — and it is a natural RQ6 differentiator.

---

## Part IV — Where Your Six RQs Fit

| RQ | Question | Nearest prior work | Overlap risk | What would differentiate |
|---|---|---|---|---|
| **RQ1** When does abstraction emerge? | Layer of transition from surface- to structure-sensitivity | [Tenney 2019](https://aclanthology.org/P19-1452/), [Hewitt & Manning 2019](https://arxiv.org/html/2402.16168), [Derivational Probing 2025](https://arxiv.org/html/2506.21861v1) | 🔴 **High** — this exact question is answered for syntax (decodability) | Controlled *surface-matched/structure-contrastive* pairs (decodability ≠ invariance); decoder-agnostic measures; multi-model |
| **RQ2** What trajectory does it follow? | Shape/order of development across depth | [Voita 2019](https://lena-voita.github.io/posts/emnlp19_evolution.html), [Derivational Probing (bottom-up)](https://arxiv.org/html/2506.21861v1) | 🟠 Medium-high — "bottom-up construction" already claimed for syntax | Trajectory of *invariance* (not just decodability); quantitative trajectory typology; across objectives/architectures |
| **RQ3** What happens geometrically? | Dimension/cluster/subspace changes at transition | [Valeriani 2023](https://arxiv.org/pdf/2302.00294), [ICLR 2025 similarity](https://proceedings.iclr.cc/paper_files/paper/2025/file/03d113a060c0ac93a5859517a0f07271-Paper-Conference.pdf), [Layer-by-Layer 2025](https://arxiv.org/abs/2502.02013), [Invariant features 2026](https://www.alphaxiv.org/fr/abs/2605.06458) | 🟠 Medium — toolkit is established; 2026 invariant-subspace paper is very close | Tie geometry specifically to the surface/structure transition windows from RQ1–2; add representation-level intervention (the 2026 paper shows this bar is now expected) |
| **RQ4** Is structure preserved? | Retention of structural relations through depth | [Structural probe lineage](https://arxiv.org/html/2402.16168), [EACL 2021 evolution](https://aclanthology.org/2021.eacl-main.191.pdf) | 🟡 Medium | Define preservation as *relational* isomorphism across transition, with controls from [probing critiques](https://ar5iv.labs.arxiv.org/html/2102.12452) (selectivity/MDL) baked in |
| **RQ5** Does it generalize? | Does depth-wise invariance predict behavioral generalization across surface forms? | [TACL argument structure](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00608/118116/How-Abstract-Is-Linguistic-Generalization-in-Large) (behavioral), [abstract geometry 2026](https://pith.science/paper/2606.23345) (representational, single task) | 🟡 Medium — **the link itself is under-studied** | The *same* controlled items measured at both levels: per-layer invariance ↔ per-item generalization. This is your strongest Level-1 RQ |
| **RQ6** Architecture-general? | Do the stages/order hold across model families? | [Universality synthesis](https://learnmechinterp.com/topics/universality/), [CKAΔ 2026](https://arxiv.org/html/2606.16897) | 🟡 Medium — endpoint universality is busy; **trajectory universality is not** | Frame as universality of the *ontogenetic profile* (stage order), not of features; seeds + scales + architectures (the three universality dimensions, mapped onto stages) |

**Honest summary:** none of RQ1–RQ6 is individually novel as a *question form*; all are established question forms. The defensible contribution of Level 1 is the **coherent controlled design** (surface variants ↔ matched structural representations measured in one framework) + **cross-axis linking (RQ5)** + **trajectory-level universality (RQ6)**. The contribution that survives review is Level 2: mechanism of the transition.

---

## Part V — Question 12: What would genuinely be a *better question*?

A better question is not a better metric on the same question. Test: *does it change what counts as an answer?*

### Candidate framings (in decreasing order of defensibility)

1. **Transition-anchored mechanism (recommended as the long-run target):**
   > *"Which component-level computations in layers ℓ..ℓ′ causally implement the transition from surface-sensitive to structure-sensitive representation?"*
   - Changes the answer type: from a decodability curve to a causal decomposition (heads/MLPs/subspaces, necessity vs. sufficiency).
   - Tooling exists: [activation patching](https://learnmechinterp.com/topics/activation-patching/), [causal abstraction/interchange interventions](https://arxiv.org/abs/2106.02997), [path patching](https://leonardbereska.github.io/blog/2024/mechinterpreview/), [attribution patching](https://proceedings.neurips.cc/paper_files/paper/2023/file/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Paper-Conference.pdf).
   - The novelty is the **anchor**: interventions targeted at a *representational transition window* identified by RQ1–RQ4, not at a behavior.

2. **Representational sufficiency/causal role:**
   > *"Is the structural representation at layer ℓ causally sufficient — interchange the structural content between matched inputs at layer ℓ and does downstream structure-sensitive behavior follow?"*
   - Directly upgrades RQ4 ("is structure preserved?" → "is structure *used*?").

3. **Two-axis ontogenesis (framing-level):**
   > *"How does a system construct abstraction twice over — across depth within a pass, and across training between passes — and do the two constructions share stage structure?"*
   - Bigger claim; requires both literatures; possible as a position-paper contribution, not as the first empirical paper.

4. **Mechanism-portability (links RQ6 to Level 2):**
   > *"Is the causal mechanism of the transition itself universal across architectures — same stages, same computation, different components?"*
   - Builds on the finding that [algorithms stabilize while components swap across training](https://arxiv.org/html/2407.10827v1) — ask the depth-axis analog.

### What NOT to do (per your own trap)

- Don't add RQ7 yet as decoration. Add it when Level 1 has *localized the window* — mechanism questions need a target region to intervene on.
- Don't let "low-level" become "inspect everything": [layer-wise attribution can invert causal importance](https://pith.science/paper/2602.01442), and probing can only ever show *present*, never *used* ([lens synthesis](https://learnmechinterp.com/topics/logit-lens-and-tuned-lens/); [probing critiques](https://ar5iv.labs.arxiv.org/html/2102.12452)).

---

## Part VI — Answers to Your 12 Questions (condensed)

1. **Is the term used?** Not in ML in your sense. Near-usages in theoretical biology (self-replication) and Simondonian philosophy of AI images ([2026](https://link.springer.com/article/10.1007/s13347-026-01141-1)).
2. **Who uses adjacent versions?** Philosophy-of-technology (Simondon), self-replication theory (Buckley), developmental-biology reviewers.
3. **What do they mean?** Artifact/organism individuation over time — *not* layer-wise construction of representations.
4. **Adjacent terms for the same idea?** "Developmental interpretability" (training axis), "representation evolution across layers" (depth axis), "iterative inference," "mechanistic interpretability," "universality."
5. **Who studies emergence of representations across computation/depth?** Clusters A–C: Tenney, Hewitt & Manning, Voita, Derivational Probing, Valeriani, Layer-by-Layer, tuned lens.
6. **Who studies mechanisms rather than trajectories?** Cluster E: patching/ACDC, causal abstraction (Geiger), Painters, sparse feature circuits — but anchored on *behaviors*, not depth-transitions.
7. **Who uses geometry?** Cluster B (Valeriani ID; ICLR'25 similarity; invariant-subspace eigen-decomposition; abstract geometry 2026).
8. **Who uses causal interventions?** Cluster E + [geometric/representation interventions 2026](https://pith.science/paper/2606.23345) + [invariant-feature interventions](https://www.alphaxiv.org/fr/abs/2605.06458).
9. **Who studies abstraction/invariance?** Cluster D + the invariant-features line.
10. **Who studies generalization?** TACL argument-structure experiments (behavioral); AAT training-shaping (training axis); little work *links* in-depth invariance to behavioral generalization.
11. **Where are the gaps?** (a) **Depth × mechanism cell** — causal explanation of representational transitions; (b) the **invariance↔generalization link** under one controlled design; (c) **trajectory-level universality**; (d) unification of phenomenology and intervention on the same items.
12. **What's a genuinely better question?** Section V, framing 1: transition-anchored causal mechanism — with RQ1–RQ4 serving to *localize the target window* and RQ5–RQ6 to establish *what's at stake and whether it's general*.

---

## Part VII — Recommended Next Steps

**Priority reading (the five papers that most constrain your positioning):**
1. [Derivational Probing (2025)](https://arxiv.org/html/2506.21861v1) — nearest competitor for RQ1/RQ2.
2. [Tenney et al. (ACL 2019)](https://aclanthology.org/P19-1452/) — canonical "when does it emerge" methodology.
3. [Invariant Features in LMs (2026)](https://www.alphaxiv.org/fr/abs/2605.06458) — nearest competitor for RQ3 + the new intervention bar.
4. [TACL argument structure generalization](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00608/118116/How-Abstract-Is-Linguistic-Generalization-in-Large) — behavioral anchor for RQ5.
5. [Developmental interpretability review (2025)](https://pith.science/paper/2508.15841) — framing collision to write against.

**Sequencing (matches your instinct):**
1. ✅ Map territory (this document).
2. Deep-read the five above; extract their exact measures/baselines → decide what "measures" your RQ1–RQ6 must *replace* rather than reproduce.
3. Draft the positioning statement: *phenomenology scaffold (RQ1–RQ6) → transition-anchored mechanism (RQ7), with "Computational Ontogenesis" explicitly scoped to the depth axis, differentiated from developmental interpretability.*
4. Only then: finalize the dataset design (it must instantiate the surface/structure contrasts that make RQ1 non-redundant) and the preregistration.

**Naming recommendation:** Use "Computational Ontogenesis" as the *umbrella framing* with an explicit coinage sentence and a differentiation sentence against developmental interpretability — never as a claim of an established field. Re-run the term check at submission time (the Springer 2026 Simondon paper shows the phrase family is warming up).
