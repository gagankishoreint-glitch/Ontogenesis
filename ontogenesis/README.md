# Computational Ontogenesis — Empirical Pipeline

Implements `paper_blueprint.md`: the depth-wise invariance transition (S_l / T_l / l*),
geometry, preservation, behavioral link (RQ5), and profile universality (RQ6).

## Quickstart

```bash
pip install -r requirements.txt

# 1) build the instrument (data/items.csv)
python data/generate_dataset.py

# 2) extract hidden states (start with the G1 pilot)
python src/extract.py --model bert_base

# 3) invariance curves + go/no-go summary
python src/invariance.py --model bert_base
python src/g1_curves.py  --model bert_base   # -> figures/g1_bert_base.png

# full panel later:
python src/extract.py --model all
```

Device is auto-detected in `src/config.py`:
- **RTX 3050 (4 GB):** CUDA; keep `EXTRACT_BATCH = 4`.
- **M5 Pro Mac 16 GB:** MPS; batch 4–8 is safe.
- Raise `EXTRACT_BATCH` to 16 on machines with ≥32 GB RAM.

## What G1 means (go/no-go gate)

- **GO** if: `S_l` declines with depth **and** cross-form probe accuracy rises
  significantly above chance (BH-FDR < 0.05) — i.e. `go_no_go: true` in
  `results/{model}_summary.json`.
- **NO-GO / fix instrument** if curves are flat: check span mapping
  (extract.py prints failures), filler splits, or form grammar first.

## Pipeline status

- [x] Dataset generator (TRANS 4 forms + swap, DAT alternation, ambiguity controls)
- [x] Extraction (span-level, all layers, fp16)
- [x] S_l / T_l / l* + G1 figure
- [ ] Full model panel (roberta, distilbert, gpt2, pythia + random-init BERT control)
- [ ] Geometry module (RQ3): ID, neighborhood overlap, CKA
- [ ] Preservation module (RQ4): retention/leakage + relational correlation
- [ ] Behavioral assay (RQ5): active-only fine-tune → cross-form eval
- [ ] Universality module (RQ6): normalized-depth profile alignment
