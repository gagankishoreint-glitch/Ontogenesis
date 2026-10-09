"""Central config: paths, device, model panel."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ITEMS_CSV = DATA / "items.csv"

MAX_LEN = 64
EXTRACT_BATCH = 4  # conservative for 16 GB machines; raise to 16 on 32 GB+

# Model panel (blueprint D2). 'bert_base' is the G1 pilot.
# 'bert_random' = same architecture + tokenizer, FULLY random weights
# (control: if l* appears without pretraining, our phenomenon is an artifact).
MODELS = {
    "bert_base": "bert-base-uncased",
    "roberta_base": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "gpt2": "gpt2",
    "pythia_410m": "EleutherAI/pythia-410m",
    "bert_random": "bert-base-uncased",  # special-cased in extract.py
    # P1 extras: more decoders for the final-layer-collapse claim (F3)
    "gpt2_medium": "gpt2-medium",
    "opt_125m": "facebook/opt-125m",
    "pythia_160m": "EleutherAI/pythia-160m",
}

# Analysis forms for the TRANS structure (S_l family grouping)
TRANS_FORMS_S = ["active", "passive", "cleft", "topical"]
# Probe train forms / in-form test / cross-form test (T_l)
PROBE_TRAIN_FORMS = ["active", "passive"]
PROBE_CROSS_FORMS = ["cleft", "topical"]

SPLIT_TRAIN, SPLIT_TEST = "P1", "P2"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dirs():
    for d in (DATA, FEATURES, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
