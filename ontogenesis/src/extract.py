#!/usr/bin/env python3
"""
Extract span-level hidden states for every item, every layer.

Usage:
  python extract.py --model bert_base
  python extract.py --model all          # full panel

Saves features/{model}.npz with:
  X : float16 [n_items, n_layers, 4, hidden]
      slots: 0=noun1 (surface order), 1=noun2, 2=predicate, 3=last non-pad token
  ids: item ids (row order matches items.csv)
  layer_indices: [0..L]  (layer 0 = embeddings output)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (EXTRACT_BATCH, FEATURES, ITEMS_CSV, MAX_LEN, MODELS,
                    device, ensure_dirs)


def load_items():
    with open(ITEMS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def span_to_token(offsets, start, end):
    """First subword token overlapping char span [start,end)."""
    if start < 0:
        return None
    for t, (s, e) in enumerate(offsets):
        if e > s and s < end and e > start:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in MODELS or 'all'")
    args = ap.parse_args()
    ensure_dirs()

    keys = list(MODELS) if args.model == "all" else [args.model]
    items = load_items()
    texts = [it["text"] for it in items]
    dev = device()
    print(f"device={dev} | items={len(items)} | models={keys}")

    for key in keys:
        if key not in MODELS:
            raise SystemExit(f"unknown model key {key}; choose from {list(MODELS)}")
        name = MODELS[key]
        t0 = time.time()
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(name)
        if key == "bert_random":
            from transformers import BertConfig
            model = AutoModel.from_config(BertConfig.from_pretrained(name))
            print("[bert_random] weights randomized — pretraining removed")
        model.eval().to(dev)
        n_layers = model.config.num_hidden_layers + 1  # + embeddings
        hidden = model.config.hidden_size

        out = np.zeros((len(items), n_layers, 4, hidden), dtype=np.float16)
        bad = 0

        with torch.no_grad():
            for i in range(0, len(items), EXTRACT_BATCH):
                batch_items = items[i:i + EXTRACT_BATCH]
                enc = tok([b["text"] for b in batch_items],
                          return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LEN,
                          return_offsets_mapping=True)
                offsets = enc.pop("offset_mapping").tolist()
                enc = {k: v.to(dev) for k, v in enc.items()}
                hs = model(**enc, output_hidden_states=True).hidden_states
                # hs: (n_layers,) each [B, T, H]
                H = torch.stack(hs, dim=1)  # [B, L, T, H]
                for j, b in enumerate(batch_items):
                    offs = offsets[j]
                    i1 = span_to_token(offs, int(b["noun1_span_start"]), int(b["noun1_span_end"]))
                    i2 = span_to_token(offs, int(b["noun2_span_start"]), int(b["noun2_span_end"]))
                    ip = span_to_token(offs, int(b["predicate_span_start"]), int(b["predicate_span_end"]))
                    last = int(enc["attention_mask"][j].sum().item()) - 1
                    slots = [i1, i2, ip, last]
                    if any(s is None for s in slots[:3]):
                        bad += 1
                    for s_idx, s in enumerate(slots):
                        if s is None:
                            continue
                        out[i + j, :, s_idx, :] = H[j, :, s, :].float().cpu().numpy()
                if (i // EXTRACT_BATCH) % 25 == 0:
                    print(f"  [{key}] {i + len(batch_items)}/{len(items)} "
                          f"({time.time() - t0:.0f}s)", flush=True)

        # sanity: layers where mapping failed are zero -> flag rows
        failed_ids = [items[r]["id"] for r in range(len(items))
                      if not out[r].any()]
        np.savez_compressed(FEATURES / f"{key}.npz",
                            X=out,
                            ids=np.array([it["id"] for it in items]),
                            model=name,
                            failed=np.array(failed_ids))
        del model, hs, H
        torch.cuda.empty_cache() if dev == "cuda" else None
        print(f"[{key}] done in {time.time() - t0:.0f}s -> features/{key}.npz "
              f"| shape={out.shape} | span-mapping failures: {len(failed_ids)} (examples: {failed_ids[:5]})")


if __name__ == "__main__":
    main()
