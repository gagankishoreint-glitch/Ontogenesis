#!/usr/bin/env python3
"""
Naturalistic protocol v3 — LLM co-annotator pre-screen.

Annotates every candidate in data/natural_candidates.csv with a PINNED model
at temperature 0 (reproducible), writes data/llm_prescreen.csv.  The human
then only reviews proposed KEEPS via:  python data/annotate.py --review-llm

Backends (first match wins):
  1. OPENAI_API_KEY      -> OpenAI chat completions (model: env OPENAI_MODEL)
  2. ANTHROPIC_API_KEY   -> Anthropic messages      (model: env ANTHROPIC_MODEL)
  3. ollama at localhost:11434 (default model qwen2.5:7b-instruct-q4_K_M,
     override with --model)   <- recommended: free, offline, reproducible
        brew install ollama && ollama serve &
        ollama pull qwen2.5:7b-instruct-q4_K_M

Model id + temperature are recorded in data/llm_prescreen.meta.json so the
paper can name the exact co-annotator.  The human remains the verifier of
every kept sentence (annotate.py --review-llm); agreement is reported by
annotate.py --recheck.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve_candidates import NON_AGENT          # means/instrument nouns

HERE = Path(__file__).resolve().parent
CANDS = HERE / "natural_candidates.csv"
OUT = HERE / "llm_prescreen.csv"
META = HERE / "llm_prescreen.meta.json"

PROMPT = """You are a linguistics annotator for a study of two-participant \
clauses. Words of one sentence are numbered 1..N below.

Decide KEEP (keep=1) and form, or REJECT (keep=0):
- cleft: "It is/was <NP focus> who|that|which <clause>"; participants = the \
focus NP and the object NP inside the relative clause; the relative clause \
must contain a transitive verb with a full-NP object and a full-NP subject.
- passive: subject is the patient; the agent appears in a "by <full NP>" \
phrase (reject if the by-phrase is not a participant: by means/aid/force/\
chance/now, or a pronoun).
- canonical: active transitive clause, subject NP + object NP.
- topical: a fronted object NP before the subject ("The prize, John salted.") \
- keep only if unambiguous.
REJECT (keep=0) when: any participant is a pronoun or a coordination; roles \
ambiguous; intransitive or copular clause; fragment, heading, quotation \
fragment, or interrupted sentence; more than one main clause with different \
participants; no two full-NP participants.
Participants must be full noun phrases. pred = the single main verb word that \
indexes the two roles (passives: the participle). n1/n2 = the two participant \
word ranges in SURFACE order. agent = 1 if n1 is the doer else 2.

WORDS: {words}

Respond with ONLY this JSON, no prose:
{{"form":"cleft|passive|canonical|topical|none","keep":0,"n1":[0,0],\
"n2":[0,0],"pred":0,"agent":1}}"""


def http_json(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={**headers,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def call_openai(model, prompt):
    h = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    r = http_json(os.environ.get("OPENAI_BASE_URL",
                                 "https://api.openai.com/v1") + "/chat/completions",
                  {"model": model, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}, h)
    return r["choices"][0]["message"]["content"], r["model"]


def call_anthropic(model, prompt):
    h = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
         "anthropic-version": "2023-06-01"}
    r = http_json("https://api.anthropic.com/v1/messages",
                  {"model": model, "max_tokens": 300, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}, h)
    return r["content"][0]["text"], r["model"]


def call_ollama(model, prompt):
    r = http_json("http://localhost:11434/api/generate",
                  {"model": model, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0}}, {})
    return r["response"], r.get("model", model)


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        return dict(form=str(d.get("form", "none")), keep=int(d.get("keep", 0)),
                    n1=[int(x) for x in d.get("n1", [0, 0])],
                    n2=[int(x) for x in d.get("n2", [0, 0])],
                    pred=int(d.get("pred", 0)), agent=int(d.get("agent", 1)))
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="politeness delay for API backends")
    args = ap.parse_args()

    if os.environ.get("OPENAI_API_KEY"):
        backend, model = "openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        call = call_openai
    elif os.environ.get("ANTHROPIC_API_KEY"):
        backend, model = "anthropic", os.environ.get(
            "ANTHROPIC_MODEL", "claude-sonnet-4-5")
        call = call_anthropic
    else:
        backend, model = "ollama", args.model
        call = call_ollama
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        except Exception:
            raise SystemExit(
                "ollama is not running.  Start it:\n"
                "  brew install ollama && ollama serve &\n"
                f"  ollama pull {model}\n"
                "or set OPENAI_API_KEY / ANTHROPIC_API_KEY for API backends.")

    cands = list(csv.DictReader(open(CANDS, newline="")))
    if args.limit:
        cands = cands[: args.limit]
    META.write_text(json.dumps(dict(backend=backend, model=model,
                                    temperature=0, n=len(cands),
                                    date=time.strftime("%Y-%m-%d")), indent=2))
    print(f"[prescreen] backend={backend} model={model} n={len(cands)}")

    rows, bad = [], 0
    for i, c in enumerate(cands, 1):
        words = json.loads(c["words_json"])
        numbered = " ".join(f"{j}:{w}" for j, w in enumerate(words, 1))
        try:
            text, used = call(model, PROMPT.format(words=numbered))
        except Exception as e:
            print(f"  ! {c['cand_id']} API error: {e}"); bad += 1
            rows.append(dict(cand_id=c["cand_id"], form="none", keep=0,
                             n1a=0, n1b=0, n2a=0, n2b=0, pred=0, agent=1,
                             raw="API_ERROR"))
            continue
        d = parse_json(text)
        n = len(words)
        ok = d and d["form"] in {"cleft", "passive", "canonical", "topical"} \
            and d["keep"] == 1 and 1 <= d["n1"][0] <= d["n1"][1] <= n \
            and 1 <= d["n2"][0] <= d["n2"][1] <= n and 1 <= d["pred"] <= n \
            and d["n1"][0] < d["n2"][0] \
            and not (d["n1"][1] >= d["n2"][0] and d["n2"][1] >= d["n1"][0])
        # passive with a means/instrument by-phrase is not a keep
        if ok and d["form"] == "passive":
            low = [w.lower().strip(",.;:") for w in words]
            bi = next((k for k in range(d["pred"], n) if low[k] == "by"), None)
            if bi is None or any(t in NON_AGENT
                                 for t in low[bi + 1:bi + 4]):
                ok = False
        if not ok:
            d = dict(form="none", keep=0, n1=[0, 0], n2=[0, 0], pred=0,
                     agent=1)
        rows.append(dict(cand_id=c["cand_id"], form=d["form"], keep=d["keep"],
                         n1a=d["n1"][0], n1b=d["n1"][1], n2a=d["n2"][0],
                         n2b=d["n2"][1], pred=d["pred"], agent=d["agent"],
                         raw=text[:200].replace("\n", " ")))
        if i % 25 == 0:
            print(f"  {i}/{len(cands)}  keeps so far="
                  f"{sum(r['keep'] for r in rows)}")
        if args.sleep:
            time.sleep(args.sleep)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    kc = Counter(r["form"] for r in rows if r["keep"])
    print(f"wrote {OUT}: {sum(r['keep'] for r in rows)} proposed keeps "
          f"({dict(kc)}), rejects={sum(not r['keep'] for r in rows)}, "
          f"errors={bad}")


if __name__ == "__main__":
    main()
