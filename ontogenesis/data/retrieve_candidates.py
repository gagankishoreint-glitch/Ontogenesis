#!/usr/bin/env python3
"""
T1 of the naturalistic validation protocol (see
/home/user/naturalistic_validation_protocol.md §3).

Retrieves candidate sentences for human annotation from attested corpora.

  python data/retrieve_candidates.py --source brown            # default
  python data/retrieve_candidates.py --source wikipedia        # escalation
  python data/retrieve_candidates.py --source all --per-form 200

Sources
  brown      : data/corpora/brown (NLTK Brown corpus, plain `word/POS` files,
               no nltk dependency — raw reader).  Download once with:
                 curl -L -o data/corpora/brown.zip \
                   https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip
                 unzip data/corpora/brown.zip -d data/corpora/
  wikipedia   : random article extracts via the MediaWiki API (stdlib only).

Filters (protocol §3): 6..22 words; >=2 noun-ish phrase hints; no quoted
speech / trace tokens (*u* etc.); construction prescreen -> form_guess
(cleft | topical | canonical; priority cleft > topical > canonical).
Predicate is NOT required here — the annotator marks it.  lexicon_overlap=1
iff the sentence contains a surface form of a TRANS_VERBS lemma from
generate_dataset.py (dative verbs excluded: 3 participants muddle roles).

Output: data/natural_candidates.csv
  cand_id, source, form_guess, lexicon_overlap, text, words_json
Deterministic: SEED 20261007.
"""
import argparse
import csv
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent            # data/
ROOT = HERE.parent                                # repo root
sys.path.insert(0, str(HERE))
from generate_dataset import TRANS_VERBS          # module has a main-guard

SEED = 20261007
OUT = ROOT / "data" / "natural_candidates.csv"
BROWN_DIR = ROOT / "data" / "corpora" / "brown"
BROWN_ZIP_URL = ("https://raw.githubusercontent.com/nltk/nltk_data/"
                 "gh-pages/packages/corpora/brown.zip")

MIN_W, MAX_W = 6, 22
FINAL_TAGS = {".", "!", "?"}
DROP_TOKENS = {"``", "''", "*u*", "*exp*", "*on*"}

DET = {"the", "a", "an", "this", "that", "these", "those", "my", "your",
       "his", "her", "its", "our", "their", "some", "any", "each", "every",
       "no", "another", "either", "neither", "both", "half", "all", "most",
       "many", "few", "several", "one", "two", "three"}
# sentence-initial openers that are NOT NP participants
OPEN_EXCL = {"it", "there", "this", "that", "what", "when", "where", "while",
             "who", "whom", "which", "how", "if", "as", "but", "and", "or",
             "nor", "so", "yet", "in", "on", "at", "for", "by", "from", "to",
             "of", "with", "after", "before", "although", "though", "because",
             "indeed", "however", "moreover", "nevertheless", "thus", "hence",
             "then", "also", "yes", "no", "oh", "well", "he", "she", "they",
             "we", "you", "i", "him", "her", "them", "us", "me", "his", "my",
             "our", "your", "their", "among", "over", "under", "between",
             "through", "during", "without", "within", "beyond", "until",
             "toward", "towards", "unlike", "upon", "across", "along",
             "around", "behind", "below", "beside", "besides", "except",
            "via", "per", "down", "up", "out", "off", "into", "onto",
             "despite", "given", "using", "considering", "including",
             "regarding", "since", "until", "unless", "whether",
             "like", "instead", "rather", "together", "apart", "along",
             "next", "just", "above", "save", "near", "far", "much",
             "whatever", "whenever", "wherever", "however"}
# sentence-initial verbs (imperatives / dialogue openers) — not NP topics
INIT_VERB = {"leave", "give", "let", "take", "make", "come", "go", "look",
             "listen", "remember", "note", "consider", "think", "say",
             "suppose", "assume", "imagine", "call", "send", "put", "keep",
             "hold", "walk", "cry", "shout", "sigh", "laugh", "smile", "nod",
             "shrug", "answer", "reply", "read", "write", "turn", "open",
             "close", "dear", "thank", "believe", "trust", "bless", "pray"}
PRE_VERB = {"is", "was", "are", "were", "be", "been", "has", "have", "had",
            "do", "does", "did", "got", "get", "gave", "give", "went", "go",
            "came", "come", "took", "take", "made", "make", "said", "say",
            "told", "tell", "thought", "knew", "know", "felt", "feel",
            "became", "become", "seemed", "seem", "looked", "look", "found",
            "find", "heard", "hear", "saw", "see", "kept", "keep", "let",
            "set", "put", "ran", "run", "fell", "fall", "began", "begin",
            "held", "hold", "stood", "stand", "sat", "sit", "lay", "lie",
            "paid", "pay", "left", "leave", "meant", "mean", "let", "bid",
            "read", "rose", "rise", "drove", "drive", "brought", "bring",
            "wrote", "write", "lost", "lose", "chose", "choose",
            "woke", "awoke", "sank", "swam"}
MODALS = {"must", "can", "will", "would", "should", "could", "may", "might",
          "shall", "need", "ought"}
# unambiguous past-tense forms — safe to require inside parentheticals
# (bare forms like set/read/put double as common nouns: "as a set")
PAST_UNAMBIG = {"was", "were", "had", "said", "told", "thought", "knew",
                "felt", "became", "seemed", "looked", "found", "heard",
                "saw", "kept", "stood", "sat", "left", "meant", "held",
                "ran", "fell", "began", "paid", "rose", "drove", "brought",
                "wrote", "lost", "chose", "got", "gave", "went", "came",
                "took", "made", "sold", "sent", "spent", "built", "bought",
                "caught", "taught", "sought", "won", "bore", "tore", "wore",
                "shook", "lay", "led", "fed", "met", "beat"}
# sentence-initial adverbials / time expressions that are not NP topics
INIT_ADV = {"even", "still", "perhaps", "maybe", "often", "sometimes",
            "usually", "generally", "finally", "first", "second", "next",
            "later", "meanwhile", "accordingly", "fortunately",
            "unfortunately", "suddenly", "quietly", "carefully", "indeed",
            "ten", "four", "five", "six", "seven", "eight", "nine",
            "eleven", "twelve", "fifteen", "twenty", "thirty", "hundred",
            "presently", "immediately"}
TIME_UNITS = {"ago", "years", "year", "months", "month", "weeks", "week",
              "days", "day", "hours", "hour", "minutes", "minute", "times",
              "morning", "afternoon", "evening", "night", "winter",
              "summer", "spring", "fall"}
# word allowed right after the topical comma?
POST_COMMA_EXCL = {"which", "who", "that", "and", "but", "or", "nor", "when",
                   "while", "if", "after", "because", "since", "though",
                   "although", "is", "was", "were", "are", "has", "had",
                   "have", "will", "would", "could", "should", "might",
                   "may", "can", "do", "does", "did", "not"}


# ---------------------------------------------------------------- lexicon
def verb_variants(v):
    s = "es" if re.search(r"(s|x|z|ch|sh|o)$", v) else "s"
    ed = v[:-1] + "ed" if v.endswith("ss") else (v + "d" if v.endswith("e") else v + "ed")
    ing = v[:-1] + "ing" if v.endswith("e") else v + "ing"
    return {v, v + s, ed, ing}


LEX_RE = re.compile(r"\b(" + "|".join(sorted(
    w for v in TRANS_VERBS for w in verb_variants(v))) + r")\b", re.I)

# ---------------------------------------------------------------- brown


def _is_final(toks):
    if not toks:
        return False
    if toks[-1][1] in FINAL_TAGS:
        return True
    # sentence-final punctuation swallowed by a closing quote
    if len(toks) >= 2 and toks[-1][1] in {"''", '"', ")", "--"} \
            and toks[-2][1] in FINAL_TAGS:
        return True
    return False


def read_brown(root=BROWN_DIR):
    """Raw Brown reader: sentence = line, with continuation-line joining
    as a safety net.  Returns list of token lists (POS stripped)."""
    if not root.is_dir():
        raise SystemExit(
            f"Brown corpus not found at {root}.\n"
            f"  mkdir -p {root.parent} && curl -L -o {root.parent}/brown.zip \\\n"
            f"    {BROWN_ZIP_URL}\n"
            f"  unzip {root.parent}/brown.zip -d {root.parent}/")
    sents, pending, nofinal = [], [], 0
    for f in sorted(root.iterdir()):
        if not f.is_file() or f.name in {"CONTENTS", "README", "INDEX"}:
            continue
        for raw in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                if pending:
                    nofinal += 1
                    sents.append([w for w, _ in pending])
                    pending = []
                continue
            toks = []
            for t in line.split():
                w, _, tag = t.rpartition("/")
                toks.append((w, tag) if tag else (t, ""))
            pending.extend(toks)
            if _is_final(toks):
                sents.append([w for w, _ in pending])
                pending = []
        if pending:                       # file ended mid-sentence
            nofinal += 1
            sents.append([w for w, _ in pending])
            pending = []
    print(f"[brown] {len(sents)} sentences parsed "
          f"({nofinal} non-final joins — expect ~0 if line=sentence holds)")
    return sents


# ------------------------------------------------------------- gutenberg
# Versioned shelves (reproducible: book = Project Gutenberg eBook id, fetched
# from https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt).
#   v1 (2026-10-07, original): 24 curated dialogue-rich novels — the shelf
#        the committed natural_candidates.csv was retrieved with.
#   v2 (2026-10-09, Phase-1 cleft retrieval repair): v1 + 8 explicitly listed
#        additions (dialogue-dense classics with high it-cleft density).
#        Added as a bounded, documented escalation per protocol §3; no
#        candidate counts are claimed for v2 — dry runs report actuals.
GUTENBERG_SHELF_V1 = [
    1342, 1260, 98, 84, 76, 74, 345, 1661, 11, 55, 36, 1483,
    730, 16, 45, 768, 1458, 531, 35, 422,   # Austen, Bronte x2,
    # Dickens x3, Shelley, Twain x2, Stoker, Carroll, Baum, Wells,
    # Doyle, Anne, Wuthering Heights, Middlemarch, Tess, Time
    # Machine, Kim
    46, 105, 158, 2701]           # Christmas Carol, Silas Marner,
                                  # Emma, Moby Dick
GUTENBERG_SHELF_V2 = GUTENBERG_SHELF_V1 + [
    100,    # Shakespeare, Complete Works (dialogue-dense; highest cleft density)
    43,     # Stevenson, The Strange Case of Dr. Jekyll and Mr. Hyde
    1023,   # Dickens, Bleak House
    1184,   # Dumas, The Count of Monte Cristo
    1400,   # Dickens, Great Expectations
    2542,   # Ibsen, A Doll's House (dialogue-dense drama)
    2600,   # Tolstoy, War and Peace
    5200,   # Kafka, The Metamorphosis
]
GUTENBERG_SHELVES = {"v1": GUTENBERG_SHELF_V1, "v2": GUTENBERG_SHELF_V2}


def gutenberg_sentences(shelf=None):
    """Curated Gutenberg shelf: narrative fiction with dialogue — where
    it-clefts and NP-fronting actually occur in attested writing.
    shelf=None -> GUTENBERG_SHELF_V1 (unchanged default behaviour)."""
    import time
    if shelf is None:
        shelf = GUTENBERG_SHELF_V1
    sents = []
    for gid in shelf:
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers={
                        "User-Agent": "ontogenesis-naturalistic/1.0"}),
                    timeout=60) as r:
                raw = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"[gutenberg] skip {gid}: {e}")
            continue
        m0 = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG"
                       r"[^*]*\*\*\*", raw)
        m1 = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG"
                       r"[^*]*\*\*\*", raw)
        body = raw[m0.end(): m1.start()] if m0 and m1 else raw
        body = re.sub(r"\[[^\]]{0,80}\]", " ", body)   # editor notes
        body = body.replace("_", "")                   # pg emphasis markup
        n0 = len(sents)
        for para in re.split(r"\n\s*\n", body):
            para = " ".join(para.split())
            if not para:
                continue
            for s in re.split(r"(?<=[.!?][\"'])\s+(?=[A-Z])|"
                              r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", para):
                w = s.split()
                if MIN_W <= len(w) <= MAX_W + 6:
                    sents.append(w)
        print(f"[gutenberg] {gid}: +{len(sents) - n0} sentences")
        time.sleep(1.0)                     # Gutenberg politeness: 1 req/s
    return sents


def wiki_sentences(n_pages=40):
    """Random Wikipedia article extracts -> sentences (stdlib only)."""
    ua = {"User-Agent": "ontogenesis-naturalistic/1.0 (research protocol)"}
    def api(params):
        import urllib.parse
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=ua)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    rnd = api({"action": "query", "list": "random", "rnnamespace": 0,
               "rnlimit": n_pages, "format": "json"})
    titles = [p["title"] for p in rnd["query"]["random"]]
    texts = []
    import time
    for t in titles:    # batched extracts come back empty for all but the
        try:            # first title — fetch one at a time, politely
            q = api({"action": "query", "prop": "extracts", "explaintext": 1,
                     "titles": t, "format": "json", "redirects": 1})
            for p in q.get("query", {}).get("pages", {}).values():
                if len(p.get("extract", "")) >= 3000:   # skip stubs
                    texts.append(p["extract"])
            time.sleep(0.8)
        except Exception as e:      # transient API error: skip the page
            print(f"[wikipedia] skip {t!r}: {e}")
            time.sleep(3.0)
    sents = []
    for t in texts:
        for line in t.splitlines():
            if line.startswith("=") or line.startswith("*") or line.startswith("["):
                continue                    # section headers, bullets, refs
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", line):
                s = s.replace("[note 1]", "").replace("[citation needed]", "")
                w = s.split()
                if MIN_W <= len(w) <= MAX_W + 8:   # cheap pre-trim
                    sents.append(w)
    print(f"[wikipedia] {n_pages} pages -> {len(sents)} raw sentence splits")
    return sents


# --------------------------------------------------------------- filters
def np_score(words):
    """Cheap >=2 noun-phrase hint count (no POS tagger; enrichment only)."""
    s = 0
    low = [w.lower().strip(",.;:") for w in words]
    for i in range(len(words) - 1):
        if low[i] in DET and words[i + 1].isalpha() and len(words[i + 1]) > 1:
            s += 1
    for i in range(1, len(words)):         # mid-sentence proper nouns
        w = words[i].strip("\"'([]")
        if w[:1].isupper() and w.isalpha() and len(w) > 1:
            s += 1
    w0 = words[0].strip("\"'([]")
    if w0[:1].isupper() and w0.isalpha() and w0.lower() not in OPEN_EXCL:
        s += 1
    return s


def topical_guess(words):
    """Fronted-NP topic, tokenization-agnostic (comma standalone as in Brown
    or attached as in Gutenberg).  Two accepted patterns:
    A)  NP , SUBJECT VERB ...            (single comma)
    B)  NP , parenthetical(<=5 words w/
        finite verb) , VERB ...          (interrupter, both commas early)
    Rejects: clause-initial openers (As/When/In...), imperatives, verbs in
    the fronted-NP segment, appositive subject-NPs, participle modifiers,
    time expressions."""
    low = [w.lower().strip(",.;:!?\"'“”‘’([]") for w in words]
    commas = [k for k in range(min(15, len(words))) if words[k].endswith(",")]

    def verbish(w):
        return (w in PRE_VERB or w in MODALS or
                (len(w) > 3 and (w.endswith("ed") or w.endswith("ing"))))

    def subjectish(w):
        return (w[:1].isupper() and w[:1].isalpha()) or w.lower() in DET or \
            w.lower() in {"my", "your", "his", "her", "its", "our", "their",
                          "he", "she", "they", "we", "you", "i"}

    ADJUNCT = {"never", "always", "often", "not", "again", "soon", "quite",
               "rather", "very", "so", "too", "only", "just", "still",
               "yet", "also", "perhaps", "maybe", "yesterday", "today",
               "tomorrow", "then", "now", "once", "twice", "ever", "here",
               "there", "home", "away", "back", "out", "up", "down",
               "off", "it", "well", "enough"}   # particles/resumptives = gap-fillers

    def object_gap(seq, vi):
        """True topicalization of an object leaves a GAP: after the finite
        verb, at most one non-adjunct word (adverb / resumptive 'it' /
        second object of a ditransitive)."""
        tail = [w for w in seq[vi + 1:]
                if w not in ADJUNCT and w not in {".", "!", "?"}]
        return len(tail) <= 1

    if not commas:
        return False                # comma-marked topicalization only
    j = commas[0]
    pre = [w for w in low[:j + 1] if w]      # words up to & incl. comma token
    if not (2 <= len(pre) <= 9):
        return False
    if low[0] in OPEN_EXCL or low[0] in INIT_VERB or low[0] in INIT_ADV \
            or low[0].endswith("ly") or \
            low[0] in {"have", "do", "does", "did", "can", "could", "will",
                       "would", "shall", "should", "may", "might", "is",
                       "are", "was", "were", "has", "had"}:
        return False                         # adverbials & questions
    if len(low) > 1 and (low[1] in OPEN_EXCL or low[1] in INIT_ADV):
        return False                         # "Halfway across ... ,"
    if any(w in {"why", "how", "when", "where", "what", "who", "whom",
                 "which", "whether"} for w in pre):
        return False                         # exclamative/interrogative fronting
    if low[0] in {"whatever", "whenever", "wherever", "yesterday", "today",
                  "tomorrow", "tonight", "here", "now"}:
        return False
    for w in pre:                            # no finite/modal verb in topic
        if w in PRE_VERB or w in MODALS or w in TIME_UNITS or \
                (len(w) > 3 and (w.endswith("ed") or w.endswith("ing"))):
            return False
    w0 = words[0].strip("\"'([]“”‘’")
    if not (w0.lower() in DET or
            w0.lower() in {"my", "your", "his", "her", "its", "our", "their"} or
            (w0[:1].isupper() and w0.isalpha())):
        return False

    if len(commas) == 1:                    # pattern A
        seq, raw = [], []
        for w in words[j + 1:j + 6]:
            lw = w.lower().strip(",.;:")
            if not lw:
                continue
            seq.append(lw)
            raw.append(w.strip("\"'([]“”‘’"))
            if len(seq) == 5:
                break
        if not seq or seq[0] in POST_COMMA_EXCL or not subjectish(raw[0]):
            return False
        for vi in (1, 2, 3):
            if vi < len(seq) and verbish(seq[vi]):
                return object_gap(seq, vi)
        return False
    # pattern B: interrupter parenthetical with a FINITE verb (not participle)
    j2 = commas[1]
    mid = [w for w in low[j + 1:j2 + 1] if w]  # incl. comma-2 token's verb
    if not (1 <= len(mid) <= 5):
        return False
    if any(w in {"which", "who", "that", "when", "where", "while", "if",
                 "though", "although", "because", "and", "but", "or", "nor",
                 "unless", "whether", "since"} for w in mid):
        return False                         # dependent-clause parenthetical
    m0 = words[j + 1].strip("\"'([]“”‘’") if j + 1 < len(words) else ""
    if not ((m0[:1].isupper() and m0[:1].isalpha()) or mid[0] in
            {"he", "she", "they", "we", "you", "i", "the", "a", "an",
             "this", "that"}):
        return False                         # parenthetical starts with subject
    finite = [w for w in mid if (w in PAST_UNAMBIG or w in MODALS)]
    if not finite or any(len(w) > 4 and w.endswith(("ed", "ing"))
                         for w in mid if w not in finite):
        return False                        # kill ", advanced notably by X ,"
    seq = [w for w in low[j2 + 1:j2 + 6] if w][:5]
    if not seq:
        return False
    for vi in range(min(4, len(seq))):
        if verbish(seq[vi]):
            return object_gap(seq, vi)
    return False


# it-cleft: focus must be an NP (det-led or Capitalized), not an extraposed
# clausal that-clause ("it was learned/natural/clear that ...") and not a
# pronoun focus ("it was he who ...") — both unusable for participant spans.
CLEFT_RE = re.compile(
    r"\b(?:It|it)\s+(?:is|was|were|has\s+been|had\s+been)\s+"
    r"(?!(?:only|also|always|never|not|just|about|then|there|so|rather|"
    r"more|less|much|very|already|still|even|perhaps|probably|possibly|"
    r"indeed|clear|obvious|natural|likely|possible|said|reported|found|"
    r"learned|announced|argued|claimed|believed|expected|assumed|true|"
    r"false|apparent|evident|strange|funny|interesting)\b)"
    r"(?:the|a|an|this|that|these|those|my|your|his|her|our|their|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})"
    r"\b[^.?!]{0,60}\b(?:who|which|that)\b")


PASSIVE_RE = re.compile(
    r"\b(?:was|were|is|are|been|being|was\s+being|were\s+being|has\s+been|"
    r"have\s+been|had\s+been)\s+"
    r"[a-z]{3,}(?:ed|en|wn|lt|ken|sed|zed|t)\b[^.?!]{0,50}\bby\b", re.I)


NON_AGENT = {"means", "mean", "aid", "way", "ways", "virtue", "chance",
             "accident", "mistake", "force", "law", "nature", "definition",
             "contrast", "comparison", "analogy", "inference", "proxy",
             "association", "extension", "courtesy", "necessity", "design",
             "hand", "hands", "now", "then", "far", "which", "whom",
             "whose", "him", "her", "them", "us", "me", "you", "it",
             "himself", "herself", "themselves", "myself", "itself",
             "one", "ones", "others", "another", "some", "any", "all",
             "both", "each", "either", "neither", "nothing", "anything",
             "something",
             # instruments / methods (not doers)
             "integration", "technique", "techniques", "method", "methods",
             "analysis", "calculation", "measurement", "procedure",
             "procedures", "process", "processes", "algorithm", "algorithms",
             "microscope", "amplifier", "treatment", "surgery", "saline",
             "albumin", "blows", "blow", "weapon", "weapons", "tool",
             "tools", "instrument", "instruments", "machine", "machines",
             "device", "devices", "test", "tests", "assay", "assays",
             "experiment", "experiments", "observation", "observations",
             "study", "studies", "survey", "surveys", "interview",
             "interviews", "drowning", "fire", "wind", "gravity", "heat",
             "cold", "pressure", "friction", "impact", "explosion",
             "collapse", "fall", "flood", "storm", "earthquake", "disease",
             "illness", "infection", "virus", "cancer", "age", "hunger",
             "thirst", "fatigue", "fear", "anger", "grief", "love", "hatred",
             "time", "war", "peace", "poverty", "famine", "drought"}
INTRANS = {"replied", "answered", "said", "spoke", "talked", "went", "came",
           "stood", "sat", "lay", "died", "lived", "arrived", "appeared",
           "vanished", "slept", "wept", "laughed", "smiled", "nodded",
           "hesitated", "paused", "waited", "listened", "cried", "shouted",
           "fell", "rose", "grew", "remained", "stayed", "existed"}
REL_PRON = {"he", "she", "they", "we", "you", "i", "him", "her", "them",
            "us", "me", "his", "hers", "theirs", "its", "their", "my",
            "your", "our", "whose", "whoever", "whatever", "to", "for",
            "in", "on", "at", "by", "with", "from", "of", "unless",
            "because", "when", "where", "while", "if", "though",
            "won't", "wouldn't", "couldn't", "can't", "don't", "doesn't",
            "didn't", "not", "never"}
INTRANS = INTRANS | {"talk", "talked", "talking", "won", "win", "work",
                     "worked", "run", "ran", "fight", "fought", "argue",
                     "argued", "agree", "agreed", "object", "objected"}


def passive_ok(words):
    """Agentive single-clause passive with full-NP subject and by-agent."""
    text = " ".join(words)
    if ", and " in text or " ; " in text or ": " in text:
        return False
    low = [w.lower().strip(",.;:") for w in words]
    if low[0] in {"it", "there", "these", "those", "this", "that", "they",
                  "he", "she", "we", "you", "i"}:
        return False
    bi = text.lower().rfind(" by ")          # the by-phrase
    if bi < 0:
        return False
    after = [w.lower().strip(",.;:") for w in text[bi + 4:].split()[:3]]
    return not any(w in NON_AGENT for w in after)   # skip dets/adjs: 3-word window


def cleft_ok(words):
    """Cleft whose relative clause has a full-NP subject and transitive verb."""
    low = [w.lower().strip(",.;:") for w in words]
    for r in range(2, len(low)):
        if low[r] in {"who", "that", "which"}:
            clause = low[r + 1:r + 7]
            if not clause:
                return False
            if clause[0] in REL_PRON:
                return False                # pronoun participant
            if clause[0] in INTRANS or (len(clause) > 1 and clause[1] in INTRANS):
                return False                # intransitive -> one participant
            return True
    return False


def classify(words):
    text = " ".join(words)
    if CLEFT_RE.search(text) and cleft_ok(words):
        return "cleft"
    if PASSIVE_RE.search(text) and passive_ok(words):
        return "passive"
    if topical_guess(words):
        return "topical"
    return "canonical"


def norm_key(words):
    """Documented normalized-text comparison used for ALL candidate dedup
    (retrieve_candidates.py and retrieve_clefts.py): the token list joined
    with single spaces and lowercased.  Two sentences are duplicates iff
    their norm_keys are equal.  Deterministic; no punctuation stripping —
    punctuation tokens are part of the key, matching the committed pool."""
    return " ".join(words).lower()


def clean(words):
    if not (MIN_W <= len(words) <= MAX_W):
        return None
    if any(w in DROP_TOKENS or w.startswith("*") for w in words):
        return None                      # quoted speech / traces / exp
    if any("/" in w for w in words):
        return None
    # sentence-likeness: must END in terminal punctuation (kills fragments,
    # chapter headings, list items)
    tail = words[-1].strip("\"'”’)]")
    if not tail or tail[-1] not in ".!?":
        return None
    # must START like a sentence (kills mid-sentence split artifacts)
    w0 = words[0].strip("\"'“‘([]")
    if not w0[:1].isupper():
        return None
    # ALL-CAPS headings
    first3 = [w.strip("\"'“‘([,") for w in words[:3]]
    if all(w.isupper() and w.isalpha() and len(w) > 1 for w in first3):
        return None
    if np_score(words) < 2:
        return None
    return words


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                    choices=["brown", "gutenberg", "wikipedia", "all"],
                    default="brown",
                    help="all = brown + gutenberg (wikipedia only on explicit request)")
    ap.add_argument("--per-form", type=int, default=200,
                    help="candidate quota per form_guess (cleft/canonical)")
    ap.add_argument("--topical-per-form", type=int, default=300,
                    help="quota for topical (lower-precision prescreen -> higher oversample)")
    ap.add_argument("--wiki-pages", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(SEED)
    pool = []                              # (words, source)
    if args.source in ("brown", "all"):
        for w in read_brown():
            c = clean(w)
            if c:
                pool.append((c, "brown"))
    if args.source in ("gutenberg", "all"):
        for w in gutenberg_sentences():
            c = clean(w)
            if c:
                pool.append((c, "gutenberg"))
    if args.source == "wikipedia":
        for w in wiki_sentences(args.wiki_pages):
            c = clean(w)
            if c:
                pool.append((c, "wikipedia"))

    seen, uniq = set(), []
    for w, src in pool:
        key = norm_key(w)
        if key not in seen:
            seen.add(key)
            uniq.append((w, src))
    print(f"[filter] {len(pool)} -> {len(uniq)} unique candidates past filters")

    buckets = {"cleft": [], "passive": [], "topical": [], "canonical": []}
    for w, src in uniq:
        buckets[classify(w)].append((w, src))
    print("[counts by form_guess] "
          + "  ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    picks = []
    order = ["passive", "canonical", "cleft", "topical"]   # precision-ordered
    for form in order:
        items = buckets[form]
        rng.shuffle(items)
        q = args.topical_per_form if form == "topical" else args.per_form
        picks.extend((form, w, src) for w, src in items[:q])

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cand_id", "source", "form_guess", "lexicon_overlap",
                     "text", "words_json"])
        for i, (form, w, src) in enumerate(picks, 1):
            wr.writerow([f"c{i:04d}", src, form,
                         int(bool(LEX_RE.search(" ".join(w)))),
                         " ".join(w), json.dumps(w)])
    print(f"wrote {args.out}  ({len(picks)} candidates)")

    n_cleft_avail = len(buckets["cleft"])
    if n_cleft_avail < 120 and args.source in ("brown", "all"):
        print(f"WARNING: only {n_cleft_avail} cleft candidates (<120) — "
              "protocol §3 escalation: rerun with --source wikipedia (or all).")


if __name__ == "__main__":
    main()
