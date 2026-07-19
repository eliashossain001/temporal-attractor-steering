#!/usr/bin/env python3
"""Bucket 3: deterministic alias-aware matching for free-generation evaluation.

Fetches English labels + aliases (and follows redirects) for the gold Wikidata
entities from the Wikidata API, caches them to
results/free_generation/alias_mapping.json, and matches a generated string to a
gold entity under deterministic normalization (Unicode NFKC, case, punctuation,
whitespace, diacritics). Used by evaluate_free_generation.py; also runnable
standalone to (re)build the alias cache and to self-test the matcher.

Usage:
  python scripts/evaluate_alias_matching.py --build            # build cache for benchmark gold entities
  python scripts/evaluate_alias_matching.py --selftest         # run matcher unit checks
"""
from __future__ import annotations
import argparse, json, re, sys, time, unicodedata, urllib.parse, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "large" / "combined_all.jsonl"
OUT = REPO / "results" / "free_generation"
CACHE = OUT / "alias_mapping.json"
API = "https://www.wikidata.org/w/api.php"
UA = "tas-benchmark-audit/1.0 (research)"


def normalize(s: str) -> str:
    """Deterministic normalization: NFKC, lowercase, strip diacritics, drop
    punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")   # strip diacritics
    s = re.sub(r"[^\w\s]", " ", s)                     # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_aliases(qids, sleep=0.1):
    """Return {qid: {"label":str, "aliases":[...]}} via wbgetentities (batched 50)."""
    out = {}
    qids = list(dict.fromkeys(qids))
    for i in range(0, len(qids), 50):
        batch = qids[i:i + 50]
        params = {"action": "wbgetentities", "ids": "|".join(batch),
                  "props": "labels|aliases", "languages": "en", "format": "json"}
        url = API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        for attempt in range(3):
            try:
                d = json.loads(urllib.request.urlopen(req, timeout=40).read())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  WARN batch {i}: {type(e).__name__}", file=sys.stderr)
                    d = {"entities": {}}
                time.sleep(1.0)
        for qid, ent in d.get("entities", {}).items():
            # redirects: wbgetentities resolves them; the returned id is canonical
            lbl = (ent.get("labels", {}).get("en", {}) or {}).get("value", "")
            als = [a["value"] for a in ent.get("aliases", {}).get("en", [])]
            out[qid] = {"label": lbl, "aliases": als}
        time.sleep(sleep)
        print(f"  fetched {min(i+50,len(qids))}/{len(qids)} entities", file=sys.stderr)
    return out


def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def norm_alias_set(entry):
    """All normalized surface forms for a gold entity."""
    forms = [entry.get("label", "")] + entry.get("aliases", [])
    return {normalize(f) for f in forms if f and normalize(f)}


def match(generated: str, gold_qid: str, alias_map: dict):
    """Return (status, matched_alias): status in {exact, alias, none}.
    exact: normalized gold label == normalized generation (or generation is exactly a form).
    alias: a normalized alias appears as a whole-token span in the generation."""
    entry = alias_map.get(gold_qid)
    if not entry:
        return ("none", None)
    g = normalize(generated)
    if not g:
        return ("none", None)
    label_n = normalize(entry.get("label", ""))
    forms = norm_alias_set(entry)
    # exact: generation equals the label (or any single form)
    if g == label_n or g in forms:
        return ("exact", entry.get("label"))
    # alias: any form appears as a contiguous whole-word span in the generation
    gtok = " " + g + " "
    for f in sorted(forms, key=len, reverse=True):
        if f and (" " + f + " ") in gtok:
            return ("alias", f)
    return ("none", None)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(BENCH)]
    qids = set()
    for r in recs:
        qids.add(r["a_old_qid"]); qids.add(r["a_new_qid"])
    print(f"building alias map for {len(qids)} gold entities...", file=sys.stderr)
    amap = fetch_aliases(sorted(qids))
    CACHE.write_text(json.dumps(amap, indent=0))
    print(f"wrote {CACHE} ({len(amap)} entities)")


def build_for(qids):
    """Fetch+cache aliases for a specific qid set (used by the generation eval)."""
    OUT.mkdir(parents=True, exist_ok=True)
    amap = load_cache()
    missing = [q for q in dict.fromkeys(qids) if q not in amap]
    if missing:
        amap.update(fetch_aliases(missing))
        CACHE.write_text(json.dumps(amap, indent=0))
    return amap


def selftest():
    # A tiny fixture, no network: Joe Biden aliases.
    amap = {"Q6279": {"label": "Joe Biden",
                      "aliases": ["Joseph R. Biden", "Joseph Robinette Biden Jr.",
                                  "Joseph Biden", "Biden"]},
            "Q76": {"label": "Barack Obama", "aliases": ["Barack Hussein Obama II", "Obama"]}}
    cases = [
        ("Joe Biden", "Q6279", "exact"),
        ("joseph robinette biden jr.", "Q6279", "exact"),   # equals an alias form
        ("The answer is Joseph R. Biden.", "Q6279", "alias"),  # form is a substring
        ("Joséph  Biden", "Q6279", "exact"),      # diacritic + spacing -> equals "joseph biden"
        ("Barack Obama", "Q6279", "none"),        # different entity
        ("", "Q6279", "none"),
        ("Barack Obama", "Q76", "exact"),
    ]
    ok = True
    for gen, qid, exp in cases:
        st, _ = match(gen, qid, amap)
        flag = "ok" if st == exp else "FAIL"
        if st != exp:
            ok = False
        print(f"[{flag}] match({gen!r}, {qid}) = {st} (expected {exp})")
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.build:
        build()
    else:
        ap.print_help()
