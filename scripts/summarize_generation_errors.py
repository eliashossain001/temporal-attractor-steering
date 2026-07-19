#!/usr/bin/env python3
"""Bucket 3: score free-generation outputs into behavioural categories + disagreement.

Categories per generation (5, mutually exclusive):
  new         -> alias-matches the newer answer a_new
  old         -> alias-matches the outdated answer a_old
  other_entity-> names some other person/entity (not old/new)
  definitional-> role/relational continuation with no entity answer (base-model verbosity)
  empty       -> no text / truncated to nothing

Reports per (model, method): rate of each category, alias-resolved rate
(new+old+other), and the free-generation NEW-ANSWER rate conditional on producing a
resolvable entity. Also builds the candidate-scoring-vs-generation disagreement
analysis (per-record for standard & temporal-cue via phase1 candidate scores;
aggregate for TAS/ITI). Emits a stratified manual-audit sample.

Outputs (results/free_generation/): metrics.csv/.json, category_rates.csv,
error_breakdown.csv, disagreement.csv, comparison_prob_vs_free.csv,
manual_review.csv.

Usage:  python scripts/summarize_generation_errors.py
"""
from __future__ import annotations
import csv, json, collections, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_alias_matching as A

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "free_generation"
PHASE1 = REPO / "results" / "phase1"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
METHODS = ["standard", "date_prefix", "instruction", "tas", "iti", "gated"]

# Aggregate candidate-scoring held-out Recovery (= my eval set) from the paper.
PROB_RECOVERY = {
    "qwen-2.5-1.5b": {"standard": 0.0, "date_prefix": 0.614, "instruction": 0.416, "tas": 0.469, "iti": 0.406},
    "qwen-2.5-7b":   {"standard": 0.0, "date_prefix": 0.782, "instruction": 0.278, "tas": 0.538, "iti": 0.538},
    "mistral-7b-v0.3":{"standard": 0.0, "date_prefix": 0.794, "instruction": 0.154, "tas": 0.324, "iti": 0.380},
    "llama-3.1-8b":  {"standard": 0.0, "date_prefix": 0.805, "instruction": 0.193, "tas": 0.356, "iti": 0.287},
}
ROLE_WORDS = {"head", "government", "state", "president", "prime", "minister", "chancellor",
              "governor", "chairperson", "chairman", "chair", "ceo", "chief", "executive",
              "officer", "coach", "republic", "federal", "the", "of", "is", "was", "are",
              "a", "an", "and", "in", "mr", "mrs", "dr", "since", "current", "as", "who",
              "leader", "director", "manager", "board", "company", "country", "city", "state"}


def name_spans(text):
    """Title-Case runs of length>=1 that look like proper nouns (candidate names)."""
    spans, cur = [], []
    for tok in re.findall(r"[A-Za-zÀ-ÿ.'-]+", text):
        if tok[:1].isupper() and tok.lower().strip(".'-") not in ROLE_WORDS:
            cur.append(tok)
        else:
            if len(cur) >= 1:
                spans.append(" ".join(cur))
            cur = []
    if cur:
        spans.append(" ".join(cur))
    # keep spans with >=2 tokens (likely a full name) or a single non-role capitalized token
    return [s for s in spans if len(s.split()) >= 2]


def _label_hit(gen_norm, label):
    """Benchmark gold label as a whole-word span in the (normalized) generation."""
    ln = A.normalize(label or "")
    return bool(ln) and (" " + ln + " ") in (" " + gen_norm + " ")


def classify(gen, new_qid, old_qid, subject_label, amap, new_label="", old_label=""):
    if not gen or not A.normalize(gen):
        return "empty"
    gnorm = A.normalize(gen)
    # match Wikidata label/aliases OR the benchmark's stored gold label (handles
    # entities whose current Wikidata English label is empty / has drifted).
    if A.match(gen, new_qid, amap)[0] != "none" or _label_hit(gnorm, new_label):
        return "new"
    if A.match(gen, old_qid, amap)[0] != "none" or _label_hit(gnorm, old_label):
        return "old"
    subj_n = A.normalize(subject_label or "")
    spans = [s for s in name_spans(gen) if A.normalize(s) != subj_n and A.normalize(s) not in subj_n]
    return "other_entity" if spans else "definitional"


def load_phase1_candidate(model):
    """record_id-ish -> {standard_pref, temporal_pref} from phase1 candidate scores."""
    p = PHASE1 / model / "per_instance.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        s = r.get("scores", {})
        def pref(block):
            if not block:
                return None
            return "new" if block["a_new"]["mean_logprob"] > block["a_old"]["mean_logprob"] else "old"
        out[r.get("instance_id")] = {"standard": pref(s.get("standard")),
                                     "temporal": pref(s.get("temporal"))}
    return out


def main():
    rows = []
    for m in MODELS:
        p = OUT / f"generated_outputs_{m}.csv"
        if p.exists():
            rows += list(csv.DictReader(open(p)))
    if not rows:
        print("no outputs"); return
    qids = set()
    for r in rows:
        qids.add(r["a_new_qid"]); qids.add(r["a_old_qid"])
    amap = A.build_for(sorted(qids))
    # subject label per record from benchmark (for definitional detection)
    subj = {}
    for line in open(REPO / "data/large/combined_all.jsonl"):
        b = json.loads(line)
        rid = f'{b["relation_pid"]}:{b["subject_qid"]}:{b["a_old_qid"]}->{b["a_new_qid"]}@{b.get("t_update")}'
        subj[rid] = b["subject_label"]
    for r in rows:
        r["_cat"] = classify(r["generated"], r["a_new_qid"], r["a_old_qid"],
                             subj.get(r["record_id"], ""), amap,
                             r["a_new_label"], r["a_old_label"])

    # per (model, method) category rates on PTC set; PA on controls
    metrics, catrates = [], []
    for m in MODELS:
        for meth in METHODS:
            ptc = [r for r in rows if r["model"] == m and r["method"] == meth and r["is_control"] == "0"]
            ctrl = [r for r in rows if r["model"] == m and r["method"] == meth and r["is_control"] == "1"]
            if not ptc and not ctrl:
                continue
            n = len(ptc)
            c = collections.Counter(r["_cat"] for r in ptc)
            resolvable = c["new"] + c["old"] + c["other_entity"]
            row = {"model": m, "method": meth, "n_ptc": n,
                   "new_rate": round(c["new"]/n, 4) if n else None,
                   "old_rate": round(c["old"]/n, 4) if n else None,
                   "other_entity_rate": round(c["other_entity"]/n, 4) if n else None,
                   "definitional_rate": round(c["definitional"]/n, 4) if n else None,
                   "empty_rate": round(c["empty"]/n, 4) if n else None,
                   "alias_resolved_rate": round(resolvable/n, 4) if n else None,
                   "new_given_resolvable": round(c["new"]/resolvable, 4) if resolvable else None,
                   "preservation_acc_ctrl": (round(sum(1 for r in ctrl if r["_cat"]=="new")/len(ctrl), 4)
                                             if ctrl else None), "n_control": len(ctrl)}
            metrics.append(row); catrates.append(row)

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (OUT / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys())); w.writeheader(); w.writerows(metrics)
    with (OUT / "category_rates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(catrates[0].keys())); w.writeheader(); w.writerows(catrates)

    # disagreement: candidate scoring (phase1) vs generation, per record, for standard & date_prefix
    cand = {m: load_phase1_candidate(m) for m in MODELS}
    dis = []
    for m in MODELS:
        for meth, cand_key in (("standard", "standard"), ("date_prefix", "temporal")):
            recs = [r for r in rows if r["model"] == m and r["method"] == meth and r["is_control"] == "0"]
            cnt = collections.Counter()
            for r in recs:
                cp = cand.get(m, {}).get(r["record_id"], {}).get(cand_key)
                if cp is None:
                    cnt["cand_unknown"] += 1; continue
                g = r["_cat"]
                if g in ("definitional", "empty"):
                    cnt[f"cand_{cp}__gen_noentity"] += 1
                elif g == "new":
                    cnt[f"cand_{cp}__gen_new"] += 1
                elif g == "old":
                    cnt[f"cand_{cp}__gen_old"] += 1
                else:
                    cnt[f"cand_{cp}__gen_other"] += 1
            row = {"model": m, "method": meth, "n": len(recs), **cnt}
            dis.append(row)
    keys = sorted({k for d in dis for k in d if k not in ("model", "method", "n")})
    with (OUT / "disagreement.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "n"] + keys, extrasaction="ignore")
        w.writeheader()
        for d in dis:
            w.writerow({**{k: d.get(k, 0) for k in ["model", "method", "n"] + keys}})

    # aggregate candidate-Recovery vs free-gen new-rate (all methods)
    with (OUT / "comparison_prob_vs_free.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "method", "candidate_recovery", "freegen_new_rate",
                                       "freegen_new_given_resolvable", "alias_resolved_rate"])
        for r in metrics:
            if r["method"] == "gated":
                continue
            pr = PROB_RECOVERY.get(r["model"], {}).get(r["method"])
            w.writerow([r["model"], r["method"], pr, r["new_rate"], r["new_given_resolvable"], r["alias_resolved_rate"]])

    # stratified manual-audit sample (spread across models, methods, categories)
    import random
    random.seed(0)
    buckets = collections.defaultdict(list)
    for r in rows:
        if r["is_control"] == "0":
            buckets[(r["model"], r["_cat"])].append(r)
    sample = []
    for key, lst in buckets.items():
        random.shuffle(lst)
        sample += lst[:8]     # up to 8 per (model,category)
    random.shuffle(sample)
    with (OUT / "manual_review.csv").open("w", newline="") as f:
        cols = ["model", "method", "record_id", "auto_category", "generated", "gold_new", "gold_old",
                "human_category", "notes"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sample[:180]:
            w.writerow({"model": r["model"], "method": r["method"], "record_id": r["record_id"],
                        "auto_category": r["_cat"], "generated": r["generated"],
                        "gold_new": r["a_new_label"], "gold_old": r["a_old_label"],
                        "human_category": "", "notes": ""})

    # console
    print("model            method       new    old   other  defin  empty  resolv  new|res  PA")
    for r in metrics:
        if r["method"] == "gated" and r["n_ptc"] == 0:
            continue
        print("%-16s %-11s %5s %5s %6s %6s %6s %6s %7s %s" % (
            r["model"], r["method"], r["new_rate"], r["old_rate"], r["other_entity_rate"],
            r["definitional_rate"], r["empty_rate"], r["alias_resolved_rate"],
            r["new_given_resolvable"], r["preservation_acc_ctrl"]))
    print("\nwrote metrics/category_rates/disagreement/comparison/manual_review + error breakdown")
    # error breakdown
    err = collections.Counter((r["method"], r["_cat"]) for r in rows if r["is_control"] == "0")
    with (OUT / "error_breakdown.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "category", "count"])
        for (meth, cat), ct in sorted(err.items()):
            w.writerow([meth, cat, ct])


if __name__ == "__main__":
    main()
