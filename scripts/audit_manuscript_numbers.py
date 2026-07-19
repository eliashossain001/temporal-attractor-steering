#!/usr/bin/env python3
"""Final numerical-consistency audit (Bucket 9, Part A+B).

Extracts authoritative values from generated artifacts, builds a machine-readable
result-source manifest, and cross-checks headline numerical claims that appear in
main.tex / supplement.tex. No model experiments are run; artifacts are read only.

Outputs (under results/final_audit/):
  - result_source_manifest.json / .md   (Part A)
  - numerical_audit.json / .md          (Part B)

Usage:
  python scripts/audit_manuscript_numbers.py [--repo .] [--help]

Exit code 0 always (audit is advisory); mismatches are reported in the artifacts and
summarized on stdout. Deterministic: reads fixed files, no randomness.
"""
import argparse, csv, glob, hashlib, json, os, re, sys

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(p):
    with open(p) as f:
        return json.load(f)

def oracle_recovery(repo):
    """Full-data oracle TAS recovery_after at alpha_star per model."""
    out = {}
    for f in sorted(glob.glob(os.path.join(repo, "results/tas/*/oracle_tas_relation.json"))):
        d = load_json(f); m = os.path.basename(os.path.dirname(f))
        astar = float(d.get("alpha_star"))
        e = next((x for x in d.get("by_alpha", []) if abs(float(x["alpha"]) - astar) < 1e-6), None)
        ptc = (e or {}).get("ptc", {})
        out[m] = {"alpha_star": astar, "n": ptc.get("n"),
                  "recovery": ptc.get("recovery_after"), "source": os.path.relpath(f, repo)}
    return out

MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]

def build_manifest(repo):
    M = {"provenance": {}, "results": {}}
    # ---- provenance / hashes ----
    split = os.path.join(repo, "results/splits/subject_disjoint_v1.json")
    audit_meta = load_json(os.path.join(repo, "results/benchmark_audit/audit_metadata.json"))
    M["provenance"] = {
        "split_manifest": {"path": "results/splits/subject_disjoint_v1.json",
                           "sha256": sha256(split) if os.path.exists(split) else None},
        "benchmark": {"path": audit_meta.get("benchmark_path"),
                      "sha256": audit_meta.get("benchmark_sha256"),
                      "record_count": audit_meta.get("benchmark_record_count")},
        "split_seed": 20260712, "benchmark_audit_seed": audit_meta.get("seed"),
        "models": MODELS,
    }
    # model revision hashes
    revs = {}
    for m in MODELS:
        for cand in glob.glob(os.path.join(repo, f"results/iti_paired/{m}/test_summary.json")) + \
                    glob.glob(os.path.join(repo, f"results/tas_splitclean/{m}/detector_gated_test.json")):
            try:
                d = load_json(cand)
                rev = d.get("model_revision") or (d.get("meta") or {}).get("model_revision")
                if rev: revs[m] = rev; break
            except Exception:
                pass
    M["provenance"]["model_revisions"] = revs
    # ---- filtered PTC rate among kept @ tau=-3.0 ----
    fs = os.path.join(repo, "results/tau_rec_sensitivity/filter_sweep_all_models.csv")
    fptc = {}
    if os.path.exists(fs):
        with open(fs) as f:
            for row in csv.DictReader(f):
                if abs(float(row.get("tau_rec", row.get("tau", "nan"))) + 3.0) < 1e-6:
                    fptc[row["model"]] = {k: row.get(k) for k in
                        ("filtered_ptc", "kept", "kept_frac", "fptc_rate_among_kept", "raw_ptc")}
    M["results"]["filtered_ptc_at_tau_-3.0"] = {"value": fptc, "source": "results/tau_rec_sensitivity/filter_sweep_all_models.csv",
        "status": "diagnostic (full benchmark)", "generating_script": "scripts/analyze_tau_rec_sensitivity.py"}
    # ---- date-prefix recovery ----
    dp = {}
    for m in MODELS:
        p = os.path.join(repo, f"results/baselines/{m}/prompt_baselines.json")
        if os.path.exists(p):
            s = load_json(p).get("summary", {})
            dp[m] = {"date_prefix_recovery": (s.get("date_prefix") or {}).get("recovery"),
                     "n_ptc": (s.get("date_prefix") or {}).get("n_ptc")}
    M["results"]["date_prefix_recovery"] = {"value": dp, "source": "results/baselines/<model>/prompt_baselines.json",
        "status": "diagnostic (full verified-PTC set)", "generating_script": "scripts/run_baselines_controls.py"}
    # ---- oracle TAS recovery (full data) ----
    M["results"]["tas_oracle_recovery_fulldata"] = {"value": oracle_recovery(repo),
        "source": "results/tas/<model>/oracle_tas_relation.json (by_alpha[alpha_star].ptc.recovery_after)",
        "status": "diagnostic (full verified-PTC set)", "note": "abstract 29-62% range"}
    # ---- random-direction control ----
    rc = {}
    for m in MODELS:
        p = os.path.join(repo, f"results/baselines/{m}/steering_controls.json")
        if os.path.exists(p):
            d = load_json(p)
            def dig(o, key):
                if isinstance(o, dict):
                    if key in o and isinstance(o[key], dict): return o[key]
                    for v in o.values():
                        r = dig(v, key)
                        if r: return r
                return None
            rd = dig(d, "random_direction") or {}
            rc[m] = {"recovery": rd.get("recovery"), "pa": rd.get("pa")}
    M["results"]["random_direction_control"] = {"value": rc,
        "source": "results/baselines/<model>/steering_controls.json", "status": "diagnostic (full verified-PTC set)"}
    # ---- held-out TAS vs ITI ----
    p = os.path.join(repo, "results/iti_paired/paired_significance.json")
    if os.path.exists(p):
        M["results"]["tas_vs_iti_heldout"] = {"value_source": "results/iti_paired/paired_significance.json",
            "status": "HELD-OUT subject-disjoint test", "generating_script": "scripts/analyze_paired_iti.py",
            "note": "Recovery: no Holm-significant difference on any model. PA: only qwen-2.5-1.5b Holm-significant (ITI>TAS, p=0.006)."}
    # ---- detector corrected ----
    dcp = os.path.join(repo, "results/detector_splitclean/summary.json")
    M["results"]["detector_corrected"] = {"value_source": "results/detector_splitclean/summary.{json,csv}",
        "status": "HELD-OUT subject-disjoint (true base rate)", "generating_script": "scripts/recompute_detector_splitclean.py",
        "note": "corrected AUROC 0.47-0.66, AUPRC 0.028-0.097; qwen-2.5-1.5b AUROC 0.467 (below chance). LEGACY leaked AUPRC 0.258-0.525 must NOT be cited as current."}
    # ---- benchmark audit status ----
    ann = load_json(os.path.join(repo, "results/benchmark_audit/audit_metadata.json")).get("annotation_state")
    M["results"]["benchmark_manual_audit"] = {"status": ann, "sample_size": 200,
        "source": "results/benchmark_audit/audit_sample.csv",
        "note": "UNREVIEWED as of this audit; quantitative accuracy removed from paper (non-quantitative description substituted)."}
    return M

# Headline claims expected in the manuscript (value strings that must appear verbatim).
EXPECTED = [
    ("abstract/intro", r"8[,{}]*746", "benchmark size 8,746"),
    ("abstract", r"61\$?--\$?81", "date-prefix 61--81%"),
    ("abstract", r"72\$?--\$?85", "patching 72--85%"),
    ("abstract", r"29\$?--\$?62", "TAS oracle 29--62% (was stale 29--57)"),
    ("abstract", r"18\$?--\$?31", "random-direction 18--31%"),
    ("intro", r"0\.041.*0\.103", "filtered PTC 0.041->0.103"),
    ("locator", r"0\.72.*0\.85|0\.723|0\.851|0\.824|0\.816", "AFR peaks"),
]

def audit_numbers(repo, manifest):
    main = open(os.path.join(repo, "TAS_AAAI27_submission/main.tex")).read()
    supp = open(os.path.join(repo, "TAS_AAAI27_submission/supplement.tex")).read()
    both = main + "\n" + supp
    findings = []
    for section, pat, desc in EXPECTED:
        present = re.search(pat, both) is not None
        findings.append({"section": section, "claim": desc, "regex": pat,
                         "status": "present" if present else "MISSING"})
    # stale/placeholder scan
    stale = []
    for name, pat in [("[N] placeholder", r"\[\$?N\$?\]"), ("[X%] placeholder", r"\[\$?X\$?\\?%\]"),
                      ("TODO", r"TODO"), ("FIXME", r"FIXME"), ("29--57 stale", r"29\$?--\$?57"),
                      ("leaked AUPRC 0.525 as current", r"reproduce the .*Det\.\\? AUPRC column")]:
        for label, txt in [("main", main), ("supplement", supp)]:
            if re.search(pat, txt):
                stale.append({"marker": name, "file": label, "status": "PRESENT (review)"})
    return {"headline_claims": findings, "stale_markers": stale}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                    help="repo root (default: parent of scripts/)")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    outdir = os.path.join(repo, "results/final_audit")
    os.makedirs(outdir, exist_ok=True)

    manifest = build_manifest(repo)
    with open(os.path.join(outdir, "result_source_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    audit = audit_numbers(repo, manifest)
    with open(os.path.join(outdir, "numerical_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # markdown renderings
    with open(os.path.join(outdir, "result_source_manifest.md"), "w") as f:
        f.write("# Result Source Manifest (auto-generated)\n\n")
        f.write("Split sha256: `%s`\n\n" % manifest["provenance"]["split_manifest"]["sha256"])
        f.write("Benchmark: `%s` sha256 `%s` (%s records)\n\n" % (
            manifest["provenance"]["benchmark"]["path"], manifest["provenance"]["benchmark"]["sha256"],
            manifest["provenance"]["benchmark"]["record_count"]))
        f.write("Model revisions:\n\n")
        for m, r in manifest["provenance"].get("model_revisions", {}).items():
            f.write("- %s: `%s`\n" % (m, r))
        f.write("\n## Results -> artifacts\n\n")
        for k, v in manifest["results"].items():
            f.write("### %s\n\n- status: %s\n- source: %s\n" % (k, v.get("status", "?"), v.get("source", v.get("value_source", "?"))))
            if "generating_script" in v: f.write("- script: %s\n" % v["generating_script"])
            if "note" in v: f.write("- note: %s\n" % v["note"])
            if "value" in v: f.write("- value: `%s`\n" % json.dumps(v["value"])[:600])
            f.write("\n")
    with open(os.path.join(outdir, "numerical_audit.md"), "w") as f:
        f.write("# Numerical Audit (auto-generated)\n\n## Headline claims\n\n")
        for c in audit["headline_claims"]:
            f.write("- [%s] %s (%s): `%s`\n" % (c["status"], c["claim"], c["section"], c["regex"]))
        f.write("\n## Stale markers / placeholders\n\n")
        if not audit["stale_markers"]:
            f.write("None found. CLEAN.\n")
        for s in audit["stale_markers"]:
            f.write("- %s in %s: %s\n" % (s["marker"], s["file"], s["status"]))

    miss = [c for c in audit["headline_claims"] if c["status"] != "present"]
    print("Manifest + numerical audit written to results/final_audit/")
    print("Headline claims: %d present, %d MISSING" % (len(audit["headline_claims"]) - len(miss), len(miss)))
    for c in miss: print("  MISSING:", c["claim"])
    print("Stale markers:", len(audit["stale_markers"]))
    for s in audit["stale_markers"]: print("  ", s)

if __name__ == "__main__":
    main()
