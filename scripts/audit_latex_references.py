#!/usr/bin/env python3
"""LaTeX reference + citation audit (Bucket 7, Part J).

Parses the manuscript TeX sources and the bibliography and reports every broken
internal reference (including cross-document \\cref that cannot resolve when
main.tex and supplement.tex compile as separate PDFs), duplicate labels,
missing citation keys, and literal unresolved markers. Writes machine-readable
and human-readable audit files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUB = REPO / "TAS_AAAI27_submission"
DOCS = {"main": SUB / "main.tex", "supplement": SUB / "supplement.tex"}
BIB = SUB / "ref.bib"
OUT = REPO / "results" / "latex_audit"

REF_CMDS = ["cref", "Cref", "ref", "eqref", "autoref", "nameref"]
CITE_CMDS = ["cite", "citep", "citet", "citealp", "citeauthor", "citeyear"]
MARKERS = ["??", "TODO", "FIXME", "TBD", "placeholder", "XXX"]


def strip_comments(t: str) -> str:
    # remove % comments (not escaped \%)
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in t.splitlines())


def labels_in(t: str) -> list[str]:
    return re.findall(r"\\label\{([^}]+)\}", t)


def refs_in(t: str) -> list[str]:
    out = []
    for cmd in REF_CMDS:
        for m in re.findall(r"\\" + cmd + r"\{([^}]+)\}", t):
            out += [x.strip() for x in m.split(",")]
    return out


def cites_in(t: str) -> list[str]:
    out = []
    for cmd in CITE_CMDS:
        # allow optional [..] args: \citep[e.g.][]{key}
        for m in re.findall(r"\\" + cmd + r"(?:\[[^\]]*\])*\{([^}]+)\}", t):
            out += [x.strip() for x in m.split(",")]
    return out


def bib_keys(t: str) -> tuple[list[str], list[str]]:
    keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", t)
    dups = [k for k in set(keys) if keys.count(k) > 1]
    return keys, dups


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc_text = {k: strip_comments(p.read_text()) for k, p in DOCS.items()}
    doc_labels = {k: labels_in(t) for k, t in doc_text.items()}
    doc_refs = {k: refs_in(t) for k, t in doc_text.items()}
    doc_cites = {k: cites_in(t) for k, t in doc_text.items()}

    bib_raw = BIB.read_text()
    keys, dup_keys = bib_keys(bib_raw)
    bibset = set(keys)

    report = {"per_doc": {}, "bib": {}, "summary": {}}
    total_missing_internal = total_dup = total_missing_cite = total_markers = 0

    for k in DOCS:
        labelset = set(doc_labels[k])
        dup_labels = sorted({l for l in doc_labels[k] if doc_labels[k].count(l) > 1})
        # a ref resolves only if defined in the SAME doc (separate PDFs)
        missing = sorted({r for r in doc_refs[k] if r not in labelset})
        # classify: is it defined in the OTHER doc? -> cross-document
        other = "supplement" if k == "main" else "main"
        other_labels = set(doc_labels[other])
        cross_doc = sorted([r for r in missing if r in other_labels])
        truly_missing = sorted([r for r in missing if r not in other_labels])
        missing_cites = sorted({c for c in doc_cites[k] if c not in bibset})
        markers = {}
        raw = DOCS[k].read_text()
        for mk in MARKERS:
            n = raw.count(mk)
            if n:
                markers[mk] = n
        report["per_doc"][k] = {
            "n_labels": len(doc_labels[k]),
            "duplicate_labels": dup_labels,
            "n_refs": len(doc_refs[k]),
            "missing_internal_refs": truly_missing,
            "cross_document_refs": cross_doc,
            "n_cites": len(doc_cites[k]),
            "missing_citation_keys": missing_cites,
            "literal_markers": markers,
        }
        total_missing_internal += len(truly_missing) + len(cross_doc)
        total_dup += len(dup_labels)
        total_missing_cite += len(missing_cites)
        total_markers += sum(markers.values())

    report["bib"] = {"n_entries": len(keys), "duplicate_keys": dup_keys,
                     "n_unique_keys": len(bibset)}
    report["summary"] = {
        "total_broken_internal_refs": total_missing_internal,
        "total_duplicate_labels": total_dup,
        "total_missing_citation_keys": total_missing_cite,
        "total_literal_markers": total_markers,
        "CLEAN": total_missing_internal == 0 and total_dup == 0
                 and total_missing_cite == 0,
    }

    (OUT / "reference_audit.json").write_text(json.dumps(report, indent=2))

    # markdown
    L = ["# LaTeX Reference Audit\n"]
    for k in DOCS:
        d = report["per_doc"][k]
        L.append(f"## {k}.tex")
        L.append(f"- labels: {d['n_labels']}  refs: {d['n_refs']}  cites: {d['n_cites']}")
        L.append(f"- duplicate labels: {d['duplicate_labels'] or 'none'}")
        L.append(f"- **broken internal refs**: {d['missing_internal_refs'] or 'none'}")
        L.append(f"- **cross-document refs (resolve to other PDF -> '?')**: {d['cross_document_refs'] or 'none'}")
        L.append(f"- **missing citation keys**: {d['missing_citation_keys'] or 'none'}")
        L.append(f"- literal markers: {d['literal_markers'] or 'none'}\n")
    L.append(f"## ref.bib\n- entries: {report['bib']['n_entries']}  "
             f"duplicate keys: {report['bib']['duplicate_keys'] or 'none'}\n")
    L.append(f"## Summary\n- broken internal refs: {total_missing_internal}\n"
             f"- duplicate labels: {total_dup}\n- missing citation keys: {total_missing_cite}\n"
             f"- literal markers: {total_markers}\n- **CLEAN: {report['summary']['CLEAN']}**")
    (OUT / "reference_audit.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
