"""Authoritative subject-disjoint split protocol (Bucket 4).

One versioned, reproducible split that every downstream consumer shares:

    train        -> detector fitting, TAS/ITI direction construction
    validation   -> alpha / operating-point selection
    calibration  -> isotonic calibration, detector threshold selection
    test         -> final detector, TAS, and paired TAS-vs-ITI metrics

Grouping unit is ``subject_qid``: every record of a subject lands in exactly
one partition, so no subject (and therefore no near-duplicate transition of the
same subject) crosses splits. Subjects are stratified by their *rarest*
relation, which protects the small relations (P35, P169) and keeps the relation
distribution stable across partitions. Assignment is deterministic given the
seed.

This module is the single source of truth for split logic; scripts and the
detector import from here rather than re-implementing it.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable

# --------------------------------------------------------------------------- #
# Protocol constants (documented, versioned)
# --------------------------------------------------------------------------- #
SPLIT_VERSION = "v1"
SPLIT_SEED = 20260712
# train / validation / calibration / test
PROPORTIONS: tuple[tuple[str, float], ...] = (
    ("train", 0.60), ("validation", 0.15),
    ("calibration", 0.10), ("test", 0.15),
)
SPLIT_NAMES = tuple(name for name, _ in PROPORTIONS)
HELDOUT_SPLITS = ("validation", "calibration", "test")
# Minimum PTC positives required in EACH held-out split for EVERY model.
MIN_POS_PER_HELDOUT = 15


# --------------------------------------------------------------------------- #
# Record identity
# --------------------------------------------------------------------------- #
def canonical_id(r: dict) -> str:
    """relation:subject:old->new (collides for 12 recurring-term records)."""
    return f"{r['relation_pid']}:{r['subject_qid']}:{r['a_old_qid']}->{r['a_new_qid']}"


def record_id(r: dict) -> str:
    """Unique id, date-anchored (matches the manual-audit scheme)."""
    return f"{canonical_id(r)}@{r.get('t_update')}"


def subject_of(r: dict) -> str:
    return r["subject_qid"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _largest_remainder(n: int, props: Iterable[tuple[str, float]]) -> dict[str, int]:
    raw = {k: n * p for k, p in props}
    base = {k: int(v) for k, v in raw.items()}
    short = n - sum(base.values())
    for k in sorted(base, key=lambda k: -(raw[k] - int(raw[k])))[:short]:
        base[k] += 1
    return base


def _norm_label(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _group_subjects(records: list[dict]) -> dict[str, str]:
    """Union subjects (subject_qid) that share a normalized surface label into
    one grouping component, and return ``{subject_qid: group_key}``.

    Distinct QIDs can share a label (homonyms, e.g. two places named
    "Fantanele"), which yields identical prompt strings. Grouping by
    subject_qid alone would let those identical prompts cross splits. Unioning
    same-label subjects guarantees no shared label -- and therefore no shared
    prompt -- crosses a split boundary, while staying subject-disjoint.
    """
    label_of: dict[str, str] = {}
    for r in records:
        label_of.setdefault(subject_of(r), _norm_label(r.get("subject_label")))
    # union subjects with the same normalized label
    by_label: dict[str, list[str]] = defaultdict(list)
    for subj, lab in label_of.items():
        by_label[lab].append(subj)
    group_key: dict[str, str] = {}
    for lab, subs in by_label.items():
        rep = min(subs)  # deterministic component representative
        for s in subs:
            group_key[s] = rep
    return group_key


def build_subject_splits(
    records: list[dict], seed: int = SPLIT_SEED,
    proportions: tuple[tuple[str, float], ...] = PROPORTIONS,
) -> dict[str, str]:
    """Return ``{subject_qid: split}``. Deterministic given seed.

    Grouping unit is a *label-merged subject component* (subjects sharing a
    normalized surface label are merged; see ``_group_subjects``) so neither a
    subject nor an identical prompt/label can cross splits. Components are
    stratified by their rarest relation so small relations (P35, P169) spread
    evenly; within each stratum components are shuffled with a stratum-keyed RNG
    and allocated by largest remainder.
    """
    group_key = _group_subjects(records)
    relcount = Counter(r["relation_pid"] for r in records)

    grp_rels: dict[str, set[str]] = defaultdict(set)
    grp_members: dict[str, set[str]] = defaultdict(set)
    for r in records:
        g = group_key[subject_of(r)]
        grp_rels[g].add(r["relation_pid"])
        grp_members[g].add(subject_of(r))

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for g, rels in grp_rels.items():
        stratum = min(rels, key=lambda rel: relcount[rel])  # rarest relation
        by_stratum[stratum].append(g)

    grp_split: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        groups = sorted(by_stratum[stratum])
        random.Random(f"{seed}:{stratum}").shuffle(groups)
        alloc = _largest_remainder(len(groups), proportions)
        i = 0
        for name, _ in proportions:
            for g in groups[i:i + alloc[name]]:
                grp_split[g] = name
            i += alloc[name]

    return {subj: grp_split[g] for subj, g in group_key.items()}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass
class SplitManifest:
    """Loaded split manifest with fast lookup + Bucket-5 verification helpers."""

    version: str
    seed: int
    benchmark_sha256: str
    by_record: dict[str, str] = field(default_factory=dict)     # record_id -> split
    by_subject: dict[str, str] = field(default_factory=dict)    # subject   -> split
    rows: list[dict] = field(default_factory=list)              # full manifest rows

    # -- lookups -----------------------------------------------------------
    def split_of_record(self, rid: str) -> str:
        return self.by_record[rid]

    def record_ids(self, split: str) -> set[str]:
        return {rid for rid, sp in self.by_record.items() if sp == split}

    def subjects(self, split: str) -> set[str]:
        return {s for s, sp in self.by_subject.items() if sp == split}

    # -- Bucket-5 guarantees ----------------------------------------------
    def assert_disjoint_from_test(self, used_record_ids: Iterable[str],
                                  what: str = "records") -> None:
        """Fail if any id used for construction/selection/calibration is in test."""
        test = self.record_ids("test")
        leaked = set(used_record_ids) & test
        if leaked:
            raise AssertionError(
                f"{len(leaked)} {what} used off-test are in the TEST split "
                f"(e.g. {list(leaked)[:5]}); this violates held-out evaluation.")

    def assert_split_only(self, used_record_ids: Iterable[str],
                          allowed: Iterable[str], what: str = "records") -> None:
        allowed = set(allowed)
        bad = {rid for rid in used_record_ids
               if self.by_record.get(rid) not in allowed}
        if bad:
            raise AssertionError(
                f"{len(bad)} {what} fall outside allowed splits {sorted(allowed)} "
                f"(e.g. {list(bad)[:5]}).")


def load_manifest(json_path: str | Path) -> SplitManifest:
    d = json.loads(Path(json_path).read_text())
    rows = d["rows"]
    return SplitManifest(
        version=d["version"], seed=d["seed"],
        benchmark_sha256=d["benchmark_sha256"],
        by_record={r["record_id"]: r["split"] for r in rows},
        by_subject={r["subject_qid"]: r["split"] for r in rows},
        rows=rows,
    )


def default_manifest_json(root: str | Path = None) -> Path:
    from pathlib import Path as _P
    base = _P(root) if root else _P(__file__).resolve().parents[2]
    return base / "results" / "splits" / f"subject_disjoint_{SPLIT_VERSION}.json"


# --------------------------------------------------------------------------- #
# Hard assertions (used by builder + tests)
# --------------------------------------------------------------------------- #
def assert_valid_partition(rows: list[dict], n_expected: int) -> None:
    rids = [r["record_id"] for r in rows]
    if len(rids) != len(set(rids)):
        dup = [k for k, v in Counter(rids).items() if v > 1]
        raise AssertionError(f"duplicate record_ids: {dup[:5]}")
    if n_expected and len(rids) != n_expected:
        raise AssertionError(f"coverage {len(rids)} != expected {n_expected}")
    if any(r["split"] not in SPLIT_NAMES for r in rows):
        bad = {r["split"] for r in rows if r["split"] not in SPLIT_NAMES}
        raise AssertionError(f"invalid split labels: {bad}")
    # zero subject overlap across splits
    subj_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        subj_splits[r["subject_qid"]].add(r["split"])
    crossing = {s: sp for s, sp in subj_splits.items() if len(sp) > 1}
    if crossing:
        raise AssertionError(
            f"{len(crossing)} subject(s) cross splits (e.g. {list(crossing)[:3]})")
    if not any(r["split"] == "test" for r in rows):
        raise AssertionError("test split is empty")
