#!/usr/bin/env python
"""Assemble the WP-A/WP-B comparison table across all models.

Reads, per model:
  results/baselines/<model>/prompt_baselines.json   (WP-A)
  results/baselines/<model>/steering_controls.json  (WP-B)
  results/tas/<model>/oracle_tas_relation.json      (TAS oracle @ alpha*)

and prints a markdown table putting the cheap prompt baselines and the negative
steering controls next to oracle TAS on the same verified-PTC subset.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]


def oracle_at_star(model: str) -> dict:
    d = json.loads((REPO / "results/tas" / model / "oracle_tas_relation.json").read_text())
    star = d["alpha_star"]
    row = next(r for r in d["by_alpha"] if r["alpha"] == star)
    return {"alpha": star, "recovery": row["ptc"]["recovery_after"],
            "pa": row["control"]["preservation_accuracy"], "n_ptc": row["ptc"]["n"]}


def main() -> None:
    rows = []
    for m in MODELS:
        pb_path = REPO / "results/baselines" / m / "prompt_baselines.json"
        sc_path = REPO / "results/baselines" / m / "steering_controls.json"
        if not pb_path.exists():
            print(f"[skip] {m}: no baseline results yet")
            continue
        pb = json.loads(pb_path.read_text())["summary"]
        sc = json.loads(sc_path.read_text())["summary"] if sc_path.exists() else {}
        orc = oracle_at_star(m)
        rows.append((m, pb, sc, orc))

    # Recovery table (verified-PTC subset).
    print("\n### Recovery on the verified-PTC subset (higher = better)\n")
    print("| Model | n_PTC | Standard | Date-prefix | Instruction | "
          "TAS oracle (a*) | rand-dir | wrong-layer |")
    print("|---|---|---|---|---|---|---|---|")
    for m, pb, sc, orc in rows:
        dp = pb.get("date_prefix", {}).get("recovery", float("nan"))
        ins = pb.get("instruction", {}).get("recovery", float("nan"))
        rd = sc.get("random_direction", {}).get("recovery", float("nan"))
        wl = sc.get("wrong_layer", {}).get("recovery", float("nan"))
        print(f"| {m} | {orc['n_ptc']} | 0.000 | {dp:.3f} | {ins:.3f} | "
              f"{orc['recovery']:.3f} | {rd:.3f} | {wl:.3f} |")

    # Preservation table (non-conflict control subset).
    print("\n### Preservation accuracy on the non-conflict control subset\n")
    print("| Model | Date-prefix | Instruction | TAS oracle | rand-dir | wrong-layer |")
    print("|---|---|---|---|---|---|")
    for m, pb, sc, orc in rows:
        dp = pb.get("date_prefix", {}).get("preservation_accuracy", float("nan"))
        ins = pb.get("instruction", {}).get("preservation_accuracy", float("nan"))
        rd = sc.get("random_direction", {}).get("preservation_accuracy", float("nan"))
        wl = sc.get("wrong_layer", {}).get("preservation_accuracy", float("nan"))
        print(f"| {m} | {dp:.3f} | {ins:.3f} | {orc['pa']:.3f} | {rd:.3f} | {wl:.3f} |")


if __name__ == "__main__":
    main()
