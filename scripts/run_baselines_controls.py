#!/usr/bin/env python
"""Full WP-A (prompt baselines) + WP-B (steering controls) run for one model.

Reuses the already-computed Phase-1 screening and the known conflict-critical
layer ell* (from afr_profile.json), so the only model work is:
  1. cache residual activations on the verified-PTC subset (Phase 2A), then
  2. rebuild the ell* steering vectors (Phase 2C) -- the expensive layer sweep
     is NOT repeated, and
  3. score the prompt baselines and steering controls on the same subsets the
     TAS evaluation used.

Usage:
    python scripts/run_baselines_controls.py --model qwen-2.5-7b [--gpu 0]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from temporal_conflict.config import ProjectConfig, RunPaths
from temporal_conflict.stages import (
    ActivationCacheStage,
    SteeringVectorBuilderStage,
    StageContext,
)
from temporal_conflict.baselines.run import run_baselines

REPO = Path(__file__).resolve().parents[1]


def seed_run_dir(model: str, runs_root: Path) -> RunPaths:
    """Populate <runs_root>/<model>/ with the cached screening + AFR profile."""
    paths = RunPaths(root=runs_root, model_key=model)
    src_screen = REPO / "results" / "phase1" / model / "per_instance.jsonl"
    src_summary = REPO / "results" / "phase1" / model / "summary.json"
    src_afr = REPO / "results" / "tas" / model / "afr_profile.json"
    for src, dst in [
        (src_screen, paths.screening_jsonl),
        (src_summary, paths.screening_summary),
        (src_afr, paths.afr_profile),
    ]:
        if not dst.exists():
            shutil.copy(src, dst)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(REPO / "configs/models.yaml"))
    ap.add_argument("--data", default=str(REPO / "data/large"))
    ap.add_argument("--runs-root", default=str(REPO / "runs/wpb"))
    ap.add_argument("--out-root", default=str(REPO / "results/baselines"))
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--control-size", type=int, default=500)
    args = ap.parse_args()

    cfg = ProjectConfig.from_yaml(args.config)
    spec = cfg.get_model(args.model)
    runs_root = Path(args.runs_root)
    paths = seed_run_dir(args.model, runs_root)

    ctx = StageContext(
        model_spec=spec, prompting=cfg.prompting, paths=paths,
        data_dir=Path(args.data), device_map=args.device_map,
    )

    # Rebuild ell* steering vectors (activation cache + vector build only).
    ActivationCacheStage(ctx, sample_cap=500).run()
    SteeringVectorBuilderStage(ctx).run()

    alpha_star = float(
        json.loads((REPO / "results/tas" / args.model / "oracle_tas_relation.json")
                   .read_text())["alpha_star"]
    )

    out_dir = Path(args.out_root) / args.model
    run_baselines(
        model_spec=spec, prompting=cfg.prompting,
        screening_path=paths.screening_jsonl, data_dir=Path(args.data),
        out_dir=out_dir, device_map=args.device_map,
        control_size=args.control_size,
        steering_vectors_path=paths.steering_vectors,
        afr_path=paths.afr_profile, alpha=alpha_star,
    )
    print(f"[done] {args.model} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
