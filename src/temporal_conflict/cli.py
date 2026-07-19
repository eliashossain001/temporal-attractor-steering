"""Single CLI entry point for the temporal-conflict pipeline.

Subcommands:
    screen   Run Phase 1 PTC screening only.
    locate   Run through Phase 2B (layer locator) for one model.
    tas      Run the full TAS pipeline (Phase 1 -> Phase 2F).
    eval     Run Phase 2F evaluation at a specific (tau, alpha).

All commands are idempotent: existing outputs are reused unless --force.

Typical reviewer flow (subset, single model):
    python -m temporal_conflict.cli tas \\
        --model qwen-2.5-1.5b \\
        --data data/subset \\
        --runs-root runs/reviewer
"""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_conflict.pipeline import Pipeline, PipelineOptions
from temporal_conflict.stages import (
    LayerLocatorStage,
    ScreeningStage,
    TASEvaluationStage,
)


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="Model key in configs/models.yaml")
    p.add_argument("--data", default="data/subset",
                   help="Directory of *.jsonl PTC instances")
    p.add_argument("--runs-root", default="runs/main",
                   help="Output root; per-model artifacts go to <root>/<model>/")
    p.add_argument("--config", default="configs/models.yaml",
                   help="Model registry YAML")
    p.add_argument("--device-map", default="auto",
                   help='HF device_map ("auto", "cuda:0", "cpu", ...)')
    p.add_argument("--force", action="store_true",
                   help="Re-run stages even if outputs already exist")


def _build_options(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        data_dir=Path(args.data),
        runs_root=Path(args.runs_root),
        device_map=args.device_map,
        sample_ptc_cap=getattr(args, "sample_ptc_cap", 500),
        alphas=getattr(args, "alphas", None) or PipelineOptions().alphas,
        taus=getattr(args, "taus", None) or PipelineOptions().taus,
        variant=getattr(args, "variant", "relation"),
    )


def cmd_screen(args: argparse.Namespace) -> None:
    pipe = Pipeline(args.model, args.config, _build_options(args))
    ScreeningStage(pipe.ctx).run(force=args.force)


def cmd_locate(args: argparse.Namespace) -> None:
    pipe = Pipeline(args.model, args.config, _build_options(args))
    for stage in pipe.stages()[:3]:  # screen + cache + locate
        stage.run(force=args.force)


def cmd_tas(args: argparse.Namespace) -> None:
    pipe = Pipeline(args.model, args.config, _build_options(args))
    pipe.run(force=args.force)


def cmd_eval(args: argparse.Namespace) -> None:
    pipe = Pipeline(args.model, args.config, _build_options(args))
    stage = TASEvaluationStage(
        pipe.ctx, tau=args.tau, alpha=args.alpha, variant=args.variant,
    )
    stage.run(force=args.force)


def cmd_baselines(args: argparse.Namespace) -> None:
    from temporal_conflict.config import ProjectConfig
    from temporal_conflict.baselines.run import run_baselines

    cfg = ProjectConfig.from_yaml(args.config)
    run_baselines(
        model_spec=cfg.get_model(args.model),
        prompting=cfg.prompting,
        screening_path=Path(args.screening),
        data_dir=Path(args.data),
        out_dir=Path(args.out_dir),
        device_map=args.device_map,
        control_size=args.control_size,
        limit=args.limit,
        run_prompts=not args.no_prompts,
        steering_vectors_path=(
            Path(args.steering_vectors) if args.steering_vectors else None
        ),
        afr_path=Path(args.afr) if args.afr else None,
        alpha=args.alpha,
        null_layer=args.null_layer,
    )


def cmd_stats(args: argparse.Namespace) -> None:
    from temporal_conflict.analysis.stats import run_stats, markdown_table

    stats = run_stats(args.results_tas, args.out)
    if args.table:
        Path(args.table).write_text(markdown_table(stats))
        print(f"Wrote {args.table}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="temporal-conflict",
        description="Run the PTC screening + TAS pipeline.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_screen = sub.add_parser("screen", help="Phase 1 only")
    _add_shared_args(p_screen)
    p_screen.set_defaults(func=cmd_screen)

    p_locate = sub.add_parser("locate", help="Through layer locator")
    _add_shared_args(p_locate)
    p_locate.add_argument("--sample-ptc-cap", type=int, default=500)
    p_locate.set_defaults(func=cmd_locate)

    p_tas = sub.add_parser("tas", help="Full pipeline through Phase 2F")
    _add_shared_args(p_tas)
    p_tas.add_argument("--sample-ptc-cap", type=int, default=500)
    p_tas.add_argument(
        "--alphas", type=float, nargs="+",
        help="Override default alpha sweep (Phase 2D)",
    )
    p_tas.add_argument(
        "--taus", type=float, nargs="+",
        help="Override default tau grid (Phase 2F)",
    )
    p_tas.add_argument(
        "--variant", default="relation",
        choices=["global", "relation", "domain"],
        help="Steering-vector variant V1/V2/V3",
    )
    p_tas.set_defaults(func=cmd_tas)

    p_eval = sub.add_parser("eval", help="Phase 2F at a chosen (tau, alpha)")
    _add_shared_args(p_eval)
    p_eval.add_argument("--tau", type=float, required=True)
    p_eval.add_argument(
        "--alpha", type=float, default=None,
        help="Steering scale; defaults to alpha* from the oracle stage",
    )
    p_eval.add_argument(
        "--variant", default="relation",
        choices=["global", "relation", "domain"],
    )
    p_eval.set_defaults(func=cmd_eval)

    # --- prompt baselines (WP-A) + steering controls (WP-B) ---
    p_base = sub.add_parser(
        "baselines", help="Prompt baselines + steering-control ablations")
    p_base.add_argument("--model", required=True)
    p_base.add_argument("--config", default="configs/models.yaml")
    p_base.add_argument("--data", default="data/large",
                        help="Directory of *.jsonl PTC instances")
    p_base.add_argument("--screening", required=True,
                        help="per_instance.jsonl from Phase 1 for this model")
    p_base.add_argument("--out-dir", required=True)
    p_base.add_argument("--device-map", default="auto")
    p_base.add_argument("--control-size", type=int, default=500)
    p_base.add_argument("--limit", type=int, default=None,
                        help="Cap PTC/control subsets (smoke testing)")
    p_base.add_argument("--no-prompts", action="store_true",
                        help="Skip WP-A prompt baselines (controls only)")
    p_base.add_argument("--steering-vectors", default=None,
                        help="steering_vectors_v2.pt (enables WP-B controls)")
    p_base.add_argument("--afr", default=None,
                        help="afr_profile.json (supplies ell*)")
    p_base.add_argument("--alpha", type=float, default=None,
                        help="Steering scale for WP-B; defaults to oracle alpha*")
    p_base.add_argument("--null-layer", type=int, default=None,
                        help="Wrong-layer control layer; defaults to ell*//8")
    p_base.set_defaults(func=cmd_baselines)

    # --- reproducible bootstrap CIs + paired significance (WP-C) ---
    p_stats = sub.add_parser(
        "stats", help="Post-hoc bootstrap CIs + paired McNemar (no model)")
    p_stats.add_argument("--results-tas", default="results/tas",
                         help="Dir with <model>/tas_eval_tau*.json")
    p_stats.add_argument("--out", default="results/tas/_stats/bootstrap_stats.json")
    p_stats.add_argument("--table", default=None,
                         help="Optional markdown table output path")
    p_stats.set_defaults(func=cmd_stats)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
