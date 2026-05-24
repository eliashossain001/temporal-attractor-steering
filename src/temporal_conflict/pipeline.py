"""End-to-end pipeline orchestrator.

`Pipeline` composes the seven stages in order. Each stage is idempotent;
re-running the pipeline only does the work that is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from temporal_conflict.config import ProjectConfig, RunPaths
from temporal_conflict.env import load_env
from temporal_conflict.stages import (
    ActivationCacheStage,
    DetectorStage,
    LayerLocatorStage,
    OracleTASStage,
    ScreeningStage,
    Stage,
    StageContext,
    SteeringVectorBuilderStage,
    TASEvaluationStage,
)


@dataclass
class PipelineOptions:
    """Per-run knobs that override stage defaults."""

    data_dir: Path = Path("data/subset")
    runs_root: Path = Path("runs/main")
    device_map: str = "auto"
    sample_ptc_cap: int | None = 500
    alphas: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    )
    taus: list[float] = field(default_factory=lambda: [0.15, 0.20, 0.30])
    variant: str = "relation"
    layers: list[int] | None = None  # None -> all layers


class Pipeline:
    """Run the full screening + TAS pipeline for one model."""

    def __init__(
        self,
        model_key: str,
        config_path: str | Path = "configs/models.yaml",
        options: PipelineOptions | None = None,
    ) -> None:
        load_env()
        self.config = ProjectConfig.from_yaml(config_path)
        self.options = options or PipelineOptions()
        self.ctx = StageContext(
            model_spec=self.config.get_model(model_key),
            prompting=self.config.prompting,
            paths=RunPaths(root=self.options.runs_root, model_key=model_key),
            data_dir=self.options.data_dir,
            device_map=self.options.device_map,
        )

    def stages(self) -> list[Stage]:
        opts = self.options
        return [
            ScreeningStage(self.ctx),
            ActivationCacheStage(self.ctx, sample_cap=opts.sample_ptc_cap),
            LayerLocatorStage(self.ctx, layers=opts.layers),
            SteeringVectorBuilderStage(self.ctx),
            OracleTASStage(self.ctx, alphas=opts.alphas, variant=opts.variant),
            DetectorStage(self.ctx),
            *[
                TASEvaluationStage(self.ctx, tau=t, variant=opts.variant)
                for t in opts.taus
            ],
        ]

    def run(self, *, force: bool = False) -> None:
        for stage in self.stages():
            stage.run(force=force)
        print("Pipeline complete.", flush=True)
