"""Project-level configuration objects.

A single place to load and validate the model registry and the per-run
options that every stage class needs. Keeps stage code free of YAML
parsing and ad-hoc dict access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """One entry from `configs/models.yaml`."""

    key: str
    hf_id: str
    family: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class PromptingSpec:
    """Default prompt formatting for candidate scoring."""

    answer_suffix: str = "\nAnswer:"
    candidate_prefix: str = " "


@dataclass
class ProjectConfig:
    """The loaded model registry plus shared prompting defaults."""

    models: dict[str, ModelSpec] = field(default_factory=dict)
    prompting: PromptingSpec = field(default_factory=PromptingSpec)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        models: dict[str, ModelSpec] = {}
        for key, spec in (raw.get("models") or {}).items():
            models[key] = ModelSpec(
                key=key,
                hf_id=spec["hf_id"],
                family=spec.get("family", "unknown"),
                dtype=spec.get("dtype", "bfloat16"),
                trust_remote_code=bool(spec.get("trust_remote_code", False)),
            )
        prompting_raw = raw.get("prompting") or {}
        prompting = PromptingSpec(
            answer_suffix=prompting_raw.get("answer_suffix", "\nAnswer:"),
            candidate_prefix=prompting_raw.get("candidate_prefix", " "),
        )
        return cls(models=models, prompting=prompting)

    def get_model(self, key: str) -> ModelSpec:
        if key not in self.models:
            raise KeyError(
                f"Model {key!r} not found in registry. Available: "
                f"{sorted(self.models)}"
            )
        return self.models[key]


@dataclass
class RunPaths:
    """Standard output layout for one (model, split) run.

    All artifacts of one model live under `<root>/<model_key>/`. Heavy
    tensors (`*.pt`, `*.pkl`) live alongside the JSON metrics; the
    repository .gitignore excludes the binaries.
    """

    root: Path
    model_key: str

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> Path:
        return self.root / self.model_key

    # Phase 1
    @property
    def screening_jsonl(self) -> Path:
        return self.model_dir / "per_instance.jsonl"

    @property
    def screening_summary(self) -> Path:
        return self.model_dir / "summary.json"

    # Phase 2A-2C
    @property
    def activations(self) -> Path:
        return self.model_dir / "activations.pt"

    @property
    def afr_profile(self) -> Path:
        return self.model_dir / "afr_profile.json"

    @property
    def steering_vectors(self) -> Path:
        return self.model_dir / "steering_vectors_v2.pt"

    # Phase 2D
    @property
    def oracle_tas(self) -> Path:
        return self.model_dir / "oracle_tas_relation.json"

    # Phase 2E
    @property
    def detector_activations(self) -> Path:
        return self.model_dir / "detector_activations.pt"

    @property
    def detector(self) -> Path:
        return self.model_dir / "detector.pkl"

    @property
    def detector_metrics(self) -> Path:
        return self.model_dir / "detector_metrics.json"

    # Phase 2F
    def tas_eval(self, tau: float) -> Path:
        return self.model_dir / f"tas_eval_tau{tau:.2f}.json"
