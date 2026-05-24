"""temporal_conflict: parametric temporal conflict (PTC) detection
and the Temporal Attractor Steering (TAS) test-time intervention pipeline.

Public entry points:
    from temporal_conflict.pipeline import Pipeline, PipelineOptions
    from temporal_conflict.config import ProjectConfig

Stages live in `temporal_conflict.stages`; lower-level scoring,
activation, and steering primitives live under
`temporal_conflict.steering`.
"""

__version__ = "1.0.0"
