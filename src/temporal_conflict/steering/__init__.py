"""TAS (Temporal Attractor Steering) framework.

Phase 2 of the proposal. Builds on the Phase 1 hidden-state machinery in
ptc.models to implement the three-stage TAS pipeline:

    Stage 1 (Detect) -- ptc.steering.detector  (Phase 2E)
    Stage 2 (Locate) -- ptc.steering.locate    (Phase 2B)
    Stage 3 (Steer)  -- ptc.steering.steer     (Phase 2C)
    Oracle eval      -- ptc.steering.eval      (Phase 2D)

All modules rely on the forward-hook utilities in ptc.steering.hooks.
"""

__all__: list[str] = []
