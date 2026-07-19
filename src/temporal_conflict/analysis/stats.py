"""Post-hoc statistical analysis for TAS evaluation outputs (checklist item 4).

This module is model-free: it reads the per-instance records already written by
the Phase 2F evaluation (``tas_eval_tau*.json``) and produces, for every model
and threshold:

  * bootstrap confidence intervals (percentile, B = 10,000, seeded) on
    Recovery (verified-PTC subset) and Preservation Accuracy (non-conflict
    control subsets), and
  * paired significance tests comparing TAS against the standard-prompt and
    temporal-prompt baselines on the *same* PTC instances (exact McNemar via
    the binomial test on discordant pairs).

The point is reproducibility: the confidence intervals quoted in the paper are
regenerated from the released per-instance decisions by a seeded routine rather
than transcribed by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import binomtest

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 0
CI_ALPHA = 0.05  # -> 95% percentile interval


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    *,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = CI_ALPHA,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(point_estimate, ci_low, ci_high)``. Empty input -> all zeros.
    Resampling is over the first axis (one row = one instance decision), so
    this works for a 0/1 recovery indicator or a continuous per-instance
    quantity alike.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    # Vectorised: draw a (b, n) index matrix once.
    idx = rng.integers(0, n, size=(b, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return point, lo, hi


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact McNemar test on two paired binary decision vectors.

    ``a`` and ``b`` are 0/1 outcomes (e.g. "recovered the new fact") for the
    same instances under two methods. We count discordant pairs and run a
    two-sided exact binomial test (p = 0.5) on them. This is preferred over
    the chi-square approximation because the discordant counts here are small.

    Returns the discordant counts, the two-sided p-value, and the paired
    accuracy difference ``mean(a) - mean(b)``.
    """
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    assert a_arr.shape == b_arr.shape, "paired vectors must align"
    n10 = int(np.sum((a_arr == 1) & (b_arr == 0)))  # a wins
    n01 = int(np.sum((a_arr == 0) & (b_arr == 1)))  # b wins
    discordant = n10 + n01
    if discordant == 0:
        p = 1.0
    else:
        p = float(binomtest(n10, discordant, 0.5, alternative="two-sided").pvalue)
    n = a_arr.shape[0]
    return {
        "n_pairs": n,
        "a_wins": n10,
        "b_wins": n01,
        "discordant": discordant,
        "delta_mean": (float(a_arr.mean()) - float(b_arr.mean())) if n else 0.0,
        "p_value": p,
    }


# ---------------------------------------------------------------------------
# Deriving per-instance decisions from a tas_eval record
# ---------------------------------------------------------------------------


def _prefers_new(rec: dict, old_key: str, new_key: str) -> int:
    return int(rec[new_key] > rec[old_key])


@dataclass
class ModelThresholdStats:
    model: str
    tau: float
    alpha: float
    n_ptc: int
    n_non_ptc: int
    n_non_ptc_clean: int
    recovery_tas: tuple[float, float, float]
    recovery_standard: tuple[float, float, float]
    recovery_temporal: tuple[float, float, float]
    pa_non_ptc: tuple[float, float, float]
    pa_non_ptc_clean: tuple[float, float, float]
    tas_vs_standard: dict = field(default_factory=dict)
    tas_vs_temporal: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def ci(t: tuple[float, float, float]) -> dict:
            return {"point": t[0], "ci_low": t[1], "ci_high": t[2]}

        return {
            "model": self.model,
            "tau": self.tau,
            "alpha": self.alpha,
            "n_ptc": self.n_ptc,
            "n_non_ptc": self.n_non_ptc,
            "n_non_ptc_clean": self.n_non_ptc_clean,
            "recovery_tas": ci(self.recovery_tas),
            "recovery_standard": ci(self.recovery_standard),
            "recovery_temporal": ci(self.recovery_temporal),
            "pa_non_ptc": ci(self.pa_non_ptc),
            "pa_non_ptc_clean": ci(self.pa_non_ptc_clean),
            "tas_vs_standard_mcnemar": self.tas_vs_standard,
            "tas_vs_temporal_mcnemar": self.tas_vs_temporal,
        }


def analyze_tas_eval(path: str | Path) -> ModelThresholdStats:
    """Compute CIs and paired tests from a single ``tas_eval_tau*.json``."""
    payload = json.loads(Path(path).read_text())
    per = payload["per_instance"]
    cfg = payload["summary"]["tas_config"]

    ptc = [r for r in per if r["is_ptc_label"]]
    non_ptc = [r for r in per if not r["is_ptc_label"]]
    # "clean" = non-PTC records that are also knowledge-present for the model.
    non_ptc_clean = [r for r in non_ptc if not r.get("knowledge_absent", False)]

    # Recovery indicators on the PTC subset (paired across methods).
    rec_tas = [_prefers_new(r, "old_lp_tas", "new_lp_tas") for r in ptc]
    rec_std = [_prefers_new(r, "old_lp_std_baseline", "new_lp_std_baseline") for r in ptc]
    rec_tmp = [_prefers_new(r, "old_lp_temporal_baseline", "new_lp_temporal_baseline") for r in ptc]

    # Preservation: TAS argmax matches the standard-baseline argmax.
    def preserved(rows: list[dict]) -> list[int]:
        return [
            int((r["new_lp_tas"] > r["old_lp_tas"])
                 == (r["new_lp_std_baseline"] > r["old_lp_std_baseline"]))
            for r in rows
        ]

    pa_all = preserved(non_ptc)
    pa_clean = preserved(non_ptc_clean)

    return ModelThresholdStats(
        model=cfg_model(payload),
        tau=float(cfg["tau"]),
        alpha=float(cfg["alpha"]),
        n_ptc=len(ptc),
        n_non_ptc=len(non_ptc),
        n_non_ptc_clean=len(non_ptc_clean),
        recovery_tas=bootstrap_ci(rec_tas),
        recovery_standard=bootstrap_ci(rec_std),
        recovery_temporal=bootstrap_ci(rec_tmp),
        pa_non_ptc=bootstrap_ci(pa_all),
        pa_non_ptc_clean=bootstrap_ci(pa_clean),
        tas_vs_standard=mcnemar_exact(rec_tas, rec_std),
        tas_vs_temporal=mcnemar_exact(rec_tas, rec_tmp),
    )


def cfg_model(payload: dict) -> str:
    return payload["summary"].get("model", "unknown")


# ---------------------------------------------------------------------------
# Driver over a results directory
# ---------------------------------------------------------------------------


def find_tas_evals(results_tas_dir: Path) -> list[Path]:
    """All ``tas_eval_tau*.json`` under ``<results_tas_dir>/<model>/``."""
    return sorted(results_tas_dir.glob("*/tas_eval_tau*.json"))


def run_stats(results_tas_dir: str | Path, out_path: str | Path | None = None) -> dict:
    """Analyze every tas_eval file under ``results_tas_dir`` and return a nested
    ``{model: {tau: stats}}`` dict; optionally write it as JSON.
    """
    results_tas_dir = Path(results_tas_dir)
    files = find_tas_evals(results_tas_dir)
    if not files:
        raise FileNotFoundError(f"No tas_eval_tau*.json under {results_tas_dir}")

    out: dict[str, dict[str, dict]] = {}
    for f in files:
        s = analyze_tas_eval(f)
        out.setdefault(s.model, {})[f"{s.tau:.2f}"] = s.to_dict()
        print(
            f"{s.model:16s} tau={s.tau:.2f}  "
            f"Rec(TAS)={s.recovery_tas[0]:.3f} "
            f"[{s.recovery_tas[1]:.3f},{s.recovery_tas[2]:.3f}]  "
            f"PA={s.pa_non_ptc[0]:.3f} "
            f"[{s.pa_non_ptc[1]:.3f},{s.pa_non_ptc[2]:.3f}]  "
            f"TAS>std p={s.tas_vs_standard['p_value']:.1e}",
            flush=True,
        )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {out_path}", flush=True)
    return out


def markdown_table(stats: dict, taus: Iterable[float] | None = None) -> str:
    """Compact per-(model, tau) markdown table of the reproducible CIs."""
    lines = [
        "| Model | tau | Recovery (TAS) 95% CI | PA (non-PTC) 95% CI | "
        "TAS vs standard (McNemar p) |",
        "|---|---|---|---|---|",
    ]
    for model, by_tau in stats.items():
        for tau_str, s in sorted(by_tau.items()):
            if taus is not None and float(tau_str) not in set(taus):
                continue
            rt = s["recovery_tas"]
            pa = s["pa_non_ptc"]
            mc = s["tas_vs_standard_mcnemar"]
            lines.append(
                f"| {model} | {tau_str} | "
                f"{rt['point']:.3f} [{rt['ci_low']:.3f}, {rt['ci_high']:.3f}] | "
                f"{pa['point']:.3f} [{pa['ci_low']:.3f}, {pa['ci_high']:.3f}] | "
                f"{mc['delta_mean']:+.3f}, p={mc['p_value']:.1e} |"
            )
    return "\n".join(lines)
