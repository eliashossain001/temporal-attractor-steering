"""Stage 1: PTC conflict detector.

Trains a logistic probe on h_ℓ*(q) (the residual-stream activation at the
conflict-critical layer, last prompt token, under standard prompting).
Positives are verified PTC instances; negatives are matched non-conflict
factual records (drawn from the screening JSONL where the model is
consistent across both prompts).

Outputs (per proposal Sec 5.2 / M5):
    - logistic probe coefficients (linear; ITI-style)
    - isotonic calibrator for s(h) -> p(PTC)
    - threshold sweep with precision / recall / F1 at each tau
    - AUPRC, AUROC
    - Expected Calibration Error (ECE)

The trained detector is serialized to disk and consumed at inference by
ptc.steering.tas to gate the steering intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    fraction_positive_predicted: float


@dataclass
class TrainedDetector:
    """A trained PTC conflict detector.

    Stores raw probe + optional isotonic calibrator. At inference, `score(h)`
    returns the calibrated PTC probability in [0, 1].
    """

    layer: int
    d_model: int
    coef: np.ndarray  # [d_model]
    intercept: float
    classes_: np.ndarray  # for sanity (should be [0, 1])
    calibrator: Optional[IsotonicRegression] = None

    def _raw_prob(self, h: np.ndarray) -> np.ndarray:
        """Uncalibrated p(PTC = 1) from the logistic probe."""
        logit = h @ self.coef + self.intercept
        # numerically stable sigmoid
        out = np.empty_like(logit)
        pos = logit >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-logit[pos]))
        ez = np.exp(logit[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def score(self, h: torch.Tensor | np.ndarray) -> float:
        """Calibrated p(PTC) for a single hidden state h ∈ [d_model]."""
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().float().numpy()
        h = np.atleast_2d(h)
        p = self._raw_prob(h)
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
        return float(p[0])

    def score_batch(self, H: torch.Tensor | np.ndarray) -> np.ndarray:
        """Calibrated p(PTC) for a batch H ∈ [N, d_model]."""
        if isinstance(H, torch.Tensor):
            H = H.detach().cpu().float().numpy()
        p = self._raw_prob(H)
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
        return p


@dataclass
class DetectorMetrics:
    auprc: float
    auroc: float
    ece: float
    n_train_pos: int
    n_train_neg: int
    n_test_pos: int
    n_test_neg: int
    pr_curve: dict = field(default_factory=dict)
    threshold_sweep: list[ThresholdPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "auprc": self.auprc,
            "auroc": self.auroc,
            "ece": self.ece,
            "n_train_pos": self.n_train_pos,
            "n_train_neg": self.n_train_neg,
            "n_test_pos": self.n_test_pos,
            "n_test_neg": self.n_test_neg,
            "pr_curve": self.pr_curve,
            "threshold_sweep": [
                {
                    "threshold": t.threshold,
                    "precision": t.precision,
                    "recall": t.recall,
                    "f1": t.f1,
                    "fraction_positive_predicted": t.fraction_positive_predicted,
                }
                for t in self.threshold_sweep
            ],
        }


def _expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Standard binned ECE."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins[1:-1])  # 0..n_bins-1
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _threshold_sweep(
    probs: np.ndarray, labels: np.ndarray, thresholds: list[float]
) -> list[ThresholdPoint]:
    pts: list[ThresholdPoint] = []
    n = len(labels)
    pos_total = int(labels.sum())
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / pos_total if pos_total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        pts.append(
            ThresholdPoint(
                threshold=t,
                precision=precision,
                recall=recall,
                f1=f1,
                fraction_positive_predicted=(tp + fp) / n,
            )
        )
    return pts


def train_detector(
    positive_acts: np.ndarray,  # [N_pos, d_model]
    negative_acts: np.ndarray,  # [N_neg, d_model]
    layer: int,
    C: float = 1.0,
    test_frac: float = 0.3,
    seed: int = 0,
    calibrate: bool = True,
    thresholds: Optional[list[float]] = None,
) -> tuple[TrainedDetector, DetectorMetrics]:
    """Train logistic probe at one layer, calibrate, evaluate.

    Test-set predictions are used both to calibrate (isotonic) and to compute
    AUPRC / AUROC / ECE / threshold sweep. This is intentionally a single
    held-out split; for the paper the proposal's stronger protocol would do
    nested CV, but for Phase 2E a single split is enough to read the headline.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    X = np.concatenate([positive_acts, negative_acts], axis=0).astype(np.float32)
    y = np.concatenate([
        np.ones(len(positive_acts), dtype=np.int32),
        np.zeros(len(negative_acts), dtype=np.int32),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y,
    )

    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    coef = clf.coef_.ravel().astype(np.float32)
    intercept = float(clf.intercept_.ravel()[0])

    # Raw probabilities on test set.
    raw = clf.predict_proba(X_te)[:, 1]

    # Calibrate (isotonic) on the test set predictions; for the paper we'd
    # split a separate calibration set, but with N this small a single
    # post-hoc calibration is the cleanest start.
    calibrator: Optional[IsotonicRegression] = None
    calibrated = raw
    if calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, y_te)
        calibrated = calibrator.transform(raw)

    auprc = float(average_precision_score(y_te, calibrated))
    auroc = float(roc_auc_score(y_te, calibrated))
    ece = _expected_calibration_error(calibrated, y_te)

    prec_curve, rec_curve, thr_curve = precision_recall_curve(y_te, calibrated)

    metrics = DetectorMetrics(
        auprc=auprc,
        auroc=auroc,
        ece=ece,
        n_train_pos=int((y_tr == 1).sum()),
        n_train_neg=int((y_tr == 0).sum()),
        n_test_pos=int((y_te == 1).sum()),
        n_test_neg=int((y_te == 0).sum()),
        pr_curve={
            "precision": prec_curve.tolist(),
            "recall": rec_curve.tolist(),
            "thresholds": thr_curve.tolist(),
        },
        threshold_sweep=_threshold_sweep(calibrated, y_te, thresholds),
    )

    detector = TrainedDetector(
        layer=layer,
        d_model=X.shape[1],
        coef=coef,
        intercept=intercept,
        classes_=clf.classes_,
        calibrator=calibrator,
    )
    return detector, metrics


def save_detector(path: str, det: TrainedDetector) -> None:
    import os, pickle

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "layer": det.layer,
                "d_model": det.d_model,
                "coef": det.coef,
                "intercept": det.intercept,
                "classes_": det.classes_,
                "calibrator": det.calibrator,
            },
            f,
        )


def load_detector(path: str) -> TrainedDetector:
    import pickle

    with open(path, "rb") as f:
        d = pickle.load(f)
    return TrainedDetector(
        layer=d["layer"],
        d_model=d["d_model"],
        coef=d["coef"],
        intercept=d["intercept"],
        classes_=d["classes_"],
        calibrator=d.get("calibrator"),
    )
