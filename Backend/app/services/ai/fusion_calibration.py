"""
Calibration for the cross-source agreement anomaly threshold.

``score_forecast_trust`` flags ``low_cross_source_agreement`` when the fusion
module's ``global_agreement_score`` drops below a threshold. That threshold was
a bare literal (0.6), and measured against real runs it could never be met: the
score ranged 0.28-0.58 across all 72 timesteps, so every single timestep was
reported as anomalous and the flag carried no information.

The reason is structural, not a bug in the score itself. Agreement is the
complement of the spread between four signals that measure different things --
a static terrain/drainage term (roughly a constant per cell), an instability
term, a satellite term and an observed-rainfall term. Their natural magnitudes
differ, so a large spread is the normal state, and comparing against a fixed
value on the wrong scale guarantees a constant verdict.

This module derives the threshold from the **calibration window** -- the
partition that exists precisely for fitting calibration constants, and which is
disjoint from both training and the held-out test set. The threshold is the low
percentile of the agreement distribution on that window, so "anomalous" means
"materially less agreement than is typical", which is what the flag claims.
"""

import os
import json
import logging
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger("VARUNA.FusionCalibration")

FUSION_CALIBRATION_FILENAME = "fusion_calibration.json"

# Used only when no calibration artifact exists (e.g. a fresh clone that has not
# trained yet). Kept at the historical value so behaviour is unchanged until a
# calibration has actually been fitted.
DEFAULT_AGREEMENT_ANOMALY_THRESHOLD = 0.6

# "Materially unusual" is read as the lowest decile of the calibration window.
DEFAULT_ANOMALY_PERCENTILE = 10.0


def _percentile(values: Iterable[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile, so no scipy dependency is needed."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def calibrate_agreement_threshold(
    agreement_scores: Iterable[float],
    percentile: float = DEFAULT_ANOMALY_PERCENTILE,
) -> Dict[str, Any]:
    """Derive the anomaly threshold from a calibration window's scores.

    Returns the threshold plus the statistics it came from, so the choice is
    auditable rather than a magic number.
    """
    scores = [float(s) for s in agreement_scores]
    if not scores:
        return {
            "agreement_anomaly_threshold": DEFAULT_AGREEMENT_ANOMALY_THRESHOLD,
            "source": "default (no calibration samples)",
            "n_calibration_samples": 0,
        }

    threshold = _percentile(scores, percentile)
    ordered = sorted(scores)
    return {
        "agreement_anomaly_threshold": round(float(threshold), 4),
        "percentile": percentile,
        "n_calibration_samples": len(scores),
        "calibration_min": round(ordered[0], 4),
        "calibration_max": round(ordered[-1], 4),
        "calibration_median": round(_percentile(scores, 50.0), 4),
        "source": "calibration window",
        "method": (
            f"lowest {percentile:g}th percentile of global_agreement_score on the "
            "calibration window; 'anomalous' means materially less agreement than "
            "is typical for this event"
        ),
    }


def save_fusion_calibration(model_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, FUSION_CALIBRATION_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_fusion_calibration(model_dir: str) -> Dict[str, Any]:
    """Load the persisted threshold, falling back to the documented default."""
    path = os.path.join(model_dir, FUSION_CALIBRATION_FILENAME)
    if not os.path.exists(path):
        return {
            "agreement_anomaly_threshold": DEFAULT_AGREEMENT_ANOMALY_THRESHOLD,
            "source": "default (no calibration artifact found)",
        }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read fusion calibration (%s); using default.", exc)
        return {
            "agreement_anomaly_threshold": DEFAULT_AGREEMENT_ANOMALY_THRESHOLD,
            "source": "default (unreadable calibration artifact)",
        }

    payload.setdefault(
        "agreement_anomaly_threshold", DEFAULT_AGREEMENT_ANOMALY_THRESHOLD
    )
    return payload
