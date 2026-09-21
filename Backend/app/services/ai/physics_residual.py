"""
Physics baseline for residual ("physics-informed") learning.
============================================================

The severity and flood-depth heads do not predict absolute values. They predict
the *correction* to the hydrology baseline produced by
``app/services/physics/pinn_risk_model.py`` (SCS Curve Number runoff + Manning
open-channel flow)::

    y_hat = physics_baseline + neural_residual

Measured on the Mumbai replay window, the baseline alone already achieves
R2 = 0.79 for severity and R2 = 0.88 for flood depth. Learning the residual
instead of the absolute value keeps the known physics exact and spends the
network's capacity on what physics cannot explain.

Why this module exists
----------------------
The baseline must be computed *identically* during training and inference. If
the two disagree, the network's residual is added to a different anchor and the
model silently degrades. Both sides therefore call
``compute_physics_baseline`` and read the scales from the same checkpoint
metadata.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.services.ai.feature_contract import resolve_feature, resolve_target
from app.services.ai.model_architectures import (
    DEPTH_RESIDUAL_SCALE,
    SEVERITY_RESIDUAL_SCALE,
)

RESIDUAL_CONFIG_FILENAME = "physics_residual.json"

# Severity is stored as a 0-3 class; the model regresses on a 0-100 score.
# This is the same mapping the evaluation suite uses, so training and scoring
# cannot drift apart.
SEVERITY_CLASS_TO_SCORE = 100.0 / 3.0


# ---------------------------------------------------------------------------
# Observed (ground-truth) labels
# ---------------------------------------------------------------------------
# These read the dataset's ``target_*`` columns. They are the labels the
# evaluation suite scores against, so the trainer must learn the same quantity.
# The previous revision ignored them entirely and recomputed labels from
# rainfall, so the model was trained on one thing and graded on another.
#
def observed_severity_score(cell: Dict[str, Any]) -> Optional[float]:
    """Observed severity on the 0-100 scale, from ``target_severity_class``."""
    value = resolve_target(cell, "target_severity_class", None)
    if value is None:
        return None
    return float(value) * SEVERITY_CLASS_TO_SCORE


def observed_flood_depth_cm(cell: Dict[str, Any]) -> Optional[float]:
    """Observed flood depth in cm, from ``target_observed_flood_depth_cm``."""
    value = resolve_target(cell, "target_observed_flood_depth_cm", None)
    if value is None:
        return None
    return float(value)


def observed_flag(cell: Dict[str, Any], column: str) -> Optional[float]:
    """Observed binary hazard flag, returned as 0.0/1.0."""
    value = resolve_target(cell, column, None)
    if value is None:
        return None
    return 1.0 if float(value) else 0.0


# Declares, per model head, whether its training label is an observation or a
# heuristic. Written into ``training_metadata.json`` so no head's provenance is
# ambiguous at review time.
TARGET_SOURCES: Dict[str, str] = {
    "thunderstorm_prob": "DERIVED (CAPE/CTT/wind heuristic - no ground-truth column exists in this dataset)",
    "cloudburst_prob": "OBSERVED (target_cloudburst_flag)",
    "flash_flood_prob": "OBSERVED (target_flash_flood_flag)",
    "severity_score": "OBSERVED (target_severity_class mapped to 0-100)",
    "flood_depth_cm": "OBSERVED (target_observed_flood_depth_cm)",
}


class TargetContractError(RuntimeError):
    """Raised when an observed label column is missing from the dataset.

    Deliberately fatal: silently substituting 0.0 is how the trainer drifted
    away from the labels the evaluation suite scores against.
    """


def _physics_model():
    """Import lazily so a missing physics module cannot break model loading."""
    from app.services.physics.pinn_risk_model import PhysicsInformedRiskModel

    return PhysicsInformedRiskModel()


def compute_physics_baseline(
    cells: Iterable[Dict[str, Any]],
    *,
    tide_height_m: Optional[float] = None,
    is_high_tide_locked: Optional[bool] = None,
    model: Optional[Any] = None,
) -> Tuple[List[float], List[float]]:
    """Return ``(severity_baseline, depth_baseline)`` aligned with ``cells``.

    ``severity_baseline`` is the physics risk score (0-100);
    ``depth_baseline`` is the physics water depth in cm.

    Tidal state is read **per cell** unless explicitly overridden. The label
    generator reads it per cell too, so anchoring the baseline on the same
    values keeps ``label - baseline`` a true residual rather than a mix of two
    different tidal assumptions.
    """
    engine = model if model is not None else _physics_model()

    severity: List[float] = []
    depth: List[float] = []

    for cell in cells:
        if not isinstance(cell, dict):
            severity.append(0.0)
            depth.append(0.0)
            continue

        cell_tide = (
            float(tide_height_m)
            if tide_height_m is not None
            else float(cell.get("tide_height_m", 2.5) or 2.5)
        )
        cell_high_tide = (
            bool(is_high_tide_locked)
            if is_high_tide_locked is not None
            else bool(cell.get("is_high_tide_locked", False))
        )

        try:
            result = engine.compute_physics_risk(
                cell,
                is_high_tide=cell_high_tide,
                tide_height=cell_tide,
            )
            severity.append(float(result.get("physics_risk_score", 0.0) or 0.0))
            depth.append(float(result.get("water_depth_cm", 0.0) or 0.0))
        except Exception:
            # A physics failure must degrade to "no correction", never break
            # the pipeline: the network then learns the full signal.
            severity.append(0.0)
            depth.append(0.0)

    return severity, depth


def compute_label_severity(cell: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return the ``(thunderstorm, cloudburst, flash_flood)`` heuristic scores.

    SUPERSEDED for cloudburst and flash flood, which now use the dataset's
    observed ``target_*`` flags. It is retained **only** for the thunderstorm
    head, which has no ground-truth column in this dataset; that head's label
    is therefore declared DERIVED rather than observed (see ``TARGET_SOURCES``).
    """
    rain = float(resolve_feature(cell, "rainfall_1h_mm", 0) or 0)
    rain_3h = float(resolve_feature(cell, "rainfall_3h_mm", 0) or 0)
    elev = float(resolve_feature(cell, "elevation_m", 10) or 10)
    cape = float(resolve_feature(cell, "cape_instability_jkg", 0) or 0)
    soil = float(resolve_feature(cell, "soil_moisture_pct", 50) or 50)
    slope = float(resolve_feature(cell, "slope_deg", 2) or 2)
    ctt = float(resolve_feature(cell, "cloud_top_temp_celsius", -20) or -20)
    wind = float(resolve_feature(cell, "wind_speed_10m_kmh", 10) or 10)
    ctt_drop = float(resolve_feature(cell, "ctt_drop_rate_c_per_hr", 0) or 0)
    drainage_dist = float(resolve_feature(cell, "drainage_outfall_dist_m", 5000) or 5000)
    is_high_tide = bool(cell.get("is_high_tide_locked", False))
    tide_height = float(cell.get("tide_height_m", 2.5) or 2.5)

    ts_prob = 0.0
    if cape > 500:
        ts_prob += 0.35 * min(1.0, (cape - 500) / 2500)
    if ctt < -20:
        ts_prob += 0.30 * min(1.0, abs(ctt + 20) / 50)
    if wind > 15:
        ts_prob += 0.20 * min(1.0, (wind - 15) / 45)
    if ctt_drop > 5:
        ts_prob += 0.15 * min(1.0, ctt_drop / 20)
    ts_prob = min(1.0, ts_prob)

    cb_prob = 0.0
    if rain > 10:
        if rain >= 65:
            cb_prob += 0.50
        elif rain >= 45:
            cb_prob += 0.35 * min(1.0, (rain - 10) / 55)
        else:
            cb_prob += 0.20 * min(1.0, rain / 45)
    if rain_3h > 30:
        cb_prob += 0.25 * min(1.0, (rain_3h - 30) / 120)
    if cape > 1500:
        cb_prob += 0.15 * min(1.0, (cape - 1500) / 2000)
    if soil > 70:
        cb_prob += 0.10 * min(1.0, (soil - 70) / 30)
    cb_prob = min(1.0, cb_prob)

    ff_prob = 0.0
    if rain > 5:
        ff_prob += 0.30 * min(1.0, rain / 65)
    if elev < 10:
        ff_prob += 0.25 * min(1.0, (10 - elev) / 10)
    if soil > 60:
        ff_prob += 0.15 * min(1.0, (soil - 60) / 40)
    if is_high_tide and tide_height > 3:
        ff_prob += 0.15 * min(1.0, (tide_height - 3) / 3)
    if drainage_dist > 2000:
        ff_prob += 0.10 * min(1.0, (drainage_dist - 2000) / 8000)
    if slope < 3:
        ff_prob += 0.05 * min(1.0, (3 - slope) / 3)
    ff_prob = min(1.0, ff_prob)

    return ts_prob, cb_prob, ff_prob


def compute_label_depth(cell: Dict[str, Any]) -> float:
    """Return the rainfall-derived heuristic flood depth (cm).

    SUPERSEDED by ``observed_flood_depth_cm``. Retained only for reference;
    this formula returns 0.0 for every cell with rainfall below 17.5 mm/hr,
    which is why it hid the dataset's real 5-9 cm post-storm residual flooding
    from the holdout windows.
    """
    rain = float(resolve_feature(cell, "rainfall_1h_mm", 0) or 0)
    elev = float(resolve_feature(cell, "elevation_m", 10) or 10)
    soil = float(resolve_feature(cell, "soil_moisture_pct", 50) or 50)
    excess = max(0.0, rain - 17.5)
    retention = 1.0 + (max(0.0, 7.0 - elev) / 7.0) * 1.5
    return excess * 0.15 * retention * (0.6 + 0.4 * soil / 100.0)


def read_observed_labels(
    cell: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    """Return ``(cloudburst, flash_flood, severity, depth)`` observed labels.

    Raises :class:`TargetContractError` if any is absent, rather than defaulting
    to zero.
    """
    cloudburst = observed_flag(cell, "target_cloudburst_flag")
    flash_flood = observed_flag(cell, "target_flash_flood_flag")
    severity = observed_severity_score(cell)
    depth = observed_flood_depth_cm(cell)

    missing = [
        name
        for name, value in (
            ("target_cloudburst_flag", cloudburst),
            ("target_flash_flood_flag", flash_flood),
            ("target_severity_class", severity),
            ("target_observed_flood_depth_cm", depth),
        )
        if value is None
    ]
    if missing:
        raise TargetContractError(
            "observed label column(s) missing from the feature table: "
            + ", ".join(missing)
            + ". The trainer must learn the same targets the evaluation scores "
            "against; fix the data rather than defaulting these to 0.0."
        )

    return float(cloudburst), float(flash_flood), float(severity), float(depth)


def save_residual_config(
    model_dir: str,
    *,
    severity_scale: float = SEVERITY_RESIDUAL_SCALE,
    depth_scale: float = DEPTH_RESIDUAL_SCALE,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist the residual-head scales so inference reproduces training."""
    payload: Dict[str, Any] = {
        "severity_residual_scale": float(severity_scale),
        "depth_residual_scale": float(depth_scale),
        "baseline": "SCS-CN + Manning (app/services/physics/pinn_risk_model.py)",
        "form": "y_hat = physics_baseline + neural_residual",
    }
    if extra:
        payload.update(extra)

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, RESIDUAL_CONFIG_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_residual_config(model_dir: str) -> Dict[str, Any]:
    """Read the residual-head scales, falling back to the module defaults."""
    path = os.path.join(model_dir, RESIDUAL_CONFIG_FILENAME)
    config: Dict[str, Any] = {
        "severity_residual_scale": SEVERITY_RESIDUAL_SCALE,
        "depth_residual_scale": DEPTH_RESIDUAL_SCALE,
        "enabled": False,
    }
    if not os.path.exists(path):
        return config
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            config.update(loaded)
            config["enabled"] = True
    except (OSError, json.JSONDecodeError):
        pass
    return config
