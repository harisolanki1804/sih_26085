"""Evaluation metrics endpoints.

The four-metric-family report is produced by ``evaluation/run_evaluation.py``
and frozen as a regression baseline. Until now nothing served it, so the
dashboard could not show real skill numbers -- the frontend's skill panel had
them hardcoded. These endpoints expose the artifacts as they are, and add one
flat shape (``/metrics/skill``) that a chart can consume directly.

Everything here is read-only and reports the artifacts' own timestamps, so a
stale report is visible rather than silently presented as current.

Undefined metrics are passed through as ``null`` with the producer's
``undefined_reason``. A hazard head with no positive events in the scored
window has no defined POD/CSI, and rendering those as 0% would be a claim the
data does not support.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

router = APIRouter()

# .../Backend/app/api/v1/endpoints/metrics.py -> Backend/
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_EVALUATION_DIR = os.path.join(_BACKEND_DIR, "evaluation")

FOUR_FAMILY_PATH = os.path.join(_EVALUATION_DIR, "four_family_evaluation_results.json")
NOWCAST_PATH = os.path.join(_EVALUATION_DIR, "nowcast_accuracy.json")
BASELINE_PATH = os.path.join(_EVALUATION_DIR, "baseline_metrics.json")
PPT_READY_PATH = os.path.join(_EVALUATION_DIR, "PPT_READY_METRICS.txt")


def _load(path: str, label: str) -> Dict[str, Any]:
    """Load a JSON artifact, or fail loudly with how to produce it."""
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"{label} not found at {os.path.relpath(path, _BACKEND_DIR)}. "
                "Generate it with: python -m evaluation.run_evaluation "
                "--freeze-baseline (and python evaluation/evaluate_nowcast.py "
                "for the nowcast report)."
            ),
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"{label} unreadable: {exc}") from exc
    return payload


def _stamp(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat()


@router.get("/evaluation", summary="Four-family evaluation report (frozen split)")
def get_evaluation():
    """Verbatim four-metric-family report on the held-out test partition.

    Families: classification (POD/FAR/CSI per hazard per lead time), regression
    (RMSE/MAE/R2), probabilistic (Brier/reliability/conformal) and operational.
    """
    report = _load(FOUR_FAMILY_PATH, "four-family evaluation report")
    return {
        "artifact": os.path.basename(FOUR_FAMILY_PATH),
        "artifact_modified_utc": _stamp(FOUR_FAMILY_PATH),
        "report": report,
    }


@router.get("/nowcast", summary="6-hour nowcast accuracy (held-out split)")
def get_nowcast():
    """Per-lead-time nowcast error, with the persistence reference.

    ``inference_modes`` records which code path produced each forecast, so a
    claim of "ConvLSTM nowcast" is checkable rather than assumed.
    """
    report = _load(NOWCAST_PATH, "nowcast accuracy report")
    return {
        "artifact": os.path.basename(NOWCAST_PATH),
        "artifact_modified_utc": _stamp(NOWCAST_PATH),
        "report": report,
    }


@router.get("/skill", summary="Model skill, flat shape for dashboards")
def get_skill():
    """Chart-ready summary of the evaluation artifacts.

    Exists so a UI does not have to walk the four-family report or hardcode
    numbers. Every value is read from the artifacts at request time.
    """
    four = _load(FOUR_FAMILY_PATH, "four-family evaluation report")

    hazard_skill: List[Dict[str, Any]] = []
    for hazard, leads in (four.get("family1_classification") or {}).items():
        for lead, row in (leads or {}).items():
            hazard_skill.append(
                {
                    "hazard": hazard,
                    "lead_time": lead,
                    "pod": row.get("POD"),
                    "far": row.get("FAR"),
                    "csi": row.get("CSI"),
                    "accuracy": row.get("Accuracy"),
                    "positives": row.get("positives"),
                    "defined": row.get("defined"),
                    "undefined_reason": row.get("undefined_reason"),
                }
            )

    nowcast: Optional[Dict[str, Any]] = None
    if os.path.exists(NOWCAST_PATH):
        try:
            with open(NOWCAST_PATH, "r", encoding="utf-8") as handle:
                nowcast_raw = json.load(handle)
            nowcast = {
                "scored_timesteps": nowcast_raw.get("scored_timesteps"),
                "evaluated_timesteps": nowcast_raw.get("evaluated_timesteps"),
                "per_lead_time": nowcast_raw.get("per_lead_time"),
                "inference_modes": nowcast_raw.get("inference_modes"),
                "neural_nowcast_used": nowcast_raw.get("neural_nowcast_used"),
                "persistence_reference": nowcast_raw.get("persistence_reference"),
                "mode_note": nowcast_raw.get("mode_note"),
            }
        except (OSError, json.JSONDecodeError):
            nowcast = None

    fam3 = four.get("family3_probabilistic") or {}
    return {
        "split": four.get("split"),
        "generated_at": _stamp(FOUR_FAMILY_PATH),
        "hazard_skill": hazard_skill,
        "regression": four.get("family2_regression"),
        "probabilistic": {
            "aggregate": fam3.get("aggregate"),
            "per_head": fam3.get("per_head"),
            "conformal_per_target": fam3.get("conformal_per_target"),
            "conformal_alpha": fam3.get("conformal_alpha"),
        },
        "operational": four.get("family4_operational"),
        "nowcast": nowcast,
        "notes": [
            "A hazard head with no positive events in the scored window reports "
            "pod/csi as null with an undefined_reason; that is not a 0% score.",
            "Nowcast skill is reported against persistence on the same held-out "
            "rows; check inference_modes before attributing a forecast to the "
            "neural nowcaster.",
        ],
    }


@router.get("/baseline", summary="Frozen regression baseline and drift status")
def get_baseline():
    """The frozen metrics the run is compared against, plus artifact presence."""
    baseline = _load(BASELINE_PATH, "frozen baseline")
    return {
        "artifact": os.path.basename(BASELINE_PATH),
        "artifact_modified_utc": _stamp(BASELINE_PATH),
        "frozen_at": baseline.get("frozen_at"),
        "split": baseline.get("split"),
        "tolerance": baseline.get("tolerance"),
        "tracked_metric_count": len(baseline.get("metrics") or {}),
        "metrics": baseline.get("metrics"),
        "ppt_ready_report_available": os.path.exists(PPT_READY_PATH),
    }
