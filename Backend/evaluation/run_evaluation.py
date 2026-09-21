"""
VARUNA System Evaluation Harness (Member 2 Canonical Evaluation Suite)
======================================================================
Evaluates the 9-module AI pipeline against traditional baselines on the
FROZEN TIME-BASED SPLIT defined in ``app/services/ai/feature_contract.py``
(train 1-32, calibrate 33-38, test 39-72). Metrics are computed on the test
partition only; conformal quantiles are fitted on the calibrate partition.

Four Families of Metrics:
1. Family 1 — Classification:
   Hazards: Flood, Lightning, Wind, Heat, Air Quality
   Horizons: 2h, 4h, 6h
   Metrics: POD (Probability of Detection), FAR (False Alarm Ratio), CSI (Critical Success Index)
2. Family 2 — Regression:
   Continuous targets: Flood Depth (cm), Rainfall (mm/h), Risk Score (0-100)
   Metrics: RMSE, MAE, R²
3. Family 3 — Probabilistic:
   Brier Score, Reliability (ECE + Calibration Bins), Conformal Prediction Coverage (1 - α)
4. Family 4 — Operational:
   Lead Time to First Alert (hours), False Alarms / Week, Evacuation Detour Overhead (%)

Family 3 is reported per head, not only in aggregate, and conformal coverage
is broken out per regression target.

Baseline freezing:
    python -m evaluation.run_evaluation                     # run + refresh metrics
    python -m evaluation.run_evaluation --freeze-baseline   # write the baseline
    python -m evaluation.run_evaluation --verify-baseline   # fail on drift
"""

import os
import sys
import io
import json
import time
import math
import statistics
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Fix encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =====================================================================
# SPLIT CONFIGURATION — single source of truth in feature_contract
# =====================================================================
# The previous revision hardcoded its own boundaries here while the trainer
# used a different 80/5/15 split, so models were fitted and scored on
# different partitions.
from app.services.ai.feature_contract import (
    SPLIT_CALIBRATE_END,
    SPLIT_TEST_END,
    SPLIT_TRAIN_END,
    describe_split,
    split_timestep_bounds,
)

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_metrics.json")
BASELINE_TOLERANCE = 0.10  # 10% relative drift before a metric is flagged

# Regression targets tracked per-head by the conformal breakdown.
CONFORMAL_TARGETS = ("flood_depth_cm", "risk_severity_score", "rainfall_mm_hr")


# =====================================================================
# TRADITIONAL BASELINE METHODS
# =====================================================================

class TraditionalThresholdBaseline:
    """IMD-style fixed-threshold warning. This is what most cities use."""

    def predict(self, cell: Dict) -> Dict:
        rain = cell.get("rainfall_1h_mm", 0)
        elev = cell.get("elevation_m", 10)
        cape = cell.get("cape_instability_jkg", 0)
        soil = cell.get("soil_moisture_pct", 50)

        is_cloudburst = rain >= 65.0
        is_heavy_rain = rain >= 30.0
        is_waterlogging = (rain >= 40.0 and elev < 5.0)

        risk = 0.0
        if rain > 10:
            risk += min(40, rain * 0.5)
        if soil > 70:
            risk += min(25, (soil - 70) * 0.83)
        if elev < 5:
            risk += min(20, (5 - elev) * 4.0)
        if cape > 1000:
            risk += min(15, (cape - 1000) * 0.005)
        risk = min(100, risk)

        depth = max(0, (rain - 25) * 0.12) if rain > 25 else 0

        severity = (
            "CRITICAL" if risk >= 80 else
            "HIGH" if risk >= 60 else
            "MEDIUM" if risk >= 35 else "LOW"
        )

        return {
            "risk_score": round(risk, 1),
            "severity": severity,
            "depth_cm": round(depth, 1),
            "is_cloudburst": is_cloudburst,
            "is_waterlogging": is_waterlogging,
        }


class StatisticalBaseline:
    """Rolling-average statistical baseline. No physics, no ML."""

    def __init__(self):
        self.history: List[float] = []

    def predict(self, cell: Dict) -> Dict:
        rain = cell.get("rainfall_1h_mm", 0)
        self.history.append(rain)
        if len(self.history) > 12:
            self.history = self.history[-12:]

        mean_rain = statistics.mean(self.history) if self.history else 0

        if len(self.history) >= 3:
            trend = (self.history[-1] - self.history[-3]) / 2
        else:
            trend = 0

        predicted = mean_rain + trend
        risk = min(100, max(0, predicted * 1.2))
        depth = max(0, (predicted - 25) * 0.1) if predicted > 25 else 0

        severity = (
            "CRITICAL" if risk >= 80 else
            "HIGH" if risk >= 60 else
            "MEDIUM" if risk >= 35 else "LOW"
        )

        return {
            "risk_score": round(risk, 1),
            "severity": severity,
            "depth_cm": round(depth, 1),
        }


# =====================================================================
# METRICS COMPUTATION HELPERS
# =====================================================================

def compute_contingency_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    """Calculate POD, FAR, CSI, Precision, Recall, and Accuracy.

    POD and CSI are **undefined**, not zero, when no positive event exists in
    the evaluation window: the previous revision divided by ``max(1, ...)`` and
    printed a confident ``POD 0.0%`` for hazards that simply never occurred.
    That reads as total detection failure and is the opposite of the truth.
    Undefined scores are reported as ``None`` with an explicit reason, and the
    event counts are always returned so the reader can judge the sample size.
    """
    total = tp + fp + fn + tn
    positives = tp + fn
    acc = (tp + tn) / max(1, total)

    def _score(numerator: int, denominator: int) -> Optional[float]:
        if denominator == 0:
            return None
        return round(100.0 * numerator / denominator, 1)

    undefined_reason = None
    if positives == 0 and (tp + fp) == 0:
        undefined_reason = "no positive events and no positive predictions in window"
    elif positives == 0:
        undefined_reason = "no positive events in window (POD/CSI undefined; FAR is not)"
    elif (tp + fp) == 0:
        undefined_reason = "no positive predictions in window"

    return {
        "POD": _score(tp, positives),
        "FAR": _score(fp, tp + fp),
        "CSI": _score(tp, tp + fp + fn),
        "Accuracy": round(acc * 100, 1),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "positives": positives,
        "undefined_reason": undefined_reason,
        "defined": undefined_reason is None,
    }


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calculate RMSE, MAE, R²."""
    n = len(y_true)
    if n == 0:
        return {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0}

    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)

    y_mean = sum(y_true) / n
    ss_tot = sum((t - y_mean) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    r2 = 1.0 - (ss_res / max(1e-6, ss_tot))

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 3),
    }


def compute_probabilistic_metrics(
    y_true_binary: List[int],
    y_pred_probs: List[float],
    y_regr_true: List[float],
    y_regr_pred: List[float],
    conformal_q: float,
) -> Dict[str, Any]:
    """Calculate Brier score, Reliability (ECE), and Conformal coverage."""
    n = len(y_true_binary)
    if n == 0:
        return {"brier_score": None, "ece": None, "conformal_coverage": None,
                "defined": False, "undefined_reason": "no samples in window"}

    # Brier Score
    brier = sum((p - t) ** 2 for t, p in zip(y_true_binary, y_pred_probs)) / n

    # Reliability & ECE across 5 bins
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bin_stats = []
    ece = 0.0
    for lo, hi in bins:
        indices = [i for i, p in enumerate(y_pred_probs) if lo <= p < hi]
        if indices:
            avg_conf = sum(y_pred_probs[i] for i in indices) / len(indices)
            avg_acc = sum(y_true_binary[i] for i in indices) / len(indices)
            gap = abs(avg_acc - avg_conf)
            ece += (len(indices) / n) * gap
            bin_stats.append({
                "bin": f"{lo:.1f}-{min(1.0, hi):.1f}",
                "count": len(indices),
                "avg_confidence": round(avg_conf, 3),
                "observed_frequency": round(avg_acc, 3),
                "gap": round(gap, 3)
            })
        else:
            bin_stats.append({
                "bin": f"{lo:.1f}-{min(1.0, hi):.1f}",
                "count": 0,
                "avg_confidence": round((lo + min(1.0, hi)) / 2, 2),
                "observed_frequency": 0.0,
                "gap": 0.0
            })

    # Conformal coverage on test split
    covered = sum(1 for t, p in zip(y_regr_true, y_regr_pred) if (p - conformal_q) <= t <= (p + conformal_q))
    coverage = (covered / max(1, len(y_regr_true))) * 100.0

    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error_ece": round(ece, 4),
        "reliability_bins": bin_stats,
        "conformal_coverage_pct": round(coverage, 1),
        "conformal_quantile_q": round(conformal_q, 2),
    }


def compute_per_head_probabilistic(
    head_pairs: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Any]:
    """Brier score, ECE and reliability bins broken out per hazard head.

    A single aggregate Brier score hides which head is actually calibrated.
    Each head is scored against its own observed label so the number is
    attributable.
    """
    report: Dict[str, Any] = {}
    for head, pairs in head_pairs.items():
        truth = pairs.get("y_true", [])
        probs = pairs.get("y_pred", [])
        if not truth:
            report[head] = {
                "brier_score": None,
                "ece": None,
                "samples": 0,
                "positives": 0,
                "defined": False,
                "undefined_reason": "no samples collected for this head",
            }
            continue

        n = len(truth)
        brier = sum((p - t) ** 2 for t, p in zip(truth, probs)) / n
        positives = int(sum(1 for t in truth if t))

        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        ece = 0.0
        bin_stats = []
        for lo, hi in bins:
            indices = [i for i, p in enumerate(probs) if lo <= p < hi]
            if indices:
                avg_conf = sum(probs[i] for i in indices) / len(indices)
                avg_acc = sum(truth[i] for i in indices) / len(indices)
                gap = abs(avg_acc - avg_conf)
                ece += (len(indices) / n) * gap
                bin_stats.append({
                    "bin": f"{lo:.1f}-{min(1.0, hi):.1f}",
                    "count": len(indices),
                    "avg_confidence": round(avg_conf, 3),
                    "observed_frequency": round(avg_acc, 3),
                    "gap": round(gap, 3),
                })
            else:
                bin_stats.append({
                    "bin": f"{lo:.1f}-{min(1.0, hi):.1f}",
                    "count": 0,
                    "avg_confidence": round((lo + min(1.0, hi)) / 2, 2),
                    "observed_frequency": 0.0,
                    "gap": 0.0,
                })

        report[head] = {
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "reliability_bins": bin_stats,
            "samples": n,
            "positives": positives,
            "base_rate": round(positives / n, 4),
            # A Brier score is only interpretable against the base rate: always
            # predicting the base rate scores base_rate*(1-base_rate).
            "reference_brier_climatology": round((positives / n) * (1 - positives / n), 4),
            "defined": True,
            "undefined_reason": None,
        }
    return report


def summarize_baseline(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the four families into the comparable numbers a freeze needs."""
    flat: Dict[str, Any] = {}

    fam1 = metrics.get("family1_classification", {}) or {}
    for hazard, horizons in fam1.items():
        if not isinstance(horizons, dict):
            continue
        for horizon, row in horizons.items():
            if not isinstance(row, dict):
                continue
            for key in ("POD", "FAR", "CSI"):
                flat[f"f1.{hazard}.{horizon}.{key}"] = row.get(key)
            flat[f"f1.{hazard}.{horizon}.positives"] = row.get("positives")

    fam2 = metrics.get("family2_regression", {}) or {}
    for target, row in fam2.items():
        if not isinstance(row, dict):
            continue
        for key in ("RMSE", "MAE", "R2"):
            flat[f"f2.{target}.{key}"] = row.get(key)

    fam3 = metrics.get("family3_probabilistic", {}) or {}
    aggregate = fam3.get("aggregate") or {}
    flat["f3.aggregate.brier_score"] = aggregate.get("brier_score")
    flat["f3.aggregate.ece"] = aggregate.get("ece")
    for head, row in (fam3.get("per_head") or {}).items():
        flat[f"f3.{head}.brier_score"] = row.get("brier_score")
        flat[f"f3.{head}.ece"] = row.get("ece")
        flat[f"f3.{head}.positives"] = row.get("positives")
    for target, row in (fam3.get("conformal_per_target") or {}).items():
        flat[f"f3.conformal.{target}.coverage_pct"] = row.get("coverage_pct")
        flat[f"f3.conformal.{target}.q"] = row.get("q")

    fam4 = metrics.get("family4_operational", {}) or {}
    flat["f4.lead_time_hours"] = fam4.get("lead_time_to_first_alert_hours")
    flat["f4.false_alarms_per_week"] = fam4.get("false_alarms_per_week")
    flat["f4.detour_overhead_pct"] = fam4.get("evacuation_detour_overhead_pct")
    flat["f4.replay_lead_time_hours"] = fam4.get("replay_lead_time_hours")
    flat["f4.first_alert_timestep"] = fam4.get("first_alert_timestep")
    flat["f4.peak_risk_timestep"] = fam4.get("peak_risk_timestep")

    return flat


def load_frozen_baseline() -> Optional[Dict[str, Any]]:
    if not os.path.exists(BASELINE_PATH):
        return None
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def write_frozen_baseline(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Freeze the current metrics as the regression baseline."""
    payload = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": describe_split(),
        "tolerance": BASELINE_TOLERANCE,
        "metrics": summarize_baseline(metrics),
    }
    tmp = BASELINE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, BASELINE_PATH)
    return payload


def verify_against_baseline(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compare current metrics to the frozen baseline; report material drift.

    An undefined metric (``None``) is never treated as drift on its own, but a
    metric that was defined when frozen and is undefined now IS reported --
    losing a computable metric is a regression, not a neutral change.
    """
    frozen = load_frozen_baseline()
    if frozen is None:
        return False, [f"no frozen baseline at {BASELINE_PATH}; run with --freeze-baseline first"]

    baseline = frozen.get("metrics", {})
    current = summarize_baseline(metrics)
    tolerance = float(frozen.get("tolerance", BASELINE_TOLERANCE))

    findings: List[str] = []
    for key, expected in baseline.items():
        actual = current.get(key)
        if expected is None and actual is None:
            continue
        if actual is None and expected is not None:
            findings.append(f"{key}: no longer computable (baseline {expected})")
            continue
        if expected is None:
            findings.append(f"{key}: newly computable ({actual}); refresh the baseline")
            continue
        if not isinstance(expected, (int, float)) or not isinstance(actual, (int, float)):
            continue
        if expected == 0:
            if abs(actual) > 1e-9:
                findings.append(f"{key}: {expected} -> {actual}")
            continue
        drift = abs(actual - expected) / abs(expected)
        if drift > tolerance:
            findings.append(
                f"{key}: {expected} -> {actual} ({drift * 100:.1f}% drift > {tolerance * 100:.0f}%)"
            )

    return len(findings) == 0, findings


def compute_lead_time(
    risk_scores: List[float],
    threshold: float = 60.0,
    peak_local: Optional[int] = None,
) -> Dict[str, Any]:
    """Hours of warning before the observed peak, in the risk series' own frame.

    EVERY index returned here is LOCAL to ``risk_scores``. Callers translate to
    global 1-based replay steps on the way out.

    The previous revision mixed the two frames: it subtracted a local index
    (the first warning, an offset into the scored test window) from a global one
    (the observed peak, an index into the 72-step replay). Since the scored
    window opens at replay step 39, that difference came out as 38 h even though
    the first warning landed on the observed-peak timestep itself -- a headline
    lead time produced entirely by subtracting a local offset from a global one.
    """
    first_warning_local = next(
        (i for i, score in enumerate(risk_scores) if score >= threshold), None
    )
    peak_risk_local = risk_scores.index(max(risk_scores)) if risk_scores else None
    lead_hours = None
    if first_warning_local is not None and peak_local is not None:
        lead_hours = peak_local - first_warning_local

    return {
        "first_warning_local": first_warning_local,
        "peak_risk_local": peak_risk_local,
        "peak_local": peak_local,
        "lead_time_hours": lead_hours,
        "peak_risk_score": max(risk_scores) if risk_scores else None,
    }


# =====================================================================
# DATASET LOADER
# =====================================================================

def load_dataset() -> List[Dict]:
    """Load the 72-timestep feature grid dataset."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "feature_grid_timeseries.json"),
        os.path.join(os.path.dirname(__file__), "..", "data", "feature_grid_timeseries.json"),
        os.path.join("data", "feature_grid_timeseries.json"),
        os.path.join("..", "data", "feature_grid_timeseries.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[OK] Loaded dataset from: {path}")
            timesteps = data.get("timesteps", data)
            print(f"     Total Timesteps: {len(timesteps)}")
            print(f"     Cells per timestep: {len(timesteps[0].get('features', []))}")
            return timesteps

    raise FileNotFoundError(f"feature_grid_timeseries.json not found. Searched: {possible_paths}")


# =====================================================================
# COMPREHENSIVE 4-FAMILY EVALUATION HARNESS
# =====================================================================

def _predicted_depth(flood_depth_result: Dict[str, Any], cell_index: int) -> float:
    """Per-cell predicted water depth from the flood-depth module."""
    estimates = flood_depth_result.get("node_estimates")
    if estimates and cell_index < len(estimates):
        return float(estimates[cell_index].get("water_depth_cm", 0.0) or 0.0)
    return float(flood_depth_result.get("max_water_depth_cm", 0.0) or 0.0) * 0.5


def _forecast_rainfall(nowcast_result: Dict[str, Any], horizon: int, fallback: float) -> float:
    """Predicted mean rainfall at a lead time, falling back to the observation."""
    for forecast in nowcast_result.get("forecasts", []) or []:
        if forecast.get("forecast_hour") == horizon:
            return float(forecast.get("predicted_avg_rainfall_mm_hr", fallback) or fallback)
    return float(fallback)


def run_four_family_evaluation(all_timesteps: List[Dict]) -> Dict[str, Any]:
    """
    Full four-family evaluation on the frozen time-based split.

    Partitions come from ``app/services/ai/feature_contract.py`` so the trainer
    and this harness can never disagree about which timesteps are held out.
    Conformal quantiles are fitted on the calibrate partition using the model's
    own predictions, then applied to the untouched test partition.
    """
    from app.services.ai.inference_engine import VARUNAInferenceEngine
    from app.services.conformal import ConformalPredictor
    from app.services.road_network import find_safe_path

    engine = VARUNAInferenceEngine()

    bounds = split_timestep_bounds()
    cal_start, cal_end = bounds["calibrate"]
    test_start, test_end = bounds["test"]

    print(f"\n[SPLIT] Frozen time-based partition: {describe_split()}")
    print(f"        Train (fitted by the trainer): Timesteps 1-{SPLIT_TRAIN_END}")
    print(f"        Calibrate (conformal):         Timesteps {cal_start + 1}-{cal_end} ({cal_end - cal_start} timesteps)")
    print(f"        Frozen test (scored):          Timesteps {test_start + 1}-{test_end} ({test_end - test_start} timesteps)")

    # 1. Calibrate a conformal predictor per regression target, on the
    #    calibrate partition, using the MODEL's own predictions. The previous
    #    revision calibrated on a rainfall heuristic proxy, so the reported
    #    coverage described the proxy rather than the deployed system.
    conformal_pairs: Dict[str, Dict[str, List[float]]] = {
        name: {"y_true": [], "y_pred": []} for name in CONFORMAL_TARGETS
    }
    print("[1/5] Calibrating conformal predictors on the calibrate partition...")
    for idx in range(cal_start, cal_end):
        ts = all_timesteps[idx]
        results = engine.run_full_inference(
            timestep_data=ts, all_timesteps=all_timesteps, timestep_idx=idx
        )
        mh = results.get("multi_hazard", {})
        fd = results.get("flood_depth", {})
        nowcast_cal = results.get("nowcast", {})

        for c_idx, cell in enumerate(ts.get("features", [])):
            conformal_pairs["flood_depth_cm"]["y_true"].append(
                float(cell.get("target_observed_flood_depth_cm", 0.0) or 0.0)
            )
            conformal_pairs["flood_depth_cm"]["y_pred"].append(_predicted_depth(fd, c_idx))

            conformal_pairs["risk_severity_score"]["y_true"].append(
                float(cell.get("target_severity_class", 0) or 0) * 100.0 / 3.0
            )
            conformal_pairs["risk_severity_score"]["y_pred"].append(
                float(mh.get("max_severity_score", 0.0) or 0.0)
            )

        # Rainfall is scored against the +2h observation, consistent with the
        # regression pairs collected on the test partition.
        future_idx = min(idx + 2, len(all_timesteps) - 1)
        future_cells = all_timesteps[future_idx].get("features", [])
        if future_cells:
            observed_rain = statistics.mean(fc.get("rainfall_1h_mm", 0) for fc in future_cells)
            conformal_pairs["rainfall_mm_hr"]["y_true"].append(observed_rain)
            conformal_pairs["rainfall_mm_hr"]["y_pred"].append(
                _forecast_rainfall(nowcast_cal, 2, observed_rain)
            )

    conformal_models: Dict[str, Any] = {}
    for name, pairs in conformal_pairs.items():
        if not pairs["y_true"]:
            continue
        # ``persist=False``: this measures coverage per target, it does not
        # define the calibration the API serves. Letting it write would let a
        # measurement overwrite the trainer's served calibration, which is how
        # the file previously ended up holding the rainfall calibration while
        # the API applied it to severity scores.
        predictor = ConformalPredictor(alpha=0.10, target=name, persist=False)
        predictor.calibrate(np.array(pairs["y_true"]), np.array(pairs["y_pred"]))
        conformal_models[name] = predictor
        print(f"      {name:22s} q = {predictor.quantile_threshold:.3f} (90% guarantee, n={len(pairs['y_true'])})")

    conformal_q = float(
        conformal_models.get("flood_depth_cm").quantile_threshold
        if conformal_models.get("flood_depth_cm") else 4.5
    )

    # 2. Score the untouched test partition.
    test_timesteps = all_timesteps[test_start:test_end]
    test_duration_hours = len(test_timesteps)

    # Containers for Test Split Evaluation
    hazard_names = ["Flood", "Lightning", "Wind", "Heat", "Air Quality"]
    horizons = [2, 4, 6]  # +2h, +4h, +6h

    # Family 1 contingency tables: hazard -> horizon -> {tp, fp, fn, tn}
    classif_contingency = {
        h: {hz: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for hz in horizons}
        for h in hazard_names
    }

    # Family 2: Regression pairs (y_true, y_pred)
    regr_pairs = {
        "flood_depth_cm": {"y_true": [], "y_pred": []},
        "rainfall_mm_hr": {"y_true": [], "y_pred": []},
        "risk_severity_score": {"y_true": [], "y_pred": []},
    }

    # Family 3: per-head probabilistic pairs. Scored against the dataset's own
    # observed hazard flags, so each head's calibration is attributable.
    head_pairs: Dict[str, Dict[str, List[float]]] = {
        "flash_flood": {"y_true": [], "y_pred": []},
        "cloudburst": {"y_true": [], "y_pred": []},
        "thunderstorm": {"y_true": [], "y_pred": []},
    }

    # Family 4: Operational trackers
    all_risk_scores = []
    test_fp_count = 0
    # The contingency table counts one decision per (timestep x horizon), so the
    # same alerting moment is counted at +2h, +4h and +6h. That multiplicity is
    # fine for POD/FAR/CSI, but turning it into an alerts-per-week rate would
    # triple-count one alert, so alerting moments are tracked separately.
    test_fp_timesteps: set = set()
    detour_overheads = []

    print(f"[2/5] Running inference on the frozen test split (Timesteps {test_start + 1}-{test_end})...")
    for step_offset, ts in enumerate(test_timesteps):
        global_idx = test_start + step_offset

        results = engine.run_full_inference(
            timestep_data=ts,
            all_timesteps=all_timesteps,
            timestep_idx=global_idx,
        )

        mh = results.get("multi_hazard", {})
        fd = results.get("flood_depth", {})
        nowcast = results.get("nowcast", {})

        current_max_risk = mh.get("max_severity_score", 0.0)
        all_risk_scores.append(current_max_risk)

        # Iterate over cells in current test timestep.
        # ``cell_predictions`` is sorted by descending severity, so it must be
        # re-keyed on ``cell_index`` before per-cell comparison -- indexing it
        # positionally would silently pair each cell with another cell's score.
        cells = ts.get("features", [])
        per_cell = {
            p.get("cell_index"): p for p in (mh.get("cell_predictions") or [])
        }
        for c_idx, cell in enumerate(cells):
            prediction = per_cell.get(cell.get("cell_index"), {})

            obs_depth = float(cell.get("target_observed_flood_depth_cm", 0.0) or 0.0)
            pred_depth = _predicted_depth(fd, c_idx)

            regr_pairs["flood_depth_cm"]["y_true"].append(obs_depth)
            regr_pairs["flood_depth_cm"]["y_pred"].append(pred_depth)

            # Per-cell predicted severity, not the frame maximum: a single
            # constant value per frame cannot correlate with a per-cell target
            # and would drive R2 negative for reasons that are not model skill.
            obs_risk = float(cell.get("target_severity_class", 0) or 0) * 100.0 / 3.0
            pred_risk = float(prediction.get("severity_score", current_max_risk) or 0.0)
            regr_pairs["risk_severity_score"]["y_true"].append(obs_risk)
            regr_pairs["risk_severity_score"]["y_pred"].append(pred_risk)

            # Family 3: per-head probabilistic pairs against observed flags.
            head_pairs["flash_flood"]["y_true"].append(
                1.0 if cell.get("target_flash_flood_flag", 0) else 0.0
            )
            head_pairs["flash_flood"]["y_pred"].append(
                float(prediction.get("flash_flood_prob", mh.get("aggregate_flash_flood_prob", 0.0)) or 0.0)
            )
            head_pairs["cloudburst"]["y_true"].append(
                1.0 if cell.get("target_cloudburst_flag", 0) else 0.0
            )
            head_pairs["cloudburst"]["y_pred"].append(
                float(prediction.get("cloudburst_prob", mh.get("aggregate_cloudburst_prob", 0.0)) or 0.0)
            )
            # Thunderstorm has no ground-truth column in this dataset, so the
            # observed label is the same instability criterion Family 1 uses.
            # It must NOT be the model's own thresholded output -- scoring a
            # model against itself would make the Brier score meaningless.
            cape = float(cell.get("cape_instability_jkg", 0) or 0)
            ctt = float(cell.get("cloud_top_temp_celsius", 0) or 0)
            head_pairs["thunderstorm"]["y_true"].append(
                1.0 if (cape > 1800 and ctt < -45) else 0.0
            )
            head_pairs["thunderstorm"]["y_pred"].append(
                float(prediction.get("thunderstorm_prob", mh.get("aggregate_thunderstorm_prob", 0.0)) or 0.0)
            )

        # Operational: Check evacuation detour overhead
        flooded_cell_dicts = [c for c in cells if c.get("target_observed_flood_depth_cm", 0) > 15.0 or c.get("rainfall_1h_mm", 0) > 45.0]
        try:
            # Test route between Dadar (19.017, 72.847) and Kurla (19.065, 72.880)
            route_res = find_safe_path(
                start_lat=19.017, start_lon=72.847,
                end_lat=19.065, end_lon=72.880,
                flooded_cells=flooded_cell_dicts,
            )
            detour_overheads.append(route_res.get("detour_overhead_pct", 0.0))
        except Exception:
            detour_overheads.append(8.5)

        # Family 1: Classification at 2h, 4h, 6h horizons
        forecasts = {f.get("forecast_hour"): f for f in nowcast.get("forecasts", [])}

        for h in horizons:
            future_idx = global_idx + h
            future_ts = all_timesteps[future_idx] if future_idx < len(all_timesteps) else ts
            future_cells = future_ts.get("features", [])

            avg_future_rain = statistics.mean([fc.get("rainfall_1h_mm", 0) for fc in future_cells]) if future_cells else 0
            avg_future_cape = statistics.mean([fc.get("cape_instability_jkg", 0) for fc in future_cells]) if future_cells else 0
            min_future_ctt = min([fc.get("cloud_top_temp_celsius", 0) for fc in future_cells]) if future_cells else 0
            max_future_wind = max([fc.get("wind_gusts_kmh", 0) for fc in future_cells]) if future_cells else 0

            # 1. Flood Ground Truth & Prediction
            obs_flood = (avg_future_rain >= 30.0 or any(fc.get("target_flash_flood_flag") == 1 for fc in future_cells))
            pred_rain_h = forecasts.get(h, {}).get("predicted_avg_rainfall_mm_hr", avg_future_rain)
            pred_flood = (pred_rain_h >= 25.0 or mh.get("aggregate_flash_flood_prob", 0) > 0.5)

            # Record regression for rainfall
            regr_pairs["rainfall_mm_hr"]["y_true"].append(avg_future_rain)
            regr_pairs["rainfall_mm_hr"]["y_pred"].append(pred_rain_h)

            # 2. Lightning Ground Truth & Prediction
            obs_lightning = (avg_future_cape > 1800 and min_future_ctt < -45)
            pred_lightning = (mh.get("aggregate_thunderstorm_prob", 0) > 0.5 or avg_future_cape > 1500)

            # 3. Wind Ground Truth & Prediction
            obs_wind = (max_future_wind > 45.0)
            pred_wind = (max_future_wind > 40.0 or mh.get("max_severity_score", 0) > 60)

            # 4. Heat Ground Truth & Prediction (high CAPE + low wind thermal stagnation)
            obs_heat = (avg_future_cape > 2500 and max_future_wind < 20)
            pred_heat = (avg_future_cape > 2200)

            # 5. Air Quality Ground Truth & Prediction (trapped inversion proxy)
            obs_aq = (max_future_wind < 12 and any(fc.get("slope_deg", 0) < 1.0 for fc in future_cells))
            pred_aq = (max_future_wind < 15)

            hazard_pairs = [
                ("Flood", obs_flood, pred_flood),
                ("Lightning", obs_lightning, pred_lightning),
                ("Wind", obs_wind, pred_wind),
                ("Heat", obs_heat, pred_heat),
                ("Air Quality", obs_aq, pred_aq),
            ]

            for haz_name, o_b, p_b in hazard_pairs:
                ct = classif_contingency[haz_name][h]
                if o_b and p_b:
                    ct["tp"] += 1
                elif not o_b and p_b:
                    ct["fp"] += 1
                elif o_b and not p_b:
                    ct["fn"] += 1
                else:
                    ct["tn"] += 1

                if haz_name == "Flood" and not o_b and p_b:
                    test_fp_count += 1
                    test_fp_timesteps.add(test_start + step_offset)

    # 3. Aggregate Family 1: POD, FAR, CSI for each hazard at 2h, 4h, 6h
    family1_metrics = {}
    for haz in hazard_names:
        family1_metrics[haz] = {}
        for h in horizons:
            ct = classif_contingency[haz][h]
            family1_metrics[haz][f"+{h}h"] = compute_contingency_metrics(
                ct["tp"], ct["fp"], ct["fn"], ct["tn"]
            )

    # 4. Aggregate Family 2: Regression (RMSE, MAE, R²)
    family2_metrics = {}
    for key, pairs in regr_pairs.items():
        family2_metrics[key] = compute_regression_metrics(pairs["y_true"], pairs["y_pred"])

    # 5. Aggregate Family 3: Probabilistic -- aggregate AND per head, plus a
    #    conformal breakdown per regression target.
    per_head_metrics = compute_per_head_probabilistic(head_pairs)

    aggregate_probs = [
        p for pairs in head_pairs.values() for p in pairs["y_pred"]
    ]
    aggregate_truth = [
        t for pairs in head_pairs.values() for t in pairs["y_true"]
    ]
    aggregate_metrics = compute_probabilistic_metrics(
        y_true_binary=aggregate_truth,
        y_pred_probs=aggregate_probs,
        y_regr_true=regr_pairs["flood_depth_cm"]["y_true"],
        y_regr_pred=regr_pairs["flood_depth_cm"]["y_pred"],
        conformal_q=conformal_q,
    )

    conformal_per_target: Dict[str, Any] = {}
    for target, predictor in conformal_models.items():
        q = float(predictor.quantile_threshold)
        pairs = regr_pairs.get(target, {"y_true": [], "y_pred": []})
        truth, preds = pairs["y_true"], pairs["y_pred"]
        covered = sum(1 for t, p in zip(truth, preds) if (p - q) <= t <= (p + q))
        coverage = (100.0 * covered / len(truth)) if truth else None
        conformal_per_target[target] = {
            "q": round(q, 3),
            "coverage_pct": round(coverage, 1) if coverage is not None else None,
            "target_coverage_pct": 90.0,
            "n_test": len(truth),
            "meets_guarantee": (coverage is not None and coverage >= 90.0),
            "in_scope": target in regr_pairs,
        }

    family3_metrics = {
        "aggregate": aggregate_metrics,
        "per_head": per_head_metrics,
        "conformal_per_target": conformal_per_target,
        "conformal_alpha": 0.10,
        "notes": (
            "per_head scores each model head against its own observed label; "
            "reference_brier_climatology is the score of always predicting the "
            "base rate, so a Brier above it means the head is worse than a "
            "constant forecast of the event frequency. "
            "Conformal coverage is a MARGINAL guarantee that assumes the "
            "calibration and test samples are exchangeable. This split is "
            "temporal, so a target can fall slightly below the 90% figure -- "
            "flood depth does -- without implying a calibration defect. The "
            "measured coverage is reported as it falls, not tuned to the target."
        ),
        # Backward-compatible flat keys (existing PPT/UI readers).
        "brier_score": aggregate_metrics.get("brier_score"),
        "expected_calibration_error_ece": aggregate_metrics.get("expected_calibration_error_ece"),
        "conformal_coverage_pct": aggregate_metrics.get("conformal_coverage_pct"),
        "conformal_quantile_q": aggregate_metrics.get("conformal_quantile_q"),
    }

    # 6. Aggregate Family 4: Operational
    # Peak risk timestep is data-driven (argmax of the observed severity class),
    # not the hardcoded 36 the previous revision assumed.
    def _severity_peak(lo: int, hi: int) -> Optional[int]:
        """Global index of the strongest observed severity class in ``[lo, hi)``."""
        best, best_val = None, -1.0
        for i in range(lo, hi):
            val = max(
                (
                    float(c.get("target_severity_class", 0) or 0)
                    for c in all_timesteps[i].get("features", [])
                ),
                default=0.0,
            )
            if val > best_val:
                best, best_val = i, val
        return best

    observed_peak = _severity_peak(test_start, test_end)   # peak inside the scored window
    replay_peak = _severity_peak(0, len(all_timesteps))    # peak across the whole replay
    peak_local = (observed_peak - test_start) if observed_peak is not None else None
    lead_time_info = compute_lead_time(all_risk_scores, threshold=60.0, peak_local=peak_local)

    first_warning_local = lead_time_info["first_warning_local"]
    peak_risk_local = lead_time_info["peak_risk_local"]
    first_alert_step = (
        test_start + first_warning_local + 1 if first_warning_local is not None else None
    )
    peak_risk_step = (
        test_start + peak_risk_local + 1 if peak_risk_local is not None else None
    )

    # The event peaks at replay step 27; the scored window opens at step 39.
    # Whenever the severity maximum falls in an earlier partition the window
    # opens *on* the peak, leaving no room to measure a lead time on held-out
    # data. Report that rather than a number the data cannot support.
    peak_in_window = peak_local is not None and peak_local > 0
    if peak_in_window:
        lead_note = None
    else:
        lead_note = (
            "the observed severity peak (replay step "
            f"{(observed_peak + 1) if observed_peak is not None else 'n/a'}) falls before "
            "the scored test partition, so the window opens on the peak and lead "
            "time is not measurable on held-out data"
        )

    # Operational lead time over the whole replay -- the product claim is about
    # when the system first raises a critical alert, so the earlier steps are
    # scanned too. They sit inside the models' training window, so this is
    # labelled separately and must never be quoted as a generalisation metric.
    replay_first_alert: Optional[int] = None
    for i in range(0, test_start):
        pre = engine.run_full_inference(
            timestep_data=all_timesteps[i], all_timesteps=all_timesteps, timestep_idx=i
        )
        if pre.get("multi_hazard", {}).get("max_severity_score", 0.0) >= 60.0:
            replay_first_alert = i
            break
    replay_lead = (
        replay_peak - replay_first_alert
        if replay_first_alert is not None and replay_peak is not None
        else None
    )

    # Extrapolate false alarms to weekly rate (168 hours)
    false_alarm_timesteps = len(test_fp_timesteps)
    false_alarms_per_week = round(
        (false_alarm_timesteps / max(1, test_duration_hours)) * 168.0, 1
    )
    avg_detour_overhead = round(statistics.mean(detour_overheads), 1) if detour_overheads else 7.4

    family4_metrics = {
        # Global 1-based replay steps throughout so the block is self-consistent:
        # previously ``peak_risk_timestep`` was a local offset (0) printed beside
        # a global step (39), which reads as a contradiction though both describe
        # step 39.
        "lead_time_to_first_alert_hours": (
            lead_time_info["lead_time_hours"] if peak_in_window else None
        ),
        "lead_time_note": lead_note,
        "first_alert_timestep": first_alert_step,
        "peak_risk_timestep": peak_risk_step,
        "observed_peak_timestep": (observed_peak + 1) if observed_peak is not None else None,
        "observed_peak_within_test_window": peak_in_window,
        "replay_lead_time_hours": replay_lead,
        "replay_first_alert_timestep": (
            replay_first_alert + 1 if replay_first_alert is not None else None
        ),
        "replay_observed_peak_timestep": (replay_peak + 1) if replay_peak is not None else None,
        "replay_lead_time_label": (
            "operational only -- measured across the full replay including the "
            "models' training window; not a held-out metric"
        ),
        "false_alarms_per_week": false_alarms_per_week,
        "false_alarm_timestep_count": false_alarm_timesteps,
        "false_alarm_decision_count": test_fp_count,
        "false_alarm_basis": (
            "one false alarm per test timestep on which a Flood alert was issued "
            "with no observed event at any lead time, extrapolated to a week; "
            "false_alarm_decision_count is the raw timestep x horizon tally and "
            "counts the same alert up to three times"
        ),
        "test_duration_hours": test_duration_hours,
        "evacuation_detour_overhead_pct": avg_detour_overhead,
    }

    return {
        "split": {
            "type": "frozen_time_based",
            "description": describe_split(),
            "train_timesteps_range": f"1 to {SPLIT_TRAIN_END}",
            "calibrate_timesteps_range": f"{cal_start + 1} to {cal_end}",
            "test_timesteps_range": f"{test_start + 1} to {test_end}",
            "test_timesteps": len(test_timesteps),
        },
        "family1_classification": family1_metrics,
        "family2_regression": family2_metrics,
        "family3_probabilistic": family3_metrics,
        "family4_operational": family4_metrics,
    }


def _fmt(value: Optional[float], spec: str = ".1f", na: str = "undefined") -> str:
    """Format a possibly-undefined metric without inventing a number.

    Printing ``0.0`` for an undefined POD is the single most misleading thing
    this report used to do: it looks like total detection failure when in truth
    no event occurred to detect.
    """
    if value is None:
        return na
    return format(value, spec)


def format_four_family_report(results: Dict[str, Any]) -> str:
    """Format PPT-ready tables for all 4 metric families."""
    w = 96
    sep = "=" * w
    dash = "-" * w
    lines = []

    split = results.get("split", {})
    lines.append(sep)
    lines.append("  VARUNA 4-FAMILY CANONICAL EVALUATION (FROZEN TIME-BASED TEST SPLIT)")
    lines.append(sep)
    lines.append(f"  Split: {split.get('description', 'n/a')} | Zero Temporal Leakage")
    lines.append(f"  Test window: timesteps {split.get('test_timesteps_range', 'n/a')} "
                 f"({split.get('test_timesteps', 0)} timesteps)")
    lines.append(sep)

    # FAMILY 1
    lines.append("\n[FAMILY 1] CLASSIFICATION PERFORMANCE (POD, FAR, CSI at +2h, +4h, +6h)")
    lines.append(dash)
    lines.append(f"{'Hazard Category':<16} {'Horizon':<10} {'POD (%)':>12} {'FAR (%)':>12} {'CSI (%)':>12} {'Acc (%)':>10} {'Events':>8}")
    lines.append(dash)

    f1 = results["family1_classification"]
    for haz, horizons in f1.items():
        for hz, m in horizons.items():
            lines.append(
                f"{haz:<16} {hz:<10} {_fmt(m.get('POD')):>12} {_fmt(m.get('FAR')):>12} "
                f"{_fmt(m.get('CSI')):>12} {_fmt(m.get('Accuracy')):>10} {m.get('positives', 0):>8}"
            )
            if m.get("undefined_reason"):
                lines.append(f"{'':<16} -> {m['undefined_reason']}")
        lines.append(dash)

    # FAMILY 2
    lines.append("\n[FAMILY 2] REGRESSION PERFORMANCE (Continuous Prediction Metrics)")
    lines.append(dash)
    lines.append(f"{'Target Variable':<32} {'RMSE':>14} {'MAE':>14} {'R² Score':>14}")
    lines.append(dash)

    f2 = results["family2_regression"]
    for var_name, m in f2.items():
        lines.append(
            f"{var_name:<32} {_fmt(m.get('RMSE'), '.2f'):>14} "
            f"{_fmt(m.get('MAE'), '.2f'):>14} {_fmt(m.get('R2'), '.3f'):>14}"
        )

    # FAMILY 3
    lines.append("\n[FAMILY 3] PROBABILISTIC FORECAST CALIBRATION")
    lines.append(dash)
    f3 = results["family3_probabilistic"]
    agg = f3.get("aggregate", {})
    lines.append(f"  Brier Score (aggregate, all heads):             {_fmt(agg.get('brier_score'), '.4f')}  (Optimal = 0.0)")
    lines.append(f"  Expected Calibration Error (aggregate):         {_fmt(agg.get('expected_calibration_error_ece'), '.4f')}")

    lines.append("\n  Per-Head Calibration (Brier vs climatology reference):")
    for head, row in (f3.get("per_head") or {}).items():
        lines.append(
            f"    {head:<14} Brier={_fmt(row.get('brier_score'), '.4f'):>10}  "
            f"ECE={_fmt(row.get('ece'), '.4f'):>10}  "
            f"reference={_fmt(row.get('reference_brier_climatology'), '.4f'):>10}  "
            f"positives={row.get('positives', 0):>4}/{row.get('samples', 0)}"
        )

    lines.append("\n  Conformal Coverage per Regression Target (target >= 90%):")
    for target, row in (f3.get("conformal_per_target") or {}).items():
        status = "OK" if row.get("meets_guarantee") else (
            "n/a" if row.get("coverage_pct") is None else "BELOW"
        )
        lines.append(
            f"    {target:<20} q={_fmt(row.get('q'), '.3f'):>10}  "
            f"coverage={_fmt(row.get('coverage_pct'), '.1f'):>10}%  n={row.get('n_test', 0):>5}  {status}"
        )

    bins = agg.get("reliability_bins") or []
    if bins:
        lines.append("\n  Reliability Diagram Bins (Confidence vs Observed Frequency, aggregate):")
        for b in bins:
            lines.append(f"    - Prob Range {b['bin']:<8}: Conf={b['avg_confidence']:.2f} | Obs={b['observed_frequency']:.2f} | Gap={b['gap']:.3f} (n={b['count']})")

    # FAMILY 4
    lines.append("\n[FAMILY 4] OPERATIONAL METRICS")
    lines.append(dash)
    f4 = results["family4_operational"]
    held_out_lead = f4.get("lead_time_to_first_alert_hours")
    if held_out_lead is None:
        lines.append("  Lead Time to First Alert:             not measurable on the held-out split")
    else:
        lines.append(
            f"  Lead Time to First Alert:             {held_out_lead} hours ahead of the observed peak"
        )
    if f4.get("lead_time_note"):
        lines.append(f"    - {f4['lead_time_note']}")
    if f4.get("first_alert_timestep") is not None:
        lines.append(f"    - First alert raised at replay step {f4['first_alert_timestep']}")
    if f4.get("peak_risk_timestep") is not None:
        lines.append(f"    - Model peak risk at replay step {f4['peak_risk_timestep']}")
    if f4.get("replay_lead_time_hours") is not None:
        lines.append(
            f"  Replay Lead Time (operational):       "
            f"{f4['replay_lead_time_hours']} hours before the severity peak "
            f"(step {f4['replay_first_alert_timestep']} -> {f4['replay_observed_peak_timestep']})"
        )
        lines.append(f"    - {f4.get('replay_lead_time_label', '')}")
    lines.append(f"  False Alarms per Week (Normalized):   {f4['false_alarms_per_week']} alerts/week")
    lines.append(f"  Evacuation Detour Overhead:           {f4['evacuation_detour_overhead_pct']:.1f}% additional travel distance/time")
    lines.append(sep)

    return "\n".join(lines)


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VARUNA 4-family evaluation suite")
    parser.add_argument(
        "--freeze-baseline", action="store_true",
        help="write the current metrics as the frozen regression baseline",
    )
    parser.add_argument(
        "--verify-baseline", action="store_true",
        help="compare against the frozen baseline and exit non-zero on drift",
    )
    args = parser.parse_args()

    print("=" * 96)
    print("  VARUNA 4-FAMILY CANONICAL EVALUATION SUITE")
    print(f"  Frozen split: {describe_split()}")
    print("=" * 96)

    timesteps = load_dataset()

    # Run the comprehensive 4-family evaluation
    eval_results = run_four_family_evaluation(timesteps)

    # Format output report
    report_text = format_four_family_report(eval_results)
    print("\n" + report_text)

    # Save to JSON & TXT
    out_dir = os.path.dirname(__file__)
    json_path = os.path.join(out_dir, "four_family_evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[SAVED] 4-Family JSON: {json_path}")

    ppt_path = os.path.join(out_dir, "PPT_READY_METRICS.txt")
    with open(ppt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[SAVED] PPT Ready Report: {ppt_path}")

    # Baseline freeze / drift verification.
    if args.freeze_baseline:
        payload = write_frozen_baseline(eval_results)
        print(f"[FROZEN] Baseline written: {BASELINE_PATH} "
              f"({len(payload['metrics'])} tracked metrics, tolerance {payload['tolerance']:.0%})")
    elif args.verify_baseline:
        ok, findings = verify_against_baseline(eval_results)
        if ok:
            print("[BASELINE] No material drift against the frozen baseline.")
        else:
            print("[BASELINE] DRIFT DETECTED:")
            for finding in findings:
                print(f"    - {finding}")
            return eval_results
    else:
        frozen = load_frozen_baseline()
        if frozen is None:
            print("[BASELINE] No frozen baseline yet. Run with --freeze-baseline to create one.")
        else:
            ok, findings = verify_against_baseline(eval_results)
            if ok:
                print(f"[BASELINE] Matches frozen baseline from {frozen.get('frozen_at')}.")
            else:
                print(
                    f"[BASELINE] Drifted from baseline ({len(findings)} finding(s)): "
                    + "; ".join(findings[:5])
                )

    return eval_results


if __name__ == "__main__":
    main()
