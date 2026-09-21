"""
VARUNA Conformal Prediction Engine
====================================
Provides statistically calibrated confidence intervals for every prediction
using conformal prediction — a distribution-free method that guarantees
marginal coverage with probability 1 - α.

Unlike heuristic confidence labels (High/Medium/Low), conformal prediction
produces prediction sets/intervals with mathematically guaranteed coverage:
- P(true value ∈ prediction set) ≥ 1 - α
- No distributional assumptions required
- Works with any base model

References:
- Vovk et al. (2005), "Algorithmic Learning in a Random World"
- Romano et al. (2020), "Conformalized Quantile Regression"
- Angelopoulos & Bates (2023), "Conformal Prediction: A Gentle Introduction"
"""

import os
import json
import math
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("VARUNA.Conformal")

# Default significance levels
ALPHA_DEFAULT = 0.1       # 90% coverage guarantee
ALPHA_95 = 0.05           # 95% coverage guarantee
ALPHA_99 = 0.01           # 99% coverage guarantee


class ConformalPredictor:
    """
    Split conformal prediction for regression tasks.

    Algorithm:
    1. Split data into proper training set and calibration set
    2. Train model on proper training set
    3. Compute nonconformity scores on calibration set:
       score_i = |y_i - ŷ_i| (absolute residual)
    4. For new predictions, compute quantile of scores:
       q = quantile(scores, ceil((n+1)(1-α)) / n)
    5. Prediction interval = [ŷ - q, ŷ + q]

    This guarantees:
    P(y_{n+1} ∈ [ŷ - q, ŷ + q]) ≥ 1 - α
    """

    def __init__(
        self,
        alpha: float = ALPHA_DEFAULT,
        target: Optional[str] = None,
        persist: bool = True,
    ):
        """
        Args:
            alpha: Significance level (1 - alpha = coverage guarantee)
            target: Name of the quantity this predictor is calibrated for, e.g.
                ``"risk_severity_score"`` (0-100) or ``"flood_depth_cm"``.
                Calibrations are persisted per target. A residual quantile is
                only valid in the unit it was measured in, so one shared
                calibration file for every quantity silently produced
                meaningless intervals: the file ended up holding the rainfall
                calibration (mm/hr, n=6) while the API applied it to severity
                scores, yielding intervals ~97% of the 0-100 scale and a
                constant ``HIGH_UNCERTAINTY`` label for every cell.
            persist: When False this instance measures coverage without writing
                a calibration artifact. The evaluation suite reports coverage
                per target but must not overwrite the calibration the API
                serves; only the trainer owns the served artifacts.
        """
        self.alpha = alpha
        self.target = target
        self.persist = persist
        self.calibration_scores = []
        self.quantile_threshold = None
        self.is_calibrated = False
        self.coverage_guarantee = 1 - alpha

        # Persistence
        self._calibration_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "conformal_calibration.json"
        )

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate the conformal predictor using held-out calibration data.

        Args:
            y_true: Ground truth values (n_samples,)
            y_pred: Model predictions (n_samples,)

        Returns:
            Calibration report with coverage statistics
        """
        n = len(y_true)

        # Compute nonconformity scores (absolute residuals)
        residuals = np.abs(y_true - y_pred)
        self.calibration_scores = residuals.tolist()

        # Compute quantile threshold
        # ceil((n+1)(1-α)) / n ensures finite-sample coverage guarantee
        quantile_idx = math.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_idx = min(quantile_idx, 1.0)  # clamp to [0, 1]

        sorted_scores = np.sort(residuals)
        idx = int(quantile_idx * (n - 1))
        idx = min(idx, n - 1)
        self.quantile_threshold = float(sorted_scores[idx])

        self.is_calibrated = True

        # Compute empirical coverage on calibration set
        lower = y_pred - self.quantile_threshold
        upper = y_pred + self.quantile_threshold
        in_interval = (y_true >= lower) & (y_true <= upper)
        empirical_coverage = float(np.mean(in_interval))

        # Average interval width
        avg_width = float(np.mean(upper - lower))

        report = {
            "alpha": self.alpha,
            "guaranteed_coverage": self.coverage_guarantee,
            "empirical_coverage": round(empirical_coverage, 4),
            "quantile_threshold": round(self.quantile_threshold, 4),
            "avg_interval_width": round(avg_width, 2),
            "calibration_samples": n,
            "residual_stats": {
                "mean": round(float(np.mean(residuals)), 4),
                "std": round(float(np.std(residuals)), 4),
                "median": round(float(np.median(residuals)), 4),
                "p90": round(float(np.percentile(residuals, 90)), 4),
                "p95": round(float(np.percentile(residuals, 95)), 4),
                "p99": round(float(np.percentile(residuals, 99)), 4),
            },
        }

        logger.info(
            f"Conformal calibration: {empirical_coverage:.1%} empirical coverage "
            f"(guaranteed ≥{self.coverage_guarantee:.0%}), "
            f"threshold={self.quantile_threshold:.2f}"
        )

        # Save calibration (measurement-only instances deliberately do not)
        if self.persist:
            self._save_calibration(report)
        else:
            logger.info(
                "Conformal calibration computed for measurement only (target=%s); "
                "no artifact written",
                self.target or "unlabelled",
            )

        return report

    def predict_interval(
        self, y_pred: np.ndarray, confidence_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate conformal prediction intervals for new predictions.

        Args:
            y_pred: Model point predictions (n_samples,) or scalar
            confidence_level: Override alpha for this prediction (e.g., 0.95)

        Returns:
            dict with point prediction, lower/upper bounds, and width
        """
        if not self.is_calibrated:
            # Use fallback heuristic if not calibrated
            return self._fallback_interval(y_pred)

        # Adjust threshold for different confidence levels
        if confidence_level is not None:
            alpha = 1 - confidence_level
            # Recompute quantile for new alpha
            n = len(self.calibration_scores)
            scores = np.sort(self.calibration_scores)
            quantile_idx = math.ceil((n + 1) * (1 - alpha)) / n
            quantile_idx = min(quantile_idx, 1.0)
            idx = int(quantile_idx * (n - 1))
            idx = min(idx, n - 1)
            threshold = float(scores[idx])
        else:
            threshold = self.quantile_threshold

        y_pred = np.atleast_1d(y_pred).astype(float)
        lower = y_pred - threshold
        upper = y_pred + threshold

        # Clip to valid range [0, 100] for risk scores
        lower = np.clip(lower, 0, 100)
        upper = np.clip(upper, 0, 100)

        return {
            "point_prediction": [round(float(v), 2) for v in y_pred],
            "lower_bound": [round(float(v), 2) for v in lower],
            "upper_bound": [round(float(v), 2) for v in upper],
            "interval_width": [round(float(u - l), 2) for u, l in zip(upper, lower)],
            "confidence_level": round(self.coverage_guarantee, 2),
            "quantile_threshold": round(threshold, 4),
        }

    def _fallback_interval(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """Fallback intervals when not calibrated (wider, conservative)."""
        y_pred = np.atleast_1d(y_pred).astype(float)
        # Conservative: ±20% of prediction value, min ±10 points
        margin = np.maximum(np.abs(y_pred) * 0.20, 10)
        lower = np.clip(y_pred - margin, 0, 100)
        upper = np.clip(y_pred + margin, 0, 100)

        return {
            "point_prediction": [round(float(v), 2) for v in y_pred],
            "lower_bound": [round(float(v), 2) for v in lower],
            "upper_bound": [round(float(v), 2) for v in upper],
            "interval_width": [round(float(u - l), 2) for u, l in zip(upper, lower)],
            "confidence_level": 0.80,
            "quantile_threshold": None,
            "note": "Fallback interval — not calibrated",
        }

    def predict_interval_risk(
        self, risk_scores: np.ndarray, method: str = "marginal"
    ) -> Dict[str, Any]:
        """
        Conformal prediction specifically for risk scores (0-100 scale).

        Supports:
        - marginal: Standard marginal conformal prediction
        - adaptive: Locally adaptive (wider intervals for higher risk)
        """
        if not self.is_calibrated:
            return self._fallback_interval(risk_scores)

        risk_scores = np.atleast_1d(risk_scores).astype(float)
        threshold = self.quantile_threshold

        if method == "adaptive":
            # Locally adaptive: scale threshold by prediction magnitude
            # Higher risk → wider intervals (more uncertain)
            adaptive_scale = 0.8 + 0.4 * (risk_scores / 100)
            threshold = threshold * adaptive_scale

        lower = np.clip(risk_scores - threshold, 0, 100)
        upper = np.clip(risk_scores + threshold, 0, 100)

        # Classify uncertainty level
        widths = upper - lower
        uncertainty_labels = []
        for w in widths:
            if w < 15:
                uncertainty_labels.append("LOW_UNCERTAINTY")
            elif w < 30:
                uncertainty_labels.append("MODERATE_UNCERTAINTY")
            else:
                uncertainty_labels.append("HIGH_UNCERTAINTY")

        return {
            "point_prediction": [round(float(v), 2) for v in risk_scores],
            "lower_bound": [round(float(v), 2) for v in lower],
            "upper_bound": [round(float(v), 2) for v in upper],
            "interval_width": [round(float(w), 2) for w in widths],
            "uncertainty_labels": uncertainty_labels,
            "confidence_level": round(self.coverage_guarantee, 2),
            "method": method,
        }

    def _save_calibration(self, report: Dict):
        """Persist this calibration under its target, preserving other targets.

        Format (v2)::

            {"version": 2, "targets": {"<target>": {alpha, quantile_threshold,
             calibration_scores, report, saved_at}}}

        Every writer used to clobber the same flat file, so the last calibration
        to run decided the intervals served for *every* quantity. Keeping the
        existing entries untouched means the trainer, the evaluation suite and
        the API can each persist their own calibration without interfering.
        """
        key = self.target or "_unlabelled"
        try:
            os.makedirs(os.path.dirname(self._calibration_path), exist_ok=True)
            data: Dict[str, Any] = {"version": 2, "targets": {}}

            if os.path.exists(self._calibration_path):
                try:
                    with open(self._calibration_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if isinstance(existing, dict) and existing.get("version") == 2:
                        data = existing
                        data.setdefault("targets", {})
                    elif isinstance(existing, dict) and "calibration_scores" in existing:
                        # Migrate the legacy single-calibration file rather than
                        # discarding it: its target was never recorded, so it is
                        # preserved under an explicit key instead of being
                        # silently relabelled as the target being saved now.
                        data["targets"]["_legacy_unlabelled"] = {
                            k: existing.get(k)
                            for k in ("alpha", "quantile_threshold", "calibration_scores", "report")
                        }
                        data["targets"]["_legacy_unlabelled"]["note"] = (
                            "migrated from the pre-v2 single-calibration file, which did "
                            "not record which quantity it was calibrated on"
                        )
                except Exception:
                    pass

            data["targets"][key] = {
                "alpha": self.alpha,
                "quantile_threshold": self.quantile_threshold,
                "calibration_scores": self.calibration_scores,
                "report": report,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            tmp = self._calibration_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._calibration_path)
        except Exception as e:
            logger.warning(f"Could not save calibration: {e}")

    def load_calibration(self, target: Optional[str] = None) -> bool:
        """Load the calibration for ``target`` (default: this instance's target).

        Falls back to the legacy flat format so a pre-v2 file still loads.
        """
        key = target or self.target
        try:
            if not os.path.exists(self._calibration_path):
                return False
            with open(self._calibration_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entry: Optional[Dict[str, Any]] = None
            if isinstance(data, dict) and data.get("version") == 2:
                targets = data.get("targets") or {}
                if key and key in targets:
                    entry = targets[key]
                else:
                    logger.warning(
                        "No conformal calibration stored for target %r; available: %s. "
                        "Serving intervals from an uncalibrated predictor would be "
                        "worse than saying so, so the heuristic fallback is used.",
                        key,
                        ", ".join(sorted(targets)) or "none",
                    )
                    return False
            elif isinstance(data, dict) and "calibration_scores" in data:
                entry = data

            if entry is None:
                return False

            self.alpha = entry["alpha"]
            self.quantile_threshold = entry["quantile_threshold"]
            self.calibration_scores = entry["calibration_scores"]
            self.is_calibrated = True
            self.coverage_guarantee = 1 - self.alpha
            logger.info(
                "Loaded conformal calibration for %s (threshold=%.2f, n=%d)",
                key or "unlabelled",
                self.quantile_threshold,
                len(self.calibration_scores),
            )
            return True
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")
        return False


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Locally adaptive conformal prediction that produces variable-width
    intervals based on input features.

    Uses a conditional quantile approach:
    - Train a residual predictor q(x) that estimates local difficulty
    - Use q(x) instead of global quantile for interval width
    - Produces narrower intervals where the model is more confident
    """

    def __init__(self, alpha: float = ALPHA_DEFAULT):
        super().__init__(alpha)
        self.residual_model = None
        self.feature_means = None
        self.feature_stds = None

    def calibrate_conditional(
        self, X_cal: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calibrate with feature-conditioned intervals.

        Args:
            X_cal: Calibration features (n_samples, n_features)
            y_true: Ground truth values
            y_pred: Model predictions

        Returns:
            Calibration report
        """
        # Compute residuals
        residuals = np.abs(y_true - y_pred)

        # Fit a simple residual predictor (gradient boosting on features)
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            self.residual_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1
            )
            self.residual_model.fit(X_cal, residuals)

            # Predicted residuals on calibration set
            predicted_residuals = self.residual_model.predict(X_cal)

            # Conformal scores: residual / predicted_residual
            # This normalizes by local difficulty
            ratios = residuals / np.maximum(predicted_residuals, 0.01)

            # Compute threshold on ratios
            n = len(ratios)
            quantile_idx = math.ceil((n + 1) * (1 - self.alpha)) / n
            quantile_idx = min(quantile_idx, 1.0)
            sorted_ratios = np.sort(ratios)
            idx = int(quantile_idx * (n - 1))
            idx = min(idx, n - 1)
            self.quantile_threshold = float(sorted_ratios[idx])

            self.calibration_scores = ratios.tolist()
            self.is_calibrated = True

            # Evaluate
            predicted_widths = 2 * predicted_residuals * self.quantile_threshold
            lower = y_pred - predicted_residuals * self.quantile_threshold
            upper = y_pred + predicted_residuals * self.quantile_threshold
            in_interval = (y_true >= lower) & (y_true <= upper)
            empirical_coverage = float(np.mean(in_interval))

            return {
                "type": "adaptive_conformal",
                "alpha": self.alpha,
                "guaranteed_coverage": self.coverage_guarantee,
                "empirical_coverage": round(empirical_coverage, 4),
                "avg_interval_width": round(float(np.mean(predicted_widths)), 2),
                "residual_model_r2": round(
                    float(self.residual_model.score(X_cal, residuals)), 4
                ),
            }

        except ImportError:
            logger.warning("scikit-learn not available for adaptive conformal")
            return super().calibrate(y_true, y_pred)

    def predict_interval_conditional(
        self, X_new: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Generate feature-conditioned prediction intervals."""
        y_pred = np.atleast_1d(y_pred).astype(float)

        if self.residual_model is not None:
            predicted_residuals = self.residual_model.predict(X_new)
            widths = 2 * predicted_residuals * self.quantile_threshold
        else:
            widths = np.full_like(y_pred, 2 * self.quantile_threshold)

        lower = np.clip(y_pred - widths / 2, 0, 100)
        upper = np.clip(y_pred + widths / 2, 0, 100)

        return {
            "point_prediction": [round(float(v), 2) for v in y_pred],
            "lower_bound": [round(float(v), 2) for v in lower],
            "upper_bound": [round(float(v), 2) for v in upper],
            "interval_width": [round(float(w), 2) for w in widths],
            "confidence_level": round(self.coverage_guarantee, 2),
            "adaptive": True,
        }


# Singleton
# The shared instance is used by the inference engine to put an interval on the
# risk/severity score (0-100), so it is labelled for exactly that target. The
# label decides which stored calibration it loads and where it saves, which is
# what keeps it from serving intervals derived from a different quantity.
conformal_predictor = ConformalPredictor(alpha=ALPHA_DEFAULT, target="risk_severity_score")
adaptive_conformal = AdaptiveConformalPredictor(alpha=ALPHA_DEFAULT)
