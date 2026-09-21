"""
VARUNA AI Model Training Pipeline
===================================
Trains all 9 AI/ML modules on the existing 72-timestep Mumbai dataset.
Saves checkpoints to models/checkpoints/ for use by inference engine.

Uses:
- PyTorch for neural network training
- scikit-learn for data splitting and metrics
- Conformal prediction for calibration
- SHAP for post-hoc explainability
"""

import os
import sys
import json
import math
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger("VARUNA.Trainer")

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — training disabled")

try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


from app.services.ai.feature_contract import (
    CANONICAL_FEATURES,
    NOWCAST_CHANNELS,
    NOWCAST_HORIZON,
    NOWCAST_NORMALIZATION_PATH,
    NOWCAST_TREND_RELAX_HOURS,
    NOWCAST_WINDOW,
    SPLIT_CALIBRATE_END,
    SPLIT_TEST_END,
    SPLIT_TRAIN_END,
    build_model_row,
    describe_split,
    resolve_feature,
    split_sample_bounds,
    split_timestep_bounds,
    validate_feature_window,
)
from app.services.ai.physics_residual import (
    TARGET_SOURCES,
    compute_label_severity,
    compute_physics_baseline,
    read_observed_labels,
    save_residual_config,
)

# Feature columns used by the models — sourced from canonical contract (26 features).
# Inference builds its input with the same helper, so the two can never disagree
# on the feature vector a checkpoint expects.
FEATURE_COLS = list(CANONICAL_FEATURES)

GRID_ROWS = 10
GRID_COLS = 9
GRID_CELLS = GRID_ROWS * GRID_COLS  # 90

# Cap on positive-class weighting for the hazard heads.
#
# The hazard labels are now the dataset's observed flags, which are rare:
# flash-flood fires on ~6% of training cells and cloudburst on ~4.5%. Under
# plain MSE (the previous approach) the loss-minimising prediction for an
# imbalanced binary target is a near-constant probability at the base rate, so
# the heads learn to output ~0.04 everywhere and detect nothing -- POD collapses
# while the loss looks excellent. A capped positive weight keeps the rare-event
# gradient comparable to the common class without letting it dominate.
MAX_POSITIVE_WEIGHT = 20.0

# Fixed seed for every stochastic step in training (weight init, dropout,
# shuffling). Without it a retrain on the identical frozen split produced
# different weights each run, which had two concrete consequences: the
# "retrain on the frozen split" deliverable could not be reproduced, and the
# frozen-baseline regression guard fired spuriously -- a no-op retrain reported
# "drifted from baseline (7 findings)" (flood depth R2 0.622 -> 0.726,
# MAE 3.74 -> 2.72 cm) purely from run-to-run variance. A guard that cries wolf
# on unchanged code cannot detect a real regression, so reproducibility is a
# correctness requirement here rather than a convenience. The seed is recorded
# in training_metadata.json.
DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG the training path draws from, and return the seed used."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic kernels where the backend supports them; the trainer
        # runs on CPU for hackathon compatibility, so this is only for parity.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def _weighted_bce(pred, target, pos_weight: float = 1.0):
    """Binary cross-entropy with the positive class up-weighted.

    ``pos_weight`` multiplies the loss contribution of positive samples only,
    which is what lets a rare observed flag (flash flood, cloudburst) still
    move the gradient. Clamped away from 0/1 so ``log`` stays finite.
    """
    import torch

    eps = 1e-6
    p = pred.clamp(eps, 1.0 - eps)
    loss = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
    weight = torch.where(
        target > 0.5,
        torch.full_like(target, float(pos_weight)),
        torch.ones_like(target),
    )
    return (loss * weight).mean()


class VARUNATrainer:
    """
    End-to-end training pipeline for all VARUNA AI modules.

    Data flow:
    1. Load feature_grid_timeseries.json
    2. Extract feature vectors and targets for each cell/timestep
    3. Create sliding window datasets
    4. Train each module
    5. Save checkpoints
    6. Calibrate conformal predictor
    7. Compute SHAP values on trained model
    """

    def __init__(self, data_dir: str = None, model_dir: str = None, seed: int = DEFAULT_SEED):
        # Seed first: weight initialisation happens in the training methods, so
        # seeding here covers every entry path, including callers that import
        # the trainer rather than using the CLI.
        self.seed = seed_everything(seed)

        if data_dir is None:
            # Navigate from backend/app/services/ai/ to project root/data
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "checkpoints"))

        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = torch.device("cpu")  # CPU for hackathon compatibility

        # Data storage
        self.raw_data = None
        # Physics baseline and the residual targets the regression heads learn.
        self.physics_severity = None
        self.physics_depth = None
        self.residual_severity = None
        self.residual_depth = None
        # Slice of the (time-ordered) sample axis held out as the test split,
        # so evaluation can align the physics baseline with the test rows.
        self._test_slice = None
        self.depth_residual_scale = None
        self.feature_matrix = None
        self.target_matrix = None
        self.scaler = StandardScaler()

    def load_data(self) -> Dict[str, Any]:
        """Load and preprocess the feature grid timeseries."""
        data_path = os.path.join(self.data_dir, "feature_grid_timeseries.json")

        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        timesteps = self.raw_data["timesteps"]
        n_timesteps = len(timesteps)

        logger.info(f"Loaded {n_timesteps} timesteps with {GRID_CELLS} cells each")

        # Validation gate against canonical contract
        report = validate_feature_window(timesteps, label="VARUNATrainer.load_data")
        if not report.ok:
            logger.warning("Feature contract validation warnings:\n%s", report.summary())

        # Build feature matrix: (n_timesteps * n_cells, n_features)
        all_features = []
        all_targets = []
        all_physics_severity = []
        all_physics_depth = []

        for ts in timesteps:
            cells = [c for c in ts.get("features", []) if isinstance(c, dict)]

            # Physics baseline per cell (SCS-CN runoff + Manning flow). The
            # regression heads learn the residual against this, so it must be
            # computed from exactly the same cell values the labels use.
            baseline_severity, baseline_depth = compute_physics_baseline(
                cells,
                model=None,
            )

            for idx, cell in enumerate(cells):
                # Identical construction path to inference (see feature_contract).
                all_features.append(build_model_row(cell))

                # Labels are the dataset's OBSERVED targets -- the same columns
                # the evaluation suite scores against. Only the thunderstorm head
                # falls back to a heuristic, because this dataset carries no
                # thunderstorm ground truth (see TARGET_SOURCES).
                cb_prob, ff_prob, severity, depth = read_observed_labels(cell)
                ts_prob = compute_label_severity(cell)[0]

                all_targets.append([ts_prob, cb_prob, ff_prob, severity, depth])
                all_physics_severity.append(baseline_severity[idx])
                all_physics_depth.append(baseline_depth[idx])

        self.feature_matrix = np.array(all_features, dtype=np.float32)
        self.target_matrix = np.array(all_targets, dtype=np.float32)
        self.physics_severity = np.array(all_physics_severity, dtype=np.float32)
        self.physics_depth = np.array(all_physics_depth, dtype=np.float32)

        # Residual targets: what physics cannot explain. This is what the
        # severity and depth heads are actually trained to predict.
        self.residual_severity = self.target_matrix[:, 3] - self.physics_severity
        self.residual_depth = self.target_matrix[:, 4] - self.physics_depth

        # Separate targets for convenience
        self.depth_targets = self.target_matrix[:, 4].copy()
        self.severity_targets = self.target_matrix[:, 3].copy()
        self.hazard_probs = self.target_matrix[:, :3].copy()  # ts, cb, ff

        # Report how much of the signal physics already carries — this is the
        # justification for residual learning, and it is measured, not assumed.
        physics_r2_severity = self._r2(self.severity_targets, self.physics_severity)
        physics_r2_depth = self._r2(self.depth_targets, self.physics_depth)
        logger.info(
            f"Physics baseline R2: severity={physics_r2_severity:.4f}, "
            f"depth={physics_r2_depth:.4f} — heads now learn the residual"
        )

        logger.info(
            f"Feature matrix: {self.feature_matrix.shape}, "
            f"Target matrix: {self.target_matrix.shape} (ts, cb, ff, severity, depth)"
        )

        return {
            "n_timesteps": n_timesteps,
            "n_cells": GRID_CELLS,
            "n_samples": len(self.feature_matrix),
            "n_features": self.feature_matrix.shape[1],
            "physics_baseline_r2": {
                "severity": round(physics_r2_severity, 4),
                "depth": round(physics_r2_depth, 4),
            },
            "residual_stats": {
                "severity_mean": round(float(self.residual_severity.mean()), 4),
                "severity_p99_abs": round(float(np.percentile(np.abs(self.residual_severity), 99)), 4),
                "depth_mean": round(float(self.residual_depth.mean()), 4),
                "depth_p99_abs": round(float(np.percentile(np.abs(self.residual_depth), 99)), 4),
            },
            "target_distribution": {
                int(k): int(v) for k, v in zip(*np.unique(self.target_matrix, return_counts=True))
            },
        }

    def train_all(self, epochs: int = 200, lr: float = 0.001) -> Dict[str, Any]:
        """Train all models and return training reports."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")

        if self.feature_matrix is None:
            self.load_data()

        results = {}

        # Frozen time-based split -- single source of truth in feature_contract.
        # The previous revision computed 80/5/15 from sample counts here while
        # run_evaluation.py used its own boundaries, so the two disagreed.
        n_samples = len(self.feature_matrix)
        bounds = split_sample_bounds(GRID_CELLS)
        train_idx = bounds["train"][1]
        val_idx = bounds["test"][0]
        test_end = bounds["test"][1]
        logger.info("Frozen split: %s", describe_split())

        X_train = self.feature_matrix[:train_idx]
        y_train = self.target_matrix[:train_idx]
        X_val = self.feature_matrix[train_idx:val_idx]
        y_val = self.target_matrix[train_idx:val_idx]
        X_test = self.feature_matrix[val_idx:test_end]
        y_test = self.target_matrix[val_idx:test_end]

        # Depth regression split (time-based)
        d_train = self.depth_targets[:train_idx]
        d_val = self.depth_targets[train_idx:val_idx]
        d_test = self.depth_targets[val_idx:test_end]

        # Physics baseline splits. The regression heads are trained on
        # (label - baseline), so the baseline is sliced on the same time axis.
        p_sev_train = self.physics_severity[:train_idx]
        p_sev_val = self.physics_severity[train_idx:val_idx]
        p_sev_test = self.physics_severity[val_idx:test_end]
        p_dep_train = self.physics_depth[:train_idx]
        p_dep_val = self.physics_depth[train_idx:val_idx]
        p_dep_test = self.physics_depth[val_idx:test_end]

        resid_d_train = d_train - p_dep_train
        resid_d_val = d_val - p_dep_val
        resid_d_test = d_test - p_dep_test

        # Remember which rows form the test split (used by _evaluate_all).
        self._test_slice = slice(val_idx, test_end)

        # Characterise the holdout windows before trusting any metric computed
        # on them. A split that puts the whole storm event inside the train
        # window leaves a holdout with zero flooded cells -- where a constant-zero
        # predictor scores perfectly and depth/severity error looks like skill.
        # Recording the actual event counts makes that failure mode visible
        # instead of flattering, and is the sole basis for the caveat written
        # into the metadata: the caveat is DERIVED from these numbers, never
        # asserted, so it cannot go stale when the feature table changes.
        #
        # ``rain``/``flooding`` are flat over (timestep, cell), the same axis
        # train_idx/val_idx index, so the slices are in consistent units.
        holdout_note = {}
        for label, sl in (("validation", slice(train_idx, val_idx)), ("test", slice(val_idx, None))):
            rain = np.asarray(
                [resolve_feature(c, "rainfall_1h_mm", 0) or 0
                 for ts in self.raw_data["timesteps"] for c in ts["features"]],
                dtype=np.float32,
            )[sl]
            depth = np.asarray(self.depth_targets, dtype=np.float32)[sl]
            severity = np.asarray(self.severity_targets, dtype=np.float32)[sl]
            # hazard_probs columns are (thunderstorm, cloudburst, flash_flood).
            hazards = np.asarray(self.hazard_probs, dtype=np.float32)[sl]
            flooding = depth > 0
            holdout_note[label] = {
                "rainfall_max_mm_hr": round(float(rain.max()), 2) if rain.size else 0.0,
                "rainfall_mean_mm_hr": round(float(rain.mean()), 2) if rain.size else 0.0,
                "cells_with_flood_depth": int(flooding.sum()),
                "cells_total": int(flooding.size),
                "max_flood_depth_cm": round(float(depth.max()), 2) if depth.size else 0.0,
                # severity_targets come from observed_severity_score(), which is
                # the 0-100 score scale (class x 100/3), NOT the raw 0-3 class.
                # Record both so neither number is read at the wrong scale.
                "max_severity_score": round(float(severity.max()), 1) if severity.size else 0.0,
                "max_severity_class": (
                    int(round(float(severity.max()) * 3.0 / 100.0)) if severity.size else 0
                ),
                "flash_flood_cells": int((hazards[:, 2] > 0).sum()) if hazards.size else 0,
                "cloudburst_cells": int((hazards[:, 1] > 0).sum()) if hazards.size else 0,
                "degenerate_for_depth_and_severity": bool(flooding.sum() == 0),
            }
        self._holdout_note = holdout_note

        if any(v["degenerate_for_depth_and_severity"] for v in holdout_note.values()):
            logger.warning(
                "HOLDOUT WINDOW CAVEAT: %s holdout window(s) contain zero cells with "
                "flood depth, so depth/severity error there is near-zero for any "
                "predictor. Quote these metrics only alongside this caveat.",
                ", ".join(k for k, v in holdout_note.items() if v["degenerate_for_depth_and_severity"]),
            )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Save scaler
        import pickle
        with open(os.path.join(self.model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        logger.info(
            "Train: %d (timesteps 1-%d), Val: %d (%d-%d), Test: %d (%d-%d)",
            len(X_train), SPLIT_TRAIN_END,
            len(X_val), SPLIT_TRAIN_END + 1, SPLIT_CALIBRATE_END,
            len(X_test), SPLIT_CALIBRATE_END + 1, SPLIT_TEST_END,
        )

        # ─── Train Multi-Hazard Predictor (Module 4) ───
        logger.info("Training Multi-Hazard Predictor (physics-residual regression heads)...")
        results["multi_hazard"] = self._train_multi_hazard(
            X_train_scaled, y_train, X_val_scaled, y_val, epochs, lr,
            physics_severity_train=p_sev_train,
            physics_depth_train=p_dep_train,
            physics_severity_val=p_sev_val,
            physics_depth_val=p_dep_val,
        )

        # ─── Train Nowcaster (Module 3) ───
        logger.info("Training Nowcaster...")
        results["nowcaster"] = self._train_nowcaster(epochs, lr)

        # ─── Train Storm Cell Detector (Module 1) ───
        logger.info("Training Storm Cell Detector...")
        results["storm_cell"] = self._train_storm_cell_detector(
            X_train_scaled, y_train, X_val_scaled, y_val, epochs, lr
        )

        # ─── Train Crowd Report NLP (Module 9) ───
        logger.info("Training Crowd Report NLP...")
        results["crowd_nlp"] = self._train_crowd_nlp(epochs)

        # ─── Train Trust Scorer (Module 7) ───
        logger.info("Training Trust Scorer...")
        results["trust_scorer"] = self._train_trust_scorer(
            X_train_scaled, X_val_scaled, epochs, lr
        )

        # ─── Train XAI Attention Layer (Module 8) ───
        logger.info("Training XAI Attention Layer...")
        results["xai"] = self._train_xai(
            X_train_scaled, y_train, epochs, lr
        )

        # ─── Train Flood Depth Estimator (Module 6) ───
        logger.info("Training Flood Depth Estimator (physics-residual)...")
        results["flood_depth"] = self._train_flood_depth(
            X_train_scaled, resid_d_train, X_val_scaled, resid_d_val, epochs, lr
        )

        # ─── Calibrate Conformal Predictor ───
        logger.info("Calibrating Conformal Predictor...")
        results["conformal"] = self._calibrate_conformal(
            X_train_scaled, y_train, X_val_scaled, y_val,
            physics_severity_val=p_sev_val,
            physics_depth_val=p_dep_val,
        )

        # ─── Compute SHAP values ───
        logger.info("Computing SHAP values...")
        results["shap"] = self._compute_shap(X_train_scaled, y_train)

        # ─── Evaluate on test set ───
        # ``resid_d_test`` is the residual target; _evaluate_all adds the
        # physics baseline back before computing the reported cm error.
        logger.info("Evaluating on test set...")
        results["test_evaluation"] = self._evaluate_all(
            X_test_scaled, y_test, resid_d_test
        )

        # Persist the residual-head scales so inference reproduces training
        # exactly (y_hat = physics_baseline + scale * tanh(head)).
        residual_config_path = save_residual_config(
            self.model_dir,
            extra={
                "physics_baseline_r2_severity": round(
                    float(self._r2(self.severity_targets, self.physics_severity)), 4
                ),
                "physics_baseline_r2_depth": round(
                    float(self._r2(self.depth_targets, self.physics_depth)), 4
                ),
                "trained_on_timesteps": len(self.raw_data["timesteps"]),
            },
        )

        # ─── Calibrate the cross-source agreement anomaly threshold ───
        #
        # The flag used a bare 0.6 threshold that the score could never reach,
        # so every timestep came back anomalous. The threshold is derived from
        # the calibration window -- disjoint from training and from the frozen
        # test set -- so it is fitted, not guessed, and not test-tuned.
        results["fusion_calibration"] = self._calibrate_fusion_threshold()

        # Save training metadata
        metadata = {
            "training_date": str(np.datetime64("now")),
            "n_timesteps": len(self.raw_data["timesteps"]),
            "n_features": len(FEATURE_COLS),
            "feature_columns": FEATURE_COLS,
            "model_input_features": list(FEATURE_COLS),
            "residual_config": residual_config_path,
            "split": describe_split(),
            "seed": getattr(self, "seed", DEFAULT_SEED),
            "fusion_calibration": results.get("fusion_calibration"),
            "target_sources": dict(TARGET_SOURCES),
            "physics_baseline_r2": {
                "severity": round(float(self._r2(self.severity_targets, self.physics_severity)), 4),
                "depth": round(float(self._r2(self.depth_targets, self.physics_depth)), 4),
            },
            "holdout_windows": holdout_note,
            "evaluation_caveat": self._describe_holdout(holdout_note),
            "results": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
                       for k, v in results.items()},
        }

        with open(os.path.join(self.model_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Training complete! All checkpoints saved.")
        return results

    @staticmethod
    def _describe_holdout(holdout_note: Dict[str, Dict[str, Any]]) -> str:
        """Derive the holdout caveat from the measured window contents.

        This used to be a fixed sentence asserting that the holdout windows
        held "zero flooded cells". After the feature table was rebuilt on the
        2022-07-05 replay that claim became false -- the test window holds 1803
        flooded cells, 272 cloudburst and 103 flash-flood cell-instants -- while
        the computed ``holdout_windows`` block beside it said
        ``degenerate_for_depth_and_severity: false``. Two contradictory claims in
        one artifact. Deriving the sentence from the numbers keeps them agreeing
        and keeps the honest warning (degenerate windows are not skill) intact.
        """
        degenerate = [k for k, v in holdout_note.items() if v["degenerate_for_depth_and_severity"]]

        if degenerate:
            head = (
                "Degenerate holdout window(s) -- "
                + ", ".join(sorted(degenerate))
                + " contain zero cells with flood depth, so depth/severity error there is "
                "near-zero for any predictor, including a constant-zero one. Treat those "
                "metrics as regression checks, not as skill estimates."
            )
        else:
            head = (
                "Neither holdout window is degenerate: both contain flooded cells and "
                "hazard events, so the depth/severity error reported on them is a genuine "
                "held-out skill estimate rather than a constant-zero regression check."
            )

        detail = " ".join(
            f"{label}: {v['cells_with_flood_depth']}/{v['cells_total']} cells flooded, "
            f"{v['flash_flood_cells']} flash-flood and {v['cloudburst_cells']} cloudburst "
            f"cell-instants, max depth {v['max_flood_depth_cm']} cm, max severity "
            f"{v['max_severity_score']}/100 (class {v['max_severity_class']} of 3), "
            f"mean rainfall {v['rainfall_mean_mm_hr']} mm/hr."
            for label, v in holdout_note.items()
        )
        return f"{head} {detail}"

    @staticmethod
    def _r2(y_true, y_pred) -> float:
        """Coefficient of determination, used to quantify the physics baseline."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        total = float(np.sum((y_true - y_true.mean()) ** 2))
        if total <= 0:
            return 0.0
        return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total

    def _calibrate_fusion_threshold(self) -> Dict[str, Any]:
        """Fit the agreement anomaly threshold on the calibration window.

        Uses the inference engine's own fusion code path, so the calibration and
        the runtime score cannot diverge. Falls back to the documented default
        if the engine (or Torch) is unavailable, and says so in the payload.
        """
        from app.services.ai.fusion_calibration import (
            calibrate_agreement_threshold,
            save_fusion_calibration,
        )

        try:
            from app.services.ai.inference_engine import VARUNAInferenceEngine

            timesteps = self.raw_data["timesteps"]
            bounds = split_timestep_bounds()["calibrate"]
            window = timesteps[bounds[0]:bounds[1]]
            if not window:
                raise ValueError("calibration window is empty")

            engine = VARUNAInferenceEngine()
            scores = [
                engine.fuse_multi_source(ts)["global_agreement_score"]
                for ts in window
            ]
            payload = calibrate_agreement_threshold(scores)
            payload["calibration_timesteps"] = (
                f"{bounds[0] + 1}-{bounds[1]}"
            )
        except Exception as exc:  # noqa: BLE001 - calibration must never block training
            logger.warning(
                "Fusion threshold calibration failed (%s); keeping default.", exc
            )
            payload = {
                "agreement_anomaly_threshold": 0.6,
                "source": f"default (calibration failed: {exc})",
            }

        save_fusion_calibration(self.model_dir, payload)
        logger.info(
            "Cross-source agreement anomaly threshold calibrated: %s on %s",
            payload.get("agreement_anomaly_threshold"),
            payload.get("calibration_timesteps", "n/a"),
        )
        return payload

    def _train_multi_hazard(
        self, X_train, y_train, X_val, y_val, epochs, lr,
        physics_severity_train=None,
        physics_depth_train=None,
        physics_severity_val=None,
        physics_depth_val=None,
    ) -> Dict[str, Any]:
        """Train the multi-hazard risk predictor.

        The hazard heads learn their probabilities directly. The severity and
        depth heads learn the **residual** against the SCS-CN + Manning physics
        baseline (``target = y_true - physics_baseline``), because that baseline
        already explains most of the variance. At inference the baseline is
        added back, so the reported values stay absolute while the network only
        has to model the correction.
        """
        from app.services.ai.model_architectures import MultiHazardPredictor

        model = MultiHazardPredictor(in_features=len(FEATURE_COLS)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        # y_train columns: [ts_prob, cb_prob, ff_prob, severity, depth]
        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_tr = torch.FloatTensor(y_train).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)
        y_v = torch.FloatTensor(y_val).to(self.device)

        # Class weights for the hazard heads, from the training split's own
        # positive rates.
        def _pos_weight(targets):
            positives = float(targets.sum())
            if positives <= 0:
                return 1.0
            negatives = float(targets.numel()) - positives
            return float(min(negatives / positives, MAX_POSITIVE_WEIGHT))

        pos_weight = [
            _pos_weight(y_tr[:, 0]),
            _pos_weight(y_tr[:, 1]),
            _pos_weight(y_tr[:, 2]),
        ]
        logger.info(
            "Hazard positive weights (ts/cb/ff): %s",
            [round(w, 2) for w in pos_weight],
        )

        # Physics baselines, and the residual targets the regression heads learn.
        baselines_train: Dict[str, Any] = {}
        baselines_val: Dict[str, Any] = {}
        residual_sev_tr = y_tr[:, 3]
        residual_dep_tr = y_tr[:, 4]
        residual_sev_v = y_v[:, 3]
        residual_dep_v = y_v[:, 4]

        if physics_severity_train is not None:
            base_sev_tr = torch.FloatTensor(physics_severity_train).to(self.device)
            base_dep_tr = torch.FloatTensor(physics_depth_train).to(self.device)
            baselines_train = {"severity": base_sev_tr, "depth": base_dep_tr}
            residual_sev_tr = y_tr[:, 3] - base_sev_tr
            residual_dep_tr = y_tr[:, 4] - base_dep_tr
        if physics_severity_val is not None:
            base_sev_v = torch.FloatTensor(physics_severity_val).to(self.device)
            base_dep_v = torch.FloatTensor(physics_depth_val).to(self.device)
            baselines_val = {"severity": base_sev_v, "depth": base_dep_v}
            residual_sev_v = y_v[:, 3] - base_sev_v
            residual_dep_v = y_v[:, 4] - base_dep_v

        # Loss normalisation for the two regression heads.
        #
        # The heads emit PHYSICAL units (the architecture multiplies the Tanh
        # output by the residual scale), so their raw MSE is ~258 for severity
        # and ~74 for depth, while the hazard BCE is ~0.6. Even at the 0.05
        # secondary weight, raw MSE let the regression terms carry ~97% of the
        # gradient -- which starved the three hazard heads, the primary product
        # of this model. Dividing each regression loss by the
        # trivial-predictor MSE (head output = 0) puts all five terms on the
        # same 0..1 footing, so the weights below mean what they claim.
        severity_loss_norm = max(float((residual_sev_tr ** 2).mean()), 1e-6)
        depth_loss_norm = max(float((residual_dep_tr ** 2).mean()), 1e-6)
        logger.info(
            "Regression loss normalisers: severity=%.2f, depth=%.2f "
            "(trivial-predictor MSE)",
            severity_loss_norm,
            depth_loss_norm,
        )

        best_val_loss = float("inf")
        best_state = None
        patience = 30
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(X_tr, physics_baseline=baselines_train or None)
            # Multi-task loss.
            #   * hazard heads: class-weighted BCE, because the observed flags
            #     are rare and MSE would collapse them to the base rate.
            #   * severity / depth: MSE on the RESIDUAL -- physics already
            #     supplies the absolute part, so the head only learns the
            #     correction.
            loss_ts = _weighted_bce(output["thunderstorm_prob"], y_tr[:, 0], pos_weight[0])
            loss_cb = _weighted_bce(output["cloudburst_prob"], y_tr[:, 1], pos_weight[1])
            loss_ff = _weighted_bce(output["flash_flood_prob"], y_tr[:, 2], pos_weight[2])
            loss_sev = criterion(output["severity_residual"], residual_sev_tr) / severity_loss_norm
            loss_depth = criterion(output["depth_residual"], residual_dep_tr) / depth_loss_norm

            # Weighted loss: probabilities are primary, severity and depth secondary
            loss = 0.30 * loss_ts + 0.30 * loss_cb + 0.30 * loss_ff + 0.05 * loss_sev + 0.05 * loss_depth

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_v, physics_baseline=baselines_val or None)
                val_loss_ts = _weighted_bce(val_output["thunderstorm_prob"], y_v[:, 0], pos_weight[0])
                val_loss_cb = _weighted_bce(val_output["cloudburst_prob"], y_v[:, 1], pos_weight[1])
                val_loss_ff = _weighted_bce(val_output["flash_flood_prob"], y_v[:, 2], pos_weight[2])
                # Include the regression heads so early stopping is driven by
                # the residual fit too, not just the hazard probabilities.
                val_loss_sev = criterion(val_output["severity_residual"], residual_sev_v) / severity_loss_norm
                val_loss_dep = criterion(val_output["depth_residual"], residual_dep_v) / depth_loss_norm
                val_loss = (
                    0.30 * val_loss_ts + 0.30 * val_loss_cb + 0.30 * val_loss_ff
                    + 0.05 * val_loss_sev + 0.05 * val_loss_dep
                )

                # Compute accuracy: dominant hazard matches
                val_ts = val_output["thunderstorm_prob"]
                val_cb = val_output["cloudburst_prob"]
                val_ff = val_output["flash_flood_prob"]
                val_dominant = torch.stack([val_ts, val_cb, val_ff], dim=1).argmax(dim=1)
                true_dominant = y_v[:, :3].argmax(dim=1)
                val_acc = (val_dominant == true_dominant).float().mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 50 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f}, val_acc: {val_acc:.3f}")

        # Load best model
        if best_state:
            model.load_state_dict(best_state)

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(self.model_dir, "multi_hazard_predictor.pt"))

        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_output = model(X_v, physics_baseline=baselines_val or None)
            val_ts = test_output["thunderstorm_prob"]
            val_cb = test_output["cloudburst_prob"]
            val_ff = test_output["flash_flood_prob"]
            val_dominant = torch.stack([val_ts, val_cb, val_ff], dim=1).argmax(dim=1)
            true_dominant = y_v[:, :3].argmax(dim=1)
            final_acc = (val_dominant == true_dominant).float().mean().item()

            # Absolute (baseline + residual) regression error, which is what
            # the API actually reports.
            abs_sev_err = float(
                torch.mean(torch.abs(test_output["severity_score"] - y_v[:, 3]))
            )
            abs_depth_err = float(
                torch.mean(torch.abs(test_output["flood_depth_cm"] - y_v[:, 4]))
            )

        report: Dict[str, Any] = {
            "final_val_accuracy": round(final_acc, 4),
            "best_val_loss": round(best_val_loss.item(), 4),
            "epochs_trained": epoch + 1,
            "checkpoint": "multi_hazard_predictor.pt",
            "residual_learning": True,
            "mae_severity_absolute": round(abs_sev_err, 4),
            "mae_depth_absolute_cm": round(abs_depth_err, 4),
        }

        # Compare against the physics baseline alone so the improvement is
        # explicit rather than asserted.
        if physics_severity_val is not None:
            base_sev = np.asarray(physics_severity_val, dtype=np.float32)
            base_dep = np.asarray(physics_depth_val, dtype=np.float32)
            report["baseline_mae_severity"] = round(
                float(np.mean(np.abs(base_sev - y_val[:, 3]))), 4
            )
            report["baseline_mae_depth_cm"] = round(
                float(np.mean(np.abs(base_dep - y_val[:, 4]))), 4
            )

        return report

    def _train_nowcaster(self, epochs, lr) -> Dict[str, Any]:
        """Train the spatiotemporal nowcaster."""
        from app.services.ai.model_architectures import SpatiotemporalNowcaster

        timesteps = self.raw_data["timesteps"]
        window_size = NOWCAST_WINDOW
        forecast_horizon = NOWCAST_HORIZON

        # Build sequences: input = 6 timesteps, target = next 6 timesteps.
        #
        # Sequences are drawn from the train partition only. The previous
        # revision swept every timestep (range(len-11)), so the nowcaster
        # trained on the same steps the evaluation then scored -- temporal
        # leakage in the headline nowcast metric.
        train_steps = split_timestep_bounds()["train"][1]
        last_start = train_steps - window_size - forecast_horizon
        sequences = []
        targets = []

        for i in range(max(0, last_start + 1)):
            inp_features = []
            tgt_features = []

            for t in range(i, i + window_size):
                ts_features = []
                for cell in timesteps[t]["features"]:
                    ts_features.append([cell.get(col, 0) for col in NOWCAST_CHANNELS])
                inp_features.append(ts_features)

            for t in range(i + window_size, i + window_size + forecast_horizon):
                ts_features = []
                for cell in timesteps[t]["features"]:
                    ts_features.append([cell.get(col, 0) for col in NOWCAST_CHANNELS])
                tgt_features.append(ts_features)

            sequences.append(inp_features)
            targets.append(tgt_features)

        if not sequences:
            return {"note": "Insufficient data for nowcaster training", "skipped": True}

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        C = len(NOWCAST_CHANNELS)

        # --- Per-channel normalization, fitted on the train partition --------
        # Channel scales span three orders of magnitude (soil moisture ~tens,
        # CAPE ~thousands). Trained un-normalized, the MSE was dominated by
        # CAPE: the loss sat at ~6.7e4 and the network collapsed to a constant,
        # emitting the same rainfall (21.16 mm/hr) for every hour from +2h to
        # +6h and ignoring the observed level. Standardizing per channel puts
        # every channel on equal footing in the loss.
        ch_mean = X.reshape(-1, C).mean(axis=0).astype(np.float32)
        ch_std = X.reshape(-1, C).std(axis=0).astype(np.float32)
        ch_std = np.where(ch_std < 1e-6, 1.0, ch_std).astype(np.float32)
        Xn = (X - ch_mean) / ch_std
        yn = (y - ch_mean) / ch_std

        # --- Baseline: damped-persistence trend, per cell per channel --------
        # The network learns the *correction* to this baseline, exactly as the
        # hazard heads learn the correction to the physics model.
        #
        # Why a baseline at all: this dataset is a single event whose build-up
        # fills the train window, so a network trained on absolute fields (or
        # on the raw change from the last frame) can only ever learn "rainfall
        # rises" -- it emitted an ascending forecast straight through the
        # recession. The baseline carries the trend direction from the observed
        # window, so the forecast falls when the input falls; the network adds
        # the learned departure from it.
        per_step_trend = (X[:, -1] - X[:, 0]) / max(1, window_size - 1)  # (N, cells, C)
        hours = np.arange(1, forecast_horizon + 1, dtype=np.float32)
        relax = (1.0 - np.exp(-hours / NOWCAST_TREND_RELAX_HOURS)).astype(np.float32)
        base = X[:, -1][:, None, :, :] + per_step_trend[:, None, :, :] * (
            NOWCAST_TREND_RELAX_HOURS * relax
        )[None, :, None, None]
        base = np.clip(base, 0.0, None)
        base_n = (base - ch_mean) / ch_std

        yn = yn - base_n

        norm_path = os.path.join(self.model_dir, os.path.basename(NOWCAST_NORMALIZATION_PATH))
        with open(norm_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "channels": list(NOWCAST_CHANNELS),
                    "mean": [float(v) for v in ch_mean],
                    "std": [float(v) for v in ch_std],
                    "window": window_size,
                    "horizon": forecast_horizon,
                    "target": "residual_over_trend_baseline",
                    "trend_relax_hours": NOWCAST_TREND_RELAX_HOURS,
                    "fit_partition": f"train 1-{SPLIT_TRAIN_END}",
                },
                handle,
                indent=2,
            )

        # Reshape for ConvLSTM: (batch, time, channels, height, width)
        B = Xn.shape[0]
        X_tensor = torch.FloatTensor(Xn).view(B, window_size, C, GRID_ROWS, GRID_COLS).to(self.device)
        y_tensor = torch.FloatTensor(yn).view(B, forecast_horizon, C, GRID_ROWS, GRID_COLS).to(self.device)

        model = SpatiotemporalNowcaster(
            in_channels=C, hidden_dim=64, forecast_horizon=forecast_horizon
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr * 0.5)
        criterion = nn.MSELoss()

        # Split
        n_train = int(0.8 * B)
        X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
        y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(min(epochs, 150)):
            model.train()
            optimizer.zero_grad()

            output = model(X_train)
            loss = criterion(output["forecast"], y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output["forecast"], y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 30 == 0:
                logger.info(f"  Nowcaster epoch {epoch+1} — loss: {loss.item():.4f}, val: {val_loss.item():.4f}")

        if best_state:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), os.path.join(self.model_dir, "spatiotemporal_nowcaster.pt"))

        return {
            "best_val_loss": round(best_val_loss.item(), 6),
            "n_sequences": len(sequences),
            "train_timesteps": f"1-{SPLIT_TRAIN_END}",
            "val_partition": "tail 20% of the train-partition sequences",
            "channels": list(NOWCAST_CHANNELS),
            "target": "residual_over_trend_baseline",
            "normalization": os.path.basename(norm_path),
            "checkpoint": "spatiotemporal_nowcaster.pt",
            # The honest limits of this model, recorded where the numbers live.
            # The frozen train window 1-32 is a single build-up phase, so the
            # network has no decay examples to learn from: the trend baseline
            # (fitted from the observed input window at inference time) carries
            # the direction of change, and the network contributes the spatial
            # pattern. Expected consequence, visible in the evaluation: skill
            # versus persistence is ~0 at +1..+3h and only turns positive at
            # +4..+6h, and R2 is negative at long leads because the forecast
            # spread exceeds the observed spread. Fixing that needs more storm
            # scenarios in the train partition, not a different architecture.
            "data_basis": (
                f"{len(sequences)} sequences from the frozen train partition "
                f"(timesteps 1-{SPLIT_TRAIN_END}), which is a single build-up phase with "
                "no decay; the calibrate band yields no nowcast sequences at "
                f"window={window_size}/horizon={forecast_horizon}, so validation uses the "
                "train-partition tail. The trend baseline carries the direction of change; "
                "the network supplies the spatial pattern. Short-lead skill vs persistence "
                "is therefore ~0 and only turns positive at +4..+6h."
            ),
        }

    def _train_storm_cell_detector(
        self, X_train, y_train, X_val, y_val, epochs, lr
    ) -> Dict[str, Any]:
        """
        Train storm cell detection model (Module 1).

        Labels: cells with CAPE > 1500 AND CTT < -40 AND rain > 10 are storm cells.
        Architecture: Simple binary classifier.
        """
        import torch.nn as nn

        cape_idx = FEATURE_COLS.index("cape_instability_jkg") if "cape_instability_jkg" in FEATURE_COLS else 7
        rain_idx = FEATURE_COLS.index("rainfall_1h_mm") if "rainfall_1h_mm" in FEATURE_COLS else 0
        ctt_idx = FEATURE_COLS.index("cloud_top_temp_celsius") if "cloud_top_temp_celsius" in FEATURE_COLS else 10
        wind_idx = FEATURE_COLS.index("wind_speed_10m_kmh") if "wind_speed_10m_kmh" in FEATURE_COLS else 12

        # Generate storm cell labels from ACTUAL meteorological criteria
        # Storm cells: high CAPE + cold cloud tops + significant rainfall
        storm_labels_train = []
        for i in range(len(X_train)):
            cape = X_train[i][cape_idx] * 3000 if len(X_train[i]) > cape_idx else 0  # denormalize
            rain = X_train[i][rain_idx] * 80 if len(X_train[i]) > rain_idx else 0
            ctt = X_train[i][ctt_idx] * 60 + 10 if len(X_train[i]) > ctt_idx else 0  # denormalize
            wind = X_train[i][wind_idx] * 40 if len(X_train[i]) > wind_idx else 0
            # Storm = CAPE>1000 AND CTT<-30 AND rain>5 OR wind>25
            is_storm = 1.0 if (cape > 1000 and ctt < -30 and rain > 5) or (wind > 25 and rain > 10) else 0.0
            storm_labels_train.append(is_storm)
        storm_labels_train = torch.tensor(storm_labels_train, dtype=torch.float32).to(self.device)
        n_storms = int(storm_labels_train.sum().item())
        logger.info(f"Storm cell labels: {n_storms}/{len(X_train)} storm cells ({100*n_storms/len(X_train):.1f}%)")

        storm_labels_val = []
        for i in range(len(X_val)):
            cape = X_val[i][cape_idx] * 3000 if len(X_val[i]) > cape_idx else 0
            rain = X_val[i][rain_idx] * 80 if len(X_val[i]) > rain_idx else 0
            ctt = X_val[i][ctt_idx] * 60 + 10 if len(X_val[i]) > ctt_idx else 0
            wind = X_val[i][wind_idx] * 40 if len(X_val[i]) > wind_idx else 0
            is_storm = 1.0 if (cape > 1000 and ctt < -30 and rain > 5) or (wind > 25 and rain > 10) else 0.0
            storm_labels_val.append(is_storm)
        storm_labels_val = torch.tensor(storm_labels_val, dtype=torch.float32).to(self.device)

        # Binary classifier: input features → storm probability
        model = nn.Sequential(
            nn.Linear(len(FEATURE_COLS), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        # Convert numpy to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(min(epochs, 100)):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_t).squeeze()
            loss = criterion(output, storm_labels_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t).squeeze()
                val_loss = criterion(val_output, storm_labels_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), os.path.join(self.model_dir, "storm_cell_detector.pt"))

        return {
            "best_val_loss": round(best_val_loss.item(), 6),
            "checkpoint": "storm_cell_detector.pt",
        }

    def _train_crowd_nlp(self, epochs: int = 100) -> Dict[str, Any]:
        """
        Train crowd report NLP classifier (Module 9).

        Uses synthetic labeled crowd reports for training.
        Architecture: Embedding + Linear classifier.
        """
        import torch.nn as nn
        from app.services.ai.data_loader import FLOOD_VOCAB, tokenize_crowd_report

        # Synthetic labeled training data
        train_reports = [
            ("heavy rain flooding streets waterlogged help", 1),
            ("emergency water rising fast rescue needed", 1),
            ("flood in mumbai roads submerged cars stuck", 1),
            ("paani badh raha hai bachao", 1),
            ("waterlogged area near bandra underpass", 1),
            ("deep water on road cannot drive", 1),
            ("submerged vehicles need rescue", 1),
            ("overflow drain near andheri", 1),
            ("rain stopped water going down", 0),
            ("roads clear traffic normal", 0),
            ("weather fine no issues", 0),
            ("thik hai sab sukha hai", 0),
            ("dry roads no water", 0),
            ("normal day no flooding", 0),
            ("clear weather good conditions", 0),
            ("road repair work happening", 2),
            ("match tonight at stadium", 2),
            ("sale offer discount shop", 2),
            ("movie release this friday", 2),
            ("traffic jam due to construction", 2),
        ]

        # Tokenize
        max_len = 128
        X_list = []
        y_list = []
        for text, label in train_reports:
            tokens = tokenize_crowd_report(text, max_len)
            X_list.append(tokens)
            y_list.append(label)

        X = torch.tensor(X_list, dtype=torch.long).to(self.device)
        y = torch.tensor(y_list, dtype=torch.long).to(self.device)

        # Simple classifier: Embedding → Global Max Pool → Linear
        vocab_size = max(FLOOD_VOCAB.values()) + 10
        embed_dim = 32

        class CrowdNLPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.pool = nn.AdaptiveMaxPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(embed_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 3),
                )
            def forward(self, x):
                emb = self.embedding(x)  # (B, seq_len, embed_dim)
                emb = emb.permute(0, 2, 1)  # (B, embed_dim, seq_len)
                pooled = self.pool(emb).squeeze(-1)  # (B, embed_dim)
                return self.classifier(pooled)

        model = CrowdNLPModel().to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(min(epochs, 200)):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(self.model_dir, "crowd_nlp_classifier.pt"))

        return {
            "checkpoint": "crowd_nlp_classifier.pt",
            "n_reports": len(train_reports),
        }

    def _train_trust_scorer(self, X_train, X_val, epochs, lr) -> Dict[str, Any]:
        """Train the forecast trust scorer (autoencoder-based)."""
        from app.services.ai.model_architectures import ForecastTrustScorer

        model = ForecastTrustScorer(input_dim=len(FEATURE_COLS)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        mse_crit = nn.MSELoss()

        X_tr = torch.FloatTensor(X_train).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)

        for epoch in range(min(epochs, 100)):
            model.train()
            optimizer.zero_grad()

            output = model(X_tr)
            # Autoencoder reconstruction loss
            recon_loss = mse_crit(output["reconstruction_error"].unsqueeze(1).expand_as(X_tr[:, :1]), X_tr[:, :1] * 0)
            # Trust classification: use reconstruction error as signal
            trust_labels = (output["reconstruction_error"] > output["reconstruction_error"].median()).long()
            trust_loss = nn.CrossEntropyLoss()(output["trust_logits"], trust_labels)

            loss = recon_loss + trust_loss
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0:
                logger.info(f"  Trust scorer epoch {epoch+1} — loss: {loss.item():.4f}")

        torch.save(model.state_dict(), os.path.join(self.model_dir, "forecast_trust_scorer.pt"))

        return {"checkpoint": "forecast_trust_scorer.pt"}

    def _train_xai(self, X_train, y_train, epochs, lr) -> Dict[str, Any]:
        """Train the XAI attention layer."""
        from app.services.ai.model_architectures import XAIAttentionLayer

        model = XAIAttentionLayer(num_features=len(FEATURE_COLS)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        X_tr = torch.FloatTensor(X_train).to(self.device)

        for epoch in range(min(epochs, 80)):
            model.train()
            optimizer.zero_grad()

            output = model(X_tr)
            # Importance should sum to ~1
            importance_loss = (output["feature_importance"].sum(dim=-1) - 1.0).pow(2).mean()
            # Attention should be diverse (not collapse to one feature)
            attn_entropy = -(output["attention_weights"].clamp(1e-8) * output["attention_weights"].clamp(1e-8).log()).sum(dim=-1).mean()

            loss = importance_loss - 0.01 * attn_entropy
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"  XAI epoch {epoch+1} — loss: {loss.item():.6f}")

        torch.save(model.state_dict(), os.path.join(self.model_dir, "xai_attention_layer.pt"))

        # Extract learned feature importance
        model.eval()
        with torch.no_grad():
            sample = X_tr[:100]
            output = model(sample)
            learned_importance = output["feature_importance"].mean(dim=0).cpu().numpy()

        importance_map = {col: round(float(learned_importance[i]), 4) for i, col in enumerate(FEATURE_COLS)}

        return {
            "checkpoint": "xai_attention_layer.pt",
            "learned_feature_importance": importance_map,
        }

    def _train_flood_depth(
        self, X_train, d_train, X_val, d_val, epochs, lr, residual_scale=None
    ) -> Dict[str, Any]:
        """Train the flood depth estimator on the physics RESIDUAL.

        The head ends in ``Tanh`` scaled by ``residual_scale`` because a depth
        correction is signed; the old ``ReLU`` output could not represent a
        negative correction. Inference adds the physics water depth back.
        """
        from app.services.ai.model_architectures import DEPTH_RESIDUAL_SCALE

        if residual_scale is None:
            residual_scale = DEPTH_RESIDUAL_SCALE
        self.depth_residual_scale = float(residual_scale)

        model = nn.Sequential(
            nn.Linear(len(FEATURE_COLS), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_tr = torch.FloatTensor(X_train).to(self.device)
        # Target is the residual normalised into Tanh's (-1, 1) range; inference
        # multiplies back by the same scale.
        d_tr = torch.FloatTensor(d_train).unsqueeze(1).to(self.device) / float(residual_scale)
        X_v = torch.FloatTensor(X_val).to(self.device)
        d_v = torch.FloatTensor(d_val).unsqueeze(1).to(self.device) / float(residual_scale)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(min(epochs, 150)):
            model.train()
            optimizer.zero_grad()

            pred = model(X_tr)
            loss = criterion(pred, d_tr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = criterion(val_pred, d_v)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 30 == 0:
                logger.info(f"  Depth estimator epoch {epoch+1} — loss: {loss.item():.4f}")

        if best_state:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), os.path.join(self.model_dir, "flood_depth_estimator.pt"))

        # Evaluate MAE on the residual scale the head actually predicts.
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            mae = torch.mean(torch.abs(val_pred - d_v)).item()

        return {
            "checkpoint": "flood_depth_estimator.pt",
            # Reported in cm: the head trains in normalised residual units.
            "val_mae_cm": round(mae * float(residual_scale), 2),
            "val_mae_residual_normalised": round(mae, 4),
            "residual_learning": True,
            "residual_scale": float(residual_scale),
        }

    def _calibrate_conformal(
        self, X_train, y_train, X_val, y_val, physics_severity_val=None, physics_depth_val=None
    ) -> Dict[str, Any]:
        """Calibrate conformal predictor on validation set."""
        from app.services.conformal import conformal_predictor
        import pickle

        # Load the multi-hazard model to get predictions
        from app.services.ai.model_architectures import MultiHazardPredictor
        model = MultiHazardPredictor(in_features=len(FEATURE_COLS)).to(self.device)

        ckpt_path = os.path.join(self.model_dir, "multi_hazard_predictor.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        # The regression heads predict residuals, so the physics baseline must
        # be added back before the calibration scores mean anything.
        baselines = None
        if physics_severity_val is not None:
            baselines = {
                "severity": torch.FloatTensor(physics_severity_val).to(self.device),
                "depth": torch.FloatTensor(physics_depth_val).to(self.device),
            }

        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(X_val).to(self.device)
            output = model(X_v, physics_baseline=baselines)
            # Two calibrations, in the units each is actually used in, each stored
            # under its own target key. Order matters below: the served quantity
            # goes through the shared ``conformal_predictor`` (the inference engine
            # loads that instance for the risk score), while depth uses a local
            # instance so the two cannot overwrite one another.
            #
            # The previous revision passed the whole 5-column target matrix as
            # ``y_true`` against a single severity vector as ``y_pred``, so this
            # raised "operands could not be broadcast (324,5) (324,)" and
            # ``train_all()`` never completed -- no training run could ever
            # finish and write its metadata.
            depth_pred = output["flood_depth_cm"].cpu().numpy().astype(float)
            depth_true = y_val[:, 4].astype(float)
            severity_pred = output["severity_score"].cpu().numpy().astype(float)
            severity_true = y_val[:, 3].astype(float)

        # The API puts an interval on the risk/severity score (0-100), so this is
        # the calibration that must exist for the served path. Calibrated on the
        # calibrate partition, cell-level (6 timesteps x 90 cells = 540 pairs),
        # which is disjoint from both training and the frozen test window.
        severity_report = conformal_predictor.calibrate(severity_true, severity_pred)
        severity_report["calibrated_quantity"] = "risk_severity_score"
        severity_report["calibrated_on"] = "calibrate partition of the frozen time-based split"

        from app.services.conformal import ConformalPredictor

        depth_predictor = ConformalPredictor(alpha=conformal_predictor.alpha, target="flood_depth_cm")
        depth_report = depth_predictor.calibrate(depth_true, depth_pred)
        depth_report["calibrated_quantity"] = "flood_depth_cm"
        depth_report["calibrated_on"] = "calibrate partition of the frozen time-based split"

        return {
            "served_target": "risk_severity_score",
            "risk_severity_score": severity_report,
            "flood_depth_cm": depth_report,
        }

    def _compute_shap(self, X_train, y_train) -> Dict[str, Any]:
        """Compute SHAP values on the trained multi-hazard model."""
        try:
            import shap
            from app.services.ai.model_architectures import MultiHazardPredictor

            model = MultiHazardPredictor(in_features=len(FEATURE_COLS)).to(self.device)
            ckpt_path = os.path.join(self.model_dir, "multi_hazard_predictor.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

            model.eval()

            # Create a wrapper for SHAP
            def model_predict(x):
                with torch.no_grad():
                    X = torch.FloatTensor(x).to(self.device)
                    output = model(X)
                    return np.column_stack([
                        output["thunderstorm_prob"].cpu().numpy(),
                        output["cloudburst_prob"].cpu().numpy(),
                        output["flash_flood_prob"].cpu().numpy(),
                        output["severity_score"].cpu().numpy(),
                    ])

            # Use a subset for SHAP (it's slow on large datasets)
            X_sample = X_train[:min(200, len(X_train))]

            explainer = shap.KernelExplainer(model_predict, X_sample[:50])
            shap_values = explainer.shap_values(X_sample[50:100] if len(X_sample) > 50 else X_sample)
            # shap_values may be a list (one per output) or array
            if isinstance(shap_values, list):
                # Take the severity_score output (index 3)
                shap_values = shap_values[3] if len(shap_values) > 3 else shap_values[0]
            shap_values = np.array(shap_values)

            # Average absolute SHAP values per feature
            mean_shap = np.mean(np.abs(shap_values), axis=0)

            importance_map = {}
            for i, col in enumerate(FEATURE_COLS):
                importance_map[col] = round(float(mean_shap[i]), 4)

            # Normalize to sum to 1
            total = sum(importance_map.values())
            if total > 0:
                importance_map = {k: round(v / total, 4) for k, v in importance_map.items()}

            # Save
            with open(os.path.join(self.model_dir, "shap_importance.json"), "w") as f:
                json.dump(importance_map, f, indent=2)

            return {
                "method": "KernelSHAP",
                "n_samples": len(X_sample),
                "feature_importance": importance_map,
            }

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {"error": str(e), "method": "fallback_uniform"}

    def _evaluate_all(self, X_test, y_test, d_test) -> Dict[str, Any]:
        """Evaluate all trained models on test set."""
        from app.services.ai.model_architectures import MultiHazardPredictor

        results = {}

        # Evaluate multi-hazard
        model = MultiHazardPredictor(in_features=len(FEATURE_COLS)).to(self.device)
        ckpt_path = os.path.join(self.model_dir, "multi_hazard_predictor.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        # The severity head predicts a residual, so the physics baseline for the
        # test window must be added back before comparing to absolute labels.
        test_baseline = None
        if self.physics_severity is not None:
            test_baseline = {
                "severity": torch.FloatTensor(
                    self.physics_severity[self._test_slice] if self._test_slice else self.physics_severity
                ).to(self.device),
                "depth": torch.FloatTensor(
                    self.physics_depth[self._test_slice] if self._test_slice else self.physics_depth
                ).to(self.device),
            }

        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(self.device)
            output = model(X_t, physics_baseline=test_baseline)
            # Severity is graded on a 0-3 scale. Both sides must be binned the
            # same way: the previous revision compared the predicted class
            # vector against the full 5-column target matrix, which raised
            # "operands could not be broadcast (972,) (972,5)" and stopped
            # train_all() from ever completing.
            y_pred = (
                output["severity_score"] / 100 * 3
            ).round().long().clamp(0, 3).cpu().numpy()
            y_true = (
                torch.FloatTensor(np.asarray(y_test)[:, 3]).to(self.device) / 100 * 3
            ).round().long().clamp(0, 3).cpu().numpy()

            accuracy = float(np.mean(y_pred == y_true))

            # Per-class metrics
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred).tolist()

            results["multi_hazard"] = {
                "accuracy": round(accuracy, 4),
                "classification_report": report,
                "confusion_matrix": cm,
            }

        # Evaluate flood depth
        try:
            depth_model = torch.load(
                os.path.join(self.model_dir, "flood_depth_estimator.pt"),
                map_location=self.device, weights_only=True
            ) if os.path.exists(os.path.join(self.model_dir, "flood_depth_estimator.pt")) else None

            if depth_model is not None:
                import torch.nn as nn
                # Tanh head: the network predicts a normalised depth residual.
                depth_nn = nn.Sequential(
                    nn.Linear(len(FEATURE_COLS), 128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh(),
                ).to(self.device)
                depth_nn.load_state_dict(depth_model)

                depth_scale = float(getattr(self, "depth_residual_scale", 25.0))

                depth_nn.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_test).to(self.device)
                    raw = depth_nn(X_t).squeeze().cpu().numpy()

                # Rescale and add the physics water depth back to report cm.
                phys_depth_test = self.physics_depth[self._test_slice] if (
                    self._test_slice is not None and self.physics_depth is not None
                ) else np.zeros_like(d_test)
                d_pred = np.clip(raw * depth_scale + phys_depth_test, 0.0, None)
                # d_test is the residual target; recover absolute for the metric.
                d_true_abs = d_test + phys_depth_test
                mae = float(np.mean(np.abs(d_pred - d_true_abs)))
                rmse = float(np.sqrt(np.mean((d_pred - d_true_abs) ** 2)))

                results["flood_depth"] = {
                    "mae_cm": round(mae, 2),
                    "rmse_cm": round(rmse, 2),
                }
        except Exception as e:
            results["flood_depth"] = {"error": str(e)}

        return results


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train VARUNA AI models")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed; retrains with the same seed are bit-identical",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    trainer = VARUNATrainer(
        data_dir=args.data_dir, model_dir=args.model_dir, seed=args.seed
    )
    data_info = trainer.load_data()
    print(f"\nData loaded: {json.dumps(data_info, indent=2)}")

    results = trainer.train_all(epochs=args.epochs, lr=args.lr)
    print(f"\nTraining complete: {json.dumps(results, indent=2, default=str)}")


if __name__ == "__main__":
    main()
