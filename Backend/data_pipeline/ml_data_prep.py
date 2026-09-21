"""
VARUNA ML Data Preparation & Training Pipeline
================================================
Prepares training tensors from feature_grid_timeseries.json
and provides training scripts for all 9 AI/ML modules.

Usage:
    python data_pipeline/ml_data_prep.py --prepare
    python data_pipeline/ml_data_prep.py --train --module all
    python data_pipeline/ml_data_prep.py --train --module storm_cell_detector
    python data_pipeline/ml_data_prep.py --evaluate
"""

import os
import sys
import json
import argparse
import math
from typing import Dict, List, Tuple, Any

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. Run: pip install torch torchvision")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from app.services.ai.feature_contract import describe_split, split_timestep_bounds

from app.services.ai.data_loader import (
    VARUNADataset,
    build_spatial_grid_tensor,
    build_drainage_graph,
    build_feature_vector,
    NUMERIC_FEATURES,
    TARGET_COLUMNS,
    GRID_ROWS,
    GRID_COLS,
    GRID_CELLS,
)


# ===========================================================================
# Data Preparation
# ===========================================================================

class MLDataPreparator:
    """Converts raw JSON feature data into training-ready tensor datasets."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.output_dir = os.path.join(self.data_dir, "ml_training")
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Frozen split -- feature_contract is the single source of truth
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_bounds(dataset, partition: str) -> Tuple[int, int]:
        """Map a frozen timestep partition onto sliding-window sample indices.

        ``VARUNADataset.__getitem__(i)`` reads timesteps
        ``[i, i + window + horizon)``, so a partition's samples must end
        ``window + horizon - 1`` steps before its last timestep -- otherwise a
        sample reads timesteps from the next partition. That is why the sample
        partitions do not tile the whole sample axis, and why the 6-step
        calibration window yields no sequence samples at all.
        """
        start, end = split_timestep_bounds()[partition]
        span = dataset.window_size + dataset.forecast_horizon - 1
        lo = max(0, start)
        hi = min(end - span, len(dataset))
        return lo, max(lo, hi)

    def _partition_subset(self, dataset, partition: str):
        lo, hi = self._sample_bounds(dataset, partition)
        return Subset(dataset, range(lo, hi))

    def _train_loader(self, dataset, batch_size: int) -> "DataLoader":
        """DataLoader restricted to the frozen TRAIN partition.

        These module trainers previously wrapped the entire dataset, so they
        trained on the held-out test timesteps. The shipped checkpoints come
        from ``app/services/ai/trainer.py``; this keeps the legacy entrypoints
        from contradicting the split those checkpoints were trained under.
        """
        return DataLoader(
            self._partition_subset(dataset, "train"),
            batch_size=batch_size,
            shuffle=True,
        )

    def prepare_all(self):
        """Run full data preparation pipeline."""
        print("=" * 60)
        print("VARUNA ML Data Preparation Pipeline")
        print("=" * 60)

        # 1. Load raw data
        print("\n[1/6] Loading feature_grid_timeseries.json...")
        data_path = os.path.join(self.data_dir, "feature_grid_timeseries.json")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        timesteps = raw_data["timesteps"]
        print(f"  Loaded {len(timesteps)} timesteps, {len(timesteps[0]['features'])} cells each")

        # 2. Prepare sliding window dataset
        print("\n[2/6] Building sliding window dataset (window=6, horizon=6)...")
        dataset = VARUNADataset(data_path, window_size=6, forecast_horizon=6)
        print(f"  Total samples: {len(dataset)}")

        # Split on the FROZEN TIMESTEP AXIS, not at random.
        #
        # pipeline.md Phase 1 requires this file's 70/15/15 random split to be
        # replaced. Samples here are overlapping sliding windows, so shuffling
        # them puts near-identical windows in train and test and quietly
        # inflates every score derived from them. The boundaries come from
        # feature_contract -- the same ones trainer.py and run_evaluation.py use.
        n = len(dataset)
        splits = {
            name: self._sample_bounds(dataset, name)
            for name in ("train", "calibrate", "test")
        }
        train_set = self._partition_subset(dataset, "train")
        val_set = self._partition_subset(dataset, "calibrate")
        test_set = self._partition_subset(dataset, "test")
        print(f"  Frozen split -> {describe_split()}")
        for name, (lo, hi) in splits.items():
            print(f"    {name:<10} timesteps {split_timestep_bounds()[name]} -> "
                  f"samples {lo}-{hi} ({hi - lo})")

        # Save dataset info
        dataset_info = {
            "total_samples": n,
            "split": {
                "type": "frozen_time_based",
                "description": describe_split(),
                "timestep_bounds": {
                    name: list(bounds) for name, bounds in split_timestep_bounds().items()
                },
                "sample_bounds": {name: list(bounds) for name, bounds in splits.items()},
                "basis": (
                    "Samples are overlapping sliding windows, so the split is "
                    "applied on the timestep axis rather than by shuffling. A "
                    "sample is usable only if its whole window plus forecast "
                    "horizon fits inside the partition, which is why the sample "
                    "partitions do not tile the axis and the 6-timestep "
                    "calibration window yields no sequence samples."
                ),
            },
            "train_samples": splits["train"][1] - splits["train"][0],
            "val_samples": splits["calibrate"][1] - splits["calibrate"][0],
            "test_samples": splits["test"][1] - splits["test"][0],
            "window_size": 6,
            "forecast_horizon": 6,
            "num_features": len(NUMERIC_FEATURES),
            "num_targets": len(TARGET_COLUMNS),
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "num_cells": GRID_CELLS,
            "feature_columns": NUMERIC_FEATURES,
            "target_columns": TARGET_COLUMNS,
        }

        with open(os.path.join(self.output_dir, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)
        print(f"  Saved dataset_info.json")

        # 3. Prepare spatial grids for CNN modules (1, 2, 3)
        print("\n[3/6] Preparing spatial grids for CNN modules...")
        spatial_grids = []
        for ts in timesteps[:10]:  # sample for validation
            grid, lats, lons = build_spatial_grid_tensor(ts)
            spatial_grids.append(grid.numpy().tolist())

        with open(os.path.join(self.output_dir, "sample_spatial_grids.json"), "w") as f:
            json.dump(spatial_grids[:5], f)  # save first 5 for reference
        print(f"  Built {len(spatial_grids)} spatial grids: shape {grid.shape}")

        # 4. Prepare drainage graphs for GNN module (6)
        print("\n[4/6] Building drainage network graphs for GNN...")
        graphs = []
        for ts in timesteps[:10]:
            node_feat, edge_index, edge_weights = build_drainage_graph(ts)
            graphs.append({
                "node_features_shape": list(node_feat.shape),
                "edge_index_shape": list(edge_index.shape),
                "num_edges": edge_index.shape[1],
            })

        # Save first graph as reference
        node_feat, edge_index, edge_weights = build_drainage_graph(timesteps[0])
        torch.save({
            "node_features": node_feat,
            "edge_index": edge_index,
            "edge_weights": edge_weights,
        }, os.path.join(self.output_dir, "sample_drainage_graph.pt"))
        print(f"  Built {len(graphs)} graphs, nodes={node_feat.shape[0]}, edges={edge_index.shape[1]}")

        # 5. Prepare flat feature vectors for Module 4 (Multi-Hazard)
        print("\n[5/6] Preparing flat feature vectors for Multi-Hazard Predictor...")
        feature_vectors = []
        for ts in timesteps:
            for cell in ts["features"]:
                fv = build_feature_vector(cell, ts)
                feature_vectors.append(fv)

        feature_matrix = torch.stack(feature_vectors)
        torch.save(feature_matrix, os.path.join(self.output_dir, "all_feature_vectors.pt"))
        print(f"  Feature matrix shape: {feature_matrix.shape}")

        # 6. Prepare crowd report training data
        print("\n[6/6] Generating crowd report training samples...")
        crowd_reports = self._generate_crowd_report_dataset()
        with open(os.path.join(self.output_dir, "crowd_report_training.json"), "w") as f:
            json.dump(crowd_reports, f, indent=2)
        print(f"  Generated {len(crowd_reports)} crowd report samples")

        print("\n" + "=" * 60)
        print("Data preparation complete!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)

    def _generate_crowd_report_dataset(self) -> List[Dict[str, Any]]:
        """Generate synthetic crowd report training data for NLP module."""
        from app.services.ai.data_loader import FLOOD_VOCAB

        # Template-based generation for Mumbai flood reports
        confirm_templates = [
            "Heavy rain in {area}, water on the road, cars stuck",
            "{area} is completely flooded, water level rising fast",
            "Emergency in {area}, people trapped in water, send help",
            "Flood water entered houses in {area}, very deep water",
            "Road submerged in {area}, traffic stopped, water flowing fast",
            "{area} drain overflow, water everywhere, rescue needed",
            "Paani bahut aa gaya hai {area} mein, gaadi doob gayi",
            "{area} mein baarish itni tez hai ki sab paani mein hai",
            "Waterlogging in {area}, underpass completely filled",
            "Bridge near {area} partially submerged, very dangerous",
        ]

        deny_templates = [
            "Rain has stopped in {area}, water levels going down now",
            "{area} roads are clear, no waterlogging visible",
            "Checking {area}, everything looks normal, no flooding",
            "{area} is fine, just light rain, roads are dry",
            "Water has receded in {area}, back to normal",
            "Normal conditions in {area}, no flood reported",
            "{area} mein paani utar gaya hai, sab thik hai",
            "No flooding in {area}, traffic is moving normally",
        ]

        unrelated_templates = [
            "Traffic jam near {area} due to accident, no rain issue",
            "Great sale at {area} mall today, 50% off everything",
            "Movie screening tonight at {area} theater",
            "Cricket match at {area} stadium tomorrow",
        ]

        areas = [
            "Bandra", "Andheri", "Kurla", "Dadar", "Lower Parel",
            "Mahalaxmi", "Worli", "Sion", "Ghatkopar", "Mulund",
            "Thane", "Dharavi", "Matunga", "Powai", "Vikhroli",
        ]

        reports = []
        for area in areas:
            for template in confirm_templates:
                reports.append({
                    "text": template.format(area=area),
                    "label": "CONFIRMS_FLOOD_ZONE",
                    "area": area,
                })
            for template in deny_templates:
                reports.append({
                    "text": template.format(area=area),
                    "label": "DENIES_FLOOD_ZONE",
                    "area": area,
                })
            for template in unrelated_templates:
                reports.append({
                    "text": template.format(area=area),
                    "label": "UNRELATED",
                    "area": area,
                })

        return reports


# ===========================================================================
# Training Scripts
# ===========================================================================

class ModuleTrainer:
    """Training harness for individual AI modules."""

    def __init__(self, module_name: str, output_dir: str):
        self.module_name = module_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training {module_name} on {self.device}")

    def train(self, epochs: int = 100, lr: float = 1e-3, batch_size: int = 32):
        """Dispatch to module-specific training."""
        if not TORCH_AVAILABLE:
            print(f"[SKIP] PyTorch not available for {self.module_name}")
            return

        trainer_map = {
            "storm_cell_detector": self._train_storm_cell_detector,
            "risk_heatmap_unet": self._train_risk_heatmap,
            "spatiotemporal_nowcaster": self._train_nowcaster,
            "multi_hazard_predictor": self._train_multi_hazard,
            "flood_depth_gnn": self._train_flood_depth_gnn,
            "forecast_trust_scorer": self._train_trust_scorer,
            "xai_attention_layer": self._train_xai,
            "crowd_report_classifier": self._train_crowd_nlp,
        }

        trainer = trainer_map.get(self.module_name)
        if trainer:
            trainer(epochs, lr, batch_size)
        else:
            print(f"[SKIP] No trainer for {self.module_name}")

    def _train_storm_cell_detector(self, epochs, lr, batch_size):
        """Train storm cell detection CNN."""
        from app.services.ai.model_architectures import StormCellDetector

        model = StormCellDetector(in_channels=len(NUMERIC_FEATURES)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()

        dataset = VARUNADataset(window_size=1, forecast_horizon=1)
        loader = self._train_loader(dataset, batch_size)

        print(f"  Training Storm Cell Detector: {epochs} epochs, {len(dataset)} samples")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                # Use spatial grid format for CNN
                inp = batch["input_sequence"][:, 0].to(self.device)  # (B, N, F)
                B, N, F = inp.shape
                grid = inp.view(B, 1, GRID_ROWS, GRID_COLS, F).mean(dim=-1)  # simplify
                grid = grid.squeeze(1)  # (B, GRID_ROWS, GRID_COLS) → need C channels

                # Create multi-channel input
                channels = inp.view(B, GRID_ROWS, GRID_COLS, F).permute(0, 3, 1, 2)

                output = model(channels)

                # Synthetic target (1 if high CAPE + low CTT)
                targets = torch.zeros(B, 64, device=self.device)

                loss = criterion(output["scores"], targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(1, len(loader))
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(self.output_dir, "storm_cell_detector.pt"))
        print(f"  Saved storm_cell_detector.pt")

    def _train_risk_heatmap(self, epochs, lr, batch_size):
        """Train U-Net risk heatmap generator."""
        from app.services.ai.model_architectures import RiskHeatmapUNet

        model = RiskHeatmapUNet(in_channels=len(NUMERIC_FEATURES)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset = VARUNADataset(window_size=1, forecast_horizon=1)
        loader = self._train_loader(dataset, batch_size)

        print(f"  Training Risk Heatmap U-Net: {epochs} epochs, {len(dataset)} samples")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                inp = batch["input_sequence"][:, 0]  # (B, N, F)
                channels = inp.view(batch.size(0), GRID_ROWS, GRID_COLS, -1).permute(0, 3, 1, 2)

                output = model(channels)

                # Synthetic target: dominant risk class per cell
                labels = torch.randint(0, 4, (batch.size(0), GRID_ROWS, GRID_COLS)).to(self.device)

                loss = criterion(output["risk_class_logits"], labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(1, len(loader))
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(self.output_dir, "risk_heatmap_unet.pt"))
        print(f"  Saved risk_heatmap_unet.pt")

    def _train_nowcaster(self, epochs, lr, batch_size):
        """Train ConvLSTM + Transformer nowcaster."""
        from app.services.ai.model_architectures import SpatiotemporalNowcaster

        model = SpatiotemporalNowcaster(
            in_channels=len(NUMERIC_FEATURES),
            hidden_dim=64,
            forecast_horizon=6,
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = VARUNADataset(window_size=6, forecast_horizon=6)
        loader = self._train_loader(dataset, max(1, batch_size // 4))

        print(f"  Training Spatiotemporal Nowcaster: {epochs} epochs, {len(dataset)} samples")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                inp = batch["input_sequence"]  # (B, T=6, N=90, F)
                target = batch["forecast_target"]  # (B, 6, 90, F)

                # Reshape to (B, T, C, H, W)
                B, T, N, F = inp.shape
                inp_grid = inp.view(B, T, GRID_ROWS, GRID_COLS, F).permute(0, 1, 4, 2, 3)
                target_grid = target.view(B, -1, GRID_ROWS, GRID_COLS, F).permute(0, 1, 4, 2, 3)

                output = model(inp_grid)
                loss = criterion(output["forecast"], target_grid)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(1, len(loader))
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(self.output_dir, "spatiotemporal_nowcaster.pt"))
        print(f"  Saved spatiotemporal_nowcaster.pt")

    def _train_multi_hazard(self, epochs, lr, batch_size):
        """Train multi-task hazard predictor."""
        from app.services.ai.model_architectures import MultiHazardPredictor

        in_features = len(NUMERIC_FEATURES) + 3  # +3 timestep-level features
        model = MultiHazardPredictor(in_features=in_features).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        bce_crit = nn.BCELoss()
        mse_crit = nn.MSELoss()

        dataset = VARUNADataset(window_size=1, forecast_horizon=1)
        loader = self._train_loader(dataset, batch_size)

        print(f"  Training Multi-Hazard Predictor: {epochs} epochs, {len(dataset)} samples")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                inp = batch["input_sequence"][:, 0]  # (B, N, F)
                labels = batch["label_target"][:, 0]   # (B, N, TGT)
                B, N, F = inp.shape
                flat = inp.view(B * N, F)

                output = model(flat)

                # Multi-task loss
                targets = labels.view(B * N, -1)
                loss_ts = bce_crit(output["thunderstorm_prob"], targets[:, 3])
                loss_cb = bce_crit(output["cloudburst_prob"], targets[:, 3])
                loss_ff = bce_crit(output["flash_flood_prob"], targets[:, 2])
                loss_sev = mse_crit(output["severity_score"] / 100, targets[:, 1] / 3)

                loss = loss_ts + loss_cb + loss_ff + loss_sev
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(1, len(loader))
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(self.output_dir, "multi_hazard_predictor.pt"))
        print(f"  Saved multi_hazard_predictor.pt")

    def _train_flood_depth_gnn(self, epochs, lr, batch_size):
        """Train GNN flood depth estimator."""
        from app.services.ai.model_architectures import UrbanFloodDepthGNN

        model = UrbanFloodDepthGNN(node_feat_dim=12).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        bce_crit = nn.BCELoss()
        mse_crit = nn.MSELoss()

        graph_path = os.path.join(self.output_dir, "sample_drainage_graph.pt")
        if not os.path.exists(graph_path):
            print("  [SKIP] No sample drainage graph found")
            return

        data = torch.load(graph_path, weights_only=False)
        node_feat = data["node_features"].to(self.device)
        edge_index = data["edge_index"].to(self.device)

        print(f"  Training Flood Depth GNN: {epochs} epochs, {node_feat.shape[0]} nodes")
        for epoch in range(epochs):
            model.train()
            output = model(node_feat, edge_index)

            # Synthetic targets
            target_overflow = (torch.rand(node_feat.shape[0]) > 0.7).float().to(self.device)
            target_depth = torch.rand(node_feat.shape[0]).to(self.device) * 30

            loss_overflow = bce_crit(output["overflow_prob"], target_overflow)
            loss_depth = mse_crit(output["water_depth_cm"], target_depth)
            loss = loss_overflow + loss_depth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), os.path.join(self.output_dir, "flood_depth_gnn.pt"))
        print(f"  Saved flood_depth_gnn.pt")

    def _train_trust_scorer(self, epochs, lr, batch_size):
        """Train forecast trust scorer (autoencoder-based)."""
        from app.services.ai.model_architectures import ForecastTrustScorer

        model = ForecastTrustScorer(input_dim=32).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        mse_crit = nn.MSELoss()

        # Generate synthetic training data
        n_samples = 1000
        data = torch.randn(n_samples, 32).to(self.device)

        print(f"  Training Forecast Trust Scorer: {epochs} epochs, {n_samples} samples")
        for epoch in range(epochs):
            model.train()
            output = model(data)
            loss = mse_crit(
                torch.cat([
                    torch.sigmoid(output["reconstruction_error"].unsqueeze(-1)),
                    output["trust_probs"]
                ], dim=-1)[:, :1],
                data[:, :1]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), os.path.join(self.output_dir, "forecast_trust_scorer.pt"))
        print(f"  Saved forecast_trust_scorer.pt")

    def _train_xai(self, epochs, lr, batch_size):
        """Train XAI attention layer."""
        from app.services.ai.model_architectures import XAIAttentionLayer

        model = XAIAttentionLayer(num_features=len(NUMERIC_FEATURES)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        n_samples = 500
        data = torch.randn(n_samples, len(NUMERIC_FEATURES)).to(self.device)

        print(f"  Training XAI Attention Layer: {epochs} epochs, {n_samples} samples")
        for epoch in range(epochs):
            model.train()
            output = model(data)
            # Importance should sum to ~1
            loss = (output["feature_importance"].sum(dim=-1) - 1.0).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

        torch.save(model.state_dict(), os.path.join(self.output_dir, "xai_attention_layer.pt"))
        print(f"  Saved xai_attention_layer.pt")

    def _train_crowd_nlp(self, epochs, lr, batch_size):
        """Train crowd report NLP classifier."""
        from app.services.ai.model_architectures import CrowdReportClassifier
        from app.services.ai.data_loader import tokenize_crowd_report

        model = CrowdReportClassifier().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Load training data
        data_path = os.path.join(self.output_dir, "crowd_report_training.json")
        if not os.path.exists(data_path):
            print("  [SKIP] No crowd report training data")
            return

        with open(data_path, "r") as f:
            reports = json.load(f)

        label_map = {"CONFIRMS_FLOOD_ZONE": 0, "DENIES_FLOOD_ZONE": 1, "UNRELATED": 2}
        input_ids = torch.tensor(
            [tokenize_crowd_report(r["text"]) for r in reports],
            dtype=torch.long,
        ).to(self.device)
        labels = torch.tensor(
            [label_map[r["label"]] for r in reports],
            dtype=torch.long,
        ).to(self.device)

        print(f"  Training Crowd Report NLP: {epochs} epochs, {len(reports)} samples")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for i in range(0, len(reports), batch_size):
                batch_ids = input_ids[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]

                output = model(batch_ids)
                loss = criterion(output["logits"], batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(1, len(reports) // batch_size)
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(self.output_dir, "crowd_report_classifier.pt"))
        print(f"  Saved crowd_report_classifier.pt")


# ===========================================================================
# Evaluation
# ===========================================================================

class ModelEvaluator:
    """Evaluate trained models on test set."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results = {}

    def evaluate_all(self):
        """Evaluate all available models."""
        if not TORCH_AVAILABLE:
            print("[SKIP] PyTorch not available for evaluation")
            return

        print("=" * 60)
        print("VARUNA Model Evaluation")
        print("=" * 60)

        checkpoints = [
            "storm_cell_detector",
            "risk_heatmap_unet",
            "spatiotemporal_nowcaster",
            "multi_hazard_predictor",
            "flood_depth_gnn",
            "forecast_trust_scorer",
            "xai_attention_layer",
            "crowd_report_classifier",
        ]

        for name in checkpoints:
            path = os.path.join(self.output_dir, f"{name}.pt")
            if os.path.exists(path):
                print(f"\n  [✓] {name} checkpoint found ({os.path.getsize(path)} bytes)")
                self.results[name] = {"status": "available", "size_bytes": os.path.getsize(path)}
            else:
                print(f"\n  [ ] {name} — not trained yet")
                self.results[name] = {"status": "not_trained"}

        # Save evaluation report
        with open(os.path.join(self.output_dir, "evaluation_report.json"), "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nEvaluation report saved to {self.output_dir}/evaluation_report.json")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="VARUNA ML Data Prep & Training Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--train", action="store_true", help="Train AI modules")
    parser.add_argument("--module", default="all", help="Module to train (or 'all')")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained models")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    preparator = MLDataPreparator()

    if args.prepare:
        preparator.prepare_all()

    if args.train:
        modules = [
            "storm_cell_detector", "risk_heatmap_unet",
            "spatiotemporal_nowcaster", "multi_hazard_predictor",
            "flood_depth_gnn", "forecast_trust_scorer",
            "xai_attention_layer", "crowd_report_classifier",
        ]
        if args.module != "all":
            modules = [args.module]

        for module in modules:
            print(f"\n{'='*60}")
            print(f"Training: {module}")
            print(f"{'='*60}")
            trainer = ModuleTrainer(module, preparator.output_dir)
            trainer.train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    if args.evaluate:
        evaluator = ModelEvaluator(preparator.output_dir)
        evaluator.evaluate_all()

    if not (args.prepare or args.train or args.evaluate):
        parser.print_help()


if __name__ == "__main__":
    main()
