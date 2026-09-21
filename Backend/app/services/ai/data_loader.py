"""
VARUNA ML Data Loader
======================
Converts feature_grid_timeseries.json into PyTorch-compatible tensors
for training and inference across all 9 AI/ML modules.
"""

from __future__ import annotations

import json
import os
import math
from typing import Dict, List, Tuple, Optional, Any

try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.core.config import settings
from app.services.ai.feature_contract import (
    CANONICAL_FEATURES,
    TARGET_COLUMNS as CONTRACT_TARGET_COLUMNS,
    resolve_feature,
    resolve_target,
    validate_feature_window,
)

# Feature columns in order (sourced from canonical feature contract)
NUMERIC_FEATURES: List[str] = list(CANONICAL_FEATURES)
TARGET_COLUMNS: List[str] = list(CONTRACT_TARGET_COLUMNS)

# Grid dimensions for Mumbai pilot
GRID_ROWS = 10   # lat range: 18.98 to 19.16
GRID_COLS = 9    # lon range: 72.80 to 72.96
GRID_CELLS = GRID_ROWS * GRID_COLS  # 90


def normalize_feature(value: float, feature_name: str) -> float:
    """Min-max normalization using Mumbai pilot domain knowledge."""
    NORMALIZATION = {
        # Moisture
        "rainfall_1h_mm": (0, 150),
        "rainfall_3h_mm": (0, 400),
        "rainfall_6h_mm": (0, 600),
        "rainfall_24h_mm": (0, 800),
        "soil_moisture_pct": (0, 100),
        "soil_saturation_factor": (0, 1),
        "iwv_mm": (20, 80),  # IWV range for Indian monsoon
        # Instability
        "cape_instability_jkg": (0, 5000),
        "cin_jkg": (0, 500),  # CIN range
        "lifted_index": (-6, 4),  # LI range
        "cloud_top_temp_celsius": (-90, 0),
        "ctt_drop_rate_c_per_hr": (0, 30),
        # Kinematics
        "wind_speed_10m_kmh": (0, 150),
        "wind_direction_10m_deg": (0, 360),
        "u_wind_ms": (-50, 50),
        "v_wind_ms": (-50, 50),
        "wind_gusts_kmh": (0, 200),
        "vertical_wind_shear_ms": (0, 50),  # Shear range
        "low_level_convergence": (-2, 2),  # Convergence/divergence
        # Topography
        "elevation_m": (0, 150),
        "slope_deg": (0, 30),
        "runoff_coefficient": (0.3, 1.0),
        "effective_runoff_mm_hr": (0, 120),
        "drainage_outfall_dist_m": (0, 15000),
        "retention_index": (0, 5),
        "tidal_backwater_factor": (0.8, 3.0),
    }
    if feature_name in NORMALIZATION:
        lo, hi = NORMALIZATION[feature_name]
        return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-8)))
    return value


class VARUNADataset(Dataset):
    """
    PyTorch Dataset for VARUNA feature grid timeseries.

    Each sample is a sliding window of `window_size` timesteps.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        window_size: int = 6,
        forecast_horizon: int = 6,
        normalize: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VARUNADataset")

        if data_path is None:
            data_path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")

        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.timesteps = self.raw_data["timesteps"]
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.normalize = normalize

        # Validate feature contract
        report = validate_feature_window(self.timesteps, label="VARUNADataset")
        if not report.ok:
            import logging
            logging.getLogger(__name__).warning("Feature contract validation warnings:\n%s", report.summary())

        # Build feature matrix: (T, N_cells, N_features)
        self.features = self._build_feature_matrix()
        self.targets = self._build_target_matrix()

    def _extract_cell_features(self, cell: Dict[str, Any]) -> List[float]:
        """Extract numeric features from a single grid cell honoring contract aliases."""
        return [float(resolve_feature(cell, f, 0.0) or 0.0) for f in NUMERIC_FEATURES]

    def _build_feature_matrix(self) -> "torch.Tensor":
        T = len(self.timesteps)
        N = len(self.timesteps[0]["features"])
        F = len(NUMERIC_FEATURES)

        matrix = torch.zeros(T, N, F)
        for t, ts in enumerate(self.timesteps):
            for n, cell in enumerate(ts["features"]):
                raw = self._extract_cell_features(cell)
                if self.normalize:
                    raw = [
                        normalize_feature(v, NUMERIC_FEATURES[i])
                        for i, v in enumerate(raw)
                    ]
                matrix[t, n] = torch.tensor(raw, dtype=torch.float32)
        return matrix

    def _build_target_matrix(self) -> "torch.Tensor":
        T = len(self.timesteps)
        N = len(self.timesteps[0]["features"])
        TGT = len(TARGET_COLUMNS)

        matrix = torch.zeros(T, N, TGT)
        for t, ts in enumerate(self.timesteps):
            for n, cell in enumerate(ts["features"]):
                targets = [float(resolve_target(cell, tc, 0.0) or 0.0) for tc in TARGET_COLUMNS]
                matrix[t, n] = torch.tensor(targets, dtype=torch.float32)
        return matrix

    def __len__(self) -> int:
        return max(0, len(self.timesteps) - self.window_size - self.forecast_horizon + 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_seq = self.features[idx: idx + self.window_size]      # (W, N, F)
        target_seq = self.features[
            idx + self.window_size: idx + self.window_size + self.forecast_horizon
        ]  # (H, N, F)
        target_labels = self.targets[
            idx + self.window_size: idx + self.window_size + self.forecast_horizon
        ]  # (H, N, TGT)

        return {
            "input_sequence": input_seq,
            "forecast_target": target_seq,
            "label_target": target_labels,
            "timestep_idx": idx,
        }


def build_spatial_grid_tensor(
    timestep_data: Dict[str, Any],
    normalize: bool = True,
) -> Tuple["torch.Tensor", List[float], List[float]]:
    """
    Convert a single timestep's feature data into a spatial grid tensor
    suitable for CNN-based models (Modules 1, 2, 3).

    Returns:
        grid_tensor: (C_features, GRID_ROWS, GRID_COLS)
        lats: list of latitude values
        lons: list of longitude values
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    features = timestep_data["features"]
    C = len(NUMERIC_FEATURES)

    # Build spatial grid
    grid = torch.zeros(C, GRID_ROWS, GRID_COLS)
    lats = []
    lons = []

    for cell in features:
        idx = cell["cell_index"]
        row = idx // GRID_COLS
        col = idx % GRID_COLS

        raw = [cell.get(f, 0.0) for f in NUMERIC_FEATURES]
        if normalize:
            raw = [normalize_feature(v, NUMERIC_FEATURES[i]) for i, v in enumerate(raw)]

        grid[:, row, col] = torch.tensor(raw, dtype=torch.float32)

        if row == 0:
            lons.append(cell["lon"])
        if col == 0:
            lats.append(cell["lat"])

    return grid, lats, lons


def build_drainage_graph(
    timestep_data: Dict[str, Any],
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Build a graph representation of the drainage network from grid data.

    Creates edges between adjacent cells, weighted by slope and flow direction.

    Returns:
        node_features: (N, node_feat_dim) — per-cell features for GNN
        edge_index: (2, E) — source→target adjacency
        edge_weights: (E,) — edge weights based on terrain gradient
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    features = timestep_data["features"]
    N = len(features)
    NODE_DIM = 12

    node_features = torch.zeros(N, NODE_DIM)
    edge_src = []
    edge_dst = []
    edge_weights = []

    for cell in features:
        idx = cell["cell_index"]
        node_features[idx] = torch.tensor([
            cell.get("rainfall_1h_mm", 0.0) / 100.0,
            cell.get("soil_moisture_pct", 50.0) / 100.0,
            cell.get("cape_instability_jkg", 0.0) / 3000.0,
            cell.get("cloud_top_temp_celsius", -40.0) / 80.0,
            cell.get("wind_speed_10m_kmh", 20.0) / 100.0,
            cell.get("elevation_m", 5.0) / 50.0,
            cell.get("slope_deg", 1.0) / 15.0,
            cell.get("runoff_coefficient", 0.8),
            cell.get("effective_runoff_mm_hr", 3.0) / 30.0,
            cell.get("drainage_outfall_dist_m", 5000.0) / 15000.0,
            cell.get("retention_index", 1.0) / 5.0,
            cell.get("tidal_backwater_factor", 1.0) / 2.0,
        ], dtype=torch.float32)

    # Create edges between adjacent cells (4-connectivity + diagonals)
    for cell in features:
        idx = cell["cell_index"]
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        elev = cell.get("elevation_m", 0.0)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                n_idx = nr * GRID_COLS + nc
                neighbor_cell = features[n_idx]
                n_elev = neighbor_cell.get("elevation_m", 0.0)
                gradient = max(0.001, (elev - n_elev + 0.1))
                edge_src.append(idx)
                edge_dst.append(n_idx)
                edge_weights.append(1.0 / gradient)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

    # Normalize weights
    if edge_weights.max() > 0:
        edge_weights = edge_weights / edge_weights.max()

    return node_features, edge_index, edge_weights


def build_feature_vector(
    cell: Dict[str, Any],
    timestep: Dict[str, Any],
) -> torch.Tensor:
    """
    Build a flat feature vector for a single cell, suitable for
    Module 4 (Multi-Hazard Predictor).
    """
    vector = [cell.get(f, 0.0) for f in NUMERIC_FEATURES]
    # Add timestep-level features
    vector.extend([
        timestep.get("tide_height_m", 2.5),
        1.0 if timestep.get("is_high_tide_locked", False) else 0.0,
        timestep.get("avg_rainfall_1h_mm", 0.0),
    ])
    return torch.tensor(vector, dtype=torch.float32)


def get_dataloader(
    window_size: int = 6,
    forecast_horizon: int = 6,
    batch_size: int = 16,
    shuffle: bool = True,
) -> "DataLoader":
    """Create a PyTorch DataLoader for the VARUNA training dataset."""
    dataset = VARUNADataset(
        window_size=window_size,
        forecast_horizon=forecast_horizon,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ===========================================================================
# Crowd Report NLP Data Processing
# ===========================================================================

# Mumbai-specific vocabulary for flood reports
FLOOD_VOCAB = {
    # English
    "flood": 1, "water": 2, "rain": 3, "heavy": 4, "submerged": 5,
    "stuck": 6, "waterlogged": 7, "emergency": 8, "help": 9, "rescue": 10,
    "road": 11, "bridge": 12, "underpass": 13, "drain": 14, "overflow": 15,
    "danger": 16, "warning": 17, "rising": 18, "deep": 19, "flowing": 20,
    # Hindi / Hinglish
    "paani": 21, "baarish": 22, "doob": 23, "bachao": 24, "sarak": 25,
    "pul": 26, "naali": 27, "beh": 28, "khatra": 29, "dar": 30,
    # Location markers
    "mumbai": 50, "thane": 51, "bandra": 52, "andheri": 53, "kurla": 54,
    "dadar": 55, "lower_parel": 56, "mahalaxmi": 57, "worli": 58,
}


def tokenize_crowd_report(text: str, max_len: int = 128) -> List[int]:
    """Simple tokenization for crowd reports using vocabulary lookup."""
    tokens = [0] * max_len
    words = text.lower().replace(",", " ").replace(".", " ").split()
    for i, word in enumerate(words[:max_len]):
        tokens[i] = FLOOD_VOCAB.get(word, 0)
    return tokens
