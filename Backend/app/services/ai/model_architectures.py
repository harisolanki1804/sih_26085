"""
VARUNA Deep Learning Model Architectures
==========================================
PyTorch nn.Module definitions for all 9 AI/ML modules.

These are production-grade architectures designed for the Mumbai
pilot grid (90 cells, 72 timesteps) and scalable to real satellite
imagery and drainage network graphs.
"""

from __future__ import annotations  # defer annotation evaluation (avoids torch=None issues)

import math
from typing import Dict, Tuple, Optional, List

# Scaling of the physics-residual heads. A Tanh head emits (-1, 1); multiplying
# by these constants puts the correction back into physical units.
#
# Measured against the dataset's OBSERVED targets (target_severity_class scaled
# to 0-100, target_observed_flood_depth_cm):
#   severity residual |r| p99 = 32.9, max 63.5  -> scale 70
#   depth    residual |r| p99 = 38.9, max 83.6  -> scale 90
# The head must be able to represent the full observed correction, so the scale
# sits above the observed maximum rather than at the 99th percentile.
#
# Note: against these observed targets the physics baseline scores R2 = 0.596
# (severity) and 0.158 (depth) -- lower than the 0.79/0.88 measured against the
# formula-derived labels, because the formula labels were themselves computed
# from rainfall and were therefore partly circular.
SEVERITY_RESIDUAL_SCALE = 70.0
DEPTH_RESIDUAL_SCALE = 90.0

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore

    # Provide stub base class so module-level class definitions parse
    # without PyTorch installed.  Actual instantiation will fail at
    # runtime with a clear ImportError from _require_torch().
    class nn:  # type: ignore[no-redef]
        class Module:
            pass
        class Sequential:
            pass
        class Conv2d:  # type: ignore[no-redef]
            pass
        class Conv1d:
            pass
        class Linear:
            pass
        class BatchNorm1d:
            pass
        class BatchNorm2d:
            pass
        class Embedding:
            pass
        class GRUCell:
            pass
        class MultiheadAttention:
            pass
        class LayerNorm:
            pass
        class ModuleList:
            pass
        class ModuleDict:
            pass
    class F:  # type: ignore[no-redef]
        @staticmethod
        def pad(x, padding):  # type: ignore
            pass
        @staticmethod
        def max_pool2d(x, k):  # type: ignore
            pass
        @staticmethod
        def relu(x, inplace=False):  # type: ignore
            return x
        @staticmethod
        def sigmoid(x):  # type: ignore
            return x
        @staticmethod
        def softmax(x, dim=-1):  # type: ignore
            return x
        @staticmethod
        def tanh(x):  # type: ignore
            return x
        @staticmethod
        def mse_loss(x, y, reduction="mean"):  # type: ignore
            return 0.0
    TORCH_AVAILABLE = False


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for VARUNA AI models. "
            "Install with: pip install torch torchvision torch-geometric"
        )


# ===========================================================================
# MODULE 1 — Storm Cell Detection & Tracking (YOLO-style CNN + Optical Flow)
# ===========================================================================

class StormCellDetector(nn.Module):
    """
    CNN-based object detection network inspired by YOLOv8 / Faster-RCNN.

    Detects convective cloud clusters in satellite/radar frames and
    outputs bounding boxes + confidence scores for each cell.

    Input:  (B, C_in, H, W) — multi-channel satellite/radar image tile
    Output: dict with
        - boxes:   (B, max_detections, 4)  — [cx, cy, w, h] normalised
        - scores:  (B, max_detections)     — confidence 0..1
        - classes: (B, max_detections)      — 0:convective 1:stratiform 2:supercell
    """

    MAX_DETECTIONS = 64
    NUM_CLASSES = 3

    def __init__(self, in_channels: int = 8, backbone_channels: int = 256):
        super().__init__()
        _require_torch()

        # Lightweight backbone (Darknet-style feature extractor)
        self.backbone = nn.Sequential(
            # Block 1: 8 → 32
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            # Block 4: 128 → backbone_channels
            nn.Conv2d(128, backbone_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(backbone_channels, backbone_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.LeakyReLU(0.1),
        )

        # Detection head: predict bounding box + class per anchor
        anchor_dim = 5 + self.NUM_CLASSES  # x, y, w, h, obj_conf + class_probs
        self.detection_head = nn.Sequential(
            nn.Conv2d(backbone_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, anchor_dim, 1),
        )

        # Optical flow estimation head (for tracking between frames)
        self.flow_head = nn.Sequential(
            nn.Conv2d(backbone_channels * 2, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 2, 1),  # (dx, dy) per pixel
        )

    def forward(
        self, current_frame: torch.Tensor, prev_frame: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(current_frame)
        detection_raw = self.detection_head(features)
        B, _, H, W = detection_raw.shape

        # Reshape detection output: (B, 5+C, H, W) → (B, H*W, 5+C)
        det = detection_raw.view(B, -1, H * W).permute(0, 2, 1)

        # Decode boxes
        box_xy = torch.sigmoid(det[..., 0:2])
        box_wh = det[..., 2:4]
        obj_conf = torch.sigmoid(det[..., 4])
        class_probs = F.softmax(det[..., 5:], dim=-1)

        # Simplified output (for API consumption, take top-K)
        scores, top_idx = obj_conf.topk(self.MAX_DETECTIONS, dim=1)

        output = {
            "boxes": box_xy.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 2)),
            "scores": scores,
            "class_probs": class_probs.gather(
                1, top_idx.unsqueeze(-1).expand(-1, -1, self.NUM_CLASSES)
            ),
        }

        # Optical flow between consecutive frames
        if prev_frame is not None:
            prev_features = self.backbone(prev_frame)
            flow_input = torch.cat([features, prev_features], dim=1)
            flow = self.flow_head(flow_input)
            output["optical_flow"] = flow.mean(dim=[2, 3])  # spatial mean → (B, 2)

        return output


# ===========================================================================
# MODULE 2 — Risk Heatmap Generation (U-Net Semantic Segmentation)
# ===========================================================================

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        _require_torch()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RiskHeatmapUNet(nn.Module):
    """
    U-Net encoder-decoder for per-pixel risk classification.

    Input:  (B, C_in, H, W) — stacked meteorological feature channels
    Output: (B, num_classes, H, W) — per-pixel class logits
            num_classes: 0=SAFE, 1=THUNDERSTORM, 2=CLOUDBURST, 3=FLASH_FLOOD

    Architecture: 4-level U-Net with skip connections.
    """

    NUM_CLASSES = 4

    def __init__(self, in_channels: int = 8, base_filters: int = 64):
        super().__init__()
        _require_torch()

        bf = base_filters

        # Encoder
        self.enc1 = DoubleConv(in_channels, bf)
        self.enc2 = DoubleConv(bf, bf * 2)
        self.enc3 = DoubleConv(bf * 2, bf * 4)
        self.enc4 = DoubleConv(bf * 4, bf * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(bf * 8, bf * 16)

        # Decoder with skip connections
        self.up4 = nn.ConvTranspose2d(bf * 16, bf * 8, 2, stride=2)
        self.dec4 = DoubleConv(bf * 16, bf * 8)

        self.up3 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.dec3 = DoubleConv(bf * 8, bf * 4)

        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = DoubleConv(bf * 4, bf * 2)

        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = DoubleConv(bf * 2, bf)

        # Output: per-pixel risk class probabilities
        self.classifier = nn.Conv2d(bf, self.NUM_CLASSES, 1)

        # Per-pixel risk score regression head (0-100)
        self.risk_score_head = nn.Sequential(
            nn.Conv2d(bf, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def _pad_to_match(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diffY = target.size(2) - x.size(2)
        diffX = target.size(3) - x.size(3)
        return F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder path with skip connections
        d4 = self.up4(b)
        d4 = self._pad_to_match(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self._pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return {
            "risk_class_logits": self.classifier(d1),
            "risk_score_map": self.risk_score_head(d1).squeeze(1) * 100.0,
        }


# ===========================================================================
# MODULE 3 — Spatiotemporal Nowcasting Engine (ConvLSTM + Transformer)
# ===========================================================================

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatiotemporal sequence modeling."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        _require_torch()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding
        )
        self.hidden_dim = hidden_dim

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        _require_torch()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SpatiotemporalNowcaster(nn.Module):
    """
    Hybrid ConvLSTM + Transformer encoder for spatiotemporal nowcasting.

    Learns how moisture, instability, and lift fields evolve minute-to-minute
    to forecast conditions 2–6 hours ahead.

    Input:  (B, T_in, C, H, W) — historical feature grid sequence
    Output: dict with
        - forecast:       (B, T_out, C, H, W) — predicted feature grids
        - risk_forecast:  (B, T_out, H, W)    — predicted risk scores
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        forecast_horizon: int = 6,  # predict 6 hours ahead
    ):
        super().__init__()
        _require_torch()

        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)

        # ConvLSTM encoder
        self.convlstm_layers = nn.ModuleList(
            [ConvLSTMCell(hidden_dim, hidden_dim) for _ in range(num_lstm_layers)]
        )

        # Transformer for temporal attention across frames
        self.temporal_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Decoder: project back to feature channels
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, in_channels, 1),
        )

        # Risk score decoder
        self.risk_decoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Autoregressive step predictor
        self.step_predictor = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T_in, C, H, W) — input sequence of feature grids
        Returns:
            dict with forecast and risk scores
        """
        B, T_in, C, H, W = x.shape

        # --- Encoder: ConvLSTM ---
        # Each layer maintains its own (h, c) state across timesteps
        states = [None] * len(self.convlstm_layers)
        spatial_features = []

        for t in range(T_in):
            inp = self.input_proj(x[:, t])  # (B, hidden, H, W)
            for i, layer in enumerate(self.convlstm_layers):
                h_new, c_new = layer(inp, states[i])
                states[i] = (h_new, c_new)
                inp = h_new
            spatial_features.append(inp)

        # Stack: (B, T_in, hidden, H, W)
        lstm_sequence = torch.stack(spatial_features, dim=1)

        # --- Temporal Transformer ---
        # Global average pool spatial dims: (B, T_in, hidden)
        pooled = lstm_sequence.mean(dim=(-2, -1))  # (B, T_in, hidden)
        pooled = self.temporal_encoding(pooled)
        transformed = self.temporal_transformer(pooled)  # (B, T_in, hidden)
        # Expand back to spatial: (B, T_in, hidden, H, W)
        transformed = transformed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)

        # --- Autoregressive Forecast Decoder ---
        forecasts = []
        current_h = transformed[:, -1]  # (B, hidden, H, W)

        for step in range(self.forecast_horizon):
            # Predict next spatial feature map
            dec_input = current_h.view(B, self.hidden_dim, H, W)
            predicted_features = self.decoder(dec_input)  # (B, C, H, W)
            forecasts.append(predicted_features)

            # Prepare next step: pool spatial to get hidden state
            projected = self.input_proj(predicted_features)
            pooled = projected.mean(dim=(-2, -1))  # (B, hidden)
            next_h = self.step_predictor(pooled, transformed[:, -1].mean(dim=(-2, -1)))  # (B, hidden)
            current_h = next_h.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Stack forecasts
        forecast_tensor = torch.stack(forecasts, dim=1)  # (B, T_out, C, H, W)

        # Risk score from forecast
        risk_scores = self.risk_decoder(forecast_tensor.view(-1, C, H, W)).view(
            B, self.forecast_horizon, H, W
        ) * 100.0

        return {
            "forecast": forecast_tensor,
            "risk_forecast": risk_scores,
        }


# ===========================================================================
# MODULE 4 — Multi-Hazard Prediction (Multi-Task Learning)
# ===========================================================================

class MultiHazardPredictor(nn.Module):
    """
    Shared backbone with 3 output heads for simultaneous prediction:
      1. Thunderstorm risk (binary)
      2. Cloudburst risk (binary)
      3. Flash flood risk (binary)
    Plus severity and flood-depth heads that learn the PHYSICS RESIDUAL.

    Physics residual learning
    -------------------------
    The SCS-CN + Manning baseline in ``app/services/physics/pinn_risk_model.py``
    already explains most of the variance on its own (measured on the Mumbai
    window: R2 = 0.79 for severity, R2 = 0.88 for flood depth). Training a
    network to reproduce that from scratch wastes capacity and discards known
    hydrology.

    So the severity and depth heads predict the *correction* to the physics
    baseline, not the absolute value::

        target_residual = y_true - physics_baseline
        y_hat           = physics_baseline + residual_head(x)

    Callers pass ``physics_baseline`` to ``forward`` to get absolute values.
    Without it the returned ``severity_score`` / ``flood_depth_cm`` are
    residual-only, so inference must supply the baseline (see
    ``feature_contract`` and the inference engine).

    The two regression heads therefore end in ``Tanh``: a residual is signed,
    and the old ``Sigmoid``/``ReLU`` activations could not represent a negative
    correction at all.

    Input:  (B, C_features) — flattened feature vector per grid cell
    Output: dict with per-cell predictions for all 3 hazards
    """

    def __init__(
        self,
        in_features: int = 30,
        shared_dim: int = 256,
        severity_residual_scale: float = SEVERITY_RESIDUAL_SCALE,
        depth_residual_scale: float = DEPTH_RESIDUAL_SCALE,
    ):
        super().__init__()
        _require_torch()

        self.severity_residual_scale = float(severity_residual_scale)
        self.depth_residual_scale = float(depth_residual_scale)

        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(shared_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Task-specific heads
        self.thunderstorm_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.cloudburst_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.flash_flood_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Severity residual head — Tanh because a correction is signed.
        self.severity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Flood depth residual head (cm) — also signed.
        self.depth_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        physics_baseline: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict hazards, optionally anchored on a physics baseline.

        ``physics_baseline`` may supply ``severity`` and/or ``depth`` tensors
        aligned with ``x``. When omitted the regression outputs are the raw
        residuals (the network's own correction, without the physics term).
        """
        shared = self.shared_backbone(x)

        severity_residual = self.severity_head(shared).squeeze(-1) * self.severity_residual_scale
        depth_residual = self.depth_head(shared).squeeze(-1) * self.depth_residual_scale

        base_severity = 0.0
        base_depth = 0.0
        if physics_baseline is not None:
            if physics_baseline.get("severity") is not None:
                base_severity = physics_baseline["severity"]
            if physics_baseline.get("depth") is not None:
                base_depth = physics_baseline["depth"]

        # Physics provides the absolute scale; the head only corrects it.
        severity = torch.clamp(base_severity + severity_residual, 0.0, 100.0)
        depth = torch.clamp(base_depth + depth_residual, min=0.0)

        return {
            "thunderstorm_prob": self.thunderstorm_head(shared).squeeze(-1),
            "cloudburst_prob": self.cloudburst_head(shared).squeeze(-1),
            "flash_flood_prob": self.flash_flood_head(shared).squeeze(-1),
            "severity_score": severity,
            "flood_depth_cm": depth,
            "severity_residual": severity_residual,
            "depth_residual": depth_residual,
        }


# ===========================================================================
# MODULE 5 — Multi-Source Data Alignment (Cross-Attention Fusion)
# ===========================================================================

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion layer that aligns satellite, reanalysis,
    and DEM data streams at each grid point.

    Treats each data source as a sequence of tokens and learns
    cross-source attention for unified feature representation.
    """

    def __init__(
        self,
        source_dims: Dict[str, int] = None,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()
        _require_torch()

        if source_dims is None:
            source_dims = {
                "satellite": 8,      # CTT, CTT_drop, WV/IWV, TPW, CMV_u, CMV_v, VIS, IR
                "reanalysis": 12,    # CAPE, CIN, LI, temp, humidity, U_wind, V_wind, shear, convergence, geopotential, precip, soil
                "dem": 5,            # elevation, slope, aspect, roughness, drainage_dist
                "radar": 4,          # reflectivity, velocity, spectrum_width, zdr
            }

        self.source_names = list(source_dims.keys())

        # Project each source to shared d_model dimension
        self.source_projections = nn.ModuleDict({
            name: nn.Linear(dim, d_model) for name, dim in source_dims.items()
        })

        # Source type embeddings
        self.source_embeddings = nn.Embedding(len(source_dims), d_model)

        # Cross-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self, source_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            source_data: dict mapping source_name → (B, N_tokens, D_source)
        Returns:
            fused_features: (B, D_model) per grid point
        """
        B = next(iter(source_data.values())).size(0)
        all_tokens = []

        for i, (name, data) in enumerate(source_data.items()):
            if name in self.source_projections:
                projected = self.source_projections[name](data)  # (B, N, d_model)
                # Add source type embedding
                src_emb = self.source_embeddings(
                    torch.tensor(i, device=data.device)
                ).unsqueeze(0).unsqueeze(0)
                projected = projected + src_emb
                all_tokens.append(projected)

        # Concatenate all source tokens
        combined = torch.cat(all_tokens, dim=1)  # (B, sum(N_i), d_model)

        # Cross-attention fusion
        fused = self.fusion_transformer(combined)

        # Global pooling → single fused representation per sample
        fused_pooled = fused.mean(dim=1)  # (B, d_model)
        output = self.output_proj(fused_pooled)

        return {
            "fused_features": output,
            "per_token_attention": fused,  # for XAI visualization
        }


# ===========================================================================
# MODULE 6 — Urban Flood Depth Estimation (GNN)
# ===========================================================================

class DrainageGNNLayer(nn.Module):
    """Single Graph Neural Network layer for drainage network message passing."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        _require_torch()
        self.message_fn = nn.Linear(in_dim * 2, out_dim)
        self.update_fn = nn.GRUCell(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, in_dim) — features for all nodes
            edge_index: (2, E) — source → target edges
        """
        src, dst = edge_index

        # Gather source and destination features
        src_feat = node_features[src]
        dst_feat = node_features[dst]

        # Compute messages
        messages = self.message_fn(torch.cat([src_feat, dst_feat], dim=-1))

        # Aggregate messages (mean over incoming edges)
        aggregated = torch.zeros_like(node_features[:, : messages.size(-1)])
        count = torch.zeros(node_features.size(0), 1, device=node_features.device)
        aggregated.index_add_(0, dst, messages)
        count.index_add_(0, dst, torch.ones_like(src_feat[:, :1]))
        count = count.clamp(min=1)
        aggregated = aggregated / count

        # Update node features
        updated = self.update_fn(aggregated, node_features[:, : aggregated.size(-1)])
        return self.norm(updated)


class UrbanFloodDepthGNN(nn.Module):
    """
    Graph Neural Network over the urban drainage network.

    Treats manholes/pipes as graph nodes, predicts overflow nodes,
    and regresses the exact water depth per street segment.

    Input:
        node_features: (N, node_feat_dim) — per-node features
        edge_index:    (2, E) — pipe/manhole connectivity
    Output:
        dict with per-node overflow probability and water depth
    """

    def __init__(
        self,
        node_feat_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        _require_torch()

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList(
            [DrainageGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Overflow prediction head (binary)
        self.overflow_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Water depth regression head (cm)
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        # Flow velocity estimation (m/s)
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = self.input_proj(node_features)

        for gnn_layer in self.gnn_layers:
            residual = x
            x = gnn_layer(x, edge_index) + residual

        return {
            "overflow_prob": self.overflow_head(x).squeeze(-1),
            "water_depth_cm": self.depth_head(x).squeeze(-1),
            "flow_velocity_ms": self.velocity_head(x).squeeze(-1),
        }


# ===========================================================================
# MODULE 7 — Forecast Trust Scoring (Anomaly Detection + k-NN)
# ===========================================================================

class ForecastTrustScorer(nn.Module):
    """
    Anomaly detection + similarity search on historical forecast-error embeddings.

    Flags when the current pattern resembles past high-error cases,
    producing High/Medium/Low confidence labels.

    Uses an autoencoder for anomaly detection and learns an embedding
    space where k-NN search finds similar historical forecasts.
    """

    def __init__(
        self,
        input_dim: int = 32,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        _require_torch()

        # Autoencoder for anomaly detection
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Trust classifier (operates on reconstruction error + embedding)
        self.trust_head = nn.Sequential(
            nn.Linear(embedding_dim + 1, 64),  # +1 for reconstruction error
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # HIGH, MEDIUM, LOW confidence
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)

        # Reconstruction error as anomaly score
        recon_error = F.mse_loss(reconstructed, x, reduction="none").mean(dim=-1, keepdim=True)

        # Trust classification
        trust_input = torch.cat([embedding, recon_error], dim=-1)
        trust_logits = self.trust_head(trust_input)

        return {
            "embedding": embedding,
            "reconstruction_error": recon_error.squeeze(-1),
            "trust_logits": trust_logits,
            "trust_probs": F.softmax(trust_logits, dim=-1),
        }


# ===========================================================================
# MODULE 8 — Explainable AI (XAI) Layer
# ===========================================================================

class XAIAttentionLayer(nn.Module):
    """
    Attention-weight visualization layer.

    Learns which meteorological features are most important for
    each prediction, surfacing top drivers like
    'high CAPE + rapid CTT drop' behind every alert.

    This is a lightweight wrapper that computes feature importance
    scores via learned attention weights over input features.
    """

    def __init__(self, num_features: int = 30, num_heads: int = 6):
        super().__init__()
        _require_torch()

        self.num_features = num_features

        # Self-attention over feature dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=1, num_heads=1, batch_first=True
        )

        # Feature importance scoring
        self.importance_scorer = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
            nn.Softmax(dim=-1),
        )

        # SHAP-like feature interaction detector
        self.interaction_detector = nn.Sequential(
            nn.Linear(num_features * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, num_features) — per-cell feature vector
        Returns:
            dict with feature importance weights and attention maps
        """
        B = features.shape[0]

        # Per-feature importance scores
        importance = self.importance_scorer(features)  # (B, num_features)

        # Feature interaction scores (pairwise importance)
        expanded = features.unsqueeze(2).expand(-1, -1, self.num_features)
        concat_pairs = torch.cat([
            expanded, expanded.transpose(1, 2)
        ], dim=-1)  # (B, F, 2F)
        interactions = self.interaction_detector(concat_pairs)  # (B, F)

        # Attention over features for interpretability
        feat_seq = features.unsqueeze(-1)  # (B, num_features, 1)
        attn_out, attn_weights = self.attention(feat_seq, feat_seq, feat_seq)

        return {
            "feature_importance": importance,
            "feature_names": None,  # filled by caller
            "interaction_scores": interactions,
            "attention_weights": attn_weights.squeeze(-1),  # (B, F, F)
        }


# ===========================================================================
# MODULE 9 — Crowd-Report Validation (NLP Text Classification)
# ===========================================================================

class CrowdReportClassifier(nn.Module):
    """
    NLP text classifier for citizen-submitted flood reports.

    Reads WhatsApp/Telegram crowd reports and classifies them as
    confirming or denying a predicted flood zone.

    Uses a lightweight 1D CNN + embedding architecture (no heavy
    transformer required for short text classification).
    """

    NUM_CLASSES = 3  # CONFIRMS, DENIES, UNRELATED
    MAX_SEQ_LEN = 128
    VOCAB_SIZE = 30000

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        _require_torch()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)

        # Multi-scale CNN for text feature extraction
        self.conv3 = nn.Conv1d(embed_dim, 128, 3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 128, 5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, 128, 7, padding=3)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        # Sentiment confidence scorer
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, max_seq_len) — tokenized report text
        Returns:
            dict with classification logits, confidence, attention weights
        """
        embedded = self.embedding(input_ids)  # (B, seq_len, embed_dim)
        embedded = self.pos_encoding(embedded)
        x = embedded.transpose(1, 2)  # (B, embed_dim, seq_len)

        # Multi-scale convolution
        c3 = F.relu(self.conv3(x)).max(dim=2)[0]  # (B, 128)
        c5 = F.relu(self.conv5(x)).max(dim=2)[0]
        c7 = F.relu(self.conv7(x)).max(dim=2)[0]
        combined = c3 + c5 + c7  # (B, 128)

        logits = self.classifier(combined)
        confidence = self.confidence_head(combined)

        return {
            "logits": logits,
            "predictions": F.softmax(logits, dim=-1),
            "confidence": confidence.squeeze(-1),
        }
