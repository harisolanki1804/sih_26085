"""
VARUNA AI/ML Module Suite
=========================
Nine integrated AI/ML modules for multi-hazard weather prediction:

1. Storm Cell Detection & Tracking  (CNN + Optical Flow)
2. Risk Heatmap Generation          (U-Net Semantic Segmentation)
3. Spatiotemporal Nowcasting Engine  (ConvLSTM / Vision Transformer)
4. Multi-Hazard Prediction          (Multi-Task Learning)
5. Multi-Source Data Alignment      (Cross-Attention Fusion)
6. Urban Flood Depth Estimation     (Graph Neural Network)
7. Forecast Trust Scoring           (Anomaly Detection + k-NN)
8. Explainable AI (XAI) Layer       (SHAP + Attention Visualization)
9. Crowd-Report Validation          (NLP Text Classification)

All modules consume the feature_grid_timeseries.json and produce
structured outputs consumed by the backend API and alert system.
"""

from app.services.ai.inference_engine import VARUNAInferenceEngine

# Singleton inference engine — lazy-initialized on first access
_inference_engine = None


def get_inference_engine() -> VARUNAInferenceEngine:
    """Returns the singleton VARUNA inference engine, creating it if needed."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = VARUNAInferenceEngine()
    return _inference_engine


__all__ = [
    "get_inference_engine",
    "VARUNAInferenceEngine",
]
