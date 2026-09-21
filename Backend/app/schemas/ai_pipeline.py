from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class StormCellInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cell_index: int
    lat: float
    lon: float
    confidence: float
    storm_type: str
    motion_vector: Dict[str, float]
    area_km2: float
    peak_echo_top_km: float
    track_id: str
    lifetime_steps: int
    trend: str


class StormCellDetectionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    total_cells_detected: int
    cells: List[StormCellInfo]
    peak_storm_intensity: float
    tracking_id: str


class RiskPixelResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cell_index: int
    lat: float
    lon: float
    dominant_risk_class: str
    risk_class_probs: Dict[str, float]
    pixel_risk_score: float


class RiskHeatmapResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    resolution: str
    total_pixels: int
    risk_class_distribution: Dict[str, int]
    max_pixel_risk: float
    mean_pixel_risk: float
    heatmap: List[RiskPixelResult]


class NowcastForecast(BaseModel):
    model_config = ConfigDict(extra="ignore")
    forecast_hour: int
    predicted_avg_rainfall_mm_hr: float
    predicted_risk_level: str
    confidence: float
    trend: str


class NowcastResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    horizon_hours: int
    temporal_context_timesteps: int = 0
    trend: str = "insufficient_data"
    trend_rate_mm_hr_per_step: float = 0.0
    forecasts: List[NowcastForecast] = []


class MultiHazardCellPrediction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cell_index: int
    lat: float
    lon: float
    thunderstorm_prob: float
    cloudburst_prob: float
    flash_flood_prob: float
    severity_score: float
    predicted_depth_cm: float
    is_cloudburst: bool
    is_waterlogging: bool
    model: Optional[str] = None


class MultiHazardResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    total_cells: int
    aggregate_thunderstorm_prob: float
    aggregate_cloudburst_prob: float
    aggregate_flash_flood_prob: float
    max_severity_score: float
    max_predicted_depth_cm: float
    cell_predictions: List[MultiHazardCellPrediction]
    inference_mode: Optional[str] = None


class FloodDepthNode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cell_index: int
    lat: float
    lon: float
    water_depth_cm: float
    overflow_probability: float
    flow_velocity_ms: float
    is_overflow_node: bool
    drainage_bottleneck: bool


class FloodDepthResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    total_nodes: int
    max_water_depth_cm: float
    mean_water_depth_cm: float
    overflow_nodes_count: int
    critical_bottlenecks: List[FloodDepthNode]
    node_estimates: List[FloodDepthNode]
    inference_mode: Optional[str] = None


class TrustComponent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cross_source_agreement: float
    nowcast_confidence: float
    storm_detection_consistency: float
    depth_severity_agreement: float


class HistoricalCase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    case_id: str
    similarity: float
    historical_error_pct: float
    note: str


class TrustScoreResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trust_score: float
    trust_level: str
    components: TrustComponent
    is_anomalous_pattern: bool
    similar_historical_cases: List[HistoricalCase]
    uncertainty_margin_pct: float
    conformal_prediction: Optional[Dict[str, Any]] = None
    confidence_guarantee: Optional[str] = None


class XAIDriver(BaseModel):
    model_config = ConfigDict(extra="ignore")
    factor: str
    value: str
    weight: float
    contribution: float


class XAICellExplanation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cell_index: int
    lat: float
    lon: float
    top_drivers: List[XAIDriver]
    explanation_text: str
    feature_importance_map: Dict[str, float]


class XAIExplanationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    explanation_method: str
    feature_importance_global: Dict[str, float]
    total_cells_analyzed: int
    high_risk_cells_count: int
    cell_explanations: List[XAICellExplanation]
    global_summary: str


class FusedCellResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cell_index: int
    fused_risk_vector: List[float]
    cross_source_agreement: float
    dominant_source: str


class FusedFeaturesResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    fusion_method: str
    data_sources: List[str]
    source_contributions: Optional[Dict[str, float]] = None
    global_agreement_score: float
    per_cell_fusion: List[FusedCellResult]
    inference_mode: Optional[str] = None


class CrowdReportClassification(BaseModel):
    model_config = ConfigDict(extra="ignore")
    report_text: str
    classification: str
    confidence: float
    detected_location: Optional[str] = None
    confirm_score: int = 0
    deny_score: int = 0
    unrelated_score: int = 0
    actionable: bool = False
    probabilities: Optional[Dict[str, float]] = None
    inference_mode: Optional[str] = None
    flagged_for_retraining: Optional[bool] = False
    model_risk_at_location: Optional[float] = None
    agrees_with_model: Optional[bool] = None


class CrowdReportRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    report_text: str
    predicted_flood_lat: Optional[float] = None
    predicted_flood_lon: Optional[float] = None


class FullInferenceResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    """Complete AI pipeline output for a single timestep."""
    timestep_id: int
    timestamp: str
    storm_cells: StormCellDetectionResult
    risk_heatmap: RiskHeatmapResult
    nowcast: NowcastResult
    multi_hazard: MultiHazardResult
    fused_features: FusedFeaturesResult
    flood_depth: FloodDepthResult
    trust_score: TrustScoreResult
    xai_explanation: XAIExplanationResult
    physics_risk: Optional[Dict[str, Any]] = None
    satellite_data: Optional[Dict[str, Any]] = None


class TrainingStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")
    module_name: str
    status: str  # "idle", "training", "completed", "error"
    epochs_completed: int = 0
    total_epochs: int = 100
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    last_updated: Optional[str] = None


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    torch_available: bool
    using_gpu: bool
    models_loaded: List[str]
    modules_trained: List[TrainingStatus]
