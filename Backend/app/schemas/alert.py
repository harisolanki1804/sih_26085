from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ScoreDecomposition(BaseModel):
    total_score: float
    rainfall_component: float       # max 40
    soil_saturation_component: float # max 25
    topography_component: float      # max 20
    instability_component: float     # max 15
    max_possible: float = 100.0


class TrustMetrics(BaseModel):
    trust_score: float             # 0 to 100
    trust_level: str               # HIGH_CONFIDENCE, MODERATE_CONFIDENCE, CAUTIONARY
    sensor_agreement: float        # percentage agreement between satellite, radar/reanalysis, gauges
    uncertainty_margin_pct: float
    key_drivers: List[str]


class AlertBase(BaseModel):
    cell_lat: float
    cell_lon: float
    cell_id: Optional[str] = None
    locality_name: Optional[str] = None
    timestep: Optional[int] = None
    timestamp: datetime
    alert_type: str = "FLASH_FLOOD"
    severity: str = "HIGH"
    risk_score_total: float
    rainfall_score_contrib: float
    soil_saturation_score_contrib: float
    topography_score_contrib: float
    atmospheric_instability_score_contrib: float
    flood_depth_estimate_cm: float
    trust_score: float = 88.0
    trust_level: str = "HIGH_CONFIDENCE"
    trust_factors_json: Optional[Dict[str, Any]] = None
    reasoning_summary: str
    recommended_action: Optional[str] = None
    contributing_hotspot_name: Optional[str] = None
    is_active: bool = True


class AlertCreate(AlertBase):
    region_id: str
    event_id: Optional[str] = None


class AlertRead(AlertBase):
    id: str
    region_id: str
    event_id: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra='ignore')


class AlertDetailExplanation(BaseModel):
    alert: AlertRead
    score_breakdown: ScoreDecomposition
    trust_breakdown: TrustMetrics
    physical_drivers: Dict[str, Any]
    action_playbook: List[str]
