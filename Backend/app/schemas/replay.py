from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from app.schemas.alert import AlertRead
from app.schemas.risk_event import RiskEventRead


class ReplayStatus(BaseModel):
    is_running: bool
    current_timestep: int
    total_timesteps: int
    current_timestamp: str
    phase: str
    region_code: str
    active_alerts_count: int
    max_risk_score: float
    max_flood_depth_cm: float
    tide_height_m: float
    is_high_tide_locked: bool
    avg_rainfall_1h_mm: float


class ReplayStepResponse(BaseModel):
    timestep_id: int
    timestamp: str
    phase: str
    tide_height_m: float
    is_high_tide_locked: bool
    avg_rainfall_1h_mm: float
    max_rainfall_1h_mm: float
    new_alerts_count: int
    active_event: Optional[RiskEventRead] = None
    alerts_generated: List[AlertRead] = []
    ai_pipeline_results: Optional[Dict[str, Any]] = None
    summary_message: str


class ReplayControlRequest(BaseModel):
    step_to: Optional[int] = None
    speed_multiplier: Optional[float] = 1.0
