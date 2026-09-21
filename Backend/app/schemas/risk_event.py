from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class RiskEventBase(BaseModel):
    event_code: str
    title: str
    description: Optional[str] = None
    start_time: datetime
    peak_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    peak_rainfall_rate_mm_hr: float = 0.0
    max_predicted_depth_cm: float = 0.0
    severity_level: str = "LOW"
    status: str = "MONITORING"
    confidence_score: float = 85.0
    triggered_by: str = "VARUNA Ensemble Cloudburst Detector"


class RiskEventCreate(RiskEventBase):
    region_id: str


class RiskEventUpdate(BaseModel):
    title: Optional[str] = None
    peak_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    peak_rainfall_rate_mm_hr: Optional[float] = None
    max_predicted_depth_cm: Optional[float] = None
    severity_level: Optional[str] = None
    status: Optional[str] = None
    confidence_score: Optional[float] = None


class RiskEventRead(RiskEventBase):
    id: str
    region_id: str
    created_at: datetime
    active_alerts_count: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)
