from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class FeatureBase(BaseModel):
    cell_index: int
    cell_lat: float
    cell_lon: float
    # Precipitation & Moisture
    rainfall_1h_mm: float
    rainfall_3h_mm: float
    rainfall_6h_mm: float
    rainfall_24h_mm: float
    soil_moisture_pct: float
    soil_saturation_factor: float
    # Instability & CTT
    cape_instability_jkg: float
    cloud_top_temp_celsius: float = -45.0
    ctt_drop_rate_c_hr: float = 0.0
    # Kinematics & Lift
    wind_speed_10m_kmh: float = 20.0
    wind_direction_10m_deg: float = 240.0
    wind_u_ms: float = -5.0
    wind_v_ms: float = -3.0
    wind_gusts_kmh: float = 30.0
    # Topography
    elevation_m: float
    slope_deg: float
    runoff_coefficient: float
    effective_runoff_mm_hr: float
    drainage_outfall_dist_m: float
    is_depression_bowl: bool
    # Coastal
    tide_height_m: float
    is_high_tide_locked: bool
    # Ground-Truth Target Labels
    target_observed_flood_depth_cm: float = 0.0
    target_severity_class: int = 0
    target_flash_flood_flag: int = 0
    target_cloudburst_flag: int = 0
    target_waterlogging_flag: int = 0
    raw_payload_json: Optional[Dict[str, Any]] = None


class FeatureCreate(FeatureBase):
    region_id: str
    timestamp: datetime


class FeatureRead(FeatureBase):
    id: str
    region_id: str
    timestamp: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FeatureGridSnapshot(BaseModel):
    region_id: str
    region_code: str
    timestamp: datetime
    timestep_id: Optional[int] = None
    total_cells: int
    avg_rainfall_1h_mm: float
    max_rainfall_1h_mm: float
    avg_cloud_top_temp_celsius: Optional[float] = -45.0
    tide_height_m: float
    is_high_tide_locked: bool
    cells: List[FeatureRead]
