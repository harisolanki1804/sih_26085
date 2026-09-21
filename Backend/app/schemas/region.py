from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class RegionBase(BaseModel):
    code: str
    name: str
    description: Optional[str] = None
    bbox_lat_min: float
    bbox_lat_max: float
    bbox_lon_min: float
    bbox_lon_max: float
    center_lat: float
    center_lon: float
    area_sqkm: Optional[float] = None
    avg_drainage_capacity_mm_hr: float = 25.0
    high_tide_threshold_m: float = 4.5
    metadata_json: Optional[Dict[str, Any]] = None


class RegionCreate(RegionBase):
    pass


class RegionRead(RegionBase):
    id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HotspotInfo(BaseModel):
    name: str
    lat: float
    lon: float
    base_elevation_m: float
    vulnerability: str
