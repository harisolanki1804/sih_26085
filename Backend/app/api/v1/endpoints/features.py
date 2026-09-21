import os
import json
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.models.region import Region
from app.models.feature import Feature
from app.schemas.feature import FeatureRead, FeatureGridSnapshot

router = APIRouter()


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string safely, handling synthetic hour overflow (e.g. 2022-07-05T36:00:00Z)."""
    clean_str = ts_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(clean_str)
    except ValueError:
        import re
        from datetime import timedelta
        match = re.match(r"(\d{4}-\d{2}-\d{2})T(\d+):(\d{2}):(\d{2})(.*)", clean_str)
        if match:
            date_part, hours_str, mins_str, secs_str, tz_part = match.groups()
            total_hours = int(hours_str)
            days_add = total_hours // 24
            rem_hours = total_hours % 24
            base_dt = datetime.fromisoformat(f"{date_part}T{str(rem_hours).zfill(2)}:{mins_str}:{secs_str}{tz_part}")
            return base_dt + timedelta(days=days_add)
        return datetime.utcnow()


@router.get("/latest", response_model=FeatureGridSnapshot, summary="Get Latest Feature Grid Snapshot")
def get_latest_feature_grid(
    region_code: Optional[str] = Query(default=settings.DEFAULT_REGION_CODE),
    db: Session = Depends(get_db)
):
    """Returns spatial grid feature state for the current active timestep."""
    # Read directly from replay or DB
    path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feature matrix not preprocessed.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timesteps = data.get("timesteps", [])
    if not timesteps:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No timesteps available.")

    # Return first timestep by default if not specified
    ts = timesteps[0]
    region = db.query(Region).filter(Region.code == region_code).first()

    cells_read = []
    for c in ts["features"]:
        cells_read.append(
            FeatureRead(
                id=f"feat-cell-{c['cell_index']}",
                region_id=region.id if region else "default",
                timestamp=parse_iso_timestamp(ts["timestamp"]),
                cell_index=c["cell_index"],
                cell_lat=c["lat"],
                cell_lon=c["lon"],
                rainfall_1h_mm=c["rainfall_1h_mm"],
                rainfall_3h_mm=c["rainfall_3h_mm"],
                rainfall_6h_mm=c["rainfall_6h_mm"],
                rainfall_24h_mm=c["rainfall_24h_mm"],
                soil_moisture_pct=c["soil_moisture_pct"],
                soil_saturation_factor=c.get("soil_saturation_factor", 0.5),
                cape_instability_jkg=c["cape_instability_jkg"],
                cloud_top_temp_celsius=c.get("cloud_top_temp_celsius", -45.0),
                ctt_drop_rate_c_hr=c.get("ctt_drop_rate_c_hr", 0.0),
                wind_speed_10m_kmh=c.get("wind_speed_10m_kmh", 20.0),
                wind_direction_10m_deg=c.get("wind_direction_10m_deg", 240.0),
                wind_u_ms=c.get("wind_u_ms", -5.0),
                wind_v_ms=c.get("wind_v_ms", -3.0),
                wind_gusts_kmh=c.get("wind_gusts_kmh", 30.0),
                elevation_m=c["elevation_m"],
                slope_deg=c["slope_deg"],
                runoff_coefficient=c["runoff_coefficient"],
                effective_runoff_mm_hr=c["effective_runoff_mm_hr"],
                drainage_outfall_dist_m=c["drainage_outfall_dist_m"],
                is_depression_bowl=c["is_depression_bowl"],
                tide_height_m=ts.get("tide_height_m", 2.5),
                is_high_tide_locked=ts.get("is_high_tide_locked", False),
                target_observed_flood_depth_cm=c.get("target_observed_flood_depth_cm", 0.0),
                target_severity_class=c.get("target_severity_class", 0),
                target_flash_flood_flag=c.get("target_flash_flood_flag", 0),
                target_cloudburst_flag=c.get("target_cloudburst_flag", 0),
                target_waterlogging_flag=c.get("target_waterlogging_flag", 0),
                created_at=datetime.utcnow()
            )
        )

    return FeatureGridSnapshot(
        region_id=region.id if region else "default",
        region_code=region_code,
        timestamp=datetime.fromisoformat(ts["timestamp"].replace("Z", "+00:00")),
        timestep_id=ts["timestep_id"],
        total_cells=len(cells_read),
        avg_rainfall_1h_mm=ts["avg_rainfall_1h_mm"],
        max_rainfall_1h_mm=ts["max_rainfall_1h_mm"],
        avg_cloud_top_temp_celsius=ts.get("avg_cloud_top_temp_celsius", -45.0),
        tide_height_m=ts.get("tide_height_m", 2.5),
        is_high_tide_locked=ts.get("is_high_tide_locked", False),
        cells=cells_read
    )


@router.get("/history", summary="Get Historical Time Series Summary")
def get_feature_timeseries_summary(
    region_code: Optional[str] = Query(default=settings.DEFAULT_REGION_CODE)
):
    """Returns overview time-series timeline across all 72 historical timesteps."""
    path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feature dataset missing.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = []
    for ts in data.get("timesteps", []):
        summary.append({
            "timestep_id": ts["timestep_id"],
            "timestamp": ts["timestamp"],
            "phase": ts["phase"],
            "avg_rainfall_1h_mm": ts["avg_rainfall_1h_mm"],
            "max_rainfall_1h_mm": ts["max_rainfall_1h_mm"],
            "tide_height_m": ts.get("tide_height_m", 2.5),
            "is_high_tide_locked": ts.get("is_high_tide_locked", False)
        })

    return {
        "region_code": region_code,
        "total_timesteps": len(summary),
        "timeline": summary
    }
