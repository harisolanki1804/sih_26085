"""
VARUNA Historical Deluge Replay Engine
---------------------------------------
Replays historical extreme rainfall time-series as a live-stream simulation.
Ticking through timesteps computes active multi-hazard risk scores,
evaluates explainability, updates DB features, and raises dynamic alerts.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.region import Region
from app.models.feature import Feature
from app.models.risk_event import RiskEvent, SeverityLevel, EventStatus
from app.models.alert import Alert
from app.services.risk_model import risk_model_service
from app.services.explainability import explainability_service
from app.services.ai import get_inference_engine
from app.services.alert_dispatch import dispatch_alert

logger = logging.getLogger("VARUNA.ReplayEngine")

# Grid bounds matching frontend
LAT_MIN, LAT_MAX = 18.88, 19.26
LON_MIN, LON_MAX = 72.78, 73.00
ROWS, COLS = 10, 9
LOCALITY = [
  ['Sea','Colaba','Fort','Churchgate','Marine Drive','Nariman Point','Malabar Hill','Walkeshwar','Haji Ali'],
  ['Sea','Grant Road','Tardeo','Bhuleshwar','Girgaon','Parel','Mahalaxmi','Byculla','Mazgaon'],
  ['Sea','Mumbai Central','Worli','Matunga','Sion','Wadala','Sewri','Chinchpokli','Reay Road'],
  ['Mahim','Dadar West','Dadar East','Kurla','Vidyavihar','Ghatkopar','BKC','Kalina','Santacruz'],
  ['Sea','Bandra','Bandra West','Khar','Chembur','Powai','Hiranandani','Chembur East','Navi Mumbai'],
  ['Juhu Beach','Juhu','Versova','Lokhandwala','Saki Naka','Ghatkopar E','Vikhroli','Kanjurmarg','Nahur'],
  ['Amboli','Jogeshwari','Andheri West','Andheri East','Marol','Powai Lake','Chandivali','Bhandup','Mulund'],
  ['Malvani','Malad West','Goregaon','Kandivali','Borivali','Deonar','Govandi','Mulund East','Thane Creek'],
  ['Erangal','Kandivali West','Borivali West','Dahisar','Mira Road','Thane West','Wagle Estate','Thane','Kopar Khairane'],
  ['Madh Island','Marve','Manori','Vasai','Nallasopara','Vashi','Sanpada','Nerul','Belapur'],
]

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


def _latlon_to_cell(lat: float, lon: float) -> tuple:
    """Convert lat/lon to (cell_id, locality_name)."""
    r = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * ROWS)
    c = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * COLS)
    r = max(0, min(ROWS - 1, r))
    c = max(0, min(COLS - 1, c))
    idx = r * COLS + c
    cell_id = f"C{str(idx + 1).zfill(2)}"
    name = LOCALITY[r][c]
    return cell_id, name


def _is_sea_cell(lat: float, lon: float) -> bool:
    """True when the cell centre is open sea (Arabian Sea west of Mumbai).

    Matches the frontend map mask so that sea cells never raise alerts:
      - column 0 (lon ~72.79) for the southern rows is open sea; only the
        far-north island rows (Marve/Manori/Gorai ~72.79) touch land.
      - the row-3/col-1 centre sits in Mahim Bay.
    """
    c = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * COLS)
    r = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * ROWS)
    if c == 0 and r <= 6:
        return True
    if c == 1 and r == 3:
        return True
    return False


class ReplaySimulationEngine:
    def __init__(self):
        self.current_timestep: int = 1
        self.is_running: bool = False
        self.data_cache: Optional[Dict[str, Any]] = None
        self._load_dataset()

    def _load_dataset(self):
        path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data_cache = json.load(f)
            logger.info(f"Replay engine loaded {len(self.data_cache.get('timesteps', []))} timesteps.")
        else:
            logger.warning(f"Feature dataset not found at {path}. Replay will be uninitialized.")

    def get_status(self, db: Session) -> Dict[str, Any]:
        if not self.data_cache:
            self._load_dataset()

        total_steps = len(self.data_cache.get("timesteps", [])) if self.data_cache else 0
        ts_data = self._get_current_timestep_data()

        active_alerts_count = db.query(Alert).filter(Alert.is_active == True).count()
        
        max_score = 0.0
        max_depth = 0.0
        if ts_data:
            for cell in ts_data.get("features", []):
                risk = risk_model_service.calculate_cell_risk(
                    rain_1h=cell["rainfall_1h_mm"],
                    rain_3h=cell["rainfall_3h_mm"],
                    rain_24h=cell["rainfall_24h_mm"],
                    soil_moist_pct=cell["soil_moisture_pct"],
                    elevation_m=cell["elevation_m"],
                    slope_deg=cell["slope_deg"],
                    cape_jkg=cell["cape_instability_jkg"],
                    is_tide_locked=ts_data.get("is_high_tide_locked", False),
                    tide_height_m=ts_data.get("tide_height_m", 2.5),
                    drainage_outfall_dist_m=cell["drainage_outfall_dist_m"],
                    is_depression=cell["is_depression_bowl"]
                )
                if risk["total_risk_score"] > max_score:
                    max_score = risk["total_risk_score"]
                if risk["flood_depth_estimate_cm"] > max_depth:
                    max_depth = risk["flood_depth_estimate_cm"]

        return {
            "is_running": self.is_running,
            "current_timestep": self.current_timestep,
            "total_timesteps": total_steps,
            "current_timestamp": ts_data["timestamp"] if ts_data else "",
            "phase": ts_data["phase"] if ts_data else "Idle",
            "region_code": self.data_cache.get("region_code", settings.DEFAULT_REGION_CODE) if self.data_cache else settings.DEFAULT_REGION_CODE,
            "active_alerts_count": active_alerts_count,
            "max_risk_score": max_score,
            "max_flood_depth_cm": max_depth,
            "tide_height_m": ts_data.get("tide_height_m", 2.5) if ts_data else 2.5,
            "is_high_tide_locked": ts_data.get("is_high_tide_locked", False) if ts_data else False,
            "avg_rainfall_1h_mm": ts_data.get("avg_rainfall_1h_mm", 0.0) if ts_data else 0.0
        }

    def _get_current_timestep_data(self) -> Optional[Dict[str, Any]]:
        if not self.data_cache:
            return None
        timesteps = self.data_cache.get("timesteps", [])
        idx = max(0, min(len(timesteps) - 1, self.current_timestep - 1))
        return timesteps[idx] if timesteps else None

    def step(self, db: Session, target_step: Optional[int] = None) -> Dict[str, Any]:
        """Advances simulation by 1 timestep or jumps to target_step and executes real-time AI risk inference."""
        if not self.data_cache:
            self._load_dataset()
        if not self.data_cache:
            raise ValueError("Feature dataset not available.")

        total_steps = len(self.data_cache.get("timesteps", []))
        if target_step is not None:
            self.current_timestep = max(1, min(total_steps, target_step))
        else:
            self.current_timestep = self.current_timestep + 1 if self.current_timestep < total_steps else 1

        ts_data = self._get_current_timestep_data()
        if not ts_data:
            raise ValueError("Invalid timestep data.")

        # Lookup region
        region = db.query(Region).filter(Region.code == self.data_cache["region_code"]).first()
        if not region:
            # Fallback to first region
            region = db.query(Region).first()

        region_id = region.id if region else "default"

        # Lookup or create RiskEvent
        event_code = self.data_cache.get("event_code", settings.DEFAULT_EVENT_CODE)
        event = db.query(RiskEvent).filter(RiskEvent.event_code == event_code).first()
        ts_datetime = parse_iso_timestamp(ts_data["timestamp"])

        if not event and region:
            event = RiskEvent(
                region_id=region.id,
                event_code=event_code,
                title="Historical Mumbai Monsoon Cloudburst Case Study",
                description="Simulated real-time replay of extreme convective deluge over Mithi river basin.",
                start_time=ts_datetime,
                severity_level=SeverityLevel.LOW.value,
                status=EventStatus.MONITORING.value
            )
            db.add(event)
            db.flush()

        # Deactivate older alerts
        db.query(Alert).filter(Alert.region_id == region_id, Alert.is_active == True).update({"is_active": False})

        new_alerts: List[Alert] = []
        max_step_risk = 0.0
        max_step_depth = 0.0

        for cell in ts_data["features"]:
            # Never raise alerts over open sea or the creek — only land cells matter
            cell_id0, locality0 = _latlon_to_cell(cell["lat"], cell["lon"])
            if _is_sea_cell(cell["lat"], cell["lon"]) or locality0 in ("Sea", "Thane Creek"):
                continue
            # Evaluate AI Multi-Hazard Risk Model
            risk_calc = risk_model_service.calculate_cell_risk(
                rain_1h=cell["rainfall_1h_mm"],
                rain_3h=cell["rainfall_3h_mm"],
                rain_24h=cell["rainfall_24h_mm"],
                soil_moist_pct=cell["soil_moisture_pct"],
                elevation_m=cell["elevation_m"],
                slope_deg=cell["slope_deg"],
                cape_jkg=cell["cape_instability_jkg"],
                drainage_capacity_mm_hr=region.avg_drainage_capacity_mm_hr if region else 25.0,
                is_tide_locked=ts_data.get("is_high_tide_locked", False),
                tide_height_m=ts_data.get("tide_height_m", 2.5),
                drainage_outfall_dist_m=cell["drainage_outfall_dist_m"],
                is_depression=cell["is_depression_bowl"]
            )

            score = risk_calc["total_risk_score"]
            depth = risk_calc["flood_depth_estimate_cm"]
            if score > max_step_risk:
                max_step_risk = score
            if depth > max_step_depth:
                max_step_depth = depth

            # Raise explainable alert if risk score is MEDIUM, HIGH or CRITICAL (>= 35)
            if score >= 35.0:
                exp = explainability_service.generate_reasoning_and_trust(
                    risk_breakdown=risk_calc,
                    cell_lat=cell["lat"],
                    cell_lon=cell["lon"],
                    rain_1h=cell["rainfall_1h_mm"],
                    soil_moist_pct=cell["soil_moisture_pct"],
                    elevation_m=cell["elevation_m"],
                    cape_jkg=cell["cape_instability_jkg"],
                    is_tide_locked=ts_data.get("is_high_tide_locked", False),
                    tide_height_m=ts_data.get("tide_height_m", 2.5)
                )

                alert_type = "CLOUDBURST" if risk_calc["is_cloudburst"] else ("WATERLOGGING" if cell["elevation_m"] <= 4.0 else "FLASH_FLOOD")
                # Determine alert type from multi-hazard model
                try:
                    ai_engine = get_inference_engine()
                    mh_result = ai_engine.predict_multi_hazard(ts_data)
                    # Find this cell in multi-hazard predictions
                    cell_mh = None
                    for cp in mh_result.get("cell_predictions", []):
                        if cp["cell_index"] == cell.get("cell_index", -1):
                            cell_mh = cp
                            break
                    if cell_mh:
                        dominant = cell_mh["dominant_hazard"]
                        if dominant == "CLOUDBURST":
                            alert_type = "CLOUDBURST"
                        elif dominant == "THUNDERSTORM":
                            alert_type = "SEVERE_THUNDERSTORM"
                        elif dominant == "FLASH_FLOOD":
                            alert_type = "FLASH_FLOOD"
                        else:
                            alert_type = "WATERLOGGING" if cell["elevation_m"] <= 4.0 else "FLASH_FLOOD"
                    else:
                        alert_type = "CLOUDBURST" if risk_calc["is_cloudburst"] else ("WATERLOGGING" if cell["elevation_m"] <= 4.0 else "FLASH_FLOOD")
                except Exception:
                    alert_type = "CLOUDBURST" if risk_calc["is_cloudburst"] else ("WATERLOGGING" if cell["elevation_m"] <= 4.0 else "FLASH_FLOOD")

                cell_id, locality_name = _latlon_to_cell(cell["lat"], cell["lon"])
                cell_id, locality_name = cell_id0, locality0
                alert = Alert(
                    event_id=event.id if event else None,
                    region_id=region_id,
                    cell_lat=cell["lat"],
                    cell_lon=cell["lon"],
                    cell_id=cell_id,
                    locality_name=locality_name,
                    timestep=self.current_timestep,
                    timestamp=ts_datetime,
                    alert_type=alert_type,
                    severity=risk_calc["severity"],
                    risk_score_total=score,
                    rainfall_score_contrib=risk_calc["rainfall_score_contrib"],
                    soil_saturation_score_contrib=risk_calc["soil_saturation_score_contrib"],
                    topography_score_contrib=risk_calc["topography_score_contrib"],
                    atmospheric_instability_score_contrib=risk_calc["atmospheric_instability_score_contrib"],
                    flood_depth_estimate_cm=depth,
                    trust_score=exp["trust_score"],
                    trust_level=exp["trust_level"],
                    trust_factors_json=exp["trust_factors"],
                    reasoning_summary=exp["reasoning_summary"],
                    recommended_action=exp["recommended_action"],
                    is_active=True
                )
                db.add(alert)
                new_alerts.append(alert)
                # Fire webhook for HIGH/CRITICAL alerts (non-blocking background thread)
                dispatch_alert(alert)

        # Update event status
        if event:
            if max_step_risk >= 80.0:
                event.severity_level = SeverityLevel.CRITICAL.value
                event.status = EventStatus.PEAK.value
            elif max_step_risk >= 60.0:
                event.severity_level = SeverityLevel.HIGH.value
                event.status = EventStatus.ACTIVE.value
            elif max_step_risk >= 35.0:
                event.severity_level = SeverityLevel.MEDIUM.value
                event.status = EventStatus.ACTIVE.value
            else:
                event.severity_level = SeverityLevel.LOW.value
                event.status = EventStatus.MONITORING.value

            if ts_data["avg_rainfall_1h_mm"] > event.peak_rainfall_rate_mm_hr:
                event.peak_rainfall_rate_mm_hr = ts_data["avg_rainfall_1h_mm"]
                event.peak_time = ts_datetime
            if max_step_depth > event.max_predicted_depth_cm:
                event.max_predicted_depth_cm = max_step_depth

        db.commit()

        # --- Run full AI/ML inference pipeline (Modules 1-8) ---
        ai_results = {}
        try:
            ai_engine = get_inference_engine()
            ai_results = ai_engine.run_full_inference(
                timestep_data=ts_data,
                all_timesteps=self.data_cache.get("timesteps", []),
                timestep_idx=self.current_timestep - 1,
            )
        except Exception as ai_err:
            logger.warning(f"AI inference pipeline error: {ai_err}. Using heuristic fallback.")

        summary_msg = (
            f"Step {self.current_timestep}/{total_steps} [{ts_data['phase']}]: "
            f"Avg rain {ts_data['avg_rainfall_1h_mm']} mm/hr, Max depth {max_step_depth} cm. "
            f"{len(new_alerts)} active alerts raised."
        )

        return {
            "timestep_id": self.current_timestep,
            "timestamp": ts_data["timestamp"],
            "phase": ts_data["phase"],
            "tide_height_m": ts_data.get("tide_height_m", 2.5),
            "is_high_tide_locked": ts_data.get("is_high_tide_locked", False),
            "avg_rainfall_1h_mm": ts_data.get("avg_rainfall_1h_mm", 0.0),
            "max_rainfall_1h_mm": ts_data.get("max_rainfall_1h_mm", 0.0),
            "new_alerts_count": len(new_alerts),
            "alerts_generated": new_alerts,
            "ai_pipeline_results": ai_results,
            "summary_message": summary_msg
        }

    def reset(self, db: Session) -> Dict[str, Any]:
        """Resets replay simulation back to timestep 1."""
        self.current_timestep = 1
        self.is_running = False
        return self.get_status(db)


replay_engine = ReplaySimulationEngine()
