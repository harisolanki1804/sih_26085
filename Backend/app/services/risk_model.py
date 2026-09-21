"""
VARUNA Multi-Hazard Risk Model & DEM Inundation Depth Heuristic
----------------------------------------------------------------
Provides:
1. Additive multi-hazard risk score calculation (0 - 100) decomposed into transparent factors:
   - Rainfall Intensity & Accumulation (0 - 40 pts)
   - Soil Saturation & Antecedent Moisture (0 - 25 pts)
   - Topography & DEM Depression Vulnerability (0 - 20 pts)
   - Atmospheric Convective Instability / CAPE (0 - 15 pts)
2. Simplified DEM-based inundation depth heuristic (in centimeters).
"""

import math
from typing import Dict, Any, Tuple


class RiskModelService:
    @staticmethod
    def calculate_cell_risk(
        rain_1h: float,
        rain_3h: float,
        rain_24h: float,
        soil_moist_pct: float,
        elevation_m: float,
        slope_deg: float,
        cape_jkg: float,
        drainage_capacity_mm_hr: float = 25.0,
        is_tide_locked: bool = False,
        tide_height_m: float = 2.5,
        drainage_outfall_dist_m: float = 3000.0,
        is_depression: bool = False
    ) -> Dict[str, Any]:
        """
        Computes additive explainable risk score and flood depth estimate for a single grid cell.
        """

        # 1. Rainfall Score Contribution (Max 40.0)
        # S-curve scaling so extreme events still show differentiation
        # 50mm/hr = moderate, 100mm/hr = high, 200mm/hr = extreme
        r1h_norm = min(1.0, rain_1h / 200.0)
        score_1h = 25.0 * (1.0 - (1.0 - r1h_norm) ** 1.5)
        # 3h and 24h cumulative loading
        r3h_norm = min(1.0, rain_3h / 500.0)
        r24h_norm = min(1.0, rain_24h / 800.0)
        score_accum = 10.0 * r3h_norm + 5.0 * r24h_norm
        rainfall_score = round(min(40.0, score_1h + score_accum), 2)

        # 2. Soil Saturation Contribution (Max 25.0)
        # S-curve: 30% = low, 60% = moderate, 90% = saturated
        s_norm = min(1.0, max(0.0, soil_moist_pct) / 100.0)
        soil_score = round(min(25.0, 25.0 * (s_norm ** 1.8)), 2)

        # 3. Topography & DEM Depression Contribution (Max 20.0)
        # Low elevations (< 5.0m) and flat slopes (< 1.0 deg) in urban bowls
        elev_penalty = max(0.0, (12.0 - min(12.0, elevation_m)) / 12.0) * 10.0
        slope_penalty = max(0.0, (3.0 - min(3.0, slope_deg)) / 3.0) * 5.0
        depression_penalty = 5.0 if is_depression else 0.0
        topo_score = round(min(20.0, elev_penalty + slope_penalty + depression_penalty), 2)

        # 4. Atmospheric Instability (CAPE) Contribution (Max 15.0)
        # S-curve: 500 = low, 1500 = moderate, 3000 = extreme
        c_norm = min(1.0, max(0.0, cape_jkg) / 3500.0)
        instab_score = round(min(15.0, 15.0 * (c_norm ** 1.3)), 2)

        # Total Additive Risk Score
        total_risk = round(min(100.0, rainfall_score + soil_score + topo_score + instab_score), 1)

        # Severity Classification
        if total_risk >= 80.0:
            severity = "CRITICAL"
        elif total_risk >= 60.0:
            severity = "HIGH"
        elif total_risk >= 35.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # 5. DEM-based Flood Depth Heuristic (cm)
        # Net un-drained rainfall volume
        net_excess_rain = max(0.0, rain_1h - (drainage_capacity_mm_hr * 0.7))
        
        # Depression retention multiplier
        retention_mult = 1.0 + (max(0.0, 7.0 - elevation_m) / 7.0) * (2.0 if is_depression else 1.2)

        # Tidal lock backwater multiplier
        tidal_mult = 1.0
        if is_tide_locked and drainage_outfall_dist_m < 8000.0:
            tidal_mult = 1.0 + (tide_height_m - 4.2) * 0.8 * (1.0 - (drainage_outfall_dist_m / 8000.0))

        # Soil saturation runoff transfer
        sat_trans = 0.6 + 0.4 * (soil_moist_pct / 100.0)

        # Inundation depth in centimeters
        if net_excess_rain > 0.5:
            # 1 mm excess rain = approx 0.1 cm baseline depth
            depth_cm = round((net_excess_rain * 0.15) * retention_mult * tidal_mult * sat_trans + (rain_3h * 0.05), 1)
        else:
            depth_cm = 0.0

        return {
            "total_risk_score": total_risk,
            "severity": severity,
            "rainfall_score_contrib": rainfall_score,
            "soil_saturation_score_contrib": soil_score,
            "topography_score_contrib": topo_score,
            "atmospheric_instability_score_contrib": instab_score,
            "flood_depth_estimate_cm": depth_cm,
            "is_cloudburst": bool(rain_1h >= 65.0 or (rain_1h >= 45.0 and cape_jkg >= 2400)),
            "is_waterlogging": bool(depth_cm >= 15.0)
        }


risk_model_service = RiskModelService()
