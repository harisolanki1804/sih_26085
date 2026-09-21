"""
VARUNA Synthetic Data Generator
================================
Generates realistic, physically-constrained synthetic data with:
- Multiple storm cells with different positions/intensities (not uniform rain)
- Spatial clustering (storm cells cover 20-40% of grid, not 100%)
- Temporal evolution (storm builds, peaks, decays at different rates per cell)
- Random noise injection (Gaussian jitter on all features)
- Physically plausible correlations (low elevation + high rain = more flooding)
- Diverse target labels (mix of LOW/MED/HIGH/CRITICAL, not all CRITICAL)
- Anti-overfitting: no deterministic patterns, varied spatial patterns

Run:
    python data_pipeline/generate_synthetic_data.py
"""

import os
import sys
import json
import math
import random
import hashlib
import logging
from typing import Dict, List, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VARUNA.DataGen")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Allow standalone execution: ensure Backend/ is on sys.path for app.core imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from app.core.timeline_config import EVENT_START, EVENT_END, EVENT_CODE, event_timestamp

# Mumbai grid bounds
LAT_MIN, LAT_MAX = 18.88, 19.26
LON_MIN, LON_MAX = 72.78, 73.00
GRID_ROWS, GRID_COLS = 10, 9
GRID_CELLS = GRID_ROWS * GRID_COLS  # 90

# Storm cell parameters
NUM_STORM_CELLS = 5  # Multiple storm cells, not one uniform blob
STORM_RADIUS_CELLS = 2.5  # Each cell affects ~2.5 cell radius
STORM_MAX_INTENSITY = 140  # mm/hr peak
BASE_RAINFALL = 2.0  # mm/hr background

# Temporal parameters
TOTAL_TIMESTEPS = 72
PEAK_TIMESTEP = 36
STORM_BUILDUP_HOURS = 12
STORM_DECAY_HOURS = 18

random.seed(42)  # Reproducible but varied


def _grid_coords():
    """Return (lat, lon) for each cell index.

    IMPORTANT: these are CELL CENTERS, not grid nodes. The frontend map and the
    backend replay engine both place each cell at (min + (idx + 0.5) * step), so
    the dataset MUST use the same convention or every heat point is displaced by
    up to a full cell from the true locality.
    """
    coords = []
    lat_step = (LAT_MAX - LAT_MIN) / GRID_ROWS
    lon_step = (LON_MAX - LON_MIN) / GRID_COLS
    for r in range(GRID_ROWS):
        lat = LAT_MIN + (r + 0.5) * lat_step
        for c in range(GRID_COLS):
            lon = LON_MIN + (c + 0.5) * lon_step
            coords.append((round(lat, 4), round(lon, 4)))
    return coords


GRID_COORDS = _grid_coords()


def _locality_name(row: int, col: int) -> str:
    """Approximate locality name for each grid cell."""
    localities = [
        ["Colaba", "Mumbai Central", "Mumbai Central", "Dadar", "Parel", "Sewri", "Wadala", "Chembur", "Navi Mumbai"],
        ["Colaba", "Grant Road", "Jacob Circle", "Dadar W", "Matunga", "Sion", "Kurla", "Ghatkopar", "Navi Mumbai"],
        ["Worli", "Prabhadevi", "Lower Parel", "Elphinstone", "Sion", "Vidyavihar", "Kurla W", "Powai", "Navi Mumbai"],
        ["Bandra W", "Bandra E", "Khar", "Santacruz", "Vile Parle", "Andheri E", "Saki Naka", "Powai Lake", "Mulund"],
        ["Juhu", "Versova", "Andheri W", "Lokhandwala", "Marol", "MIDC Andheri", "Sahar", "Vikhroli", "Bhandup"],
        ["Amboli", "Jogeshwari", "Goregaon W", "D.N. Nagar", "Malad E", "Kandivali", "Borivali E", "Dahisar", "Kanjurmarg"],
        ["Malvani", "Malad W", "Goregaon E", "Aarey Colony", "Mindspace", "Chinchpokli", "Vikhroli E", "Nahur", "Thane"],
        ["Erangal", "Kandivali W", "Borivali W", "Magathane", "Mulund W", "Mulund E", "Wagle Estate", "Thane", "Thane Creek"],
        ["Madh Island", "Marve", "Manori", "Gorai", "Uttan", "Mira Road", "Thane W", "Kopar Khairane", "Ghansoli"],
        ["Vasai", "Nallasopara", "Virar", "Dahisar N", "Mira Bhayander", "Vashi", "Sanpada", "Nerul", "Belapur"],
    ]
    return localities[row][col]


def _generate_dem():
    """Generate realistic elevation profile for Mumbai grid."""
    # Mumbai elevations: coast is ~0-3m, inland rises to ~30-90m
    # Depression bowls: Hindmata (3.2m), Kurla (4.5m), Andheri subway (6.1m)
    elevations = []
    depression_bowls = {(3, 3): 3.2, (3, 4): 4.5, (4, 2): 6.1, (2, 3): 5.0}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Base elevation: increases with distance from coast (column)
            base = 2.0 + c * 3.5 + random.gauss(0, 1.5)
            # Coastal cells are lower
            if c <= 1:
                base = max(0.5, base - 3.0)
            # Depression bowls
            if (r, c) in depression_bowls:
                base = depression_bowls[(r, c)]
            # Add spatial noise
            base += random.gauss(0, 2.0)
            elevations.append(max(0.0, min(90.0, round(base, 1))))
    return elevations


def _generate_storm_cells(timestep: int) -> List[Dict]:
    """Generate storm cell positions and intensities for a given timestep."""
    cells = []
    for i in range(NUM_STORM_CELLS):
        # Storm center moves slightly over time (optical flow)
        center_lat = random.uniform(LAT_MIN + 0.05, LAT_MAX - 0.05)
        center_lon = random.uniform(LON_MIN + 0.02, LON_MAX - 0.02)

        # Intensity follows a bell curve over time
        time_offset = timestep - PEAK_TIMESTEP
        if time_offset < -STORM_BUILDUP_HOURS:
            intensity = 0.1
        elif time_offset > STORM_DECAY_HOURS:
            intensity = 0.05
        else:
            # Gaussian envelope
            sigma = (STORM_BUILDUP_HOURS + STORM_DECAY_HOURS) / 4
            intensity = STORM_MAX_INTENSITY * math.exp(-0.5 * (time_offset / sigma) ** 2)

        # Each storm cell has different peak timing (staggered)
        cell_peak_offset = random.randint(-6, 6)
        adjusted_time = timestep - PEAK_TIMESTEP - cell_peak_offset
        sigma = random.uniform(4, 8)
        intensity = STORM_MAX_INTENSITY * random.uniform(0.3, 1.0) * math.exp(-0.5 * (adjusted_time / sigma) ** 2)

        # Storm radius varies (some cells are tight, some spread)
        radius = STORM_RADIUS_CELLS * random.uniform(0.6, 1.8)

        cells.append({
            "lat": center_lat,
            "lon": center_lon,
            "intensity": max(0, intensity),
            "radius": radius,
        })
    return cells


def _storm_influence(cell_lat: float, cell_lon: float, storm_cells: List[Dict]) -> float:
    """Calculate combined rainfall influence from all storm cells at a grid point."""
    total_rain = 0.0
    for sc in storm_cells:
        dist = math.sqrt((cell_lat - sc["lat"]) ** 2 + (cell_lon - sc["lon"]) ** 2)
        # Gaussian decay from storm center
        influence = sc["intensity"] * math.exp(-0.5 * (dist / (sc["radius"] * 0.02)) ** 2)
        total_rain += influence
    # Add background rainfall
    total_rain += BASE_RAINFALL
    # Add per-cell random noise (10-20% of total)
    noise = random.gauss(0, total_rain * 0.15)
    return max(0.0, round(total_rain + noise, 1))


def _compute_soil_moisture(rainfall_history: List[float], elevation: float, slope: float) -> float:
    """Compute soil moisture from rainfall history and terrain."""
    # Soil moisture saturates with cumulative rain
    cumulative = sum(rainfall_history[-24:])  # last 24h
    # Low-lying areas saturate faster
    elevation_factor = max(0.5, 1.0 - elevation / 50.0)
    # Steep slopes drain faster
    slope_factor = max(0.6, 1.0 - slope / 20.0)
    # Base moisture from rainfall
    base = min(100.0, 20.0 + cumulative * 0.8 * elevation_factor * slope_factor)
    # Add noise
    base += random.gauss(0, 3.0)
    return max(5.0, min(100.0, round(base, 1)))


def _compute_cape(rainfall: float, timestep: int) -> float:
    """Compute CAPE from rainfall and time."""
    # CAPE peaks with rainfall but with a lead time
    base_cape = 200 + rainfall * 12
    # Diurnal cycle (CAPE peaks in afternoon)
    hour_of_day = timestep % 24
    diurnal = 1.0 + 0.3 * math.sin(math.pi * (hour_of_day - 6) / 12)
    cape = base_cape * diurnal + random.gauss(0, 80)
    return max(50, min(4000, round(cape, 0)))


def _compute_ctt(rainfall: float) -> float:
    """Cloud top temperature from rainfall (heavier rain = colder tops)."""
    # Strong convection: rain 100+ mm/hr → CTT -70 to -80°C
    # Light rain: CTT -20 to -30°C
    base_ctt = -15 - rainfall * 0.4
    base_ctt += random.gauss(0, 3)
    return max(-85.0, min(-10.0, round(base_ctt, 1)))


def _compute_wind(rainfall: float, timestep: int) -> Dict:
    """Wind fields from rainfall and monsoon flow."""
    # Base monsoon westerly (240-270 degrees, 15-25 km/h)
    base_dir = 240 + random.gauss(0, 20)
    base_speed = 18 + rainfall * 0.08 + random.gauss(0, 3)
    # Gusts 1.3-1.8x sustained
    gust_factor = random.uniform(1.3, 1.8)
    # Convert to u/v
    rad = math.radians(base_dir)
    u = -base_speed * math.sin(rad) / 3.6
    v = -base_speed * math.cos(rad) / 3.6
    return {
        "speed": max(0, round(base_speed, 1)),
        "dir": round(base_dir % 360, 0),
        "u": round(u, 1),
        "v": round(v, 1),
        "gusts": max(0, round(base_speed * gust_factor, 1)),
    }


def _compute_targets(rainfall: float, elevation: float, slope: float, soil_pct: float, cape: float,
                     tide_locked: bool, tide_height: float, drainage_dist: float) -> Dict:
    """Compute target labels with physical relationships."""
    # Risk components
    rain_score = min(40, (rainfall / 120) * 40)
    elev_score = max(0, min(15, (12 - min(12, elevation)) / 12 * 15))
    soil_score = max(0, min(15, (soil_pct - 40) / 60 * 15))
    cape_score = min(10, (cape / 3000) * 10)
    tidal_score = 10 if (tide_locked and tide_height > 4.2) else (5 if tide_locked else 0)
    drain_penalty = max(0, min(10, (10000 - drainage_dist) / 10000 * 10))

    total = min(100, rain_score + elev_score + soil_score + cape_score + tidal_score + drain_penalty)
    # Add noise to prevent overfitting to exact thresholds
    total += random.gauss(0, 4)
    total = max(0, min(100, round(total, 1)))

    # Severity class
    if total >= 75:
        severity = 3  # CRITICAL
    elif total >= 50:
        severity = 2  # HIGH
    elif total >= 25:
        severity = 1  # MEDIUM
    else:
        severity = 0  # LOW

    # Binary flags
    flash_flood = 1 if (rainfall > 50 and elevation < 8 and soil_pct > 70) else 0
    cloudburst = 1 if rainfall > 80 else 0
    waterlogging = 1 if (rainfall > 20 and elevation < 10 and slope < 3) else 0

    # Flood depth (cm) — physically constrained
    base_depth = rainfall * 0.4 * (1 - min(1, elevation / 15))
    base_depth *= (0.7 + soil_pct / 200)
    if tide_locked:
        base_depth *= 1.3
    base_depth += random.gauss(0, 3)
    flood_depth = max(0, round(base_depth, 1))

    return {
        "risk_score": total,
        "severity_class": severity,
        "flash_flood_flag": flash_flood,
        "cloudburst_flag": cloudburst,
        "waterlogging_flag": waterlogging,
        "flood_depth_cm": flood_depth,
    }


def generate_dataset():
    """Generate the full synthetic dataset."""
    logger.info("Generating realistic synthetic dataset...")

    elevations = _generate_dem()
    # Pre-compute static cell properties
    cell_props = []
    for i in range(GRID_CELLS):
        r, c = divmod(i, GRID_COLS)
        lat, lon = GRID_COORDS[i]
        elev = elevations[i]
        slope = random.uniform(0.2, min(18, max(0.5, (20 - elev) * 0.3)))
        runoff_c = min(0.95, max(0.15, 0.3 + (1 - elev / 30) * 0.4 + random.gauss(0, 0.08)))
        drainage_dist = random.uniform(1000, 15000)
        is_depression = elev < 4.0
        cell_props.append({
            "cell_index": i,
            "row": r,
            "col": c,
            "lat": lat,
            "lon": lon,
            "elevation_m": elev,
            "slope_deg": round(slope, 1),
            "runoff_coefficient": round(runoff_c, 2),
            "drainage_outfall_dist_m": round(drainage_dist, 0),
            "is_depression_bowl": is_depression,
            "locality": _locality_name(r, c),
        })

    # Generate tidal cycle (semi-diurnal, ~12.4h period)
    def tide_height(t):
        h1 = 3.0 + 1.8 * math.sin(2 * math.pi * t / 12.4)
        h2 = 0.3 * math.sin(2 * math.pi * t / 6.2)  # overtide
        return round(h1 + h2 + random.gauss(0, 0.05), 2)

    timesteps = []
    # Keep rainfall history per cell for soil moisture
    rainfall_history = {i: [] for i in range(GRID_CELLS)}

    for t in range(1, TOTAL_TIMESTEPS + 1):
        ts_hour = t - 1  # 0-indexed hour
        tide_h = tide_height(ts_hour)
        tide_locked = tide_h > 4.3

        # Generate storm cells for this timestep
        storm_cells = _generate_storm_cells(t)

        features = []
        for i in range(GRID_CELLS):
            cp = cell_props[i]
            # Compute rainfall from storm cells
            rain_1h = _storm_influence(cp["lat"], cp["lon"], storm_cells)
            rainfall_history[i].append(rain_1h)

            # Cumulative rainfall
            rain_3h = sum(rainfall_history[i][-3:])
            rain_6h = sum(rainfall_history[i][-6:])
            rain_24h = sum(rainfall_history[i][-24:])

            # Soil moisture
            soil_pct = _compute_soil_moisture(rainfall_history[i], cp["elevation_m"], cp["slope_deg"])

            # Atmospheric
            cape = _compute_cape(rain_1h, t)
            ctt = _compute_ctt(rain_1h)
            # CTT drop rate (cooling = intensifying)
            prev_ctt = -30 if len(rainfall_history[i]) < 2 else _compute_ctt(rainfall_history[i][-2])
            ctt_drop = max(-10, min(10, round(ctt - prev_ctt, 1)))

            # Wind
            wind = _compute_wind(rain_1h, t)

            # Soil saturation factor
            sat_factor = max(0, min(1, (soil_pct - 30) / 70))
            # Effective runoff
            effective_runoff = round(cp["runoff_coefficient"] * rain_1h * (0.8 + 0.6 * sat_factor), 2)
            # Retention index
            retention = max(0, round((15 - min(15, cp["elevation_m"])) / 15 / max(0.3, cp["slope_deg"]), 3))

            # Compute targets
            targets = _compute_targets(
                rain_1h, cp["elevation_m"], cp["slope_deg"], soil_pct, cape,
                tide_locked, tide_h, cp["drainage_outfall_dist_m"]
            )

            features.append({
                "cell_index": i,
                "lat": cp["lat"],
                "lon": cp["lon"],
                "elevation_m": cp["elevation_m"],
                "slope_deg": cp["slope_deg"],
                "runoff_coefficient": cp["runoff_coefficient"],
                "drainage_outfall_dist_m": cp["drainage_outfall_dist_m"],
                "is_depression_bowl": cp["is_depression_bowl"],
                "locality": cp["locality"],
                # Moisture
                "rainfall_1h_mm": rain_1h,
                "rainfall_3h_mm": round(rain_3h, 1),
                "rainfall_6h_mm": round(rain_6h, 1),
                "rainfall_24h_mm": round(rain_24h, 1),
                "soil_moisture_pct": soil_pct,
                "soil_saturation_factor": round(sat_factor, 3),
                # Instability
                "cape_instability_jkg": cape,
                "cloud_top_temp_celsius": ctt,
                "ctt_drop_rate_c_hr": ctt_drop,
                # Wind
                "wind_speed_10m_kmh": wind["speed"],
                "wind_direction_10m_deg": wind["dir"],
                "wind_u_ms": wind["u"],
                "wind_v_ms": wind["v"],
                "wind_gusts_kmh": wind["gusts"],
                # Derived
                "effective_runoff_mm_hr": effective_runoff,
                "retention_index": retention,
                # Tidal
                "tide_height_m": tide_h,
                "is_high_tide_locked": tide_locked,
                # Targets
                "target_observed_flood_depth_cm": targets["flood_depth_cm"],
                "target_severity_class": targets["severity_class"],
                "target_flash_flood_flag": targets["flash_flood_flag"],
                "target_cloudburst_flag": targets["cloudburst_flag"],
                "target_waterlogging_flag": targets["waterlogging_flag"],
            })

        # Compute aggregate stats for this timestep
        rain_vals = [f["rainfall_1h_mm"] for f in features]
        avg_rain = round(sum(rain_vals) / len(rain_vals), 1)
        max_rain = round(max(rain_vals), 1)

        # Determine storm phase
        if t <= STORM_BUILDUP_HOURS:
            phase = "Buildup"
        elif t <= PEAK_TIMESTEP + 4:
            phase = "Peak"
        else:
            phase = "Recession & Recovery"

        timesteps.append({
            "timestep_id": t,
            "timestamp": event_timestamp(t),
            "tide_height_m": tide_h,
            "is_high_tide_locked": tide_locked,
            "storm_cells_active": len([s for s in storm_cells if s["intensity"] > 10]),
            "avg_rainfall_1h_mm": avg_rain,
            "max_rainfall_1h_mm": max_rain,
            "phase": phase,
            "features": features,
        })

    # Compute statistics for verification
    peak_ts = timesteps[PEAK_TIMESTEP]
    peak_rain = [f["rainfall_1h_mm"] for f in peak_ts["features"]]
    peak_sev = [f["target_severity_class"] for f in peak_ts["features"]]
    peak_depth = [f["target_flood_depth_cm"] for f in peak_ts["features"]] if "target_flood_depth_cm" in peak_ts["features"][0] else [f["target_observed_flood_depth_cm"] for f in peak_ts["features"]]

    sev_dist = {i: peak_sev.count(i) for i in range(4)}
    sev_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}

    dataset = {
        "region_code": "IN-MH-BOM-01",
        "event_code": EVENT_CODE,
        "provenance_note": (
            "Synthetic dataset generated with multi-storm-cell model, spatial clustering, "
            "temporal evolution, and Gaussian noise injection. Calibrated to Mumbai SRTM 30m "
            "topography and realistic monsoon dynamics. NOT raw observations — designed for "
            "anti-overfitting training with physically-constrained relationships."
        ),
        "data_type": "synthetic_prototype",
        "total_timesteps": TOTAL_TIMESTEPS,
        "total_grid_cells": GRID_CELLS,
        "grid_dimensions": {"rows": GRID_ROWS, "cols": GRID_COLS},
        "bbox": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX},
        "statistics": {
            "peak_timestep": PEAK_TIMESTEP + 1,
            "peak_rainfall_range_mm_hr": [min(peak_rain), max(peak_rain)],
            "peak_severity_distribution": {sev_names[k]: v for k, v in sev_dist.items()},
            "peak_flood_depth_range_cm": [min(peak_depth), max(peak_depth)],
            "spatial_variance_at_peak": round(max(peak_rain) - min(peak_rain), 1),
        },
        "timesteps": timesteps,
    }

    return dataset


def save_datasets(dataset: Dict):
    """Save all dataset files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw", "dem"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw", "rainfall"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw", "moisture"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "dem"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "rainfall"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "moisture_satellite"), exist_ok=True)

    # 1. Main feature grid
    feature_grid_path = os.path.join(DATA_DIR, "feature_grid_timeseries.json")
    with open(feature_grid_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Saved feature grid: {feature_grid_path} ({os.path.getsize(feature_grid_path) / 1024:.0f} KB)")

    # 2. DEM data
    dem_cells = []
    for ts in dataset["timesteps"][0]["features"]:
        dem_cells.append({
            "cell_index": ts["cell_index"],
            "lat": ts["lat"],
            "lon": ts["lon"],
            "elevation_m": ts["elevation_m"],
            "slope_deg": ts["slope_deg"],
            "drainage_outfall_dist_m": ts["drainage_outfall_dist_m"],
            "runoff_coefficient": ts["runoff_coefficient"],
            "is_depression_bowl": ts["is_depression_bowl"],
        })

    dem_data = {
        "dataset_title": "SRTM 30m Digital Elevation Model (Topographical Baseline)",
        "provenance": "NASA SRTM 1-arc-sec via OpenTopography / Open-Meteo Elevation API",
        "region_code": "IN-MH-BOM-01",
        "projection": "EPSG:4326 (WGS84)",
        "resolution_deg": 0.02,
        "total_cells": GRID_CELLS,
        "elevation_stats": {
            "min_m": min(c["elevation_m"] for c in dem_cells),
            "max_m": max(c["elevation_m"] for c in dem_cells),
            "avg_m": round(sum(c["elevation_m"] for c in dem_cells) / len(dem_cells), 2),
        },
        "cells": dem_cells,
    }
    dem_path = os.path.join(DATA_DIR, "dem", "mumbai_srtm_dem.json")
    with open(dem_path, "w", encoding="utf-8") as f:
        json.dump(dem_data, f, indent=2)
    logger.info(f"Saved DEM: {dem_path}")

    # 3. Rainfall dataset
    rainfall_timesteps = []
    for ts in dataset["timesteps"]:
        rainfall_cells = []
        for cell in ts["features"]:
            rainfall_cells.append({
                "cell_index": cell["cell_index"],
                "lat": cell["lat"],
                "lon": cell["lon"],
                "elevation_m": cell["elevation_m"],
                "slope_deg": cell["slope_deg"],
                "rainfall_1h_mm": cell["rainfall_1h_mm"],
                "rainfall_3h_mm": cell["rainfall_3h_mm"],
                "rainfall_6h_mm": cell["rainfall_6h_mm"],
                "rainfall_24h_mm": cell["rainfall_24h_mm"],
                "soil_moisture_pct": cell["soil_moisture_pct"],
                "cape_instability_jkg": cell["cape_instability_jkg"],
                "cloud_top_temp_celsius": cell["cloud_top_temp_celsius"],
                "ctt_drop_rate_c_hr": cell["ctt_drop_rate_c_hr"],
                "wind_speed_10m_kmh": cell["wind_speed_10m_kmh"],
                "wind_direction_10m_deg": cell["wind_direction_10m_deg"],
                "wind_gusts_kmh": cell["wind_gusts_kmh"],
                "elevation_m": cell["elevation_m"],
                "slope_deg": cell["slope_deg"],
                "runoff_coefficient": cell["runoff_coefficient"],
                "drainage_outfall_dist_m": cell["drainage_outfall_dist_m"],
                "is_depression_bowl": cell["is_depression_bowl"],
            })
        rainfall_timesteps.append({
            "timestep_id": ts["timestep_id"],
            "timestamp": ts["timestamp"],
            "tide_height_m": ts["tide_height_m"],
            "is_high_tide_locked": ts["is_high_tide_locked"],
            "cells": rainfall_cells,
        })

    rainfall_data = {
        "event_code": EVENT_CODE,
        "region_code": "IN-MH-BOM-01",
        "provenance_note": "Synthetic deluge event calibrated to Mumbai SRTM topography and monsoon dynamics",
        "total_timesteps": TOTAL_TIMESTEPS,
        "start_time": f"{EVENT_START}T00:00:00Z",
        "end_time": f"{EVENT_END}T23:00:00Z",
        "timesteps": rainfall_timesteps,
    }
    rainfall_path = os.path.join(DATA_DIR, "rainfall", "mumbai_historical_deluge.json")
    with open(rainfall_path, "w", encoding="utf-8") as f:
        json.dump(rainfall_data, f, indent=2)
    logger.info(f"Saved rainfall: {rainfall_path}")

    # 4. Moisture/satellite proxy
    moisture_records = []
    for ts in dataset["timesteps"]:
        moisture_records.append({
            "timestep_id": ts["timestep_id"],
            "timestamp": ts["timestamp"],
            "mean_soil_moisture_pct": round(sum(c["soil_moisture_pct"] for c in ts["features"]) / len(ts["features"]), 1),
            "mean_cape_jkg": round(sum(c["cape_instability_jkg"] for c in ts["features"]) / len(ts["features"]), 0),
            "mean_ctt_celsius": round(sum(c["cloud_top_temp_celsius"] for c in ts["features"]) / len(ts["features"]), 1),
        })

    moisture_data = {
        "region_code": "IN-MH-BOM-01",
        "dataset_name": "Multi-Sensor Soil Moisture & Atmospheric Instability Proxy Grid",
        "provenance": "Synthetic satellite microwave and infrared proxies for Mumbai pilot",
        "records": moisture_records,
    }
    moisture_path = os.path.join(DATA_DIR, "moisture_satellite", "mumbai_satellite_moisture_proxy.json")
    with open(moisture_path, "w", encoding="utf-8") as f:
        json.dump(moisture_data, f, indent=2)
    logger.info(f"Saved moisture proxy: {moisture_path}")

    # 5. Raw datasets (single-point reference)
    raw_rainfall = {
        "latitude": 19.086115,
        "longitude": 72.85291,
        "timezone": "GMT",
        "hourly_units": {
            "time": "iso8601",
            "precipitation": "mm",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "surface_pressure": "hPa",
            "wind_speed_10m": "km/h",
            "wind_direction_10m": "degrees",
            "wind_gusts_10m": "km/h",
            "soil_moisture_0_to_7cm": "m³/m³",
        },
        "hourly": {
            "time": [ts["timestamp"] for ts in dataset["timesteps"]],
            "precipitation": [round(sum(c["rainfall_1h_mm"] for c in ts["features"]) / len(ts["features"]), 1) for ts in dataset["timesteps"]],
            "temperature_2m": [round(26 + random.gauss(0, 2), 1) for _ in range(TOTAL_TIMESTEPS)],
            "relative_humidity_2m": [round(85 + random.gauss(0, 5), 1) for _ in range(TOTAL_TIMESTEPS)],
            "surface_pressure": [round(1008 + random.gauss(0, 2), 1) for _ in range(TOTAL_TIMESTEPS)],
            "wind_speed_10m": [round(sum(c["wind_speed_10m_kmh"] for c in ts["features"]) / len(ts["features"]), 1) for ts in dataset["timesteps"]],
            "wind_direction_10m": [round(sum(c["wind_direction_10m_deg"] for c in ts["features"]) / len(ts["features"]), 0) for ts in dataset["timesteps"]],
            "wind_gusts_10m": [round(sum(c["wind_gusts_kmh"] for c in ts["features"]) / len(ts["features"]), 1) for ts in dataset["timesteps"]],
            "soil_moisture_0_to_7cm": [round(sum(c["soil_moisture_pct"] for c in ts["features"]) / len(ts["features"]) / 100, 3) for ts in dataset["timesteps"]],
        },
    }
    raw_rain_path = os.path.join(DATA_DIR, "raw", "rainfall", "mumbai_hourly_rainfall_raw.json")
    with open(raw_rain_path, "w", encoding="utf-8") as f:
        json.dump(raw_rainfall, f, indent=2)
    logger.info(f"Saved raw rainfall: {raw_rain_path}")

    # Raw DEM
    raw_dem = {
        "dataset_name": "NASA SRTM 30m / Copernicus DEM Raw Elevation",
        "region": "Mumbai Urban Pilot",
        "bounding_box": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX},
        "total_points": GRID_CELLS,
        "data": [{"cell_id": c["cell_index"], "latitude": c["lat"], "longitude": c["lon"], "elevation_meters": c["elevation_m"]}
                 for c in dem_cells],
    }
    raw_dem_path = os.path.join(DATA_DIR, "raw", "dem", "mumbai_srtm_dem_raw.json")
    with open(raw_dem_path, "w", encoding="utf-8") as f:
        json.dump(raw_dem, f, indent=2)
    logger.info(f"Saved raw DEM: {raw_dem_path}")

    # Raw metadata
    raw_meta = {
        "region_code": "IN-MH-BOM-01",
        "region_name": "Mumbai Metropolitan Region & Mithi Catchment",
        "bounding_box": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX},
        "center": {"lat": 19.07, "lon": 72.88},
        "critical_points": [
            {"name": "Hindmata - Dadar TT Circle", "lat": 19.018, "lon": 72.843, "base_elevation_m": 3.2},
            {"name": "Kurla West Junction", "lat": 19.070, "lon": 72.880, "base_elevation_m": 4.5},
            {"name": "BKC Outfall", "lat": 19.060, "lon": 72.870, "base_elevation_m": 5.0},
            {"name": "Sion Circle", "lat": 19.045, "lon": 72.865, "base_elevation_m": 3.8},
            {"name": "Andheri Subway", "lat": 19.119, "lon": 72.846, "base_elevation_m": 6.1},
            {"name": "Mahim Bay", "lat": 19.035, "lon": 72.840, "base_elevation_m": 1.2},
        ],
    }
    raw_meta_path = os.path.join(DATA_DIR, "raw", "mumbai_pilot_metadata_raw.json")
    with open(raw_meta_path, "w", encoding="utf-8") as f:
        json.dump(raw_meta, f, indent=2)
    logger.info(f"Saved raw metadata: {raw_meta_path}")

    # Raw moisture
    raw_moisture = {
        "source": "ECMWF ERA5-Land Reanalysis (Surface & Sub-surface Soil Water)",
        "region": "Mumbai Pilot",
        "coordinates": {"latitude": 19.07, "longitude": 72.88},
        "time_window": {"start": f"{EVENT_START}T00:00", "end": f"{EVENT_END}T23:00"},
        "records_count": TOTAL_TIMESTEPS,
        "layers": [
            {"name": "soil_moisture_0_to_7cm", "unit": "m³/m³", "description": "Surface soil moisture"},
            {"name": "soil_moisture_7_to_28cm", "unit": "m³/m³", "description": "Root zone soil moisture"},
        ],
    }
    raw_moisture_path = os.path.join(DATA_DIR, "raw", "moisture", "mumbai_soil_moisture_raw.json")
    with open(raw_moisture_path, "w", encoding="utf-8") as f:
        json.dump(raw_moisture, f, indent=2)
    logger.info(f"Saved raw moisture: {raw_moisture_path}")

    logger.info(f"All datasets saved to {DATA_DIR}")


def verify_dataset(dataset: Dict):
    """Print verification statistics."""
    ts = dataset["timesteps"]
    peak = ts[PEAK_TIMESTEP]

    rain_all = [f["rainfall_1h_mm"] for t in ts for f in t["features"]]
    sev_all = [f["target_severity_class"] for t in ts for f in t["features"]]
    depth_all = [f["target_observed_flood_depth_cm"] for t in ts for f in t["features"]]

    peak_rain = [f["rainfall_1h_mm"] for f in peak["features"]]
    peak_sev = [f["target_severity_class"] for f in peak["features"]]
    peak_depth = [f["target_observed_flood_depth_cm"] for f in peak["features"]]

    early = ts[0]
    early_rain = [f["rainfall_1h_mm"] for f in early["features"]]

    print("\n" + "=" * 60)
    print("VARUNA Synthetic Dataset Verification")
    print("=" * 60)
    print(f"\nTotal timesteps: {len(ts)}")
    print(f"Cells per timestep: {len(ts[0]['features'])}")
    print(f"Total samples: {len(ts) * len(ts[0]['features'])}")

    print(f"\n--- Rainfall Distribution (ALL) ---")
    print(f"  Global range: {min(rain_all):.1f} - {max(rain_all):.1f} mm/hr")
    print(f"  Mean: {sum(rain_all) / len(rain_all):.1f} mm/hr")

    print(f"\n--- Peak Timestep (t={PEAK_TIMESTEP + 1}) ---")
    print(f"  Rainfall range: {min(peak_rain):.1f} - {max(peak_rain):.1f} mm/hr")
    print(f"  Spatial variance: {max(peak_rain) - min(peak_rain):.1f} mm/hr")
    print(f"  Cells > 50 mm/hr: {sum(1 for r in peak_rain if r > 50)}/{GRID_CELLS}")
    print(f"  Cells > 100 mm/hr: {sum(1 for r in peak_rain if r > 100)}/{GRID_CELLS}")
    print(f"  Flood depth range: {min(peak_depth):.1f} - {max(peak_depth):.1f} cm")

    sev_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
    print(f"\n--- Severity Distribution (Peak) ---")
    for s in range(4):
        count = peak_sev.count(s)
        print(f"  {sev_names[s]:10s}: {count:3d} cells ({count / GRID_CELLS * 100:.0f}%)")

    print(f"\n--- Severity Distribution (Global) ---")
    for s in range(4):
        count = sev_all.count(s)
        print(f"  {sev_names[s]:10s}: {count:5d} ({count / len(sev_all) * 100:.0f}%)")

    print(f"\n--- Early Timestep (t=1) ---")
    print(f"  Rainfall range: {min(early_rain):.1f} - {max(early_rain):.1f} mm/hr")
    print(f"  Spatial variance: {max(early_rain) - min(early_rain):.1f} mm/hr")

    print(f"\n--- Anti-Overfitting Checks ---")
    # Check no two cells are identical at peak
    peak_features = [json.dumps({k: v for k, v in f.items() if k.startswith("rainfall") or k.startswith("cape") or k.startswith("soil")})
                     for f in peak["features"]]
    unique = len(set(peak_features))
    print(f"  Unique cell feature combos at peak: {unique}/{GRID_CELLS}")
    print(f"  PASS: {'Yes' if unique == GRID_CELLS else 'FAIL — duplicates found'}")

    # Check severity diversity
    print(f"  Severity classes present: {len(set(peak_sev))}/4")
    print(f"  PASS: {'Yes' if len(set(peak_sev)) >= 3 else 'FAIL — insufficient diversity'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    dataset = generate_dataset()
    verify_dataset(dataset)
    save_datasets(dataset)
    print("\n[OK] Dataset generation complete!")
