"""
VARUNA Real-Time Data API
==========================
Serves live weather data for India-wide flood prediction.

Endpoints:
- GET /realtime/india — Full India grid with live data
- GET /realtime/city/{name} — Detailed city-level data
- GET /realtime/mumbai — Mumbai detailed (90 cells)
- GET /realtime/summary — Quick summary for dashboard

Data is refreshed every 5 minutes from free APIs.
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException

router = APIRouter()


@router.get(
    "/india",
    summary="Live weather data for all of India (930 grid cells)",
)
def get_india_realtime():
    """
    Fetch current weather conditions for entire India.

    Uses Open-Meteo API (free, no key) to get:
    - Rainfall (1h, 3h, 6h, 24h accumulation)
    - Temperature, Humidity, Wind
    - CAPE (convective instability)
    - Soil moisture
    - Cloud top temperature estimate

    Grid: 31 rows × 30 cols = 930 cells
    Resolution: 1° × 1° (~110km)
    Coverage: 6°N to 37°N, 68°E to 98°E
    """
    from app.services.realtime.data_ingester import realtime_ingester
    return realtime_ingester.fetch_current_conditions(region="india")


@router.get(
    "/city/{city_name}",
    summary="Detailed weather data for a specific city",
)
def get_city_realtime(city_name: str):
    """
    Fetch detailed weather data for a specific city.

    Automatically zooms into the city region with finer resolution.
    """
    from app.services.realtime.india_grid import STATE_CENTERS

    # Find city center
    city_lower = city_name.lower()
    center = None
    for state, coords in STATE_CENTERS.items():
        if city_lower in state.lower() or state.lower() in city_lower:
            center = coords
            break

    if not center:
        # Default to Mumbai
        center = (19.08, 72.88)

    # Create fine grid around city center (±0.5°)
    from app.services.realtime.data_ingester import realtime_ingester
    return realtime_ingester.fetch_current_conditions(
        region="city",
        lat_min=center[0] - 0.5,
        lat_max=center[0] + 0.5,
        lon_min=center[1] - 0.5,
        lon_max=center[1] + 0.5,
    )


@router.get(
    "/mumbai",
    summary="Detailed Mumbai weather data (90 cells at 0.02° resolution)",
)
def get_mumbai_realtime():
    """
    Fetch detailed Mumbai data with 90-cell grid.

    Resolution: 0.02° × 0.02° (~2km × 2km)
    Same grid as the pilot dataset.
    """
    from app.services.realtime.data_ingester import realtime_ingester
    return realtime_ingester.fetch_mumbai_detailed()


@router.get(
    "/summary",
    summary="Quick India-wide weather summary for dashboard",
)
def get_india_summary():
    """
    Quick summary: top 10 highest-risk areas in India.

    Useful for the dashboard header/alerts.
    """
    from app.services.realtime.data_ingester import realtime_ingester
    data = realtime_ingester.fetch_current_conditions(region="india")

    # Sort by rainfall (proxy for risk)
    cells = data.get("cells", [])
    cells_sorted = sorted(cells, key=lambda c: c.get("rainfall_1h_mm", 0), reverse=True)

    # Top 10 wettest areas
    top_areas = []
    for c in cells_sorted[:10]:
        rain = c.get("rainfall_1h_mm", 0)
        if rain > 50:
            severity = "CRITICAL"
        elif rain > 25:
            severity = "HIGH"
        elif rain > 10:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        top_areas.append({
            "city": c.get("city", "Unknown"),
            "lat": c.get("lat"),
            "lon": c.get("lon"),
            "rainfall_mm_hr": rain,
            "temperature_c": c.get("temperature_celsius", 25),
            "wind_kmh": c.get("wind_speed_10m_kmh", 10),
            "severity": severity,
        })

    return {
        "source": "OPEN_METEO_LIVE",
        "timestamp": data.get("timestamp"),
        "total_cells_monitored": len(cells),
        "areas_with_rain": sum(1 for c in cells if c.get("rainfall_1h_mm", 0) > 1),
        "areas_heavy_rain": sum(1 for c in cells if c.get("rainfall_1h_mm", 0) > 25),
        "areas_extreme_rain": sum(1 for c in cells if c.get("rainfall_1h_mm", 0) > 50),
        "top_areas": top_areas,
    }
