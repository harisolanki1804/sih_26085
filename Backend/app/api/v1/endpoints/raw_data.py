import os
import json
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from app.core.config import settings
from app.core.timeline_config import EVENT_START, EVENT_END

router = APIRouter()

RAW_DIR = os.path.join(settings.BASE_DIR, "data", "raw")


@router.get("/dem", summary="Get Raw Unprocessed DEM Topography Points")
def get_raw_dem() -> Dict[str, Any]:
    """Returns raw SRTM elevation points (cell_id, latitude, longitude, elevation_meters) without any derived slopes or indices."""
    path = os.path.join(RAW_DIR, "dem", "mumbai_srtm_dem_raw.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw DEM dataset not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/rainfall", summary="Get Raw Unprocessed Historical Hourly Rainfall & Weather")
def get_raw_rainfall() -> Dict[str, Any]:
    """Returns raw ECMWF ERA5-Land hourly precipitation, temperature, humidity, and soil moisture."""
    path = os.path.join(RAW_DIR, "rainfall", "mumbai_hourly_rainfall_raw.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw rainfall dataset not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


import urllib.request

@router.get("/catalog", summary="Get Raw Datasets File Catalog for Model Team")
def get_raw_catalog() -> Dict[str, Any]:
    """Lists raw dataset file paths available for pandas or numpy ingestion."""
    return {
        "status": "RAW_UNPROCESSED_READY",
        "description": "Untouched raw open datasets for feature engineering and prediction model team.",
        "files": {
            "dem_elevation_csv": os.path.join(RAW_DIR, "dem", "mumbai_srtm_dem_raw.csv"),
            "dem_elevation_json": os.path.join(RAW_DIR, "dem", "mumbai_srtm_dem_raw.json"),
            "rainfall_weather_csv": os.path.join(RAW_DIR, "rainfall", "mumbai_hourly_rainfall_raw.csv"),
            "rainfall_weather_json": os.path.join(RAW_DIR, "rainfall", "mumbai_hourly_rainfall_raw.json"),
            "pilot_metadata_json": os.path.join(RAW_DIR, "mumbai_pilot_metadata_raw.json")
        },
        "python_snippet": "import pandas as pd\ndf_dem = pd.read_csv(r'VARUNA/data/raw/dem/mumbai_srtm_dem_raw.csv')\ndf_rain = pd.read_csv(r'VARUNA/data/raw/rainfall/mumbai_hourly_rainfall_raw.csv')"
    }


@router.get("/fetch-dynamic", summary="Dynamically Fetch Historical Meteorological Data on Demand")
def fetch_dynamic_raw_data(
    start_date: str = EVENT_START,
    end_date: str = EVENT_END,
    latitude: float = 19.07,
    longitude: float = 72.88
) -> Dict[str, Any]:
    """
    Dynamically fetches hourly ECMWF ERA5-Land reanalysis for any custom date range and coordinates.
    - **start_date**: Start date (YYYY-MM-DD), e.g. 2024-06-01
    - **end_date**: End date (YYYY-MM-DD), e.g. 2024-09-30
    - **latitude**: Target latitude (default: 19.07 for Mumbai)
    - **longitude**: Target longitude (default: 72.88 for Mumbai)
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&"
        f"hourly=precipitation,temperature_2m,relative_humidity_2m,surface_pressure,"
        f"wind_speed_10m,wind_direction_10m,wind_gusts_10m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VARUNA-FastAPI/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        return {
            "status": "SUCCESS",
            "source": "ECMWF ERA5-Land via Open-Meteo Archive API",
            "query": {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "total_timesteps_hours": len(times)
            },
            "hourly_data": hourly
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch dynamic meteorological data from upstream provider: {str(e)}"
        )
