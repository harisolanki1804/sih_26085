import os
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="System Health & Diagnostic Status")
def get_health(db: Session = Depends(get_db)):
    """Verifies API status, database connectivity, and data availability."""
    db_status = "HEALTHY"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"ERROR: {str(e)}"

    dem_available = os.path.exists(os.path.join(settings.DATA_DIR, "dem", "mumbai_srtm_dem.json"))
    rainfall_available = os.path.exists(os.path.join(settings.DATA_DIR, "rainfall", "mumbai_historical_deluge.json"))
    feature_matrix_available = os.path.exists(os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json"))

    return {
        "status": "ONLINE",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": db_status,
        "database_url_type": "sqlite" if settings.DATABASE_URL.startswith("sqlite") else "postgresql",
        "datasets": {
            "dem_srtm_loaded": dem_available,
            "historical_rainfall_loaded": rainfall_available,
            "model_ready_features_loaded": feature_matrix_available
        },
        "server_time_utc": datetime.utcnow().isoformat() + "Z"
    }
