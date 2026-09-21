import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings
from app.core.timeline_config import EVENT_CODE, EVENT_START_DT

logger = logging.getLogger("VARUNA.Database")

# SQLite needs connect_args check_same_thread=False
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency generator for database sessions in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all database tables."""
    import app.models  # Ensure all models (including CitizenReport) are imported
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")


def seed_db():
    """
    Seed the database with the Mumbai pilot region and the 2005 deluge case-study
    event if they don't already exist.  Safe to call on every startup.
    """
    import uuid
    from datetime import datetime
    from app.models.region import Region
    from app.models.risk_event import RiskEvent, SeverityLevel, EventStatus

    db = SessionLocal()
    try:
        # ── Mumbai Pilot Region ─────────────────────────────────────
        existing_region = db.query(Region).filter(Region.code == "IN-MH-BOM-01").first()
        if not existing_region:
            region = Region(
                id=str(uuid.uuid4()),
                code="IN-MH-BOM-01",
                name="Mumbai Metropolitan Region — Mithi River Pilot Zone",
                description=(
                    "Urban flash-flood pilot covering Mumbai Central, Mithi River basin, "
                    "Kurla-Sion depression bowl, and coastal tidal-lock zones."
                ),
                bbox_lat_min=18.88,
                bbox_lat_max=19.26,
                bbox_lon_min=72.78,
                bbox_lon_max=73.00,
                center_lat=19.08,
                center_lon=72.88,
                area_sqkm=603.4,
                avg_drainage_capacity_mm_hr=25.0,
                high_tide_threshold_m=4.5,
                metadata_json={
                    "state": "Maharashtra",
                    "pilot_status": "live",
                    "grid_cells": 90,
                    "resolution_km": 2,
                },
            )
            db.add(region)
            db.flush()
            logger.info("Seeded Mumbai pilot region (IN-MH-BOM-01).")
        else:
            region = existing_region

        # ── Mumbai Deluge Case-Study Event ──────────────────────────
        if not db.query(RiskEvent).filter(
            RiskEvent.event_code == EVENT_CODE
        ).first():
            event = RiskEvent(
                id=str(uuid.uuid4()),
                region_id=region.id,
                event_code=EVENT_CODE,
                title="Mumbai Monsoon Cloudburst — Mithi Basin Deluge (Case Study)",
                description=(
                    "72-hour synthetic replay calibrated to the July 2005 Mumbai deluge "
                    "(944 mm in 24 hrs). Used for Digital Twin demonstration."
                ),
                start_time=EVENT_START_DT,
                severity_level=SeverityLevel.CRITICAL.value,
                status=EventStatus.MONITORING.value,
                peak_rainfall_rate_mm_hr=120.0,
                max_predicted_depth_cm=110.0,
                confidence_score=91.0,
                triggered_by="VARUNA Ensemble Cloudburst Detector",
            )
            db.add(event)
            logger.info("Seeded Mumbai deluge case-study event.")

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(f"DB seeding failed (non-critical): {exc}")
    finally:
        db.close()

