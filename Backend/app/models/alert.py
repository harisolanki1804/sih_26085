import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Boolean, JSON, Text, Index
from sqlalchemy.orm import relationship
from app.core.database import Base


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String(36), ForeignKey("risk_events.id", ondelete="CASCADE"), index=True, nullable=True)
    region_id = Column(String(36), ForeignKey("regions.id", ondelete="CASCADE"), index=True, nullable=False)

    cell_lat = Column(Float, nullable=False)
    cell_lon = Column(Float, nullable=False)
    cell_id = Column(String(10), nullable=True)  # e.g. "C36"
    locality_name = Column(String(100), nullable=True)  # e.g. "Andheri West"
    timestep = Column(Integer, nullable=True)  # replay timestep
    timestamp = Column(DateTime, index=True, nullable=False)

    alert_type = Column(String(50), default="FLASH_FLOOD")  # FLASH_FLOOD, WATERLOGGING, CLOUDBURST, TIDAL_LOCK
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL

    # Explainable Additive Scoring Breakdown (Total = Sum of components, max 100)
    risk_score_total = Column(Float, nullable=False)
    rainfall_score_contrib = Column(Float, default=0.0)             # Max 40 pts
    soil_saturation_score_contrib = Column(Float, default=0.0)      # Max 25 pts
    topography_score_contrib = Column(Float, default=0.0)           # Max 20 pts
    atmospheric_instability_score_contrib = Column(Float, default=0.0)  # Max 15 pts

    # DEM-based Inundation Estimate
    flood_depth_estimate_cm = Column(Float, default=0.0)

    # Trust & Confidence Scoring
    trust_score = Column(Float, default=88.0)  # 0 to 100
    trust_level = Column(String(30), default="HIGH_CONFIDENCE")
    trust_factors_json = Column(JSON, nullable=True)

    # Human-readable Explainability & Reasoning
    reasoning_summary = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=True)
    contributing_hotspot_name = Column(String(100), nullable=True)

    # Operational lifecycle
    is_active = Column(Boolean, default=True)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    event = relationship("RiskEvent", back_populates="alerts")
    region = relationship("Region", back_populates="alerts")

    __table_args__ = (
        Index("ix_alerts_region_time_severity", "region_id", "timestamp", "severity"),
    )
