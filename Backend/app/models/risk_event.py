import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class SeverityLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EventStatus(str, enum.Enum):
    MONITORING = "MONITORING"
    ACTIVE = "ACTIVE"
    PEAK = "PEAK"
    RECEDING = "RECEDING"
    RESOLVED = "RESOLVED"


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String(36), ForeignKey("regions.id", ondelete="CASCADE"), index=True, nullable=False)
    event_code = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    start_time = Column(DateTime, nullable=False)
    peak_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

    # Hazard Dynamics Summary
    peak_rainfall_rate_mm_hr = Column(Float, default=0.0)
    max_predicted_depth_cm = Column(Float, default=0.0)
    severity_level = Column(String(20), default=SeverityLevel.LOW.value)
    status = Column(String(20), default=EventStatus.MONITORING.value)
    confidence_score = Column(Float, default=85.0)  # 0 to 100
    
    triggered_by = Column(String(100), default="VARUNA Ensemble Cloudburst Detector")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    region = relationship("Region", back_populates="risk_events")
    alerts = relationship("Alert", back_populates="event", cascade="all, delete-orphan")
