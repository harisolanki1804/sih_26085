import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class Region(Base):
    __tablename__ = "regions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(150), nullable=False)
    description = Column(String(500), nullable=True)
    
    # Bounding Box
    bbox_lat_min = Column(Float, nullable=False)
    bbox_lat_max = Column(Float, nullable=False)
    bbox_lon_min = Column(Float, nullable=False)
    bbox_lon_max = Column(Float, nullable=False)
    
    # Center Centroid
    center_lat = Column(Float, nullable=False)
    center_lon = Column(Float, nullable=False)
    
    area_sqkm = Column(Float, nullable=True)
    avg_drainage_capacity_mm_hr = Column(Float, default=25.0)
    high_tide_threshold_m = Column(Float, default=4.5)
    
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    features = relationship("Feature", back_populates="region", cascade="all, delete-orphan")
    risk_events = relationship("RiskEvent", back_populates="region", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="region", cascade="all, delete-orphan")
