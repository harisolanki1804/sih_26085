"""
Citizen Report Model
====================
Stores crowd-sourced flood reports submitted by the public.

Each report is:
  - Classified by the NLP module as CONFIRMING or DENYING a predicted zone
  - Compared against current model risk for that location
  - Flagged for retraining if it disagrees with the model (active learning)
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text
from app.core.database import Base


class CitizenReport(Base):
    __tablename__ = "citizen_reports"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Report content
    report_text = Column(Text, nullable=False)
    source = Column(String(50), default="WEB_FORM")  # WEB_FORM, WHATSAPP, API

    # Location
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    locality_name = Column(String(100), nullable=True)

    # NLP Classification result
    classification = Column(String(30), nullable=True)        # CONFIRMING / DENYING / NEUTRAL / UNCERTAIN
    confidence = Column(Float, nullable=True)                  # 0.0 – 1.0
    flood_keywords = Column(String(500), nullable=True)        # comma-separated matched keywords

    # Disagreement detection (active learning signal)
    model_risk_score = Column(Float, nullable=True)            # VARUNA risk score at that location/time
    agrees_with_model = Column(Boolean, nullable=True)         # True if citizen + model agree
    flagged_for_retraining = Column(Boolean, default=False)    # True when disagreement detected

    # Operational
    reviewed_by_operator = Column(Boolean, default=False)
    operator_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
