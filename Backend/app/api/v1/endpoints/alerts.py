from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.alert import Alert
from app.schemas.alert import AlertRead, AlertCreate, AlertDetailExplanation, ScoreDecomposition, TrustMetrics
from app.services.explainability import explainability_service

router = APIRouter()


@router.get("", response_model=List[AlertRead], summary="List Active & Historical Alerts")
def list_alerts(
    is_active: Optional[bool] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    alert_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db)
):
    """Fetches alerts with optional filtering by active state, severity (LOW/MED/HIGH/CRITICAL), or type."""
    query = db.query(Alert)
    if is_active is not None:
        query = query.filter(Alert.is_active == is_active)
    if severity:
        query = query.filter(Alert.severity == severity)
    if alert_type:
        query = query.filter(Alert.alert_type == alert_type)

    return query.order_by(Alert.risk_score_total.desc(), Alert.timestamp.desc()).limit(limit).all()


@router.get("/{alert_id}", response_model=AlertRead, summary="Get Single Alert")
def get_alert(alert_id: str, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found.")
    return alert


@router.get("/{alert_id}/explain", response_model=AlertDetailExplanation, summary="Get Full Explainability & Trust Decomposition")
def explain_alert(alert_id: str, db: Session = Depends(get_db)):
    """
    Returns full transparent breakdown showing:
    - Additive scoring components (Rainfall, Soil Saturation, Topography, Instability)
    - Trust metrics (Sensor agreement %, uncertainty margin, confidence level)
    - Contributing physical drivers
    - Actionable emergency playbook for operators
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found.")

    score_decomp = ScoreDecomposition(
        total_score=alert.risk_score_total,
        rainfall_component=alert.rainfall_score_contrib,
        soil_saturation_component=alert.soil_saturation_score_contrib,
        topography_component=alert.topography_score_contrib,
        instability_component=alert.atmospheric_instability_score_contrib
    )

    trust_factors = alert.trust_factors_json or {}
    key_drivers = []
    if alert.rainfall_score_contrib >= 25.0:
        key_drivers.append(f"Excess Rainfall Contribution ({alert.rainfall_score_contrib:.1f}/40 pts)")
    if alert.soil_saturation_score_contrib >= 15.0:
        key_drivers.append(f"Soil Saturation Contribution ({alert.soil_saturation_score_contrib:.1f}/25 pts)")
    if alert.topography_score_contrib >= 10.0:
        key_drivers.append(f"Lowland Topography Trap ({alert.topography_score_contrib:.1f}/20 pts)")
    if alert.atmospheric_instability_score_contrib >= 8.0:
        key_drivers.append(f"CAPE Convective Instability ({alert.atmospheric_instability_score_contrib:.1f}/15 pts)")

    trust_metrics = TrustMetrics(
        trust_score=alert.trust_score,
        trust_level=alert.trust_level or "HIGH_CONFIDENCE",
        sensor_agreement=trust_factors.get("satellite_radar_agreement_pct", 93.5),
        uncertainty_margin_pct=trust_factors.get("model_uncertainty_margin_pct", 7.2),
        key_drivers=key_drivers or ["Sustained monsoon rain flux"]
    )

    playbook = [
        "Review live telemetry from upstream rain gauges",
        "Confirm de-watering pump operability at nearest sump",
        alert.recommended_action or "Dispatch field inspection unit"
    ]

    return AlertDetailExplanation(
        alert=AlertRead.model_validate(alert),
        score_breakdown=score_decomp,
        trust_breakdown=trust_metrics,
        physical_drivers={
            "flood_depth_cm": alert.flood_depth_estimate_cm,
            "cell_lat": alert.cell_lat,
            "cell_lon": alert.cell_lon,
            "alert_type": alert.alert_type,
            "severity": alert.severity
        },
        action_playbook=playbook
    )


@router.post("/{alert_id}/acknowledge", response_model=AlertRead, summary="Acknowledge Alert by Operator")
def acknowledge_alert(
    alert_id: str,
    operator_name: str = Query(default="Operator-1"),
    db: Session = Depends(get_db)
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found.")
    
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = operator_name
    db.commit()
    db.refresh(alert)
    return alert
