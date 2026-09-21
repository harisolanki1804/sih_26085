from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.risk_event import RiskEvent
from app.models.alert import Alert
from app.schemas.risk_event import RiskEventRead, RiskEventCreate, RiskEventUpdate

router = APIRouter()


@router.get("", response_model=List[RiskEventRead], summary="List All Hazard & Risk Events")
def list_risk_events(
    status: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    db: Session = Depends(get_db)
):
    """Returns hazard events tracked across regions."""
    query = db.query(RiskEvent)
    if status:
        query = query.filter(RiskEvent.status == status)
    if severity:
        query = query.filter(RiskEvent.severity_level == severity)
    
    events = query.order_by(RiskEvent.start_time.desc()).all()
    results = []
    for evt in events:
        active_alerts = db.query(Alert).filter(Alert.event_id == evt.id, Alert.is_active == True).count()
        evt_dict = RiskEventRead.model_validate(evt)
        evt_dict.active_alerts_count = active_alerts
        results.append(evt_dict)
    return results


@router.get("/{event_id_or_code}", response_model=RiskEventRead, summary="Get Risk Event Details")
def get_risk_event(event_id_or_code: str, db: Session = Depends(get_db)):
    evt = (
        db.query(RiskEvent)
        .filter((RiskEvent.id == event_id_or_code) | (RiskEvent.event_code == event_id_or_code))
        .first()
    )
    if not evt:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Risk event not found.")
    active_alerts = db.query(Alert).filter(Alert.event_id == evt.id, Alert.is_active == True).count()
    res = RiskEventRead.model_validate(evt)
    res.active_alerts_count = active_alerts
    return res


@router.post("", response_model=RiskEventRead, status_code=status.HTTP_201_CREATED, summary="Create New Risk Event")
def create_risk_event(payload: RiskEventCreate, db: Session = Depends(get_db)):
    evt = RiskEvent(**payload.model_dump())
    db.add(evt)
    db.commit()
    db.refresh(evt)
    return evt
