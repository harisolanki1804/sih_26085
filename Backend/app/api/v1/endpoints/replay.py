from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.replay import ReplayStatus, ReplayStepResponse, ReplayControlRequest
from app.services.replay_service import replay_engine

router = APIRouter()


@router.get("/status", response_model=ReplayStatus, summary="Get Current Historical Replay Simulation Status")
def get_replay_status(db: Session = Depends(get_db)):
    """Returns the current timestep, active storm phase, max risk score, and active alerts count."""
    return replay_engine.get_status(db)


@router.post("/step", response_model=ReplayStepResponse, summary="Advance Simulation by 1 Timestep (or Jump)")
def step_replay(
    step_to: int = Query(default=None, ge=1, le=72),
    db: Session = Depends(get_db)
):
    """
    Ticks the simulation 1 hour forward (or jumps to step_to).
    Executes real-time multi-hazard risk model, updates features, and raises dynamic explainable alerts.
    """
    try:
        return replay_engine.step(db, target_step=step_to)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/reset", response_model=ReplayStatus, summary="Reset Simulation to Timestep 1")
def reset_replay(db: Session = Depends(get_db)):
    """Resets the simulation back to pre-storm baseline."""
    return replay_engine.reset(db)
