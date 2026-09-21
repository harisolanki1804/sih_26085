import os
import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.models.region import Region
from app.schemas.region import RegionRead, RegionCreate, HotspotInfo

router = APIRouter()


@router.get("", response_model=List[RegionRead], summary="List All Monitored Pilot Regions")
def list_regions(db: Session = Depends(get_db)):
    """Returns all geographic regions registered in VARUNA."""
    return db.query(Region).all()


@router.get("/{region_id_or_code}", response_model=RegionRead, summary="Get Pilot Region Details")
def get_region(region_id_or_code: str, db: Session = Depends(get_db)):
    """Fetches single region by UUID or region code (e.g. IN-MH-BOM-01)."""
    region = (
        db.query(Region)
        .filter((Region.id == region_id_or_code) | (Region.code == region_id_or_code))
        .first()
    )
    if not region:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Region not found.")
    return region


@router.get("/{region_id_or_code}/hotspots", response_model=List[HotspotInfo], summary="Get Regional Flood Hotspots")
def get_region_hotspots(region_id_or_code: str, db: Session = Depends(get_db)):
    """Returns critical urban infrastructure hotspots (underpasses, basins, transit hubs) for the pilot region."""
    meta_path = os.path.join(settings.DATA_DIR, "pilot_mumbai_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [HotspotInfo(**h) for h in data.get("critical_hotspots", [])]
    return []


@router.post("", response_model=RegionRead, status_code=status.HTTP_201_CREATED, summary="Create New Pilot Region")
def create_region(payload: RegionCreate, db: Session = Depends(get_db)):
    existing = db.query(Region).filter(Region.code == payload.code).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Region with this code already exists.")
    region = Region(**payload.model_dump())
    db.add(region)
    db.commit()
    db.refresh(region)
    return region


@router.get("/roadmap", summary="🗺️ All-India State Selector — Live vs Roadmap Status")
def get_state_roadmap():
    """
    Returns all major Indian states with their VARUNA deployment status.

    - status=live : Mumbai pilot fully operational
    - status=roadmap : Planned for future phases — no data shown

    Used by the frontend state/city selector to show the "Roadmap" badge
    on non-operational states, so the dashboard never shows placeholder data.
    """
    states = [
        # ── Live Pilot ────────────────────────────────────────────────
        {
            "code": "IN-MH-BOM-01", "state": "Maharashtra", "city": "Mumbai",
            "status": "live", "pilot_since": "2024",
            "grid_cells": 90, "resolution_km": 2,
            "center_lat": 19.08, "center_lon": 72.88,
            "note": "Full 90-cell Digital Twin — Mumbai Pilot Zone",
        },
        # ── Phase 2 Roadmap ───────────────────────────────────────────
        {
            "code": "IN-MH-PUN-01", "state": "Maharashtra", "city": "Pune",
            "status": "roadmap", "available_from": "Phase 2",
            "center_lat": 18.52, "center_lon": 73.86,
            "note": "Planned — Mula-Mutha river basin pilot",
        },
        {
            "code": "IN-KA-BLR-01", "state": "Karnataka", "city": "Bengaluru",
            "status": "roadmap", "available_from": "Phase 2",
            "center_lat": 12.97, "center_lon": 77.59,
            "note": "Planned — urban lake overflow pilot",
        },
        {
            "code": "IN-TN-CHN-01", "state": "Tamil Nadu", "city": "Chennai",
            "status": "roadmap", "available_from": "Phase 2",
            "center_lat": 13.08, "center_lon": 80.27,
            "note": "Planned — Adyar & Cooum river basin pilot",
        },
        {
            "code": "IN-WB-KOL-01", "state": "West Bengal", "city": "Kolkata",
            "status": "roadmap", "available_from": "Phase 3",
            "center_lat": 22.57, "center_lon": 88.36,
            "note": "Planned — Hooghly tidal flood zone pilot",
        },
        {
            "code": "IN-GJ-AHM-01", "state": "Gujarat", "city": "Ahmedabad",
            "status": "roadmap", "available_from": "Phase 3",
            "center_lat": 23.03, "center_lon": 72.58,
            "note": "Planned — Sabarmati river flood pilot",
        },
        {
            "code": "IN-OD-BHU-01", "state": "Odisha", "city": "Bhubaneswar",
            "status": "roadmap", "available_from": "Phase 3",
            "center_lat": 20.30, "center_lon": 85.84,
            "note": "Planned — cyclone + coastal flood pilot",
        },
        {
            "code": "IN-AS-GHY-01", "state": "Assam", "city": "Guwahati",
            "status": "roadmap", "available_from": "Phase 3",
            "center_lat": 26.14, "center_lon": 91.74,
            "note": "Planned — Brahmaputra basin flood pilot",
        },
        {
            "code": "IN-HP-SML-01", "state": "Himachal Pradesh", "city": "Shimla",
            "status": "roadmap", "available_from": "Phase 4",
            "center_lat": 31.10, "center_lon": 77.17,
            "note": "Planned — cloudburst & landslide pilot",
        },
        {
            "code": "IN-UK-DRD-01", "state": "Uttarakhand", "city": "Dehradun",
            "status": "roadmap", "available_from": "Phase 4",
            "center_lat": 30.32, "center_lon": 78.03,
            "note": "Planned — Himalayan flash-flood pilot",
        },
    ]
    live_count = sum(1 for s in states if s["status"] == "live")
    roadmap_count = sum(1 for s in states if s["status"] == "roadmap")
    return {
        "total_states": len(states),
        "live": live_count,
        "roadmap": roadmap_count,
        "states": states,
    }

