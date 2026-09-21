"""
VARUNA Flood-Safe Route Suggestion API
--------------------------------------
Given start/end coordinates, suggests a safe route that avoids
flooded street segments (cells with high water depth estimates).
"""

import math
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.alert import Alert
from app.services.risk_model import risk_model_service
from app.services.road_network import find_safe_path, get_road_graph

router = APIRouter()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def get_flooded_cells(db: Session) -> List[Dict[str, Any]]:
    """Get currently active flooded cells (depth > 10cm or risk > 60)."""
    alerts = db.query(Alert).filter(
        Alert.is_active == True,
        (Alert.flood_depth_estimate_cm > 10.0) | (Alert.risk_score_total > 60.0)
    ).all()
    return [
        {"lat": a.cell_lat, "lon": a.cell_lon, "depth_cm": a.flood_depth_estimate_cm, "risk": a.risk_score_total}
        for a in alerts
    ]


def is_flooded(lat: float, lon: float, flooded_cells: List[Dict], threshold_km: float = 0.005) -> bool:
    """Check if a point is near a flooded cell."""
    for cell in flooded_cells:
        dist = haversine_km(lat, lon, cell["lat"], cell["lon"]) * 1000  # convert to meters
        if dist < threshold_km * 111000:  # approx meters for 0.005 degrees
            return True
    return False


def generate_safe_waypoints(
    start_lat: float, start_lon: float,
    end_lat: float, end_lon: float,
    flooded_cells: List[Dict],
    num_waypoints: int = 8
) -> List[Dict[str, Any]]:
    """
    Generate waypoints from start to end, detouring around flooded areas.
    Simple approach: divide the path into segments, check each for flooding,
    and nudge waypoints away from flooded cells.
    """
    waypoints = []
    
    for i in range(num_waypoints + 1):
        t = i / num_waypoints
        # Linear interpolation
        lat = start_lat + t * (end_lat - start_lat)
        lon = start_lon + t * (end_lon - start_lon)
        
        # Check if this point is flooded
        flooded = is_flooded(lat, lon, flooded_cells)
        detour = False
        
        if flooded and i > 0 and i < num_waypoints:
            # Nudge perpendicular to the route direction
            dlat = end_lat - start_lat
            dlon = end_lon - start_lon
            norm = math.sqrt(dlat**2 + dlon**2) + 1e-10
            # Perpendicular direction
            perp_lat = -dlon / norm * 0.008  # ~800m nudge
            perp_lon = dlat / norm * 0.008
            
            # Try both perpendicular directions, pick the non-flooded one
            alt1_lat, alt1_lon = lat + perp_lat, lon + perp_lon
            alt2_lat, alt2_lon = lat - perp_lat, lon - perp_lon
            
            if not is_flooded(alt1_lat, alt1_lon, flooded_cells):
                lat, lon = alt1_lat, alt1_lon
                detour = True
            elif not is_flooded(alt2_lat, alt2_lon, flooded_cells):
                lat, lon = alt2_lat, alt2_lon
                detour = True
            # If both are flooded, keep original (best we can do)
        
        waypoints.append({
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "is_flooded": flooded,
            "detour": detour,
            "label": "Start" if i == 0 else ("End" if i == num_waypoints else f"Waypoint {i}")
        })
    
    return waypoints


@router.get("/alternate", summary="Get Flood-Safe Alternate Route")
def get_flood_safe_route(
    start_lat: float = Query(..., description="Start latitude"),
    start_lon: float = Query(..., description="Start longitude"),
    end_lat: float = Query(..., description="End latitude"),
    end_lon: float = Query(..., description="End longitude"),
    db: Session = Depends(get_db)
):
    """
    Given start and end coordinates, returns a safe route that avoids
    flooded street segments. Uses active alert data to identify flooded areas.
    """
    flooded_cells = get_flooded_cells(db)
    
    # Use road network graph for realistic routing
    result = find_safe_path(start_lat, start_lon, end_lat, end_lon, flooded_cells)
    
    # Ensure required fields
    result.setdefault("flooded_cells_count", len(flooded_cells))
    result.setdefault("safety_note", "Route follows actual road network and avoids flooded streets.")
    
    return result
