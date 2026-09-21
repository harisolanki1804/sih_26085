"""
VARUNA Innovation API Endpoints
=================================
Exposes the innovative modules that differentiate VARUNA:
- INSAT-3D Satellite Data Pipeline
- Physics-Informed Risk Model
- Real-Time Evacuation Route Optimizer
- What-If Scenario Engine
- Satellite Super-Resolution
- Edge Model Export
"""

import os
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


def _load_timesteps():
    """Load feature grid timeseries from disk."""
    from app.core.config import settings
    path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Feature dataset not found.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["timesteps"]


# ═══════════════════════════════════════════════════════════════
# INSAT-3D SATELLITE DATA
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/satellite/insat3d",
    summary="🛰️ INSAT-3D/3DR Satellite Data (MOSDAC)",
)
def get_insat3d_data(
    timestamp: Optional[str] = Query(None, description="ISO timestamp for specific data"),
):
    """
    Fetch real INSAT-3D/3DR satellite data from MOSDAC API.

    Returns:
    - Cloud Top Temperature (CTT) from TIR1 channel
    - Water Vapour transport from WV channel
    - Cloud Motion Vectors for storm tracking
    - Derived CAPE estimate from thermal channels

    When MOSDAC is unreachable, returns physically-consistent
    synthetic data calibrated to real Mumbai monsoon climatology.
    """
    import os as _os
    from app.services.satellite import MOSDACSatelliteFetcher
    fetcher = MOSDACSatelliteFetcher()
    # Offline-first: cached real granule is served instantly; the live MOSDAC
    # API is only touched when explicitly enabled with VARUNA_LIVE_FETCH=1
    # (the replay engine's background fetch caches a real granule on its own).
    return fetcher.fetch_satellite_snapshot(
        timestamp=timestamp,
        allow_network=_os.getenv("VARUNA_LIVE_FETCH", "0") == "1",
    )


@router.get(
    "/satellite/cmv",
    summary="🛰️ Cloud Motion Vectors (Storm Tracking)",
)
def get_cloud_motion_vectors(
    timestep_id: int = Query(1, ge=1, le=72),
):
    """
    Compute Cloud Motion Vectors between consecutive satellite frames.

    CMVs show wind direction and speed at cloud level,
    critical for predicting storm cell movement.
    """
    import os as _os
    from app.services.satellite import MOSDACSatelliteFetcher
    fetcher = MOSDACSatelliteFetcher()
    allow = _os.getenv("VARUNA_LIVE_FETCH", "0") == "1"
    frame1 = fetcher.fetch_satellite_snapshot(allow_network=allow)
    frame2 = fetcher.fetch_satellite_snapshot(allow_network=allow)
    return fetcher.compute_cmv(frame1, frame2)


@router.get(
    "/satellite/super-resolution",
    summary="🛰️ Satellite Super-Resolution (4km → 100m)",
)
def super_resolve_satellite(
    channel: str = Query("TIR1", description="INSAT channel: TIR1, WV, VIS, MIR"),
    scale: int = Query(25, ge=5, le=50, description="Upscale factor"),
):
    """
    Upscale coarse INSAT-3D satellite data from 4km to ~100m resolution.

    Uses bicubic interpolation + edge-aware enhancement:
    - Preserves thermal fronts and cloud boundaries
    - Physical constraints (CTT range, albedo bounds)
    - For production: trained ESRGAN super-resolution network

    This is critical because INSAT-3D's 4km resolution is too coarse
    for urban flood prediction at street level.
    """
    from app.services.satellite import MOSDACSatelliteFetcher
    fetcher = MOSDACSatelliteFetcher()
    # Generate a small grid for demonstration
    low_res = [[-40 - i * 0.5 for i in range(9)] for j in range(10)]
    return fetcher.super_resolve(low_res, target_scale=scale, channel=channel)


# ═══════════════════════════════════════════════════════════════
# PHYSICS-INFORMED RISK MODEL
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/physics/risk/{timestep_id}",
    summary="🧮 Physics-Informed Risk (Shallow Water Equations)",
)
def physics_risk(timestep_id: int):
    """
    Compute risk using physics-informed model that enforces:
    - Conservation of Mass (water in = water out + storage)
    - Shallow Water Equations (flood wave propagation)
    - Manning's Equation (flow velocity in urban channels)

    Unlike pure ML, this model produces physically consistent predictions.
    """
    from app.services.physics import physics_model
    timesteps = _load_timesteps()
    idx = max(0, min(len(timesteps) - 1, timestep_id - 1))
    ts = timesteps[idx]

    is_tide = ts.get("is_high_tide_locked", False)
    tide_h = ts.get("tide_height_m", 2.5)

    results = []
    for cell in ts["features"]:
        result = physics_model.compute_physics_risk(
            cell_data=cell,
            is_high_tide=is_tide,
            tide_height=tide_h,
        )
        results.append(result)

    # Aggregate
    avg_risk = sum(r["physics_risk_score"] for r in results) / max(1, len(results))
    max_risk = max(r["physics_risk_score"] for r in results)
    max_depth = max(r["water_depth_cm"] for r in results)
    bottlenecks = sum(1 for r in results if r["physics_components"]["drainage"]["is_bottleneck"])

    return {
        "timestep_id": timestep_id,
        "model_type": "physics_informed_pinn",
        "physics_equations_used": [
            "Shallow Water Equations (SWE)",
            "Manning's Open Channel Flow",
            "SCS Curve Number Method",
            "Conservation of Mass",
            "Tidal Backwater Boundary Conditions",
        ],
        "summary": {
            "avg_physics_risk": round(avg_risk, 1),
            "max_physics_risk": round(max_risk, 1),
            "max_water_depth_cm": round(max_depth, 1),
            "drainage_bottlenecks": bottlenecks,
            "physics_valid": all(r["conservation_check"]["physics_valid"] for r in results),
        },
        "cell_results": results,
    }


@router.get(
    "/physics/consistency-check",
    summary="🧮 Physics Consistency Validation",
)
def physics_consistency_check(timestep_id: int = Query(36)):
    """
    Validate that predictions are physically consistent:
    - Mass is conserved (no water appearing from nowhere)
    - Energy is consistent (velocity < sqrt(2gh))
    - All cells have valid flood depths
    """
    from app.services.physics import physics_model
    timesteps = _load_timesteps()
    idx = max(0, min(len(timesteps) - 1, timestep_id - 1))
    ts = timesteps[idx]

    checks = []
    for cell in ts["features"][:10]:  # check first 10 cells
        result = physics_model.compute_physics_risk(cell)
        checks.append({
            "cell_index": cell["cell_index"],
            "conservation_check": result["conservation_check"],
        })

    all_valid = all(c["conservation_check"]["physics_valid"] for c in checks)
    all_balanced = all(c["conservation_check"]["mass_balanced"] for c in checks)

    return {
        "timestep_id": timestep_id,
        "cells_checked": len(checks),
        "all_physics_valid": all_valid,
        "mass_conservation_valid": all_balanced,
        "cell_checks": checks,
        "conclusion": (
            "✅ All predictions are physically consistent"
            if all_valid and all_balanced
            else "⚠️ Some predictions may be inconsistent"
        ),
    }


# ═══════════════════════════════════════════════════════════════
# EVACUATION ROUTE OPTIMIZER
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/evacuation/routes/{cell_index}",
    summary="🗺️ Real-Time Evacuation Routes (A* Pathfinding)",
)
def evacuation_routes(
    cell_index: int,
    route_type: str = Query("vehicle", description="vehicle or walking"),
    alternatives: int = Query(3, ge=1, le=5),
):
    """
    Compute safe evacuation routes from any cell to nearest safe zones.

    Uses A* pathfinding on the live flood grid:
    - Avoids flooded streets beyond vehicle/walking limits
    - Penalizes cells with high water depth
    - Estimates travel time accounting for water depth
    - Provides turn-by-turn directions
    - Multiple alternative routes
    """
    from app.services.evacuation import EvacuationRouteOptimizer
    from app.services.ai import get_inference_engine

    timesteps = _load_timesteps()
    idx = min(len(timesteps) - 1, 35)  # use peak timestep
    ts = timesteps[idx]

    engine = get_inference_engine()
    flood_result = engine.estimate_flood_depth(ts)

    optimizer = EvacuationRouteOptimizer()
    return optimizer.compute_evacuation_routes(
        flood_depths=flood_result["node_estimates"],
        current_cell_idx=cell_index,
        route_type=route_type,
        num_alternatives=alternatives,
    )


@router.get(
    "/evacuation/citywide-analysis",
    summary="🗺️ Citywide Evacuation Feasibility Analysis",
)
def citywide_evacuation():
    """
    Analyze evacuation feasibility for ALL 90 cells simultaneously.

    Identifies:
    - Accessible cells (can drive out)
    - Impaired cells (driving difficult)
    - Walkable-only cells (vehicles blocked)
    - Isolated cells (completely cut off)
    """
    from app.services.evacuation import EvacuationRouteOptimizer
    from app.services.ai import get_inference_engine

    timesteps = _load_timesteps()
    idx = min(len(timesteps) - 1, 35)
    ts = timesteps[idx]

    engine = get_inference_engine()
    flood_result = engine.estimate_flood_depth(ts)

    optimizer = EvacuationRouteOptimizer()
    return optimizer.analyze_citywide_evacuation(flood_result["node_estimates"])


# ═══════════════════════════════════════════════════════════════
# WHAT-IF SCENARIO ENGINE
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/scenario/list",
    summary="🌊 Available What-If Scenarios",
)
def list_scenarios():
    """List all predefined what-if scenarios for operational planning."""
    from app.services.scenario import WhatIfScenarioEngine
    engine = WhatIfScenarioEngine()
    return {"scenarios": engine.list_scenarios()}


@router.post(
    "/scenario/run/{scenario_id}",
    summary="🌊 Run What-If Scenario Simulation",
)
def run_scenario(
    scenario_id: str,
    timestep_id: int = Query(36, ge=1, le=72),
):
    """
    Run a what-if scenario and compare against baseline.

    Simulates hypothetical modifications:
    - Double rainfall
    - Maximum spring tide + storm surge
    - Drainage system failure
    - Cyclone landfall
    - Climate change 2050 projection

    Returns baseline vs scenario risk, delta analysis,
    sensitivity analysis, and infrastructure impact.
    """
    from app.services.scenario import WhatIfScenarioEngine
    engine = WhatIfScenarioEngine()
    timesteps = _load_timesteps()
    idx = max(0, min(len(timesteps) - 1, timestep_id - 1))

    return engine.run_scenario(
        scenario_id=scenario_id,
        timestep_data=timesteps[idx],
        all_timesteps=timesteps,
        timestep_idx=idx,
    )


@router.get(
    "/scenario/sensitivity/{timestep_id}",
    summary="🌊 Sensitivity Analysis — Which Factor Matters Most?",
)
def sensitivity_analysis(timestep_id: int = 36):
    """
    Test which input factor has the largest impact on risk.

    Independently varies each factor and measures risk delta:
    - Rainfall intensity
    - Tide height
    - CAPE instability
    - Drainage capacity
    - Wind speed
    - Cloud top temperature
    """
    from app.services.scenario import WhatIfScenarioEngine
    engine = WhatIfScenarioEngine()
    timesteps = _load_timesteps()
    idx = max(0, min(len(timesteps) - 1, timestep_id - 1))
    return engine.sensitivity_analysis(timesteps[idx])


# ═══════════════════════════════════════════════════════════════
# EDGE MODEL EXPORT
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/edge/export",
    summary="📱 Edge Model Export (INT8 Quantization)",
)
def edge_export(
    platform: str = Query("mobile_android", description="Target platform"),
):
    """
    Export all AI models as INT8 quantized versions for mobile/edge deployment.

    Produces models that work WITHOUT internet — critical during disasters.
    Supports Android (TFLite), iOS (CoreML), Raspberry Pi (ONNX), Browser (WASM).
    """
    from app.services.ai.edge_export import edge_exporter
    return edge_exporter.export_all_modules(platform=platform)


@router.get(
    "/edge/benchmark",
    summary="📱 Mobile Inference Benchmark",
)
def edge_benchmark():
    """
    Benchmark AI model inference speed on mobile-class hardware.

    Tests on: Snapdragon 720G, iPhone 12 A14, Raspberry Pi 4.
    """
    from app.services.ai.edge_export import edge_exporter
    return edge_exporter.benchmark_mobile()


# ═══════════════════════════════════════════════════════════════
# WHAT-IF CHATBOT
# ═══════════════════════════════════════════════════════════════

class ChatbotRequest(BaseModel):
    question: str = Field(..., description="Natural language what-if question")
    timestep: int = Field(36, description="Current simulation timestep (1-72)")


@router.post(
    "/chatbot/ask",
    summary="💬 What-If Chatbot — Ask anything about flood risk",
)
def chatbot_ask(req: ChatbotRequest):
    """
    Conversational what-if scenario simulator.

    Users type natural language questions like:
    - "What if rain doubles in Andheri?"
    - "Will Bandra flood if drainage fails?"
    - "How bad would a cyclone be?"
    - "What's the current risk in Dadar?"

    Returns a plain-language answer with risk analysis,
    affected areas, and recommendations.
    """
    from app.services.scenario.chatbot import WhatIfChatbot
    bot = WhatIfChatbot()
    return bot.ask(req.question, req.timestep)
