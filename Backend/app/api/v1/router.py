from fastapi import APIRouter
from app.api.v1.endpoints import health, regions, features, events, alerts, replay, raw_data
from app.api.v1.endpoints import ai_pipeline, innovations, realtime, routes, metrics

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Diagnostics"])
api_router.include_router(raw_data.router, prefix="/data/raw", tags=["Raw Datasets for Model Team"])
api_router.include_router(regions.router, prefix="/regions", tags=["Regions & Pilot Areas"])
api_router.include_router(features.router, prefix="/features", tags=["Feature Grids & Hydro Data"])
api_router.include_router(events.router, prefix="/events", tags=["Hazard Events"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["Explainable Alerts & Trust"])
api_router.include_router(replay.router, prefix="/replay", tags=["Historical Deluge Replay Engine"])
api_router.include_router(ai_pipeline.router, prefix="/ai", tags=["AI/ML Inference Pipeline (9 Modules)"])
api_router.include_router(innovations.router, prefix="/innovations", tags=["🚀 Innovation Modules (Satellite, PINN, Evacuation, Scenarios, Edge)"])
api_router.include_router(realtime.router, prefix="/realtime", tags=["📡 Real-Time Data (India-Wide)"])
api_router.include_router(routes.router, prefix="/routes", tags=["🗺️ Evacuation Route Planner"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["📊 Evaluation Metrics & Model Skill"])
