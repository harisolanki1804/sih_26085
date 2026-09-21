import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.database import init_db, seed_db
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database tables, then seed with pilot data
    init_db()
    seed_db()
    yield
    # Shutdown logic if any


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve React frontend static files
_frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'Frontend', 'dist')
if os.path.isdir(_frontend_dist):
    app.mount('/static', StaticFiles(directory=_frontend_dist), name='frontend-static')


@app.get("/", summary="Project VARUNA Root Gateway")
def root_gateway():
    return {
        "project": "Project VARUNA - Urban Multi-Hazard Early Warning System",
        "role": "Rudra (Dataset Sourcing & Preprocessing; FastAPI Backend + Database; Events/Alerts API)",
        "pilot_region": "Mumbai Central & Mithi River Basin (IN-MH-BOM-01)",
        "api_docs": "/docs",
        "health_check": f"{settings.API_V1_STR}/health",
        "version": settings.VERSION
    }
