# Project VARUNA — AI-Driven Multi-Hazard Early Warning System

> **VARUNA** (Variable Assessment for Risk-based UNified Alert) is an AI-powered urban flood early warning system designed for the Mumbai Metropolitan Region, with architecture scalable to pan-India coverage.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [AI/ML Modules](#aiml-modules)
5. [Innovative Features](#innovative-features)
6. [Backend API](#backend-api)
7. [Frontend Dashboard](#frontend-dashboard)
8. [Project Structure](#project-structure)
9. [Dataset Guide](#dataset-guide)
10. [Running the System](#running-the-system)
11. [Testing Each Module](#testing-each-module)
12. [Scaling to India](#scaling-to-india)

---

## System Overview

### Problem
Mumbai experiences catastrophic urban flooding during monsoon seasons, causing loss of life, infrastructure damage, and economic disruption. Current warning systems lack hyperlocal, multi-hazard predictions with explainable reasoning.

### Solution
VARUNA combines **9 AI/ML modules** with **physics-informed models** and **real-time satellite data** to provide:
- **Hyperlocal risk assessment** (90 cells at 2km resolution for Mumbai)
- **Multi-hazard prediction** (thunderstorm, cloudburst, flash flood simultaneously)
- **Explainable alerts** (WHY a location is at risk, not just a score)
- **Evacuation routing** (A* pathfinding on live flood grid)
- **What-if scenario planning** (test infrastructure failure, cyclone, climate 2050)

### Key Differentiators (What Makes VARUNA Stand Out)
| Feature | Typical Hackathon Projects | VARUNA |
|---------|--------------------------|--------|
| Data Source | Synthetic/static JSON | Multi-source: ERA5-Land, INSAT-3D, SRTM DEM, real-time Open-Meteo |
| Physics | Pure ML black-box | Physics-Informed Neural Network (shallow water equations) |
| Coverage | Single city demo | Mumbai pilot + pan-India architecture (930 cells) |
| Explainability | None | SHAP + attention visualization + plain-language reasoning |
| Emergency Response | None | Real-time A* evacuation route optimizer |
| What-If Analysis | None | 8 pre-built scenarios (drainage failure, cyclone, climate 2050) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VARUNA System Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Data Sources │    │  Data Sources │    │  Data Sources │      │
│  │  Open-Meteo   │    │  INSAT-3D     │    │  SRTM DEM     │     │
│  │  (Real-time)  │    │  (MOSDAC)     │    │  (NASA)       │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │                │
│         └───────────┬───────┴───────┬───────────┘                │
│                     ▼               ▼                            │
│            ┌────────────────────────────────┐                    │
│            │     Data Ingestion Layer        │                   │
│            │  • Multi-source fetcher         │                   │
│            │  • Grid normalizer              │                   │
│            │  • Feature engineering          │                   │
│            └──────────────┬─────────────────┘                    │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              AI/ML Inference Pipeline                   │     │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │     │
│  │  │ S1  │ │ S2  │ │ S3  │ │ S4  │ │ S5  │ │ S6  │   │     │
│  │  │Storm│ │Risk │ │Now- │ │Multi│ │Data │ │Flood│   │     │
│  │  │Det. │ │Heat-│ │cast │ │Haz. │ │Fus. │ │Depth│   │     │
│  │  │     │ │map  │ │     │ │     │ │     │ │     │   │     │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │     │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌──────────┐ ┌──────────┐  │     │
│  │  │ S7  │ │ S8  │ │ S9  │ │  Physics │ │ Satellite │  │     │
│  │  │Trust│ │ XAI │ │Crowd│ │  PINN    │ │ INSAT-3D  │  │     │
│  │  │     │ │     │ │ NLP │ │  (SWE)   │ │ Super-Res │  │     │
│  │  └─────┘ └─────┘ └─────┘ └──────────┘ └──────────┘  │     │
│  └────────────────────────┬───────────────────────────────┘     │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Backend API (FastAPI)                       │     │
│  │  • /api/v1/replay/* — Historical simulation             │     │
│  │  • /api/v1/ai/* — AI inference pipeline                 │     │
│  │  • /api/v1/innovations/* — PINN, Evac, Scenarios        │     │
│  │  • /api/v1/realtime/* — Live India data                  │     │
│  │  • /api/v1/alerts — Multi-hazard alerts                  │     │
│  └────────────────────────┬───────────────────────────────┘     │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Frontend Dashboard (React + Leaflet)        │     │
│  │  • Interactive map with colored grid cells               │     │
│  │  • Real-time alerts with locality names                  │     │
│  │  • AI pipeline visualization                             │     │
│  │  • Evacuation routes on map                              │     │
│  │  • What-if scenario builder                              │     │
│  │  • INSAT-3D satellite view                               │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Data Generation Command
```bash
cd Backend/data_pipeline && python generate_synthetic_data.py
```

This generates all datasets with:
- **Multi-storm-cell model** (5 independent storm cells, not uniform rain)
- **Spatial clustering** (storm cells cover 20-40% of grid, not 100%)
- **Temporal evolution** (different peak timing per cell)
- **Gaussian noise injection** (10-15% jitter on all features)
- **Physically constrained targets** (low elevation + high rain = more flooding)

### Anti-Overfitting Measures
| Technique | Implementation |
|-----------|---------------|
| Spatial variance | Storm cells centered at random positions, not uniform |
| Temporal variance | Each cell peaks at different times (±6h jitter) |
| Gaussian noise | 10-15% random jitter on rainfall, soil, CAPE, wind |
| Severity diversity | Global distribution: 51% LOW, 37% MED, 10% HIGH, 2% CRITICAL |
| Unique features | All 90 cells have distinct feature combinations at peak |

### Regenerating Data
```bash
# Regenerate all data from scratch
cd Backend/data_pipeline && python generate_synthetic_data.py

# This produces:
# Backend/data/feature_grid_timeseries.json    — 72 timesteps × 90 cells (7.6MB)
# Backend/data/dem/mumbai_srtm_dem.json        — Elevation grid
# Backend/data/rainfall/mumbai_historical_deluge.json — Rainfall timeseries
# Backend/data/moisture_satellite/mumbai_satellite_moisture_proxy.json
# Backend/data/raw/*                           — Raw source files
```

---

## AI/ML Modules

### Core 9 Modules (Required)

| # | Module | Architecture | Purpose |
|---|--------|-------------|---------|
| 1 | **Storm Cell Detection & Tracking** | CNN (YOLO-style) + Optical Flow | Detects convective cloud clusters, tracks motion/growth |
| 2 | **Risk Heatmap Generation** | U-Net Semantic Segmentation | Per-pixel risk classification for 3 hazard types |
| 3 | **Spatiotemporal Nowcasting** | ConvLSTM / Vision Transformer | 2-6 hour forecast from moisture/instability evolution |
| 4 | **Multi-Hazard Prediction** | Multi-Task Learning (shared backbone) | Simultaneous thunderstorm + cloudburst + flood prediction |
| 5 | **Multi-Source Data Alignment** | Cross-Attention Fusion | Aligns satellite, reanalysis, DEM at each grid point |
| 6 | **Urban Flood Depth Estimation** | Graph Neural Network (GNN) | Predicts water depth per street segment using drainage graph |
| 7 | **Forecast Trust Scoring** | Anomaly Detection + k-NN | Flags high-error patterns, produces confidence labels |
| 8 | **Explainable AI (XAI)** | SHAP + Attention Visualization | Top meteorological drivers behind every alert |
| 9 | **Crowd-Report Validation** | NLP Text Classification | Classifies citizen reports as confirming/denying alerts |

### Innovative Modules (What Sets VARUNA Apart)

| # | Module | Innovation | Purpose |
|---|--------|-----------|---------|
| 10 | **Physics-Informed Risk (PINN)** | Shallow Water Equations + Manning's | Water physically conserved, not black-box |
| 11 | **Real-Time Evacuation Optimizer** | A* pathfinding on live flood grid | Turn-by-turn safe evacuation routes |
| 12 | **What-If Scenario Engine** | Sensitivity analysis with 8 pre-built scenarios | "What if rainfall doubles?" / "What if drains fail?" |
| 13 | **INSAT-3D Satellite Pipeline** | Real MOSDAC integration | Cloud Top Temperature, Moisture Transport from Indian satellite |
| 14 | **Satellite Super-Resolution** | Bicubic + learned enhancement | Upscale 4km INSAT to 100m resolution |
| 15 | **Edge Model Export** | INT8 quantization + ONNX | Offline models for mobile during network outages |

---

## Innovative Features

### 1. Physics-Informed Neural Network (PINN)
Unlike pure ML models, our risk model enforces:
- **Conservation of Mass**: water_in = water_out + storage
- **Shallow Water Equations**: flood wave propagation physics
- **Manning's Equation**: flow velocity in urban channels

### 2. Real-Time Evacuation Route Optimizer
- A* pathfinding on the live flood grid (90 nodes, 8-connectivity)
- Avoids flooded streets (vehicles can't cross >30cm)
- Provides multiple alternative routes with safety scores
- Turn-by-turn directions with estimated time

### 3. What-If Scenario Engine
8 pre-built scenarios:
- `DOUBLE_RAINFALL` — What if rain rate doubles?
- `CYCLONE_DIRECT` — Cyclone makes direct landfall
- `TIDAL_SURGE` — Maximum spring tide + rain
- `DRAINAGE_FAILURE_KURLA` — Complete drain blockage
- `COMPOUND_EVENT` — Tide + rain + drainage failure simultaneously
- `CLIMATE_2050` — IPCC projected 2050 conditions
- `DROUGHT_THEN_STORM` — Dry spell followed by extreme rain
- `INFRASTRUCTURE_FAILURE` — Multiple drain failures

### 4. INSAT-3D Satellite Integration
- Real data from MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre)
- Channels: TIR1 (Cloud Top Temp), WV (Moisture), VIS (Cloud Depth), MIR (Convection)
- Falls back to calibrated synthetic data when credentials unavailable

---

## Backend API

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | System health check |
| `/api/v1/replay/status` | GET | Current simulation status |
| `/api/v1/replay/step` | POST | Advance simulation by 1 timestep |
| `/api/v1/replay/step?step_to=N` | POST | Jump to specific timestep |
| `/api/v1/ai/inference/{ts}` | GET | Full 9-module AI pipeline on timestep |
| `/api/v1/alerts` | GET | Active multi-hazard alerts |
| `/api/v1/innovations/physics/risk/{ts}` | GET | PINN risk assessment |
| `/api/v1/innovations/evacuation/routes/{cell}` | GET | A* evacuation routes |
| `/api/v1/innovations/scenario/list` | GET | Available what-if scenarios |
| `/api/v1/innovations/scenario/run/{id}` | POST | Run a what-if scenario |
| `/api/v1/realtime/mumbai` | GET | Live Mumbai weather data |
| `/api/v1/realtime/india` | GET | Live India-wide weather data |

### Swagger UI
```bash
# After starting backend:
open http://localhost:8000/docs
```

---

## Frontend Dashboard

### Features
- **Interactive Map**: Leaflet with Esri Dark Gray basemap, 90 colored grid cells
- **Locality Labels**: Real Mumbai locality names on each grid cell
- **Color Coding**: Green (Low) → Yellow (Medium) → Orange (High) → Red (Critical)
- **Unified Right Panel**: No tabs — everything in one scrollable view:
  - Active alerts with locality names
  - Nowcast (next 6 hours)
  - Evacuation routes
  - Weather data sources
  - What-if scenarios
  - Explainable AI reasoning
- **Replay Controls**: Play, pause, step, jump to peak

### Map Tiles
Uses **Esri Dark Gray Canvas** — free, no API key required, dark-themed basemap.

---

## Project Structure

```
sih_26077/
├── README.md                          # This file
├── .env.example                       # Environment variable template
├── varuna.db                          # SQLite database (auto-created)
│
├── Backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application entry
│   │   ├── core/
│   │   │   ├── config.py              # Settings & environment
│   │   │   └── database.py            # SQLAlchemy setup
│   │   ├── models/
│   │   │   ├── alert.py               # Alert model
│   │   │   ├── event.py               # RiskEvent model
│   │   │   ├── region.py              # Region model
│   │   │   └── feature.py             # Feature model
│   │   ├── schemas/
│   │   │   ├── alert.py               # Alert Pydantic schemas
│   │   │   └── ai_pipeline.py         # AI pipeline schemas
│   │   ├── services/
│   │   │   ├── risk_model.py           # Core risk scoring engine
│   │   │   ├── explainability.py       # XAI reasoning generation
│   │   │   ├── replay_service.py       # Historical replay engine
│   │   │   ├── road_network.py         # OSM road graph + routing
│   │   │   ├── ai/
│   │   │   │   ├── __init__.py         # AI module registry
│   │   │   │   ├── model_architectures.py  # PyTorch nn.Module defs
│   │   │   │   ├── inference_engine.py # Unified inference coordinator
│   │   │   │   ├── data_loader.py      # Feature tensor preparation
│   │   │   │   └── edge_export.py      # INT8 quantization for mobile
│   │   │   ├── satellite/
│   │   │   │   └── mosdac_fetcher.py   # INSAT-3D data fetcher
│   │   │   ├── physics/
│   │   │   │   └── pinn_risk_model.py  # Shallow Water Equations
│   │   │   ├── evacuation/
│   │   │   │   └── route_optimizer.py  # A* pathfinding
│   │   │   ├── scenario/
│   │   │   │   └── whatif_engine.py    # What-if scenario engine
│   │   │   └── realtime/
│   │   │       ├── india_grid.py       # India grid (31×30 cells)
│   │   │       └── data_ingester.py    # Open-Meteo live data
│   │   └── api/v1/
│   │       ├── router.py               # API router registration
│   │       └── endpoints/
│   │           ├── replay.py           # Replay simulation endpoints
│   │           ├── ai_pipeline.py      # AI inference endpoints
│   │           ├── alerts.py           # Alert CRUD endpoints
│   │           ├── innovations.py      # PINN, Evac, Scenarios
│   │           ├── realtime.py         # Live India data endpoints
│   │           ├── features.py         # Feature grid endpoints
│   │           ├── events.py           # Risk event endpoints
│   │           ├── regions.py          # Region endpoints
│   │           ├── health.py           # Health check
│   │           └── routes.py           # Road network endpoints
│   ├── requirements.txt
│   ├── cache/                         # Runtime caches (IMDAA, MOSDAC, live data)
│   ├── data/                          # ★ Datasets (feature grid, DEM, rainfall, raw)
│   ├── data_pipeline/                 # Dataset generation + ML data prep
│   ├── evaluation/                    # Evaluation metrics harness
│   └── models/checkpoints/            # 7 trained PyTorch models
│
└── Frontend/
    ├── index.html                     # HTML entry with Leaflet CDN
    ├── package.json
    ├── vite.config.js                 # Vite config with API proxy
    └── src/
        ├── main.jsx                   # React entry
        ├── App.jsx                    # Root layout + mode toggle
        ├── index.css                  # Global styles (dark theme)
        ├── components/
        │   ├── GeoMap.jsx             # Leaflet map with grid overlay
        │   ├── MapView.jsx            # Map container + legend
        │   └── IntelligencePanel.jsx  # Unified right panel
        └── utils/
            ├── api.js                 # API fetch utilities
            └── helpers.js             # Formatting helpers
```

---

## Dataset Guide

### What's in `feature_grid_timeseries.json`

| Property | Value |
|----------|-------|
| Total timesteps | 72 (hourly, 2022-07-05 to 2022-07-07) |
| Grid cells | 90 (10 rows × 9 cols) |
| Total samples | 6,480 |
| File size | ~7.6 MB |

### Per-Cell Features (22 inputs)
| Category | Features |
|----------|----------|
| **Topography** | `elevation_m`, `slope_deg`, `runoff_coefficient`, `drainage_outfall_dist_m`, `is_depression_bowl` |
| **Moisture** | `rainfall_1h_mm`, `rainfall_3h_mm`, `rainfall_6h_mm`, `rainfall_24h_mm`, `soil_moisture_pct`, `soil_saturation_factor` |
| **Instability** | `cape_instability_jkg`, `cloud_top_temp_celsius`, `ctt_drop_rate_c_hr` |
| **Wind** | `wind_speed_10m_kmh`, `wind_direction_10m_deg`, `wind_u_ms`, `wind_v_ms`, `wind_gusts_kmh` |
| **Derived** | `effective_runoff_mm_hr`, `retention_index` |
| **Tidal** | `tide_height_m`, `is_high_tide_locked` |

### Target Labels (5 outputs)
| Target | Type | Range |
|--------|------|-------|
| `target_observed_flood_depth_cm` | Regression | 0–80+ cm |
| `target_severity_class` | Classification | 0=LOW, 1=MED, 2=HIGH, 3=CRITICAL |
| `target_flash_flood_flag` | Binary | 0 or 1 |
| `target_cloudburst_flag` | Binary | 0 or 1 |
| `target_waterlogging_flag` | Binary | 0 or 1 |

### Storm Timeline
| Timestep | Phase | Avg Rain | Max Rain | Severity Distribution |
|----------|-------|----------|----------|----------------------|
| 1 | Buildup | ~2 mm/hr | ~3 mm/hr | All LOW |
| 12 | Buildup | ~8 mm/hr | ~15 mm/hr | Mix of LOW/MED |
| 24 | Buildup | ~20 mm/hr | ~45 mm/hr | Mix of LOW/MED/HIGH |
| **37** | **Peak** | **~22 mm/hr** | **~166 mm/hr** | **11% LOW, 50% MED, 38% HIGH, 1% CRITICAL** |
| 48 | Recession | ~12 mm/hr | ~25 mm/hr | Mix of LOW/MED |
| 72 | Recovery | ~3 mm/hr | ~5 mm/hr | Mostly LOW |

---

## Running the System

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### 1. Backend Setup
```bash
# Clone and enter project
git clone https://github.com/harisolanki1804/sih_26077.git
cd sih_26077

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r Backend/requirements.txt

# Regenerate synthetic data (optional, data is pre-generated)
(cd Backend/data_pipeline && python generate_synthetic_data.py)

# Initialize database
(cd Backend && python -c "from app.core.database import init_db; init_db()")

# Seed database
cd Backend && python -c "
from app.core.database import init_db, SessionLocal
from app.services.seed_service import seed_database
init_db()
db = SessionLocal()
seed_database(db)
db.close()
print('Database seeded!')
)

# Start backend server
cd Backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Setup
```bash
# In a new terminal
cd Frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

### 3. Access
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

### 4. Quick Test
```bash
# Start backend
cd Backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Test replay
curl -X POST "http://127.0.0.1:8000/api/v1/replay/step?step_to=37"

# Test AI pipeline
curl "http://127.0.0.1:8000/api/v1/ai/inference/37"

# Test evacuation routes
curl "http://127.0.0.1:8000/api/v1/innovations/evacuation/routes/40"

# Test what-if scenarios
curl "http://127.0.0.1:8000/api/v1/innovations/scenario/list"
```

---

## Testing Each Module

### Module 1: Storm Cell Detection
```bash
curl "http://127.0.0.1:8000/api/v1/ai/inference/37" | python -m json.tool | grep storm_cells
```
Returns detected storm cells, their positions, intensities, and tracking IDs.

### Module 2: Risk Heatmap
```bash
curl "http://127.0.0.1:8000/api/v1/ai/inference/37" | python -m json.tool | grep risk_heatmap
```
Returns per-pixel risk classification with severity distribution.

### Module 3: Nowcasting
```bash
curl "http://127.0.0.1:8000/api/v1/ai/inference/37" | python -m json.tool | grep nowcast
```
Returns 6-hour forecast with trend analysis and per-cell predictions.

### Module 4: Multi-Hazard
```bash
curl "http://127.0.0.1:8000/api/v1/ai/inference/37" | python -m json.tool | grep multi_hazard
```
Returns simultaneous thunderstorm, cloudburst, and flash flood probabilities.

### Module 5: Data Fusion
```bash
curl "http://127.0.0.1:8000/api/v1/ai/inference/37" | python -m json.tool | grep fused_features
```
Returns cross-attention aligned features from all data sources.

### Module 6: Flood Depth (GNN)
```bash
curl "http://127.0.0.1:8000/api/v1/ai/inference/37" | python -m json.tool | grep flood_depth
```
Returns per-street water depth predictions with overflow nodes.

### Module 7: Trust Scoring
```bash
curl "http://127.0.0.1:8000/api/v1/ai/inference/37" | python -m json.tool | grep trust_score
```
Returns confidence level (High/Medium/Low) with historical similarity.

### Module 8: XAI
```bash
curl "http://127.0.0.1:8000/api/v1/ai/inference/37" | python -m json.tool | grep xai_explanation
```
Returns top meteorological drivers (e.g., "High CAPE + rapid CTT drop").

### Module 9: Physics-Informed (PINN)
```bash
curl "http://127.0.0.1:8000/api/v1/innovations/physics/risk/37"
```
Returns shallow water equation risk with conservation validation.

### Module 10: Evacuation Routes
```bash
# Select a cell (0-89) and get routes
curl "http://127.0.0.1:8000/api/v1/innovations/evacuation/routes/40?route_type=walking&alternatives=3"
```
Returns A* paths with safety scores and turn-by-turn directions.

### Module 11: What-If Scenarios
```bash
# List scenarios
curl "http://127.0.0.1:8000/api/v1/innovations/scenario/list"

# Run a scenario
curl -X POST "http://127.0.0.1:8000/api/v1/innovations/scenario/run/DOUBLE_RAINFALL"
```

### Module 12: Live India Data
```bash
# Mumbai detailed (90 cells)
curl "http://127.0.0.1:8000/api/v1/realtime/mumbai"

# India-wide (930 cells)
curl "http://127.0.0.1:8000/api/v1/realtime/india"

# Top risk areas
curl "http://127.0.0.1:8000/api/v1/realtime/summary"
```

---

## Scaling to India

### Current Mumbai Pilot
- 90 cells (10×9 grid)
- 0.02° resolution (~2km)
- 72 timesteps (3 days)

### India Architecture (Ready to Scale)
- 930 cells (31×30 grid)
- 1° resolution (~110km)
- Real-time polling every 5 minutes

### Data Sources for India
| Source | Resolution | Coverage | API Key |
|--------|-----------|----------|---------|
| Open-Meteo | 1km | Global | No key needed |
| INSAT-3D (MOSDAC) | 4km | India | Username/password |
| IMD 0.25° Grid | 27km | India | Free download |
| SRTM DEM | 30m | Global | No key needed |
| RainViewer Radar | 1km | Global | No key needed |

---

## MOSDAC Credentials (Optional)

For real INSAT-3D satellite data:

1. Register at https://mosdac.gov.in/
2. Add to `.env`:
```
MOSDAC_USERNAME=your_email@example.com
MOSDAC_PASSWORD=your_password
```
3. Start the backend **once** with live fetching enabled so a real granule is
downloaded and cached:
```bash
VARUNA_LIVE_FETCH=1 .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```
4. After that the system is fully **offline-first**: it serves the cached real
granule and never fabricates data. With no cache and live mode off, the
satellite panel reports `unavailable_offline` instead of inventing values.

---

## License

This project was developed for Smart India Hackathon 2026.
