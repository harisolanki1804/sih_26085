# VARUNA — AI-Driven Hyperlocal Flood Digital Twin

## What is VARUNA?

VARUNA is an AI-powered early warning system that predicts **where** flash floods will hit, **how deep** the water will be on each street, and **how confident** the prediction is — all **2-6 hours in advance**.

### Core Intelligence
**Spatiotemporal Deep Learning + Physics-Informed AI + Multi-Source Data Fusion + Hyperlocal Risk Mapping + Explainable Forecasting**

### Key Features
- **9 AI/ML Modules** — all trained with PyTorch (214K+ parameters)
- **Real satellite data** — INSAT-3D via MOSDAC API
- **72-hour Digital Twin replay** — rewind any storm, replay step-by-step
- **What-If Simulator** — "what if rainfall doubles?"
- **Works offline** — runs on a laptop, no internet needed during floods

---

## Quick Setup (20 minutes)

### 1. Clone & Install

```bash
git clone <repo-url>
cd sih_26077

# Create virtual environment
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install backend dependencies
cd Backend
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install shap

# Install frontend dependencies
cd ../frontend
npm install
```

### 2. Configure MOSDAC Credentials

Create/edit `Backend/.env`:
```
MOSDAC_USERNAME=your_username
MOSDAC_PASSWORD=your_password
```

**How to get MOSDAC credentials:**
1. Go to https://mosdac.gov.in
2. Click **Register** → create free account
3. Verify email → login
4. Use those credentials in `.env`

### 2b. Real Satellite (MOSDAC) — how it behaves now

MOSDAC products supported (verified against the live API):
```
3RIMG_L1C_SGP   ← INSAT-3DR L1C imagery (preferred, user-selected)
3SIMG_L1B_STD   ← INSAT-3D L1B imagery   (fallback, IMG_* band layout)
3RIMG_L1B_STD   ← INSAT-3DR L1B imagery  (fallback)
3DIMG_L2I_TPW   ← L2 Total Precipitable Water (small quick-win product)
```

When `MOSDAC_USERNAME`/`MOSDAC_PASSWORD` are in `.env`, the backend performs
**one automatic background fetch on startup** (daemon thread) that downloads and
caches the newest real granule. The replay engine itself never blocks on the
network; once the real granule is cached it is served offline forever.
The dashboard INSAT-3D badge then shows **LIVE** — if the network is down it
honestly shows `Offline · fetch once` / `Retrying…` instead of fake data.

Start the backend from the **`Backend/` folder** (so `.env` is read):
```bash
cd Backend
../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
First-time granule downloads are ~90 MB; on a slow link the code auto-resumes
partial downloads and falls back to a smaller product. To force a live refresh
at any time:
```bash
VARUNA_LIVE_FETCH=1 ../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2c. IMDAA Reanalysis (you said you registered — here is exactly what to do)

IMDAA is **not an API** — you download NetCDF files from the NCMRWF portal:

1. Log in at https://rds.ncmrwf.gov.in (your registration works here)
2. Dataset page → select **IMDAA Reanalysis**
3. Pick the **Mumbai box** (18.8–19.3 N, 72.7–73.1 E) and **July 2024**
4. Download the **single-level** hourly files and **pressure-level** 3-hourly
   files (temperature, humidity, U/V wind, geopotential)
5. Place them in:
```
Backend/cache/imdaa/single_level/IMDAA_*.nc
Backend/cache/imdaa/pressure_level/IMDAA_*.nc
```
6. Restart the backend — it auto-detects the files (`source: IMDAA_ACTUAL`).
   Until then, with `VARUNA_LIVE_FETCH=1`, the backend uses genuine Open-Meteo
   ERA5 reanalysis as a real-data fallback (never labelled as IMDAA).

**Prototype shortcut:** you do NOT need full-pressure-level IMDAA for the demo —
single-level CAPE/CIN/humidity + the real rainfall event is enough; the
pressure-level profiles only refine wind-shear. Skip that download if pressed.

### 3. Start the System

**Terminal 1 — Backend:**
```bash
cd Backend
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd Frontend
npm run dev
```

**Open:** http://localhost:5173

---

## Module Status — Real vs Rule-Based (verify yourself)

Every module response carries an `inference_mode` field. After starting the
backend, run:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/replay/step?step_to=36" | \
  python -m json.tool | grep -E "inference_mode|dominant_hazard|trust_level" | sort | uniq -c
```

| Module | Backend path | Status (this repo) | How to verify |
|---|---|---|---|
| 1. Storm cell detection | `storm_cells` | **Trained NN** classifier (checkpoint `storm_cell_detector.pt`) + deterministic wind-advection tracking | `inference_mode: trained_neural_network` |
| 2. Risk heatmap | `risk_heatmap` | **Derived from trained multi-hazard model** (map and alerts always agree) | `inference_mode` = multi-hazard mode |
| 3. Nowcasting 2–6 h | `nowcast` | **Trained ConvLSTM+Transformer** (`spatiotemporal_nowcaster.pt`, bias-corrected to latest obs) | `inference_mode: trained_neural_network` |
| 4. Multi-hazard | `multi_hazard` | **Trained multi-task NN** — thunderstorm, cloudburst, flash flood heads | `inference_mode: trained_neural_network` |
| 5. Cross-source fusion | `fused_features` | Deterministic alignment of INSAT + reanalysis + DEM + QPE signals (no NN checkpoint yet) | `inference_mode: deterministic_source_alignment` |
| 6. Flood depth | `flood_depth` | **Trained NN** (`flood_depth_estimator.pt`), physics-informed fallback only when NN fails | `model: trained_neural_network` |
| 7. Trust scoring | `trust_score` | **Trained autoencoder scorer** + conformal interval calibrated on data | `inference_mode: trained_neural_network` |
| 8. XAI layer | `xai_explanation` | **Trained attention layer** (`xai_attention_layer.pt`) → top-3 drivers per alert | `explanation_method: trained_XAI_attention_model` |
| 9. Crowd-report NLP | `classify_crowd_report` | **Trained embedding classifier** (labels from keyword rules on 20 seed reports — expand with real reports) | `inference_mode: trained_neural_network` |

**Honest caveats** — read before demoing to judges:
- The 72-step feature grid driving the demo is calibrated to the real 26–29 Jul
  2024 Mumbai deluge rainfall (ERA5/Open-Meteo) but the *full* 22-column feature
  vectors (CAPE/CTT/IWV fields per cell) are reconstructed, not observed.
- Model labels are physics-rule pseudo-labels (IMD-style thresholds), which is
  the standard way to prototype when per-cell ground-truth hazard labels don't
  exist. Retrain on months of real data (below) before production.
- Module 5 fusion has no trained cross-attention checkpoint yet — it aligns
  real per-source values deterministically.

---

## How to Test Each Module

### Module 1: Storm Cell Detection & Tracking
- **What it does:** Detects convective cloud clusters and tracks their motion
- **How to test:** Click "Mumbai Replay" → click any cell on the map
- **Look for:** `inference_mode: trained_neural_network` in API response
- **API:** `GET /api/v1/ai/storm-cells/36`

### Module 2: Risk Heatmap Generation
- **What it does:** Classifies every pixel into risk zones (thunderstorm/cloudburst/flash flood)
- **How to test:** Map shows colored cells — red = critical, orange = high, yellow = medium, green = low
- **API:** `GET /api/v1/ai/risk-heatmap/36`

### Module 3: Spatiotemporal Nowcasting
- **What it does:** Forecasts rainfall 2-6 hours ahead using ConvLSTM + Transformer
- **How to test:** Check "What's Coming" panel — bars should vary across timesteps
- **API:** `GET /api/v1/ai/nowcast/36`

### Module 4: Multi-Hazard Prediction
- **What it does:** Simultaneously predicts thunderstorm, cloudburst, and flash flood risk
- **How to test:** Check Overview tab for three hazard probabilities
- **API:** `GET /api/v1/ai/multi-hazard/36`

### Module 5: Cross-Attention Fusion
- **What it does:** Aligns satellite, reanalysis, and DEM data at each grid point
- **How to test:** Check fused features in API response
- **API:** `GET /api/v1/ai/fused-features/36`

### Module 6: Urban Flood Depth Estimation
- **What it does:** Predicts exact water depth (cm) per street segment using GNN
- **How to test:** Click cells with ⚠ markers — shows overflow nodes
- **API:** `GET /api/v1/ai/flood-depth/36`

### Module 7: Forecast Trust Scoring
- **What it does:** Flags when current pattern resembles past high-error cases
- **How to test:** Check Trust tab — shows High/Medium/Low confidence with conformal prediction
- **API:** `GET /api/v1/ai/trust-score/36`

### Module 8: Explainable AI (XAI)
- **What it does:** Shows top meteorological drivers behind every alert
- **How to test:** Check XAI tab — shows rainfall, CAPE, CTT, soil moisture as top drivers
- **API:** `GET /api/v1/ai/xai/36`

### Module 9: Crowd-Report Validation (NLP)
- **What it does:** Classifies citizen flood reports as confirming/denying a predicted flood zone
- **How to test:** Type "heavy rain flooding help" — should classify as CONFIRMS_FLOOD_ZONE
- **API:** `POST /api/v1/ai/crowd-report`

### Replay Engine (All Modules Together)
- **▶ Play** — auto-advances through 72 timesteps (1.5s each)
- **◀/▶** — step-by-step control
- **🔥 Peak** — jump to timestep 36 (maximum storm)
- Each step runs the full 9-module AI pipeline

---

## Training AI Models

All models are pre-trained. To retrain:

```bash
cd Backend
.venv/Scripts/python.exe -c "
from app.services.ai.trainer import VARUNATrainer
trainer = VARUNATrainer()
trainer.train_all(epochs=100, lr=0.001)
"
```

**Training time:** ~2 minutes on CPU

**What gets trained:**
- Multi-Hazard Predictor (145K params)
- Flood Depth Estimator (10K params)
- Trust Scorer (24K params)
- XAI Attention Layer (9K params)
- Storm Cell Detector (3.5K params)
- Crowd NLP Classifier (1.3K params)
- Spatiotemporal Nowcaster (3.4M params)

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | System health check |
| `GET /api/v1/ai/inference/{ts}` | Full 9-module pipeline |
| `GET /api/v1/ai/storm-cells/{ts}` | Storm cell detection |
| `GET /api/v1/ai/risk-heatmap/{ts}` | Risk heatmap |
| `GET /api/v1/ai/nowcast/{ts}` | 6-hour nowcast |
| `GET /api/v1/ai/multi-hazard/{ts}` | Multi-hazard prediction |
| `GET /api/v1/ai/fused-features/{ts}` | Cross-source fusion |
| `GET /api/v1/ai/flood-depth/{ts}` | Flood depth estimation |
| `GET /api/v1/ai/trust-score/{ts}` | Trust scoring |
| `GET /api/v1/ai/xai/{ts}` | XAI explanation |
| `POST /api/v1/ai/crowd-report` | NLP classification |
| `GET /api/v1/ai/model-status` | Model training status |
| `GET /api/v1/replay/status` | Replay engine state |
| `POST /api/v1/replay/step?step_to={ts}` | Jump to timestep |
| `GET /api/v1/alerts` | Active alerts |
| `GET /api/v1/realtime/india` | Live India grid |
| `GET /api/v1/innovations/satellite/latest` | INSAT-3D satellite data |
| `GET /api/v1/innovations/physics/risk/{ts}` | Physics-informed risk |
| `GET /api/v1/innovations/evacuation/routes/{cell}` | Evacuation routes |
| `POST /api/v1/innovations/scenario/simulate` | What-If simulator |
| `POST /api/v1/innovations/chatbot/ask` | NLP chatbot |

---

## Project Structure

```
sih_26077/
├── Backend/                      # All AI/ML + API code
│   ├── app/
│   │   ├── api/v1/endpoints/     # 47 REST API endpoints
│   │   ├── services/
│   │   │   ├── ai/               # 9 AI/ML modules
│   │   │   │   ├── model_architectures.py  # PyTorch models
│   │   │   │   ├── inference_engine.py      # Unified inference
│   │   │   │   ├── trainer.py               # Training pipeline
│   │   │   │   └── data_loader.py           # Data preparation
│   │   │   ├── satellite/        # MOSDAC INSAT-3D fetcher
│   │   │   ├── data/             # IMDAA reanalysis
│   │   │   ├── physics/          # PINN risk model
│   │   │   ├── realtime/         # Open-Meteo live data
│   │   │   └── scenario/         # What-If engine
│   │   └── schemas/              # Pydantic models
│   ├── cache/                    # Runtime caches (IMDAA, MOSDAC, live data)
│   ├── data/                     # Datasets (feature grid, DEM, rainfall, raw)
│   ├── data_pipeline/            # Dataset generation + ML data prep
│   ├── evaluation/               # Evaluation metrics
│   └── models/checkpoints/       # 7 trained PyTorch models
├── Frontend/                     # React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx               # Main layout
│   │   ├── components/
│   │   │   ├── GeoMap.jsx        # Interactive map
│   │   │   ├── MapView.jsx       # Map container
│   │   │   ├── LeftPanel.jsx     # KPI cards
│   │   │   └── IntelligencePanel.jsx  # Right panel
│   │   └── utils/                # API client, helpers
│   └── package.json
└── RUN_GUIDE.md                  # This file
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ECONNREFUSED ::1:8000` | Start backend first: `cd Backend && uvicorn app.main:app --port 8000` |
| `MOSDAC auth failed` | Check username/password in `Backend/.env` |
| `Open-Meteo 400 error` | Already fixed — uses valid parameters only |
| `Open-Meteo 429 error` | Rate limited — wait 5 minutes |
| `Model not loading` | Run training command above |
| `Frontend blank page` | Check browser console for errors |
| `Map not showing` | Ensure Leaflet CSS is loaded in `index.html` |

---

## Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| Open-Meteo | Free API (no key) | Real-time weather |
| MOSDAC/INSAT-3D | Free (credentials) | Satellite data |
| SRTM DEM | Static JSON | Elevation, slope |
| IMDAA | Free (academic) | Reanalysis data |

---

## Tech Stack Summary

**AI/ML:** Python, PyTorch, ConvLSTM, Transformer, SHAP  
**Backend:** FastAPI, SQLAlchemy, SQLite  
**Frontend:** React.js, Leaflet, Vite  
**Data:** Open-Meteo, MOSDAC/INSAT-3D, SRTM DEM  
**Physics:** SCS-CN, Manning's Equation, Shallow Water Eq  
