# VARUNA API & Feature Contract (Day-1 Canonical Specification)

**Owner**: Member 2 (Backend / API + Evaluation Owner)  
**Consumers**: Member 1 (Satellite Downloader / Data Pipeline Owner), Member 3 (ML Model Training Owner)  
**Status**: ACTIVE / FROZEN  
**Version**: 1.0.0  
**Effective Date**: Day 1

---

## 1. Overview & Architecture Dependency

In the VARUNA multi-hazard early-warning architecture, Member 2 establishes the canonical API contracts and data schemas against which the entire team builds.

```text
                 MEMBER 2 (Backend & Evaluation)
                               │
                CANONICAL API & FEATURE CONTRACT
                  /                          \
                 ↓                            ↓
       MEMBER 1 (Data Pipeline)      MEMBER 3 (ML Training)
       - Satellite ingestion (MOSDAC) - PyTorch multi-hazard models
       - DEM & Reanalysis fusion      - Frozen input column tensors
       - Writes canonical columns     - Trains on frozen split
                 \                            /
                  └────────────┬─────────────┘
                               ↓
                 MEMBER 2 (Evaluation Harness)
                 - Frozen time-based test split
                 - 4 metric families (POD/FAR/CSI, RMSE, Brier, Lead Time)
```

**Rule 1**: Member 1 **MUST NOT** invent custom column names. All pipeline outputs must conform to the canonical names defined in Section 2.  
**Rule 2**: Member 3 **MUST** freeze model input dimensions and column order to the exact canonical sequence specified in Section 2.  
**Rule 3**: All API responses and services **MUST** distinguish actual sensor readings from synthetic or physically reconstructed fields using Section 3.

---

## 2. Canonical Feature Specification

The feature vector consists of **26 atmospheric, hydrological, and topographic inputs** plus **5 evaluation ground-truth targets**.

### 2.1 Ordered Input Features (`NUMERIC_FEATURES`)

All model architectures, dataset tensors, and data export JSONs must index these features in the exact order below:

| Index | Canonical Feature Name | Data Type | Units | Physical Domain | Source Instrument / Reanalysis |
|---|---|---|---|---|---|
| 0 | `rainfall_1h_mm` | float | mm/h | Moisture (The Fuel) | IMD Doppler Radar QPE / AWS Gauge |
| 1 | `rainfall_3h_mm` | float | mm | Moisture (The Fuel) | 3-hour rolling precipitation |
| 2 | `rainfall_6h_mm` | float | mm | Moisture (The Fuel) | 6-hour rolling precipitation |
| 3 | `rainfall_24h_mm` | float | mm | Moisture (The Fuel) | 24-hour antecedent rainfall |
| 4 | `soil_moisture_pct` | float | % | Moisture (The Fuel) | ERA5-Land / IMDAA top-layer moisture |
| 5 | `soil_saturation_factor` | float | [0.0 - 1.0] | Moisture (The Fuel) | Physical bucket infiltration model |
| 6 | `iwv_mm` | float | mm | Moisture (The Fuel) | INSAT-3D/3DR Water Vapor (6.7-7.2 µm) |
| 7 | `cape_instability_jkg` | float | J/kg | Instability (The Energy) | NCMRWF IMDAA / WRF convective energy |
| 8 | `cin_jkg` | float | J/kg | Instability (The Energy) | Convective Inhibition energy |
| 9 | `lifted_index` | float | °C | Instability (The Energy) | Atmospheric thermal stability index |
| 10 | `cloud_top_temp_celsius` | float | °C | Instability (The Energy) | INSAT-3D TIR-1 Channel (10.8 µm) |
| 11 | `ctt_drop_rate_c_per_hr` | float | °C/h | Instability (The Energy) | Temporal rate of cloud-top cooling |
| 12 | `wind_speed_10m_kmh` | float | km/h | Kinematics (The Trigger) | 10m horizontal wind velocity |
| 13 | `wind_direction_10m_deg` | float | degrees | Kinematics (The Trigger) | Wind vector azimuth (0-360°) |
| 14 | `u_wind_ms` | float | m/s | Kinematics (The Trigger) | Zonal eastward wind component |
| 15 | `v_wind_ms` | float | m/s | Kinematics (The Trigger) | Meridional northward wind component |
| 16 | `wind_gusts_kmh` | float | km/h | Kinematics (The Trigger) | Peak 3-second gust speed |
| 17 | `vertical_wind_shear_ms` | float | m/s | Kinematics (The Trigger) | 0-6 km bulk convective wind shear |
| 18 | `low_level_convergence` | float | 10⁻⁵ s⁻¹ | Kinematics (The Trigger) | Horizontal kinematic moisture convergence |
| 19 | `elevation_m` | float | meters | Topography (The Catalyst) | SRTM 30m Digital Elevation Model |
| 20 | `slope_deg` | float | degrees | Topography (The Catalyst) | First derivative of DEM surface |
| 21 | `runoff_coefficient` | float | [0.0 - 1.0] | Topography (The Catalyst) | Land-use land-cover runoff factor |
| 22 | `effective_runoff_mm_hr` | float | mm/h | Topography (The Catalyst) | Rational runoff: $C \times I$ |
| 23 | `drainage_outfall_dist_m` | float | meters | Topography (The Catalyst) | Geodesic distance to Mithi / sea outfall |
| 24 | `retention_index` | float | [0.0 - 5.0] | Topography (The Catalyst) | Morphometric depression accumulation factor |
| 25 | `tidal_backwater_factor` | float | [0.8 - 3.0] | Topography (The Catalyst) | Sluice gate blockage multiplier ($f(Tide)$) |

### 2.2 Ground-Truth Target Columns (`TARGET_COLUMNS`)

| Canonical Target Name | Data Type | Range | Meaning |
|---|---|---|---|
| `target_observed_flood_depth_cm` | float | $\ge 0.0$ cm | Inundation water depth (continuous regression) |
| `target_severity_class` | int | 0 to 3 | Hazard class: 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL |
| `target_flash_flood_flag` | int | 0 or 1 | Active flash flood binary ground truth |
| `target_cloudburst_flag` | int | 0 or 1 | Rainfall $> 65$ mm/hr convective cloudburst flag |
| `target_waterlogging_flag` | int | 0 or 1 | Urban drainage ponding / waterlogging flag |

---

## 3. Data Provenance & Realism Classification

Every data stream and response payload must be classified according to its operational realism level:

| Classification | Meaning | Production Guidance |
|---|---|---|
| `REAL` | Directly acquired from physical satellite instruments, radar, AWS weather stations, or official GIS DEMs without synthetic alterations. | Mandatory for live production mode (`VARUNA_LIVE_FETCH=1`). |
| `RECONSTRUCTED / REGRESSED` | Derived through calibrated numerical, hydrodynamic, or geospatial transfer functions (e.g. spatial super-resolution from 4km to 100m, DEM slope derivation, kinematic shear computation). | Acceptable for spatial densification; confidence penalties must be reported in `trust_score`. |
| `SYNTHETIC` | Generated for extreme stress-testing, historical benchmark re-enactment, or validation splits where physical sensor recording was unavailable. | Permitted only for offline simulation replay, scenario stress testing, and training augmentation. Must be marked in response metadata. |

### Provenance Mapping by Field

```text
[REAL]                  elevation_m, slope_deg, drainage_outfall_dist_m (SRTM 30m)
[REAL]                  cloud_top_temp_celsius, iwv_mm (INSAT-3D/3DR MOSDAC L2)
[REAL]                  cape_instability_jkg, cin_jkg, u_wind_ms, v_wind_ms (IMDAA Reanalysis)
[RECONSTRUCTED]         ctt_drop_rate_c_per_hr, low_level_convergence, vertical_wind_shear_ms
[RECONSTRUCTED]         soil_saturation_factor, effective_runoff_mm_hr, tidal_backwater_factor
[RECONSTRUCTED]         Super-resolved 100m satellite fields (from coarse 4km pixels)
[SYNTHETIC]             Simulated extreme cloudburst deluges (historical calibration scenarios)
```

---

## 4. API Endpoints & Exact JSON Response Shapes

### 4.1 Features API: `GET /api/v1/features/latest`

Returns the active spatial grid feature matrix.

- **Query Parameters**:
  - `region_code` (string, optional, default `"IN-MH-BOM-01"`)
- **HTTP Status**: `200 OK`
- **Response Structure**:

```json
{
  "region_id": "reg-mumbai-01",
  "region_code": "IN-MH-BOM-01",
  "timestamp": "2022-07-05T08:00:00Z",
  "total_cells": 90,
  "cells": [
    {
      "id": "feat-cell-0",
      "region_id": "reg-mumbai-01",
      "timestamp": "2022-07-05T08:00:00Z",
      "cell_index": 0,
      "cell_lat": 18.98,
      "cell_lon": 72.80,
      "rainfall_1h_mm": 12.4,
      "rainfall_3h_mm": 24.8,
      "rainfall_6h_mm": 38.0,
      "rainfall_24h_mm": 85.2,
      "soil_moisture_pct": 78.4,
      "soil_saturation_factor": 0.78,
      "cape_instability_jkg": 2450.0,
      "cloud_top_temp_celsius": -52.4,
      "ctt_drop_rate_c_hr": -8.5,
      "wind_speed_10m_kmh": 42.0,
      "wind_direction_10m_deg": 240.0,
      "wind_u_ms": -8.4,
      "wind_v_ms": -5.2,
      "wind_gusts_kmh": 58.0,
      "elevation_m": 4.2,
      "slope_deg": 1.1,
      "runoff_coefficient": 0.85,
      "effective_runoff_mm_hr": 10.54,
      "drainage_outfall_dist_m": 420.0,
      "is_depression_bowl": true,
      "tide_height_m": 4.2,
      "is_high_tide_locked": true,
      "target_observed_flood_depth_cm": 25.4,
      "target_severity_class": 2,
      "target_flash_flood_flag": 1,
      "target_cloudburst_flag": 0,
      "target_waterlogging_flag": 1,
      "created_at": "2026-09-15T09:30:00Z"
    }
  ]
}
```

---

### 4.2 AI Pipeline: `GET /api/v1/ai/inference/{timestep_id}`

Runs all 9 AI/ML modules on the requested timestep.

- **Path Parameters**:
  - `timestep_id` (integer, e.g. `1` to `72`)
- **HTTP Status**: `200 OK`
- **Response Structure**:

```json
{
  "timestep_id": 1,
  "timestamp": "2022-07-05T00:00:00Z",
  "storm_cells": {
    "total_cells_detected": 1,
    "cells": [
      {
        "cell_index": 42,
        "lat": 19.065,
        "lon": 72.880,
        "confidence": 0.825,
        "storm_type": "convective",
        "motion_vector": {
          "dx_deg_hr": 0.045,
          "dy_deg_hr": 0.028
        },
        "area_km2": 16.5,
        "peak_echo_top_km": 12.4,
        "track_id": "SC-42",
        "lifetime_steps": 3,
        "trend": "growing"
      }
    ],
    "peak_storm_intensity": 84.5,
    "tracking_id": "VARUNA-TRK-001"
  },
  "risk_heatmap": {
    "resolution": "1km",
    "total_pixels": 90,
    "risk_class_distribution": {
      "LOW": 45,
      "MEDIUM": 25,
      "HIGH": 15,
      "CRITICAL": 5
    },
    "max_pixel_risk": 88.5,
    "mean_pixel_risk": 42.1,
    "heatmap": [
      {
        "cell_index": 0,
        "lat": 18.98,
        "lon": 72.80,
        "dominant_risk_class": "MEDIUM",
        "risk_class_probs": { "LOW": 0.2, "MEDIUM": 0.6, "HIGH": 0.15, "CRITICAL": 0.05 },
        "pixel_risk_score": 48.2
      }
    ]
  },
  "nowcast": {
    "horizon_hours": 6,
    "temporal_context_timesteps": 12,
    "trend": "intensifying",
    "trend_rate_mm_hr_per_step": 3.8,
    "forecasts": [
      {
        "forecast_hour": 1,
        "predicted_avg_rainfall_mm_hr": 35.2,
        "predicted_risk_level": "HIGH",
        "confidence": 0.88,
        "trend": "rising"
      }
    ]
  },
  "multi_hazard": {
    "total_cells": 90,
    "aggregate_thunderstorm_prob": 0.72,
    "aggregate_cloudburst_prob": 0.45,
    "aggregate_flash_flood_prob": 0.68,
    "max_severity_score": 88.5,
    "max_predicted_depth_cm": 64.2,
    "inference_mode": "trained_neural_network",
    "cell_predictions": []
  },
  "fused_features": {
    "fusion_method": "cross_source_value_fusion",
    "data_sources": [
      "insat3d_satellite",
      "imdaa_reanalysis",
      "srtm_dem",
      "qpe_rainfall"
    ],
    "source_contributions": {
      "insat3d_satellite": 0.0714,
      "imdaa_reanalysis": 0.3083,
      "srtm_dem": 0.4042,
      "qpe_rainfall": 0.0310
    },
    "global_agreement_score": 0.684,
    "per_cell_fusion": [
      {
        "cell_index": 0,
        "fused_risk_vector": [0.08, 0.32, 0.42, 0.05],
        "cross_source_agreement": 0.63,
        "dominant_source": "srtm_dem"
      }
    ],
    "inference_mode": "deterministic_source_alignment"
  },
  "flood_depth": {
    "total_nodes": 90,
    "max_water_depth_cm": 64.2,
    "mean_water_depth_cm": 14.8,
    "overflow_nodes_count": 8,
    "critical_bottlenecks": [],
    "node_estimates": [],
    "inference_mode": "trained_neural_network"
  },
  "trust_score": {
    "trust_score": 86.4,
    "trust_level": "HIGH_CONFIDENCE",
    "components": {
      "cross_source_agreement": 88.2,
      "nowcast_confidence": 84.0,
      "storm_detection_consistency": 90.5,
      "depth_severity_agreement": 82.9
    },
    "is_anomalous_pattern": false,
    "similar_historical_cases": [
      {
        "case_id": "MUMBAI_2005_PEAK",
        "similarity": 0.88,
        "historical_error_pct": 6.2,
        "note": "High rainfall with concurrent high tide gate closure"
      }
    ],
    "uncertainty_margin_pct": 7.5,
    "conformal_prediction": {
      "alpha": 0.1,
      "coverage_guarantee": "90%",
      "lower_bound": 78.9,
      "upper_bound": 93.9
    },
    "confidence_guarantee": "Statistical 90% coverage interval [78.9, 93.9]"
  },
  "xai_explanation": {
    "explanation_method": "integrated_gradients_xai",
    "feature_importance_global": {
      "rainfall_1h_mm": 0.34,
      "soil_saturation_factor": 0.28,
      "cape_instability_jkg": 0.18,
      "elevation_m": 0.12,
      "tide_height_m": 0.08
    },
    "total_cells_analyzed": 90,
    "high_risk_cells_count": 12,
    "cell_explanations": [],
    "global_summary": "Precipitation and soil moisture account for 62% of regional risk."
  }
}
```

#### Conformal interval semantics (read before rendering uncertainty)

The `conformal_prediction` block puts a **90% coverage interval on the risk/severity
score (0-100 scale)**, not on rainfall or depth. Its half-width is the residual
quantile measured on the **calibrate partition (timesteps 33-38, n=540 cell-level
pairs)** by the trainer, stored per target in `models/conformal_calibration.json`.

- A residual quantile is only meaningful in the unit it was measured in. Every
  writer previously shared one calibration file, so the last calibration to run
decided the interval for *every* quantity: the file held the rainfall calibration
  (mm/hr, n=6) while the API applied it to severity scores, producing half-widths
  of ~53 on a 100-point scale (every cell labelled `HIGH_UNCERTAINTY`).
- Calibrations are now keyed by target (`risk_severity_score`, `flood_depth_cm`).
  The evaluation suite measures coverage with `persist=False` and therefore cannot
  overwrite the served calibration.
- If no calibration exists for the requested target, the predictor reports itself
  uncalibrated and uses `_fallback_interval` rather than silently serving an
  interval derived from an unrelated quantity.
- Do not present `uncertainty_labels` as a per-cell confidence signal unless they
  vary across the payload; a constant label means the interval is wider than the
  signal, which is a finding, not a UI detail.

---

### 4.3 Active Learning: `POST /api/v1/ai/crowd-report`

Validates crowd-sourced field reports against model predictions.

- **Request Body**:

```json
{
  "report_text": "Heavy flood and waterlogging near Kurla station, water is deep and rising fast!",
  "predicted_flood_lat": 19.065,
  "predicted_flood_lon": 72.880
}
```

- **HTTP Status**: `200 OK`
- **Response Structure**:

```json
{
  "report_text": "Heavy flood and waterlogging near Kurla station, water is deep and rising fast!",
  "classification": "CONFIRMS_FLOOD_ZONE",
  "confidence": 0.88,
  "detected_location": "kurla",
  "confirm_score": 4,
  "deny_score": 0,
  "unrelated_score": 0,
  "actionable": true,
  "inference_mode": "trained_neural_network",
  "flagged_for_retraining": false,
  "model_risk_at_location": 84.3,
  "agrees_with_model": true,
  "probabilities": {
    "confirm": 0.88,
    "deny": 0.08,
    "unrelated": 0.04
  }
}
```

---

### 4.4 Simulation Replay API

#### `GET /api/v1/replay/status`

- **Response Structure**:

```json
{
  "current_timestep": 36,
  "timestamp": "2022-07-05T12:00:00Z",
  "phase": "Peak Deluge & High Tide",
  "max_risk_score": 89.2,
  "active_alerts_count": 18,
  "simulation_complete": false
}
```

#### `POST /api/v1/replay/step?step_to={step}`

- **Response Structure**:

```json
{
  "timestep_id": 36,
  "timestamp": "2022-07-05T12:00:00Z",
  "phase": "Peak Deluge & High Tide",
  "max_risk_score": 89.2,
  "new_alerts_count": 4,
  "total_active_alerts": 18
}
```

#### `POST /api/v1/replay/reset`

- **Response Structure**: Resets simulation to timestep 1 and returns initial `ReplayStatus`.

---

### 4.5 Alerts & Explainability API

#### `GET /api/v1/alerts?is_active=true&severity=CRITICAL`

- **Response Structure**: `List[AlertRead]`

```json
[
  {
    "id": "alt-36-cell-42",
    "region_id": "reg-mumbai-01",
    "timestamp": "2022-07-05T12:00:00Z",
    "cell_index": 42,
    "lat": 19.065,
    "lon": 72.880,
    "severity": "CRITICAL",
    "alert_type": "COMPOUND_FLOOD",
    "risk_score_total": 86.5,
    "rainfall_score_contrib": 38.5,
    "soil_saturation_score_contrib": 24.2,
    "topography_score_contrib": 12.8,
    "atmospheric_instability_score_contrib": 11.0,
    "lead_time_hours": 3.5,
    "is_active": true,
    "acknowledged_by": null,
    "acknowledged_at": null,
    "created_at": "2026-09-15T09:30:00Z"
  }
]
```

#### `GET /api/v1/alerts/{alert_id}/explain`

- **Response Structure**:

```json
{
  "alert": {
    "id": "alt-36-cell-42",
    "severity": "CRITICAL",
    "risk_score": 86.5,
    "reasoning_summary": "CRITICAL compound flood hazard detected: concurrent 65mm/h deluge and high-tide sluice lock."
  },
  "score_breakdown": {
    "total_score": 86.5,
    "rainfall_component": 38.5,
    "soil_saturation_component": 24.2,
    "topography_component": 12.8,
    "instability_component": 11.0
  },
  "trust_breakdown": {
    "trust_score": 93.8,
    "trust_level": "HIGH_CONFIDENCE",
    "sensor_agreement": 94.0,
    "uncertainty_margin_pct": 6.2,
    "key_drivers": ["INSAT-3D CTT drop < -55°C", "IMD Radar 65mm/hr", "High Tide Lock 4.2m"]
  },
  "emergency_playbook": {
    "action_summary": "Deploy dewatering pumps to Kurla station outfall; redirect traffic via SCLR.",
    "evacuation_priority": "IMMEDIATE"
  }
}
```

---

### 4.6 Satellite Innovation API

#### `GET /api/v1/innovations/satellite/insat3d`

- **HTTP Status**: `200 OK`
- **Response Structure**:

```json
{
  "source": "MOSDAC_INSAT3D",
  "provenance": "REAL",
  "timestamp": "2022-07-05T08:00:00Z",
  "channels": {
    "thermal_infrared_1_celsius": {
      "channel": "TIR1_10.8um",
      "resolution_km": 4.0,
      "min": -72.4,
      "max": 28.1,
      "mean": -34.6
    },
    "water_vapor_iwv_mm": {
      "channel": "WV_6.8um",
      "resolution_km": 8.0,
      "min": 32.0,
      "max": 74.5,
      "mean": 58.2
    }
  },
  "spatial_bounds": {
    "min_lat": 18.9,
    "max_lat": 19.3,
    "min_lon": 72.7,
    "max_lon": 73.1
  }
}
```

#### `GET /api/v1/innovations/satellite/cmv`

- **Response Structure**: Cloud Motion Vector fields (`provenance: RECONSTRUCTED / REGRESSED`) derived from optical flow between consecutive 30-minute INSAT frames.

#### `GET /api/v1/innovations/satellite/super-resolution`

- **Response Structure**: Downscaled satellite infrared grid upscaled from native 4 km to 100 m using physics-guided spatial interpolation (`provenance: RECONSTRUCTED / REGRESSED`).

### 4.7 Evaluation Metrics API

Serves the four-family evaluation artifacts. Added because nothing exposed them,
so the dashboard's skill panel had its numbers hardcoded. All endpoints are
read-only and every response carries the artifact's modification time, so a
stale report is visible rather than presented as current.

#### `GET /api/v1/metrics/skill`

Chart-ready summary; the shape the skill panel should bind to.

```json
{
  "split": { "type": "frozen_time_based", "description": "train 1-32, calibrate 33-38, test 39-72" },
  "generated_at": "2026-09-18T12:07:25Z",
  "hazard_skill": [
    { "hazard": "Flood", "lead_time": "+2h", "pod": 80.0, "far": 42.9,
      "csi": 50.0, "accuracy": 88.2, "positives": 5,
      "defined": true, "undefined_reason": null }
  ],
  "regression": { "flood_depth_cm": { "RMSE": 0.0, "MAE": 0.0, "R2": 0.0 } },
  "probabilistic": {
    "aggregate": { "brier_score": 0.0, "expected_calibration_error_ece": 0.0 },
    "per_head": { "flash_flood": { "brier_score": 0.0, "ece": 0.0 } },
    "conformal_per_target": { "flood_depth_cm": { "q": 0.0, "coverage_pct": 0.0, "target_coverage_pct": 90.0, "meets_guarantee": false } }
  },
  "operational": {
    "lead_time_to_first_alert_hours": null,
    "lead_time_note": "...",
    "first_alert_timestep": 39,
    "peak_risk_timestep": 39,
    "observed_peak_timestep": 39,
    "replay_lead_time_hours": 3,
    "false_alarms_per_week": 29.6,
    "evacuation_detour_overhead_pct": 4.9
  },
  "nowcast": {
    "per_lead_time": { "+6h": { "mae_mm_hr": 0.0, "persistence_mae_mm_hr": 0.0, "mae_skill_vs_persistence_pct": 0.0 } },
    "inference_modes": { "trained_neural_network": 19, "trend_aware_extrapolation": 15 },
    "neural_nowcast_used": true
  }
}
```

**Rendering rules for the UI** (these are correctness requirements, not style):

1. A hazard head with no positive events in the scored window returns `pod`/`csi` as `null` with an `undefined_reason`. Render that as "undefined", **never as 0%**.
2. Before attributing a forecast to the ConvLSTM, read `nowcast.inference_modes`. `neural_nowcast_used: false` means every forecast came from the extrapolation fallback.
3. `operational.lead_time_to_first_alert_hours` is `null` when the observed peak falls outside the scored window (it does here); show `lead_time_note` instead of inventing a figure.

#### `GET /api/v1/metrics/evaluation`

Verbatim four-family report, plus `artifact` and `artifact_modified_utc`.

#### `GET /api/v1/metrics/nowcast`

Verbatim per-lead-time nowcast report, including the persistence reference.

#### `GET /api/v1/metrics/baseline`

The frozen regression baseline: `frozen_at`, `tolerance`, `tracked_metric_count` and the tracked metrics.

---

## 5. Team Handoff Directives

### 5.1 Directive for Member 1 (Data Pipeline & Satellite Downloader)

1. **Use Canonical Feature Names**: In all output datasets and JSON generators, write the exact 26 column names defined in Section 2.1. Do not use abbreviations like `rain_1h`, `temp`, or `elevation`.
2. **Flag Data Provenance**: In any generated feature file, include the provenance tag (`REAL`, `RECONSTRUCTED`, or `SYNTHETIC`) for each channel.
3. **Handle Missing Sensor Readings**: Missing satellite pixels must be marked `null` or NaN; downstream fallback handles interpolation. Never insert random placeholder values.

### 5.2 Directive for Member 3 (ML Model Training)

1. **Freeze Feature Input Order**: All neural networks (`MultiHazardPredictor`, `FloodDepthEstimator`, `ConvLSTM`) must take inputs formatted exactly according to the 26-element index order in Section 2.1.
2. **Evaluation Split Discipline**: Train only on the frozen training split, **Timesteps 1 to 32**. Calibrate on **33 to 38**; never train on or shuffle with **39 to 72**, the frozen test split. (This supersedes the earlier 1-50 / 51-62 / 63-72 boundaries: every flash-flood and cloudburst event sits in timesteps 21-47, so a trailing test window had zero positives and its POD was undefined. The current boundaries keep both event types in all three partitions while preserving strict temporal order.)
3. **Output Shape Compliance**: Models must output probabilities for the exact class names defined in `TARGET_COLUMNS` (`target_severity_class`, `target_flash_flood_flag`, `target_cloudburst_flag`, `target_waterlogging_flag`).
4. **Reproducible Retrains**: Training is seeded (`DEFAULT_SEED = 42`, overridable with `--seed`); two retrains on the frozen split produce identical results, and the seed is recorded in `training_metadata.json`. Keep it that way — the frozen-baseline regression guard compares metric to metric, so a non-deterministic retrain makes it report drift on unchanged code and it stops detecting real regressions.
5. **Calibrate the Served Quantity**: Persist conformal calibrations per target (`ConformalPredictor(target=...)`). The API serves `risk_severity_score`, so that target's calibration must come from the trainer; evaluation runs use `persist=False` and must not overwrite it.

---

## 6. Verification Status

| Endpoint | Method | Tested Status | Schema Verification |
|---|---|---|---|
| `/api/v1/ai/inference/{timestep_id}` | GET | **HTTP 200 PASSED** | Verified with `source_contributions` |
| `/api/v1/ai/crowd-report` | POST | **HTTP 200 PASSED** | Verified with scores, actionable, location |
| `/api/v1/features/latest` | GET | **HTTP 200 PASSED** | Verified with 90-cell spatial grid |
| `/api/v1/replay/step` | POST | **HTTP 200 PASSED** | Verified with step jump & alert generation |
| `/api/v1/alerts` | GET | **HTTP 200 PASSED** | Verified with score decomposition & trust metrics |
| `/api/v1/metrics/skill` | GET | **HTTP 200 PASSED** | Verified: 15 hazard-skill rows, nulls + reasons for undefined heads |
| `/api/v1/metrics/evaluation` | GET | **HTTP 200 PASSED** | Verified: four-family report with artifact timestamp |
| `/api/v1/metrics/nowcast` | GET | **HTTP 200 PASSED** | Verified: per-lead-time report with `inference_modes` |
| `/api/v1/metrics/baseline` | GET | **HTTP 200 PASSED** | Verified: frozen baseline, 92 tracked metrics, tolerance |
