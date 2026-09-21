"""
Synthetic satellite granule reconstruction -- calibrated on REAL granules.
=========================================================================

Why this exists
---------------
The MOSDAC ``3DIMG`` L2 archive ends 2024-06-18, and the replay window is
2022-07-05..2022-07-07. Only a handful of real granules were pulled before
the demo deadline, so most of the window has no observation to ingest.

This module fills the *missing* slots with physically-constrained
reconstructions that are **calibrated on the real granules actually
downloaded**, so the window stays internally consistent instead of turning
into constant placeholders.

Honesty contract (non-negotiable)
---------------------------------
* Real granules are **never overwritten**. A slot that already holds a real
  decoded granule is left exactly as it is.
* Every payload written here carries ``is_real_data: false``,
  ``gap_filled: true``, ``provenance: "SYNTHETIC"`` and a
  ``calibration_basis`` block naming the real granules it was fitted from.
* Nothing in this module claims to be telemetry. The downstream merge
  surfaces the synthetic origin per field, and the UI badge stays
  PARTIAL/FALLBACK rather than LIVE.

Calibration actually used (measured from the real cache, not invented)
---------------------------------------------------------------------
* ``spatial_anomaly[cell]`` -- per-cell mean CTT minus the real global mean
  across the downloaded frames (real observed -45.6 .. -26.0 degC spread).
* the value envelope, anchored on the real observed range (-51.3 .. -17.2 degC).
* the storm envelope and per-cell rainfall from the feature table, so the
  reconstruction agrees with the rest of the timeline.

Cross-field physics
-------------------
The four reconstructed fields are mutually consistent by construction:

* cloud mask   -- derived from the reconstructed cloud-top temperature
* rain rate    -- scales with how cold the cloud top is (cold cloud = deep
                  convection), anchored on the table's rainfall field
* wind shear   -- grows with storm intensity and cold-cloud fraction
* cooling rate -- computed by the *real* derivation path from consecutive
                  reconstructed CTP frames, so the maths is identical to the
                  real-data path

Usage::

    python -m data_pipeline.synthetic_fill --report    # what is missing
    python -m data_pipeline.synthetic_fill             # fill missing slots
    python -m data_pipeline.synthetic_fill --force     # rewrite synthetic fills
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
backend_dir = os.path.join(root_dir, "Backend")

for path in [backend_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from app.core.timeline_config import EVENT_END_DT, EVENT_START_DT, EVENT_STEP_HOURS  # noqa: E402

CACHE_DIR = os.path.join(backend_dir, "cache", "satellite_cache")
FEATURE_TABLE = os.path.join(backend_dir, "data", "feature_grid_timeseries.json")

GRID_CELLS = 90

# Half-hourly derivation lattice (satellite frames arrive ~30 min apart).
# Must span every table step, including the final one at EVENT_END_DT.
STEP = timedelta(hours=EVENT_STEP_HOURS) / 2
WINDOW_START = EVENT_START_DT
WINDOW_END = EVENT_END_DT

REAL_PRODUCT = "3DIMG_L2B_CTP"

# Every product the contract requests. Only CTP was pulled for real; the rest
# are reconstructed so the contract has all four derived fields populated.
PRODUCT_SPEC: Dict[str, Dict[str, str]] = {
    "3DIMG_L2B_CTP": {"variable": "CTT", "units": "degC", "frame": "ctt"},
    "3DIMG_L2G_WDP": {"variable": "SHEAR", "units": "m/s", "frame": "wind_speed_shear"},
    "3DIMG_L2B_UTH": {"variable": "UTH", "units": "%", "frame": "uth"},
    "3DIMG_L2B_HEM": {"variable": "HEM", "units": "mm/hr", "frame": "qpe_rain_rate"},
    "3DIMG_L2B_CMK": {"variable": "CMK", "units": "class", "frame": "cloud_mask"},
}

# Extra variables carried in the decoder's ``derived`` block. WDP is not a
# single field: alongside shear it yields low-level convergence, which the real
# decoder computes and the table now consumes.
SECONDARY_FIELDS: Dict[str, Dict[str, str]] = {
    "3DIMG_L2G_WDP": {"low_level_convergence_1e5": "low_level_convergence"},
}

# Cloud-top temperature envelope, anchored on the REAL observed range
# (-51.27 .. -17.20 degC across the downloaded frames). Warm/clear at the
# weak end, deep convective at the strong end.
CTT_WARM_ANCHOR = -13.0
CTT_COLD_ANCHOR = -46.0
CTT_CLAMP = (-58.0, -10.0)

# Real-granule statistics, measured (see module docstring). Refreshed from the
# cache at runtime; these are only fallbacks if the cache is empty.
FALLBACK_CTT_STD = 7.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cache_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def _stable_rng(*parts: Any, seed: int = 20220705) -> random.Random:
    """Deterministic RNG so repeated runs produce identical output."""
    digest = hashlib.sha256(("|".join(str(p) for p in parts) + f"|{seed}").encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _list_lattice() -> List[datetime]:
    slots: List[datetime] = []
    cursor = WINDOW_START
    while cursor <= WINDOW_END:
        slots.append(cursor)
        cursor += STEP
    return slots


# ---------------------------------------------------------------------------
# Real-granule calibration
# ---------------------------------------------------------------------------
def load_real_calibration(cache_dir: str = CACHE_DIR) -> Dict[str, Any]:
    """Fit the reconstruction prior on the real decoded granules on disk.

    Returns per-cell spatial anomalies plus the observed value envelope. This
    is the whole point: the synthetic field inherits the *real* spatial
    structure rather than being a smooth invented surface.
    """
    frames: List[List[float]] = []
    frame_times: Dict[datetime, List[float]] = {}
    for fname in sorted(glob.glob(os.path.join(cache_dir, f"{REAL_PRODUCT}_*.json"))):
        try:
            with open(fname, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        # Only genuinely real granules feed the prior.
        if payload.get("is_real_data") is not True:
            continue
        values = payload.get("grid_values")
        if not isinstance(values, list) or len(values) != GRID_CELLS:
            continue
        values = [float(v) for v in values]
        frames.append(values)
        stamp = payload.get("timestamp")
        if isinstance(stamp, str):
            try:
                frame_times[datetime.fromisoformat(stamp)] = values
            except ValueError:
                pass

    if not frames:
        return {
            "n_real_frames": 0,
            "spatial_anomaly": [0.0] * GRID_CELLS,
            "observed_mean": -36.1,
            "observed_std": FALLBACK_CTT_STD,
            "observed_min": -51.3,
            "observed_max": -17.2,
            "real_frames": {},
        }

    per_cell_mean = [
        sum(frame[i] for frame in frames) / len(frames) for i in range(GRID_CELLS)
    ]
    global_mean = sum(per_cell_mean) / GRID_CELLS
    all_values = [v for frame in frames for v in frame]
    variance = sum((v - global_mean) ** 2 for v in all_values) / len(all_values)

    return {
        "n_real_frames": len(frames),
        "spatial_anomaly": [mean - global_mean for mean in per_cell_mean],
        "observed_mean": global_mean,
        "observed_std": math.sqrt(variance),
        "observed_min": min(all_values),
        "observed_max": max(all_values),
        "real_frames": frame_times,
    }


# Which real product supplies which reconstructed field, and where in the
# decoded payload those values live.
REAL_FIELD_SOURCES: Dict[str, Tuple[str, Optional[str]]] = {
    "wind_speed_shear": ("3DIMG_L2G_WDP", None),
    "low_level_convergence": ("3DIMG_L2G_WDP", "low_level_convergence_1e5"),
    "uth": ("3DIMG_L2B_UTH", None),
    "qpe_rain_rate": ("3DIMG_L2B_HEM", None),
}


def load_real_field_frames(
    cache_dir: str = CACHE_DIR,
) -> Dict[str, Dict[datetime, List[float]]]:
    """Index real decoded granules by the reconstructed field they inform.

    CTT was the only field ever fitted to real granules, so the other
    reconstructions ran on amplitudes of their own invention. Where a real
    granule exists for one of them the mismatch is abrupt: measured at the
    MOSDAC-observed slot, reconstructed rain rate stepped from ~35 mm/hr down
    to the real 0.08 mm/hr and back up to ~49 mm/hr the following hour, and
    shear from ~12.6 m/s to the real 1.2 m/s. Those are artefacts of the fill,
    not weather, so the same observation nudging already applied to CTT is
    applied here.
    """
    frames: Dict[str, Dict[datetime, List[float]]] = {}
    for field_name, (product_id, derived_key) in REAL_FIELD_SOURCES.items():
        for fname in sorted(glob.glob(os.path.join(cache_dir, f"{product_id}_*.json"))):
            try:
                with open(fname, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("is_real_data") is not True:
                continue
            if derived_key:
                values = (payload.get("derived") or {}).get(derived_key)
            else:
                values = payload.get("grid_values")
            if not isinstance(values, list) or len(values) != GRID_CELLS:
                continue
            stamp = payload.get("timestamp")
            if not isinstance(stamp, str):
                continue
            try:
                when = datetime.fromisoformat(stamp)
            except ValueError:
                continue
            frames.setdefault(field_name, {})[when] = [float(v) for v in values]
    return frames


def load_storm_envelope(feature_path: str = FEATURE_TABLE) -> Dict[str, Any]:
    """Read the per-step storm envelope and per-cell rainfall from the table.

    Anchoring on the table keeps the reconstruction consistent with the
    timeline the UI and the models already see.
    """
    with open(feature_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    timesteps = raw["timesteps"] if isinstance(raw, dict) else raw
    steps: List[Dict[str, Any]] = []
    for ts in timesteps:
        if not isinstance(ts, dict):
            continue
        cells = ts.get("features") or []
        rain = [
            float(c.get("rainfall_1h_mm", 0.0) or 0.0)
            for c in cells
            if isinstance(c, dict)
        ]
        if len(rain) != GRID_CELLS:
            rain = [0.0] * GRID_CELLS
        steps.append(
            {
                "avg_rainfall_1h_mm": float(ts.get("avg_rainfall_1h_mm", 0.0) or 0.0),
                "rainfall_1h_mm": rain,
            }
        )

    peak = max((s["avg_rainfall_1h_mm"] for s in steps), default=0.0)
    return {"steps": steps, "peak_avg_rainfall": peak if peak > 0 else 1.0}


def _nearest_step_index(slot: datetime) -> int:
    """Map a half-hourly slot onto its nearest hourly table step."""
    hours = (slot - EVENT_START_DT).total_seconds() / 3600.0
    return int(round(hours))


# ---------------------------------------------------------------------------
# Field synthesis
# ---------------------------------------------------------------------------
def _interpolate_envelope(slot: datetime, envelope: Dict[str, Any]) -> Tuple[float, List[float]]:
    """Linearly interpolate storm strength and rainfall between table steps.

    Interpolating (rather than snapping to the nearest hour) keeps the
    reconstructed frames temporally smooth, so the cooling rate the derivation
    computes from consecutive frames reflects the storm's real evolution
    instead of a step-boundary jump.
    """
    steps = envelope["steps"]
    if not steps:
        return 0.0, [0.0] * GRID_CELLS

    position = _clamp((slot - EVENT_START_DT).total_seconds() / 3600.0, 0.0, len(steps) - 1)
    lo = int(math.floor(position))
    hi = min(lo + 1, len(steps) - 1)
    frac = position - lo

    strength = (
        steps[lo]["avg_rainfall_1h_mm"] * (1.0 - frac)
        + steps[hi]["avg_rainfall_1h_mm"] * frac
    ) / envelope["peak_avg_rainfall"]
    rain = [
        steps[lo]["rainfall_1h_mm"][i] * (1.0 - frac) + steps[hi]["rainfall_1h_mm"][i] * frac
        for i in range(GRID_CELLS)
    ]
    return _clamp(strength, 0.0, 1.0), rain


def _storm_intensity(slot: datetime, envelope: Dict[str, Any]) -> float:
    """Normalised storm strength (0..1) at a slot, from the table envelope."""
    return _interpolate_envelope(slot, envelope)[0]


def _cell_rainfall(slot: datetime, envelope: Dict[str, Any]) -> List[float]:
    return _interpolate_envelope(slot, envelope)[1]


def _temporal_texture(slot: datetime, cell_index: int, unit_std: float) -> float:
    """Smooth, deterministic temporal texture for one cell.

    The previous revision drew independent white noise per frame. Differencing
    two independent draws over 30 minutes and doubling for the hourly rate
    produced swings of +-60 degC/hr -- physically impossible. Convection cools
    at roughly 5-15 degC/hr. Summing a few long-period harmonics keeps the
    field evolving smoothly, so the derived rate stays in the physical range
    while still varying from cell to cell.
    """
    hours = (slot - EVENT_START_DT).total_seconds() / 3600.0
    rng = _stable_rng("texture", cell_index)
    value = 0.0
    for _ in range(3):
        period = rng.uniform(3.0, 10.0)  # hours
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amplitude = rng.uniform(0.3, 1.0)
        value += amplitude * math.sin(2.0 * math.pi * hours / period + phase)
    return value * unit_std * 0.6


def _cloud_mask_from_ctt(ctt: List[float]) -> List[int]:
    """CMK classes: 2 = deep convective, 1 = cloud, 0 = clear."""
    return [2 if v < -32.0 else (1 if v < -12.0 else 0) for v in ctt]


def _uncorrected_ctt(
    slot: datetime,
    calibration: Dict[str, Any],
    envelope: Dict[str, Any],
    noise_scale: float = 1.0,
) -> List[float]:
    """Cloud-top temperature before any observation nudging."""
    intensity, _ = _interpolate_envelope(slot, envelope)
    anomaly = calibration["spatial_anomaly"]
    observed_std = calibration.get("observed_std") or FALLBACK_CTT_STD
    unit_std = max(observed_std / 3.0, 1.0) * noise_scale
    anchor = CTT_WARM_ANCHOR + (intensity ** 0.7) * (CTT_COLD_ANCHOR - CTT_WARM_ANCHOR)
    return [
        _clamp(anchor + anomaly[i] + _temporal_texture(slot, i, unit_std), *CTT_CLAMP)
        for i in range(GRID_CELLS)
    ]


def _observation_nudge(
    slot: datetime,
    calibration: Dict[str, Any],
    envelope: Dict[str, Any],
    *,
    blend_hours: float = 3.0,
) -> Tuple[Optional[List[float]], float]:
    """Observation nudging (data assimilation) against the real granules.

    The table's synthetic storm envelope peaks at a different hour from the
    real downloaded frames, so an un-nudged reconstruction disagreed with the
    real observations by ~20 degC near them -- producing impossible 40 degC/hr
    seams where real and reconstructed frames met.

    This computes the residual between each real frame and the reconstruction
    at that same instant, then applies it with a Gaussian decay in time. The
    reconstruction therefore passes *through* the real observations and decays
    smoothly away from them, exactly like a nudged NWP field.

    Returns ``(per_cell_delta, weight)``; ``(None, 0.0)`` when no real frame is
    close enough to matter.
    """
    real_frames = calibration.get("real_frames") or {}
    if not real_frames:
        return None, 0.0

    best_weight, best_time = 0.0, None
    for real_time in real_frames:
        gap_hours = abs((slot - real_time).total_seconds()) / 3600.0
        weight = math.exp(-((gap_hours / blend_hours) ** 2))
        if weight > best_weight:
            best_weight, best_time = weight, real_time

    if best_time is None or best_weight < 0.02:
        return None, 0.0

    synthetic_at_real = _uncorrected_ctt(best_time, calibration, envelope)
    observed = real_frames[best_time]
    delta = [observed[i] - synthetic_at_real[i] for i in range(GRID_CELLS)]
    return delta, best_weight


def _uncontrolled_fields(
    slot: datetime, ctt: List[float], envelope: Dict[str, Any]
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """The four non-CTT fields, before any observation nudging.

    Extracted unchanged from the original inline loops so the same formulas can
    be evaluated at a real granule's own timestamp -- which is what the nudging
    residual needs. The RNG is drawn once per cell in the same order as before,
    so un-nudged output is bit-identical to the previous revision.
    """
    intensity = _storm_intensity(slot, envelope)
    rain_cells = _cell_rainfall(slot, envelope)
    rng = _stable_rng("frame", _cache_key(slot))

    qpe: List[float] = []
    shear: List[float] = []
    uth: List[float] = []
    convergence: List[float] = []
    for i in range(GRID_CELLS):
        cold_fraction = _clamp((-ctt[i] - 20.0) / 25.0, 0.0, 1.0)
        qpe.append(
            _clamp(
                rain_cells[i]
                * (0.70 + 0.90 * cold_fraction)
                * math.exp(rng.gauss(0.0, 0.25)),
                0.0,
                180.0,
            )
        )
        shear.append(
            _clamp(
                4.0
                + 30.0 * intensity
                + 6.0 * cold_fraction
                + _temporal_texture(slot, i, 1.0) * 2.5,
                1.0,
                48.0,
            )
        )
        uth.append(
            _clamp(
                20.0
                + 55.0 * intensity
                + 20.0 * cold_fraction
                + _temporal_texture(slot, i, 1.0) * 3.0,
                5.0,
                98.0,
            )
        )
        convergence.append(
            round(
                _clamp(
                    0.2
                    + 4.0 * intensity
                    + 1.5 * cold_fraction
                    + _temporal_texture(slot, i, 1.0) * 0.5,
                    -1.5,
                    9.0,
                ),
                3,
            )
        )
    return qpe, shear, uth, convergence


def _field_nudge(
    field_name: str,
    slot: datetime,
    calibration: Dict[str, Any],
    envelope: Dict[str, Any],
    real_field_frames: Dict[str, Dict[datetime, List[float]]],
    *,
    blend_hours: float = 3.0,
) -> Tuple[Optional[List[float]], float]:
    """Observation nudging for one non-CTT field (same scheme as CTT)."""
    frames = real_field_frames.get(field_name) or {}
    if not frames:
        return None, 0.0

    best_weight, best_time = 0.0, None
    for real_time in frames:
        gap_hours = abs((slot - real_time).total_seconds()) / 3600.0
        weight = math.exp(-((gap_hours / blend_hours) ** 2))
        if weight > best_weight:
            best_weight, best_time = weight, real_time
    if best_time is None or best_weight < 0.02:
        return None, 0.0

    ctt_at_real = _uncorrected_ctt(best_time, calibration, envelope)
    qpe, shear, uth, convergence = _uncontrolled_fields(best_time, ctt_at_real, envelope)
    synthetic_at_real = {
        "qpe_rain_rate": qpe,
        "wind_speed_shear": shear,
        "uth": uth,
        "low_level_convergence": convergence,
    }[field_name]
    observed = frames[best_time]
    delta = [observed[i] - synthetic_at_real[i] for i in range(GRID_CELLS)]
    return delta, best_weight


def synthesize_frame(
    slot: datetime,
    calibration: Dict[str, Any],
    envelope: Dict[str, Any],
    *,
    noise_scale: float = 1.0,
    real_field_frames: Optional[Dict[str, Dict[datetime, List[float]]]] = None,
) -> Dict[str, List[float]]:
    """Build the four physically coherent fields for one half-hourly slot.

    CTP carries cloud-top temperature (the input the real derivation path
    differences to get a cooling rate); the other three carry their derived
    field directly, exactly as a real decoded granule would.
    """
    # --- Cloud-top temperature: storm envelope + REAL observed anomaly field,
    #     with smooth temporal texture, nudged toward the real observations.
    ctt = _uncorrected_ctt(slot, calibration, envelope, noise_scale)
    nudge_delta, nudge_weight = _observation_nudge(slot, calibration, envelope)
    if nudge_delta is not None:
        ctt = [
            _clamp(ctt[i] + nudge_weight * nudge_delta[i], *CTT_CLAMP)
            for i in range(GRID_CELLS)
        ]

    # --- Rain rate, shear, upper-tropospheric humidity and low-level
    #     convergence: the cold-cloud fraction drives convection, the storm
    #     envelope sets the amplitude, and the field is anchored on the table's
    #     rainfall so the two agree.
    qpe, shear, uth, convergence = _uncontrolled_fields(slot, ctt, envelope)

    # --- Honour the real granules for these products too. Without this the
    #     reconstruction met the real observation with a step change -- the
    #     mismatch was in the fill's invented amplitude, not in the weather.
    if real_field_frames:
        for field_name, series, low, high in (
            ("qpe_rain_rate", qpe, 0.0, 180.0),
            ("wind_speed_shear", shear, 1.0, 48.0),
            ("uth", uth, 5.0, 98.0),
            ("low_level_convergence", convergence, -1.5, 9.0),
        ):
            delta, weight = _field_nudge(
                field_name, slot, calibration, envelope, real_field_frames
            )
            if delta is None:
                continue
            for i in range(GRID_CELLS):
                series[i] = _clamp(series[i] + weight * delta[i], low, high)

    return {
        "ctt": ctt,
        "qpe_rain_rate": qpe,
        "wind_speed_shear": shear,
        "low_level_convergence": convergence,
        "uth": uth,
        "cloud_mask": [float(v) for v in _cloud_mask_from_ctt(ctt)],
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _real_slots(cache_dir: str, product_id: str) -> set:
    """Timestamps already covered by a real decoded granule for ``product_id``."""
    covered = set()
    for fname in glob.glob(os.path.join(cache_dir, f"{product_id}_*.json")):
        stem = os.path.basename(fname)[:-5]
        key = stem[len(product_id) + 1:]
        try:
            with open(fname, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("is_real_data") is True:
            covered.add(key)
    return covered


def fill_missing_granules(
    cache_dir: str = CACHE_DIR,
    feature_path: str = FEATURE_TABLE,
    *,
    report_only: bool = False,
    overwrite_synthetic: bool = False,
) -> Dict[str, Any]:
    """Write calibrated reconstructions into every slot lacking a real granule."""
    os.makedirs(cache_dir, exist_ok=True)

    calibration = load_real_calibration(cache_dir)
    real_field_frames = load_real_field_frames(cache_dir)
    envelope = load_storm_envelope(feature_path)
    slots = _list_lattice()

    report: Dict[str, Any] = {
        "window": [slots[0].isoformat(), slots[-1].isoformat()],
        "slots": len(slots),
        "real_frames_used_for_calibration": calibration["n_real_frames"],
        "real_field_frames_used_for_nudging": {
            name: len(frames) for name, frames in sorted(real_field_frames.items())
        },
        "calibration": {
            "observed_mean_degC": round(calibration["observed_mean"], 2),
            "observed_std_degC": round(calibration["observed_std"], 2),
            "observed_range_degC": [
                round(calibration["observed_min"], 2),
                round(calibration["observed_max"], 2),
            ],
        },
        "per_product": {},
    }

    if calibration["n_real_frames"] == 0:
        print(
            "WARNING: no real granules found in the cache. The reconstruction\n"
            "         prior falls back to flat statistics, so the output will\n"
            "         NOT be calibrated on real observations."
        )

    written_total = 0
    for product_id, spec in PRODUCT_SPEC.items():
        real_slots = _real_slots(cache_dir, product_id)
        written = 0
        skipped_real = 0
        skipped_existing = 0

        for slot in slots:
            key = _cache_key(slot)
            out_path = os.path.join(cache_dir, f"{product_id}_{key}.json")

            # Real data always wins -- never overwrite an observation.
            if key in real_slots:
                skipped_real += 1
                continue
            if os.path.exists(out_path) and not overwrite_synthetic:
                skipped_existing += 1
                continue

            frame = synthesize_frame(
                slot, calibration, envelope, real_field_frames=real_field_frames
            )
            values = frame[spec["frame"]]

            # Secondary variables go in the ``derived`` block, mirroring the
            # shape the real decoder produces so consumers see one format.
            secondary = {
                out_name: [round(float(v), 3) for v in frame[frame_key]]
                for out_name, frame_key in (SECONDARY_FIELDS.get(product_id) or {}).items()
            }

            payload = {
                "product_id": product_id,
                "granule_id": None,
                "identifier": f"SYNTH_{slot.strftime('%d%b%Y_%H%M').upper()}_{product_id.split('_')[-1]}_V01R00",
                "timestamp": slot.isoformat(),
                "grid_values": [round(float(v), 3) for v in values],
                "units": spec["units"],
                "variable": spec["variable"],
                "derived": secondary,
                "is_real_data": False,
                "gap_filled": True,
                "provenance": "SYNTHETIC",
                "source": (
                    "synthetic_reconstruction calibrated on "
                    f"{calibration['n_real_frames']} real MOSDAC granule(s)"
                ),
                "calibration_basis": {
                    "real_frames": calibration["n_real_frames"],
                    "observed_ctt_mean_degC": round(calibration["observed_mean"], 2),
                    "observed_ctt_range_degC": [
                        round(calibration["observed_min"], 2),
                        round(calibration["observed_max"], 2),
                    ],
                    "spatial_anomaly_source": "per-cell mean of real downloaded frames",
                    "storm_envelope_source": "feature table rainfall_1h_mm",
                },
                "extracted_at": datetime.now().isoformat(),
            }

            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            shutil.move(tmp_path, out_path)
            written += 1

        written_total += written
        report["per_product"][product_id] = {
            "slots_total": len(slots),
            "real_slots_preserved": skipped_real,
            "synthetic_existing_kept": skipped_existing,
            "written": written,
        }
        print(
            f"  {product_id:18s} real={skipped_real:3d} preserved | "
            f"reconstructed={written:3d} written | {skipped_existing:3d} kept"
        )

    report["granules_written"] = written_total
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill satellite slots lacking real granules with calibrated reconstructions"
    )
    parser.add_argument("--report", action="store_true", help="show what is missing without writing")
    parser.add_argument("--force", action="store_true", help="rewrite existing synthetic fills")
    args = parser.parse_args(argv)

    if args.report:
        calibration = load_real_calibration()
        print(json.dumps({
            "real_frames": calibration["n_real_frames"],
            "observed_ctt_mean_degC": round(calibration["observed_mean"], 2),
            "observed_ctt_range_degC": [
                round(calibration["observed_min"], 2),
                round(calibration["observed_max"], 2),
            ],
        }, indent=2))
        return 0

    print(f"Reconstructing satellite slots for {WINDOW_START} .. {WINDOW_END} (half-hourly)")
    report = fill_missing_granules(overwrite_synthetic=args.force)
    print(json.dumps(report, indent=2))
    print(
        "\nNOTE: reconstructed granules are marked is_real_data=false. Use them\n"
        "      for continuity only -- never advertise them as telemetry."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
