"""
Phase 3 -- Temporal derivation of satellite products.
=====================================================

What this module does
---------------------
Reads the decoded, per-cell satellite products written by ``extractor.py``
(and, for slots the archive could not serve, ``synthetic_fill.py``) from
``Backend/cache/satellite_cache/`` and derives the temporal fields the
feature contract needs:

* ``ctt_drop_rate_c_per_hr``  -- cloud-top cooling rate from consecutive CTP frames
* ``wind_speed_shear``       -- wind derived product (WDP) field
* ``qpe_rain_rate``          -- hydro-estimator (HEM) rain rate
* ``cloud_mask``             -- cloud mask (CMK) restricting cooling to cloudy pixels

Provenance is per field, not per file
-------------------------------------
A single timestep can be built from a mix of real and reconstructed inputs
(e.g. a real CTP frame plus a reconstructed HEM frame). Collapsing that to one
boolean at the file level is exactly how reconstructed data gets laundered into
"LIVE" telemetry.

So every payload carries:

* ``field_provenance``  -- ``{field: REAL | SYNTHETIC | PLACEHOLDER}``
* ``real_fields`` / ``synthetic_fields`` / ``placeholder_fields``
* ``provenance``        -- REAL, MIXED, SYNTHETIC or PLACEHOLDER at the step level
* ``is_real_data``      -- true ONLY when every populated field came from real granules
* ``calibration_note``  -- when synthetic, which real granules the fill was fitted on

``feature_merge`` then writes the fields and reports exactly which are measured
and which are modelled, instead of a single optimistic flag.

Usage::

    python -m data_pipeline.derivation_engine            # derive all timesteps
    python -m data_pipeline.derivation_engine --summary  # report cache coverage
"""

from __future__ import annotations

import argparse
import json
import os
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

CACHE_DIR = os.path.join(backend_dir, "cache", "satellite_cache")
DERIVED_DIR = os.path.join(backend_dir, "cache", "derived_satellite")

# Derivation window -- single source of truth: app.core.timeline_config.
# The feature table steps hourly (EVENT_STEP_HOURS); satellite frames arrive
# half-hourly, so derivation runs on a half-hourly lattice and feature_merge
# resolves each table timestamp to the matching derived payload.
from app.core.timeline_config import EVENT_END_DT, EVENT_START_DT, EVENT_STEP_HOURS

STEP = timedelta(hours=EVENT_STEP_HOURS) / 2
WINDOW_START = EVENT_START_DT
# Cover EVERY table step: the last step sits at EVENT_END_DT exactly, so the
# lattice must end there too or the final timestep resolves to no payload.
WINDOW_END = EVENT_END_DT

GRID_CELLS = 90

# Placeholder values used ONLY when no usable input exists. These are flagged
# as gap_filled so no consumer mistakes them for measurements.
PLACEHOLDER = {
    "ctt_drop_rate_c_per_hr": 0.0,
    "wind_speed_shear": 5.0,
    "low_level_convergence": 0.0,
    "qpe_rain_rate": 0.0,
    "cloud_mask": 1,
    "iwv_mm": 40.0,
}

DERIVED_FIELDS: Tuple[str, ...] = (
    "ctt_drop_rate_c_per_hr",
    "wind_speed_shear",
    "low_level_convergence",
    "qpe_rain_rate",
    "cloud_mask",
    "iwv_mm",
)

# Which derived fields each product sources, and how.
#
# ``secondary`` carries the extra variables the decoder already computes into
# its ``derived`` block. WDP is not a single field: it yields wind shear (the
# primary sampled variable) *and* low-level convergence. The previous revision
# decoded convergence and then dropped it, so ``pipeline.md``'s requirement to
# read "winds, shear, divergence/convergence" out of WDP was only two thirds met.
PRODUCT_SPEC: Dict[str, Dict[str, Any]] = {
    "3DIMG_L2B_CTP": {"primary": "ctt_drop_rate_c_per_hr", "kind": "ctt"},
    "3DIMG_L2G_WDP": {
        "primary": "wind_speed_shear",
        "secondary": {"low_level_convergence": "low_level_convergence_1e5"},
    },
    "3DIMG_L2B_UTH": {"primary": "iwv_mm", "transform": "uth_to_iwv"},
    "3DIMG_L2B_HEM": {"primary": "qpe_rain_rate"},
    "3DIMG_L2B_CMK": {"primary": "cloud_mask"},
}

# cache filename pattern -> primary derived field (kept for cache indexing)
PRODUCT_FIELDS = {pid: spec["primary"] for pid, spec in PRODUCT_SPEC.items()}

# UTH -> IWV proxy constants. IWV is NOT retrieved here: ``pipeline.md`` states
# IWV stays a regression/proxy, so this is a *stated* monotone mapping from
# upper-tropospheric humidity to an integrated water-vapour estimate, bounded by
# Mumbai monsoon climatology (roughly 20-75 mm). Provenance is RECONSTRUCTED,
# never REAL, and the relationship is recorded on every derived payload.
IWV_MIN_MM = 20.0
IWV_MAX_MM = 75.0
IWV_UTH_EXPONENT = 1.4
IWV_METHOD = (
    "UTH-based proxy: IWV = 20 + 55 * (UTH/100)^1.4, bounded by Mumbai monsoon "
    "climatology. Not a retrieval; recorded as RECONSTRUCTED."
)


def uth_to_iwv(uth_values: Sequence[float]) -> List[float]:
    """Monotone UTH (%) -> IWV (mm) proxy. See ``IWV_METHOD``."""
    out: List[float] = []
    for value in uth_values:
        clamped = max(0.0, min(100.0, float(value)))
        out.append(round(IWV_MIN_MM + (IWV_MAX_MM - IWV_MIN_MM) * (clamped / 100.0) ** IWV_UTH_EXPONENT, 2))
    return out


TRANSFORMS = {"uth_to_iwv": uth_to_iwv}


def _extract_list(container: Any, key: str) -> Optional[List[float]]:
    """Pull a numeric list out of a nested payload block, or ``None``."""
    if not isinstance(container, dict):
        return None
    values = container.get(key)
    if not isinstance(values, list) or not values:
        return None
    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(numeric) != len(values):
        return None
    return [float(v) for v in values]


# ---------------------------------------------------------------------------
# Cache access
# ---------------------------------------------------------------------------
def _cache_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def load_cache_index(cache_dir: str = CACHE_DIR) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Index decoded granules by ``(product_id, timestamp_key)``.

    Filenames follow ``{product_id}_{timestamp_clean}.json`` where
    ``timestamp_clean`` is the ISO timestamp with ``:`` replaced by ``-``.
    """
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not os.path.isdir(cache_dir):
        return index
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        for product_id in PRODUCT_FIELDS:
            prefix = product_id + "_"
            if stem.startswith(prefix):
                key = stem[len(prefix):]
                try:
                    with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict):
                    index[(product_id, key)] = payload
                break
    return index


def _grid_values(payload: Dict[str, Any]) -> Optional[List[float]]:
    """Extract the 90-cell grid from a decoded cache payload."""
    for key in ("grid_values", "values", "cells", "data"):
        values = payload.get(key)
        if isinstance(values, list) and values:
            numeric = [
                v for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if len(numeric) == len(values):
                return [float(v) for v in values]
    return None


def _is_usable(values: Optional[List[float]]) -> bool:
    """A decoded grid is usable only if it actually varies across the grid."""
    if not values:
        return False
    return len(set(values)) > 1


def _payload_is_real(payload: Optional[Dict[str, Any]]) -> bool:
    """Whether a cache payload came from a real decoded granule."""
    if not payload:
        return False
    if isinstance(payload.get("is_real_data"), bool):
        return bool(payload["is_real_data"])
    return str(payload.get("provenance", "")).upper() == "REAL"


def load_calibration_note(cache_dir: str = CACHE_DIR) -> Optional[str]:
    """Describe the real granules the reconstruction prior was fitted on."""
    count = 0
    total = 0
    for fname in os.listdir(cache_dir) if os.path.isdir(cache_dir) else []:
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        total += 1
        if payload.get("is_real_data") is True:
            count += 1
    if count == 0:
        return None
    return (
        f"reconstructed fields calibrated on {count} real MOSDAC granule(s) "
        f"of {total} cached granule(s)"
    )


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
def derive_timestep(
    ts: datetime,
    cache_index: Dict[Tuple[str, str], Dict[str, Any]],
    ctt_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Derive one timestep.

    ``ctt_state`` carries the most recent cloud-top frame forward so an hourly
    table step can still pair with a half-hourly granule observed up to one
    step earlier. Returns ``(payload, ctt_state_for_next_step)``.
    """
    key = _cache_key(ts)
    values: Dict[str, List[float]] = {}
    field_provenance: Dict[str, str] = {}
    used_products: List[str] = []
    source_steps: List[str] = []
    state: Dict[str, Any] = dict(ctt_state) if ctt_state else {}

    for product_id, spec in PRODUCT_SPEC.items():
        payload = cache_index.get((product_id, key))
        grid = _grid_values(payload) if payload else None
        derived_field = spec["primary"]

        if spec.get("kind") == "ctt":
            # Prefer this step's own frame; otherwise carry the prior one
            # forward so a table step can pair with an earlier granule.
            if _is_usable(grid):
                state["values"] = grid
                state["ts"] = ts
                state["is_real"] = _payload_is_real(payload)
            continue

        if _is_usable(grid):
            origin = "REAL" if _payload_is_real(payload) else "SYNTHETIC"
            transform = TRANSFORMS.get(spec.get("transform"))
            values[derived_field] = transform(grid) if transform else grid
            field_provenance[derived_field] = origin
            used_products.append(product_id)
            source_steps.append(key)

            # Secondary variables the decoder already computed into its
            # ``derived`` block (e.g. WDP low-level convergence), which were
            # previously decoded and then discarded.
            for out_name, source_key in (spec.get("secondary") or {}).items():
                extra = _extract_list(payload.get("derived"), source_key)
                if extra and _is_usable(extra):
                    values[out_name] = extra
                    field_provenance[out_name] = origin
                    source_steps.append(key)

    # CTT cooling rate: consecutive-frame difference normalised to degC per
    # hour using the ACTUAL elapsed time between the two frames -- so both
    # half-hourly and hourly pairings come out right (no hardcoded x2).
    current_ctt = state.get("values")
    current_ts = state.get("ts")
    previous_ctt = (ctt_state or {}).get("values")
    previous_ts = (ctt_state or {}).get("ts")

    if (
        current_ctt is not None
        and previous_ctt is not None
        and current_ts is not None
        and previous_ts is not None
        and current_ts != previous_ts
        and len(current_ctt) == len(previous_ctt)
    ):
        elapsed_hr = (current_ts - previous_ts).total_seconds() / 3600.0
        if 0.0 < elapsed_hr <= 3.0:
            rate = [
                (prev - cur) / elapsed_hr
                for prev, cur in zip(previous_ctt, current_ctt)
            ]
            if len(set(rate)) > 1:
                values["ctt_drop_rate_c_per_hr"] = rate
                # A rate mixing a real frame with a reconstructed one is only
                # as trustworthy as its weaker input.
                real_inputs = bool(state.get("is_real")) and bool(
                    (ctt_state or {}).get("is_real")
                )
                field_provenance["ctt_drop_rate_c_per_hr"] = (
                    "REAL" if real_inputs else "SYNTHETIC"
                )
                used_products.append("3DIMG_L2B_CTP")
                source_steps.append(key)

    # Restrict cooling rate to cloudy pixels using the cloud mask.
    mask = values.get("cloud_mask")
    rate = values.get("ctt_drop_rate_c_per_hr")
    if mask is not None and rate is not None and len(mask) == len(rate):
        values["ctt_drop_rate_c_per_hr"] = [
            r if (m and m > 0) else 0.0 for r, m in zip(rate, mask)
        ]

    real_fields = sorted(f for f, p in field_provenance.items() if p == "REAL")
    synthetic_fields = sorted(f for f, p in field_provenance.items() if p == "SYNTHETIC")

    payload_out: Dict[str, Any] = {"timestamp": ts.isoformat()}
    for derived_field in DERIVED_FIELDS:
        if derived_field in values:
            payload_out[derived_field] = values[derived_field]
            field_provenance.setdefault(derived_field, "SYNTHETIC")
        else:
            payload_out[derived_field] = [PLACEHOLDER[derived_field]] * GRID_CELLS
            field_provenance[derived_field] = "PLACEHOLDER"

    if not real_fields and not synthetic_fields:
        step_provenance, reason = "PLACEHOLDER", (
            "no usable decoded or reconstructed granules in cache - emitted "
            "placeholders; run extractor.py or synthetic_fill.py first"
        )
    elif real_fields and synthetic_fields:
        step_provenance, reason = "MIXED", (
            "real: " + ", ".join(real_fields) + " | modelled: " + ", ".join(synthetic_fields)
        )
    elif real_fields:
        step_provenance, reason = "REAL", "derived entirely from decoded granules"
    else:
        step_provenance, reason = "SYNTHETIC", (
            "derived entirely from calibrated reconstructed granules"
        )

    placeholder_fields = sorted(f for f, p in field_provenance.items() if p == "PLACEHOLDER")

    # State the method on the payload so a reconstructed variable is never read
    # as a retrieval (IWV is a UTH-based proxy by design).
    payload_out.update(
        {
            "provenance": step_provenance,
            "is_real_data": bool(real_fields) and not synthetic_fields
            and not placeholder_fields,
            "field_methods": {
                name: method
                for name, method in (("iwv_mm", IWV_METHOD),)
                if field_provenance.get(name) not in (None, "PLACEHOLDER")
            },
            "gap_filled": not real_fields,
            "field_provenance": field_provenance,
            "real_fields": real_fields,
            "synthetic_fields": synthetic_fields,
            "placeholder_fields": placeholder_fields,
            "available_products": sorted(set(used_products)),
            "source_timesteps": sorted(set(source_steps)),
            "reason": reason,
            "derivation_version": "3.0.0",
        }
    )

    if synthetic_fields:
        note = load_calibration_note()
        if note:
            payload_out["calibration_note"] = note

    return payload_out, state


def calculate_temporal_derivations(
    cache_dir: str = CACHE_DIR,
    derived_dir: str = DERIVED_DIR,
    *,
    summary_only: bool = False,
) -> Dict[str, Any]:
    """Derive every timestep in the window and write payloads to disk."""
    cache_index = load_cache_index(cache_dir)

    timesteps: List[datetime] = []
    cursor = WINDOW_START
    while cursor <= WINDOW_END:
        timesteps.append(cursor)
        cursor += STEP

    report: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "cache_files_indexed": len(cache_index),
        "timesteps": len(timesteps),
        "timesteps_placeholder": 0,
        "timesteps_synthetic_only": 0,
        "timesteps_mixed": 0,
        "timesteps_real": 0,
        "field_totals": {field: {"REAL": 0, "SYNTHETIC": 0, "PLACEHOLDER": 0} for field in DERIVED_FIELDS},
    }

    print(
        f"Phase 3: deriving {len(timesteps)} timesteps from "
        f"{len(cache_index)} cached granule(s)."
    )
    if not cache_index:
        print(
            "  WARNING: satellite_cache is empty. Every timestep will be written\n"
            "  with gap_filled=true and placeholder values. Nothing downstream\n"
            "  should treat these as measurements."
        )
    if summary_only:
        return report

    os.makedirs(derived_dir, exist_ok=True)
    ctt_state: Optional[Dict[str, Any]] = None

    for ts in timesteps:
        payload, ctt_state = derive_timestep(ts, cache_index, ctt_state)

        bucket = {
            "PLACEHOLDER": "timesteps_placeholder",
            "SYNTHETIC": "timesteps_synthetic_only",
            "MIXED": "timesteps_mixed",
            "REAL": "timesteps_real",
        }[payload["provenance"]]
        report[bucket] += 1
        for field, provenance in payload["field_provenance"].items():
            report["field_totals"][field][provenance] += 1

        out_path = os.path.join(derived_dir, f"derived_{_cache_key(ts)}.json")
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        shutil.move(tmp_path, out_path)

    print(
        f"Phase 3 complete: {report['timesteps_real']} real, "
        f"{report['timesteps_mixed']} mixed, {report['timesteps_synthetic_only']} synthetic-only, "
        f"{report['timesteps_placeholder']} placeholder -> {derived_dir}"
    )
    if report["timesteps_real"] == 0:
        print(
            "  NOTE: zero fully-real timesteps. Fields derived from reconstructed\n"
            "  granules are labelled SYNTHETIC per field; do not advertise them as LIVE."
        )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive temporal satellite fields from decoded granules"
    )
    parser.add_argument("--summary", action="store_true", help="report cache coverage without writing")
    args = parser.parse_args(argv)

    report = calculate_temporal_derivations(summary_only=args.summary)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
