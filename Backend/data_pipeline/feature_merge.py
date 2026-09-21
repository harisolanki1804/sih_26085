"""
Phase 4 -- Merge derived satellite products into the feature table.
===================================================================

Contract obligations this module now honours
--------------------------------------------
1. **Shape-preserving.** ``feature_grid_timeseries.json`` keeps its existing
   container (``timesteps`` as a list, cells under ``features``, all top-level
   metadata). The previous revision replaced the container with a different
   shape and, because nothing matched, silently discarded all 26 canonical
   features and all 5 target labels. This rewrite is strictly *additive*.

2. **No fabricated provenance.** Missing granules are marked ``gap_filled`` and
   reported as ``FALLBACK``. The previous revision read a ``gap_filled`` key
   that the producer never wrote, so it defaulted to ``False`` and advertised
   constant placeholders as ``is_real_data: True`` -- i.e. a LIVE badge over
   zeros. Provenance is now *derived* from the data (explicit marker first,
   variance-based inference as a conservative fallback).

3. **Gated.** The result is validated against ``feature_contract`` before it is
   written. If a merge would damage the table, it raises and leaves the file
   untouched.

Usage::

    python -m data_pipeline.feature_merge             # merge + validate
    python -m data_pipeline.feature_merge --report    # inspect provenance only
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

from app.services.ai.feature_contract import (  # noqa: E402
    FeatureContractError,
    validate_feature_window,
)

FEATURE_TABLE = os.path.join(backend_dir, "data", "feature_grid_timeseries.json")
DERIVED_DIR = os.path.join(backend_dir, "cache", "derived_satellite")

# Requested MOSDAC products (contract priority order).
REQUESTED_PRODUCTS: Tuple[str, ...] = (
    "3DIMG_L2B_CTP",
    "3DIMG_L2G_WDP",
    "3DIMG_L2B_UTH",
    "3DIMG_L2B_HEM",
    "3DIMG_L2B_CMK",
)

# derived-file key -> canonical/auxiliary destination column
DERIVED_TO_COLUMN: Dict[str, str] = {
    "ctt_drop_rate_c_per_hr": "ctt_drop_rate_c_per_hr",   # canonical, index 11
    "wind_speed_shear": "vertical_wind_shear_ms",        # canonical, index 17
    "low_level_convergence": "low_level_convergence",    # canonical, index 18 (WDP)
    "qpe_rain_rate": "qpe_rain_rate_mm_hr",              # auxiliary (HEM)
    "cloud_mask": "cloud_mask",                          # auxiliary (CMK)
    "iwv_mm": "iwv_mm",                                  # canonical (UTH proxy)
}

# Neutral placeholder values, keyed by DESTINATION COLUMN (not by the derived
# field name) so detection cannot silently miss a constant array.
PLACEHOLDER_VALUES: Dict[str, float] = {
    "ctt_drop_rate_c_per_hr": 0.0,
    "vertical_wind_shear_ms": 5.0,
    "low_level_convergence": 0.0,
    "qpe_rain_rate_mm_hr": 0.0,
    "cloud_mask": 1,
    "iwv_mm": 40.0,
}

_SYNTHETIC_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d+):(\d{2}):(\d{2})")


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------
def expand_synthetic_timestamp(ts_str: str) -> Optional[datetime]:
    """Expand a synthetic hour-overflow timestamp into real calendar time.

    The replay table encodes timestep N as ``2022-07-05T{N}:00:00Z``, so N=71
    means "71 hours after 05 Jul 2022 00:00", not an invalid hour-of-day.
    """
    match = _SYNTHETIC_TS.match(ts_str.replace("Z", ""))
    if not match:
        return None
    date_part, hours, minutes, seconds = match.groups()
    base = datetime.strptime(date_part, "%Y-%m-%d")
    return base + timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds))


def _derived_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _normalize_timesteps_container(raw: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return ``(top_level_metadata, timestep_list)``.

    Supports the list shape (canonical) and the dict shape (legacy) so the
    module never silently drops data again.
    """
    if isinstance(raw, dict):
        meta = {k: v for k, v in raw.items() if k != "timesteps"}
        timesteps = raw.get("timesteps", [])
        if isinstance(timesteps, dict):
            timesteps = [v for v in timesteps.values() if isinstance(v, dict)]
        return meta, [t for t in timesteps if isinstance(t, dict)]
    if isinstance(raw, list):
        return {}, [t for t in raw if isinstance(t, dict)]
    raise ValueError(f"unsupported feature table type: {type(raw)!r}")


def load_derived_files(derived_dir: str = DERIVED_DIR) -> Dict[str, Dict[str, Any]]:
    """Index derived satellite payloads by their raw timestamp key."""
    payloads: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(derived_dir):
        return payloads
    for fname in sorted(os.listdir(derived_dir)):
        if not fname.endswith(".json"):
            continue
        key = fname[len("derived_"):-len(".json")] if fname.startswith("derived_") else fname[:-5]
        try:
            with open(os.path.join(derived_dir, fname), "r", encoding="utf-8") as handle:
                payloads[key] = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


# ---------------------------------------------------------------------------
# Provenance inference
# ---------------------------------------------------------------------------
def classify_derived(payload: Dict[str, Any]) -> Tuple[bool, List[str], str]:
    """Decide whether a derived payload carries real measurements.

    Trusts an explicit provenance marker when the producer supplies one;
    otherwise infers conservatively from variance. Returns
    ``(is_real, available_products, reason)``.
    """
    # 1. Explicit producer declaration wins.
    for flag in ("is_real_data", "real"):
        if isinstance(payload.get(flag), bool):
            is_real = bool(payload[flag])
            products = payload.get("available_products")
            products = list(products) if isinstance(products, list) else []
            return is_real, products, "explicit producer flag"

    if isinstance(payload.get("gap_filled"), bool):
        is_real = not payload["gap_filled"]
        products = payload.get("available_products")
        products = list(products) if isinstance(products, list) else []
        return is_real, products, "explicit gap_filled flag"

    # 2. Conservative fallback: a payload is only "real" if at least one field
    #    actually varies across the sampled cells. Constant arrays are the
    #    signature of the placeholder path in derivation_engine.
    varying: List[str] = []
    checked = 0
    for field_name in DERIVED_TO_COLUMN:
        values = payload.get(field_name)
        if not isinstance(values, list) or not values:
            continue
        checked += 1
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(set(numeric)) > 1:
            varying.append(field_name)

    if checked == 0:
        return False, [], "no recognizable satellite fields present"

    if varying:
        return True, list(REQUESTED_PRODUCTS), "inferred from non-constant field(s): " + ", ".join(varying)

    return (
        False,
        [],
        "all satellite fields are constant across the grid - placeholder values, not measurements",
    )


def _is_placeholder_field(column: str, values: Sequence[Any]) -> bool:
    """True when a field is not a usable measurement.

    Rule: **a spatially constant array is never treated as a measurement.**
    Across a 90-cell urban grid spanning ~20 km, uniform values are the
    signature of the placeholder path, not a physical field. When the constant
    equals the declared neutral value for ``column`` it is reported as a
    ``PLACEHOLDER``; otherwise it is still withheld from the table and reported
    as ``CONSTANT_UNTRASTED`` so the ambiguity is visible rather than merged.

    Being wrong in this direction only *withholds* data; the opposite mistake
    silently injects fabricated constants into training.
    """
    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        return True
    return len(set(numeric)) <= 1


def _field_status(column: str, values: Sequence[Any]) -> str:
    """Classify a live field for the provenance report."""
    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        return "ABSENT"
    if len(set(numeric)) > 1:
        return "MEASURED"
    expected = PLACEHOLDER_VALUES.get(column)
    if expected is not None and all(abs(float(v) - float(expected)) < 1e-9 for v in numeric):
        return "PLACEHOLDER"
    return "CONSTANT_UNTRASTED"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_satellite_into_features(
    feature_path: str = FEATURE_TABLE,
    derived_dir: str = DERIVED_DIR,
    *,
    strict: bool = True,
    report_only: bool = False,
) -> Dict[str, Any]:
    """Additively merge derived satellite fields into the feature table."""
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"{feature_path} is missing. Restore it before merging - this step is "
            "additive and must never be the thing that creates it."
        )

    with open(feature_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    meta, timesteps = _normalize_timesteps_container(raw)
    if not timesteps:
        raise ValueError("feature table contains no timesteps")

    derived_index = load_derived_files(derived_dir)

    # Classify every derived payload once.
    provenance_cache: Dict[str, Tuple[bool, List[str], str]] = {
        key: classify_derived(payload) for key, payload in derived_index.items()
    }
    real_keys = [k for k, (is_real, _, _) in provenance_cache.items() if is_real]

    matched = 0
    no_payload = 0
    real_columns: set = set()
    synthetic_columns: set = set()
    real_field_timesteps = 0

    for ts in timesteps:
        ts_str = str(ts.get("timestamp", ""))
        real_dt = expand_synthetic_timestamp(ts_str)
        lookup_key = _derived_key(real_dt) if real_dt else None

        # Half-hourly granules were being thrown away.
        #
        # The derivation engine steps every 30 minutes (STEP = 1h/2) because
        # MOSDAC granules arrive at :00 and :30, but the replay table is hourly.
        # An exact-key lookup therefore dropped every :30 granule -- half of all
        # real coverage. The current slot is preferred, and one fallback is
        # allowed: the *previous* half-hour slot, never the next one. A frame
        # acquired at HH:30 is genuinely in hand by the HH+1:00 analysis step,
        # whereas reaching forward to HH:30 would leak a future observation into
        # the step being scored.
        #
        # Reconstructions exist for every slot, so an exact match always
        # succeeds -- picking purely on key would never reach the real granule
        # sitting half a step away. Selection is therefore by measured content:
        # among the candidates, take the one carrying the most fields that came
        # from a real granule, with the current slot winning ties.
        candidates: List[str] = [k for k in (lookup_key,) if k]
        legacy_key = ts_str.replace("Z", "").replace(":", "-")
        if legacy_key not in candidates:
            candidates.append(legacy_key)
        if real_dt is not None:
            candidates.append(_derived_key(real_dt - timedelta(minutes=30)))

        payload = None
        best_real = -1
        for candidate in candidates:
            candidate_payload = derived_index.get(candidate)
            if candidate_payload is None:
                continue
            n_real = len(candidate_payload.get("real_fields") or [])
            if n_real > best_real:
                best_real = n_real
                payload = candidate_payload
                lookup_key = candidate

        if payload is None:
            no_payload += 1
            is_real, products, reason = False, [], "no derived satellite payload for this timestamp"
        else:
            matched += 1
            is_real, products, reason = provenance_cache.get(
                lookup_key, (False, [], "unclassified")
            )

        gap_filled = not is_real
        cells = ts.get("features")
        if not isinstance(cells, list):
            cells = []

        field_placeholder: Dict[str, bool] = {}
        field_status: Dict[str, str] = {}
        field_provenance: Dict[str, str] = {}

        # The producer declares which fields came from real granules and which
        # were reconstructed. Trust it; fall back to the step-level flag.
        declared = payload.get("field_provenance") if isinstance(payload, dict) else None
        if not isinstance(declared, dict):
            declared = {}

        for derived_name, column in DERIVED_TO_COLUMN.items():
            values = payload.get(derived_name) if payload else None
            if not isinstance(values, list) or not values:
                field_placeholder[derived_name] = True
                field_status[column] = "ABSENT"
                continue
            placeholder = _is_placeholder_field(column, values)
            field_placeholder[derived_name] = placeholder
            if placeholder:
                field_status[column] = _field_status(column, values)
                continue

            origin = declared.get(derived_name)
            if origin not in ("REAL", "SYNTHETIC"):
                origin = "REAL" if is_real else "SYNTHETIC"
            field_provenance[column] = origin
            if origin == "REAL":
                field_status[column] = "MEASURED"
                real_columns.add(column)
            else:
                field_status[column] = "SYNTHETIC"
                synthetic_columns.add(column)

        skipped_fields: List[str] = [
            derived for derived, placeholder in field_placeholder.items() if placeholder
        ]

        measured_names = sorted(c for c, o in field_provenance.items() if o == "REAL")
        modelled_names = sorted(c for c, o in field_provenance.items() if o == "SYNTHETIC")
        if measured_names:
            real_field_timesteps += 1

        for idx, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            for derived_name, column in DERIVED_TO_COLUMN.items():
                # A placeholder must never be written under a canonical name:
                # canonical spellings win alias resolution, so writing 0.0
                # placeholders would mask real values already in the table
                # (e.g. the existing varying ctt_drop_rate_c_hr column).
                if field_placeholder.get(derived_name, True):
                    continue
                values = payload.get(derived_name) if payload else None
                if isinstance(values, list) and idx < len(values):
                    cell[column] = values[idx]
            cell["gap_filled"] = gap_filled

        # Per-cell gap flag stays; placeholders are declared, not injected.
        for cell in cells:
            if isinstance(cell, dict):
                cell["satellite_gap_filled"] = gap_filled

        if measured_names and modelled_names:
            step_status = "PARTIAL"
        elif measured_names:
            step_status = "LIVE"
        elif modelled_names:
            step_status = "SYNTHETIC"
        else:
            step_status = "FALLBACK"

        # A step is only "real" when every populated field is measured.
        honest_real = bool(measured_names) and not modelled_names

        ts["satellite_provenance"] = {
            "source": "INSAT-3D/3DR (MOSDAC)",
            "status": step_status,
            "is_real_data": honest_real,
            "measured_fields": measured_names,
            "modelled_fields": modelled_names,
            "field_provenance": dict(field_provenance),
            "available_products": products,
            "requested_products": list(REQUESTED_PRODUCTS),
            "gap_filled": not honest_real,
            "reason": reason,
            "field_status": dict(field_status),
            "fields_not_merged": sorted(skipped_fields),
        }
        if modelled_names and not measured_names:
            note = payload.get("calibration_note") if isinstance(payload, dict) else None
            if note:
                ts["satellite_provenance"]["calibration_note"] = note

        # Legacy flat keys kept for existing readers; values now truthful.
        ts["source"] = "INSAT-3D/3DR (MOSDAC)"
        ts["is_real_data"] = honest_real
        ts["available_products"] = products

    # Roll-up summary for the UI badge and the audit trail.
    all_columns = set(DERIVED_TO_COLUMN.values())
    placeholder_columns = sorted(all_columns - real_columns - synthetic_columns)
    timesteps_fully_real = len([t for t in timesteps if t.get("is_real_data")])

    if real_columns and timesteps_fully_real == len(timesteps):
        overall_status = "LIVE"
    elif real_columns:
        overall_status = "PARTIAL"
    elif synthetic_columns:
        overall_status = "SYNTHETIC"
    else:
        overall_status = "FALLBACK"

    summary = {
        "status": overall_status,
        "is_real_data": bool(real_columns) and timesteps_fully_real == len(timesteps),
        "timesteps_total": len(timesteps),
        "timesteps_with_payload": matched,
        "timesteps_without_payload": no_payload,
        "timesteps_with_measured_fields": real_field_timesteps,
        "timesteps_fully_real": timesteps_fully_real,
        "timesteps_gap_filled": len(timesteps) - timesteps_fully_real,
        "derived_payloads_available": len(derived_index),
        "derived_payloads_real": len(real_keys),
        "measured_fields": sorted(real_columns),
        "modelled_fields": sorted(synthetic_columns),
        "placeholder_fields": placeholder_columns,
        "provenance_note": (
            "measured_fields come from real MOSDAC granules; modelled_fields are "
            "physically-constrained reconstructions calibrated on the real granules. "
            "placeholder_fields had no usable input and were withheld, not fabricated."
        ),
    }

    merged = dict(meta)
    merged.update({"timesteps": timesteps, "satellite_provenance_summary": summary})

    print(
        f"Merge: {matched} timestep(s) matched to a satellite payload, "
        f"{no_payload} with no payload at all."
    )
    print(f"Satellite status: {summary['status']} | fully real: {summary['is_real_data']}")
    print(
        f"  measured from real granules : {', '.join(summary['measured_fields']) or 'none'}\n"
        f"  reconstructed (modelled)    : {', '.join(summary['modelled_fields']) or 'none'}\n"
        f"  withheld (no input)         : {', '.join(summary['placeholder_fields']) or 'none'}"
    )

    if report_only:
        return merged

    # --- Gate before writing: never let a merge damage the table -----------
    report = validate_feature_window(timesteps, label="post-merge", strict=strict)
    print()
    print(report.summary())

    tmp_path = feature_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    shutil.move(tmp_path, feature_path)
    print(f"\nWrote {feature_path} (container shape preserved).")
    return merged


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge derived satellite fields into the feature table")
    parser.add_argument("--report", action="store_true", help="classify provenance without writing")
    parser.add_argument("--no-strict", action="store_true", help="downgrade gate failures to warnings")
    args = parser.parse_args(argv)

    try:
        merge_satellite_into_features(strict=not args.no_strict, report_only=args.report)
    except FeatureContractError as exc:
        print("\nFEATURE CONTRACT VIOLATION - merge aborted, feature table left untouched.")
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
