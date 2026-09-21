"""
Phase 1 -- Search-only satellite inventory (no downloads).
==========================================================

Builds ``satellite_inventory_manifest.json``: proof of what INSAT-3D/3DR
coverage actually exists over the Mumbai window before anyone downloads a byte.

Why this file was rewritten
---------------------------
The previous revision called the fetcher with keyword arguments it does not
accept::

    search_available_data(product_id=..., boundingBox=...)

The real signature is::

    search_available_data(dataset_key, start_time, end_time, bounding_box, count)

Every one of the 960 attempts raised ``TypeError: unexpected keyword argument
'product_id'``, and each failure was recorded as ``available: false`` with
``granule_id: "N/A"``. The manifest therefore looked like "MOSDAC has no data",
when in fact no request was ever sent. This version sends real requests and
records *why* something is unavailable.

Usage::

    python -m data_pipeline.inventory_scout --probe     # first few steps only
    python -m data_pipeline.inventory_scout             # full 26-29 Jul window
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
backend_dir = os.path.join(root_dir, "Backend")

for path in [backend_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from app.core.timeline_config import EVENT_START_DT, EVENT_END_DT
from app.services.satellite.mosdac_fetcher import (  # noqa: E402
    MOSDACSatelliteFetcher,
    INSAT3D_DATASETS,
)

# Products from the replay plan. Priority order: CTP -> WDP -> UTH -> HEM -> CMK.
REQUESTED_L2_PRODUCTS: Tuple[str, ...] = (
    "3DIMG_L2B_CTP",
    "3DIMG_L2G_WDP",
    "3DIMG_L2B_UTH",
    "3DIMG_L2B_HEM",
    "3DIMG_L2B_CMK",
)

# Imagery products (L1C/L1B) are NOT searched by default: the replay consumes
# only the L2 products above, and every extra catalog query adds ~20s latency.
# Pass --with-imagery to include them.
AVAILABLE_IMAGERY_PRODUCTS: Tuple[str, ...] = (
    "3RIMG_L1C_SGP",
    "3SIMG_L1B_STD",
    "3RIMG_L1B_STD",
)

PRODUCTS: Tuple[str, ...] = REQUESTED_L2_PRODUCTS

# Centralized incident window (2022-07-05 to 2022-07-07)
WINDOW_START = EVENT_START_DT
WINDOW_END = EVENT_END_DT
STEP = timedelta(minutes=30)

# MOSDAC bounding_box format is "minLon,minLat,maxLon,maxLat".
MUMBAI_BBOX = "72.80,18.90,73.10,19.30"

MANIFEST_PATH = os.path.join(backend_dir, "data_pipeline", "satellite_inventory_manifest.json")
SUMMARY_PATH = os.path.join(backend_dir, "data_pipeline", "satellite_inventory_summary.json")

# Dataset registry keys are aliases; every replay product is passed through
# verbatim as the datasetId (verified live: '3DIMG_L2B_UTH' itself returns 47
# granules/day, so remapping it to L2I_UTH is unnecessary and wrong).
DATASET_ALIASES: Dict[str, str] = {
    "3DIMG_L2B_CTP": "3DIMG_L2B_CTP",
    "3DIMG_L2G_WDP": "3DIMG_L2G_WDP",
    "3DIMG_L2B_UTH": "3DIMG_L2B_UTH",
    "3DIMG_L2B_HEM": "3DIMG_L2B_HEM",
    "3DIMG_L2B_CMK": "3DIMG_L2B_CMK",
}


def _granule_id(entry: Dict[str, Any]) -> str:
    """Return the DOWNLOADABLE record id for a search entry.

    MOSDAC's download endpoint is ``download_api/download?id={record_id}`` and
    the record id is the numeric ``entry['id']`` — NOT the filename in
    ``identifier``. Passing the filename 404s.
    """
    for key in ("id", "record_id", "metaid"):
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "UNKNOWN"


def _identifier(entry: Dict[str, Any]) -> str:
    """Return the granule filename (e.g. 3DIMG_05JUL2022_2300_L2B_CTP_V01R00.h5)."""
    for key in ("identifier", "title", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return "UNKNOWN"


def _granule_time(entry: Dict[str, Any]) -> Optional[datetime]:
    """Parse the granule's actual acquisition time.

    ``updated`` is the frame time (e.g. 2022-07-05T23:00:00Z); ``dcDate`` is a
    start/end range whose start matches it.
    """
    raw = entry.get("updated")
    if not raw and isinstance(entry.get("dcDate"), str):
        raw = entry["dcDate"].split("/")[0]
    if not raw:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=None)
    except ValueError:
        return None


def _entry_size_mb(entry: Dict[str, Any]) -> float:
    for key in ("sizeMB", "size_mb", "size"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            # MOSDAC reports bytes for some keys; normalise obvious byte counts.
            return round(float(value) / 1_048_576, 2) if key == "size" else round(float(value), 2)
    return 0.0


def _interpret(status_counts: Dict[str, int], n_available: int) -> str:
    """State plainly whether the search actually ran, so an empty manifest is
    never mistaken for absent satellite coverage."""
    if n_available:
        return "COVERAGE_CONFIRMED"
    failed = sum(
        status_counts.get(s, 0)
        for s in ("api_error", "request_failed", "credentials_not_configured", "auth_failed")
    )
    if failed and failed >= sum(status_counts.values()):
        return (
            "SEARCH_FAILED — no request succeeded; this is NOT evidence that "
            "coverage is absent. Fix credentials/auth/transport and re-run"
        )
    if status_counts.get("no_coverage_for_window"):
        return "SEARCH_RAN_OK_BUT_NO_COVERAGE_FOUND"
    return "INCONCLUSIVE"


def _query(fetcher: MOSDACSatelliteFetcher, dataset_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    """Issue one search call with the fetcher's real signature.

    count=60: half-hourly products emit ~48 granules/day, and the search page
    size defaults to 10 — without raising it, 38 of 48 frames are invisible to
    the manifest.
    """
    return fetcher.search_available_data(
        dataset_key=dataset_id,
        start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
        bounding_box=MUMBAI_BBOX,
        count=60,
    )


def _daily_windows() -> List[Tuple[datetime, datetime]]:
    """The MOSDAC catalog is date-granular, so one query per calendar day.

    The endpoint rejects every datetime form and accepts only ``YYYY-MM-DD``,
    so issuing one query per 30-minute step would repeat the same request 48
    times per day for no extra information. The per-day ``totalResults``
    already reveals the cadence (192/day == half-hourly).
    """
    windows: List[Tuple[datetime, datetime]] = []
    day = WINDOW_START.replace(hour=0, minute=0)
    while day <= WINDOW_END:
        windows.append((day, day + timedelta(days=1)))
        day += timedelta(days=1)
    return windows


def build_inventory_manifest(
    *,
    limit: Optional[int] = None,
    manifest_path: str = MANIFEST_PATH,
    summary_path: str = SUMMARY_PATH,
) -> Dict[str, Any]:
    """Search MOSDAC across the window and write the manifest + summary."""
    fetcher = MOSDACSatelliteFetcher()

    credentials_present = bool(
        getattr(fetcher, "username", None) and getattr(fetcher, "password", None)
    )

    timesteps: List[datetime] = []
    cursor = WINDOW_START
    while cursor <= WINDOW_END:
        timesteps.append(cursor)
        cursor += STEP

    day_windows = _daily_windows()
    if limit:
        day_windows = day_windows[:limit]

    print(
        f"Phase 1: searching {len(PRODUCTS)} product(s) x {len(day_windows)} day window(s) "
        f"over the Mumbai bounding box {MUMBAI_BBOX}"
    )
    if not credentials_present:
        print(
            "  WARNING: MOSDAC credentials are not configured. Every search will\n"
            "  return an error; the manifest will say 'credentials_not_configured',\n"
            "  which is NOT the same as 'no coverage exists'."
        )

    manifest: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    available_products: Dict[str, int] = {p: 0 for p in PRODUCTS}
    first_error: Optional[str] = None

    for ts, window_end in day_windows:
        for product_id in PRODUCTS:
            dataset_id = DATASET_ALIASES.get(product_id, product_id)

            base: Dict[str, Any] = {
                "product_id": product_id,
                "datasetId_requested": dataset_id,
                "window_start": ts.isoformat(),
                "window_end": window_end.isoformat(),
                "boundingBox": MUMBAI_BBOX,
            }

            try:
                result = _query(fetcher, dataset_id, ts, window_end)
            except Exception as exc:  # pragma: no cover - network dependent
                manifest.append({
                    **base,
                    "available": False,
                    "granule_id": "N/A",
                    "status": "request_failed",
                    "notes": f"search raised {type(exc).__name__}: {exc}",
                })
                status_counts["request_failed"] = status_counts.get("request_failed", 0) + 1
                continue

            error = result.get("error") if isinstance(result, dict) else "malformed response"
            entries = result.get("entries", []) if isinstance(result, dict) else []

            if error:
                lowered = str(error).lower()
                if "credential" in lowered:
                    status = "credentials_not_configured"
                elif "authentication" in lowered or "auth" in lowered:
                    status = "auth_failed"
                else:
                    status = "api_error"
                manifest.append({
                    **base,
                    "available": False,
                    "granule_id": "N/A",
                    "status": status,
                    "notes": str(error)[:300],
                })
                first_error = first_error or f"{status}: {error}"
                status_counts[status] = status_counts.get(status, 0) + 1
            elif entries:
                # One manifest record PER GRANULE so the extractor can download
                # each frame individually.
                for gran in entries:
                    g_time = _granule_time(gran)
                    record = {
                        **base,
                        "available": True,
                        "status": "available",
                        "granule_id": _granule_id(gran),
                        "identifier": _identifier(gran),
                        "granule_time": g_time.isoformat() if g_time else None,
                        "sizeMB": _entry_size_mb(gran),
                        "total_results": result.get("totalResults") or len(entries),
                    }
                    manifest.append(record)
                    status_counts["available"] = status_counts.get("available", 0) + 1
                    available_products[product_id] += 1
            else:
                manifest.append({
                    **base,
                    "available": False,
                    "granule_id": "N/A",
                    "status": "no_coverage_for_window",
                    "notes": "search accepted but the catalog has no granules for this product/day",
                })
                status_counts["no_coverage_for_window"] = (
                    status_counts.get("no_coverage_for_window", 0) + 1
                )

    n_available = sum(1 for e in manifest if e["available"])
    # Coverage actually usable by the extractor: per product, distinct frames.
    usable_by_product = {
        p: sum(
            1 for e in manifest
            if e["available"] and e["product_id"] == p and e.get("granule_time")
        )
        for p in PRODUCTS
    }
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "window": {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat(), "step_minutes": 30},
        "bounding_box": MUMBAI_BBOX,
        "credentials_present": credentials_present,
        "products_requested": list(PRODUCTS),
        "requested_l2_products": list(REQUESTED_L2_PRODUCTS),
        "available_imagery_products": list(AVAILABLE_IMAGERY_PRODUCTS),
        "days_searched": len(day_windows),
        "manifest_granularity": "per_granule",
        "entries": len(manifest),
        "entries_available": n_available,
        "usable_by_product": usable_by_product,
        "coverage_pct": round(100.0 * n_available / len(manifest), 2) if manifest else 0.0,
        "status_counts": status_counts,
        "first_error": first_error,
        "interpretation": _interpret(status_counts, n_available),
    }

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {manifest_path}")
    print(f"Wrote {summary_path}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Search-only MOSDAC inventory manifest")
    parser.add_argument("--probe", action="store_true", help="search only the first 2 timesteps")
    parser.add_argument("--limit", type=int, default=None, help="limit number of timesteps searched")
    parser.add_argument("--with-imagery", action="store_true", help="also search L1 imagery products (slow)")
    args = parser.parse_args(argv)

    if args.with_imagery:
        globals()["PRODUCTS"] = REQUESTED_L2_PRODUCTS + AVAILABLE_IMAGERY_PRODUCTS

    limit = args.limit if args.limit is not None else (2 if args.probe else None)
    build_inventory_manifest(limit=limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
