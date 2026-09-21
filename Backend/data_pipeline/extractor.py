"""
VARUNA Phase 2 -- HDF5 Satellite Data Extractor
===============================================

Downloads the MOSDAC granules listed in ``satellite_inventory_manifest.json``
(one record per granule, written by ``inventory_scout.py``), decodes each file
with :mod:`app.services.satellite.product_decoder`, and writes the 90 Mumbai
grid values to ``cache/satellite_cache/{product_id}_{YYYY-MM-DDTHH-MM-SS}.json``.

Resume behaviour
----------------
A granule already cached with ``is_real_data: true`` is skipped, so the run can
be interrupted and continued. A cache entry that was previously written by the
old placeholder path (``is_real_data`` missing/false) is re-extracted.

Provenance
----------
Every cache record says where its values came from::

    {
      "product_id":  "3DIMG_L2B_CTP",
      "granule_id":  "10303213",
      "identifier":  "3DIMG_05JUL2022_2300_L2B_CTP_V01R00.h5",
      "timestamp":   "2022-07-05T23:00:00",
      "grid_values": [...90 floats...],
      "units":       "degC",
      "variable":    "CTT",
      "is_real_data": true,
      "gap_filled":  false,
      "source":      "MOSDAC download id=10303213"
    }

Usage::

    python -m data_pipeline.extractor --limit 10      # smoke test
    python -m data_pipeline.extractor                 # full window
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("VARUNA.Extractor")

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
backend_dir = os.path.join(root_dir, "Backend")

for path in [backend_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import h5py  # noqa: F401  (kept as the hard dependency check)
except ImportError:
    logger.error(
        "h5py is not installed.  Run:  pip install h5py>=3.10.0\\n"
        "It is listed in requirements.txt under '# --- Satellite HDF5 Data ---'."
    )
    raise

from app.services.satellite.mosdac_fetcher import MOSDACSatelliteFetcher  # noqa: E402
from app.services.satellite.product_decoder import decode_product  # noqa: E402

MANIFEST_PATH = os.path.join(backend_dir, "data_pipeline", "satellite_inventory_manifest.json")
CACHE_DIR = os.path.join(backend_dir, "cache", "satellite_cache")
RAW_DIR = os.path.join(backend_dir, "cache", "satellite_raw")

# Download priority from pipeline.md: CTP -> WDP -> UTH -> HEM -> CMK.
PRIORITY = {
    "3DIMG_L2B_CTP": 1,
    "3DIMG_L2G_WDP": 2,
    "3DIMG_L2B_UTH": 3,
    "3DIMG_L2B_HEM": 4,
    "3DIMG_L2B_CMK": 5,
}


def _cache_key(product_id: str, granule_time: str) -> str:
    """Cache filename stem: product + granule acquisition time.

    The manifest's ``granule_time`` is the frame's acquisition time
    (e.g. ``2022-07-05T23:00:00``); colons are replaced so the name is
    Windows-safe. ``derivation_engine`` rebuilds the exact same key, so the
    two stages stay in lockstep.
    """
    ts = (granule_time or "").replace(":", "-")
    return f"{product_id}_{ts}"


def _already_cached_real(cache_dir: str, stem: str) -> bool:
    """True when a real decoded payload already exists for this granule."""
    path = os.path.join(cache_dir, stem + ".json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return bool(json.load(handle).get("is_real_data"))
    except (OSError, json.JSONDecodeError):
        return False


def run_extraction_pipeline(
    *,
    limit: Optional[int] = None,
    manifest_path: str = MANIFEST_PATH,
    cache_dir: str = CACHE_DIR,
    keep_raw: bool = True,
) -> Dict[str, Any]:
    fetcher = MOSDACSatelliteFetcher()
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    if not os.path.exists(manifest_path):
        logger.error("Manifest missing. Run Phase 1 (inventory_scout) first.")
        return {"error": "manifest_missing"}

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    available = [e for e in manifest if e.get("available") and e.get("granule_id") not in (None, "N/A", "UNKNOWN")]
    available.sort(key=lambda e: (
        PRIORITY.get(e["product_id"], 99),
        e.get("granule_time") or "",
    ))
    if limit:
        available = available[:limit]

    if not available:
        print(
            "No downloadable granules in the manifest. Re-run inventory_scout.py "
            "and check its summary.interpretation."
        )
        return {"downloaded": 0, "decoded": 0, "skipped": 0, "failed": 0}

    print(
        f"Phase 2: {len(available)} granule(s) to process "
        f"({len({e['product_id'] for e in available})} products); "
        f"resume-enabled cache at {os.path.relpath(cache_dir, root_dir)}"
    )

    stats = {"downloaded": 0, "reused_raw": 0, "decoded": 0, "skipped": 0, "failed": 0}
    failures: List[str] = []
    t_start = time.time()

    for item in available:
        product_id = item["product_id"]
        record_id = str(item["granule_id"])
        granule_time = item.get("granule_time") or ""
        stem = _cache_key(product_id, granule_time)

        if _already_cached_real(cache_dir, stem):
            stats["skipped"] += 1
            continue

        raw_path = os.path.join(RAW_DIR, f"{record_id}.h5")
        try:
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                stats["reused_raw"] += 1
            else:
                got = fetcher.download_granule(record_id, dest_path=raw_path)
                if not got or not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
                    stats["failed"] += 1
                    failures.append(f"{product_id}@{granule_time}: download failed ({record_id})")
                    continue
                stats["downloaded"] += 1

            decoded = decode_product(raw_path, product_id)
            payload = {
                "product_id": product_id,
                "granule_id": record_id,
                "identifier": item.get("identifier", ""),
                "timestamp": granule_time,
                "grid_values": decoded["values"],
                "units": decoded["units"],
                "variable": decoded["variable"],
                "derived": decoded.get("derived", {}),
                "is_real_data": True,
                "gap_filled": False,
                "source": f"MOSDAC download id={record_id}",
                "extracted_at": datetime.utcnow().isoformat(),
            }
            out_path = os.path.join(cache_dir, stem + ".json")
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, out_path)

            stats["decoded"] += 1
            if stats["decoded"] % 25 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  progress: {stats['decoded']} decoded, "
                    f"{stats['failed']} failed, {elapsed:.0f}s elapsed"
                )

        except Exception as exc:
            stats["failed"] += 1
            failures.append(f"{product_id}@{granule_time}: {type(exc).__name__}: {exc}")
            logger.error("Failed to extract %s @ %s: %s", product_id, granule_time, exc)
        finally:
            if not keep_raw and os.path.exists(raw_path):
                os.remove(raw_path)

    elapsed = time.time() - t_start
    print("\nPhase 2 complete in {:.0f}s".format(elapsed))
    print(
        f"  decoded:   {stats['decoded']}\n"
        f"  reused:    {stats['reused_raw']}\n"
        f"  downloaded:{stats['downloaded']}\n"
        f"  skipped:   {stats['skipped']} (already cached)\n"
        f"  failed:    {stats['failed']}"
    )
    if failures:
        show = failures[:10]
        print("  first failures:")
        for line in show:
            print(f"    - {line}")
        if len(failures) > len(show):
            print(f"    ... and {len(failures) - len(show)} more")

    result = dict(stats)
    result["failures_sample"] = failures[:20]
    result["elapsed_seconds"] = round(elapsed, 1)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download + decode MOSDAC granules into the satellite cache")
    parser.add_argument("--limit", type=int, default=None, help="process at most N granules (smoke test)")
    parser.add_argument("--no-keep-raw", action="store_true", help="delete .h5 files after decoding (saves disk)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_extraction_pipeline(limit=args.limit, keep_raw=not args.no_keep_raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
