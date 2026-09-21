"""
VARUNA Timeline Configuration — Single Source of Truth
=======================================================
Every pipeline script, endpoint default, and seed function imports from
here.  When the incident window changes, edit **only this file**.

Why this exists: the previous 2024-07-26 date was hardcoded in ~10 files
plus 192 filenames.  That duplication caused the original disaster where
the MOSDAC 3DIMG L2 archive (which ends 2024-06-18) could never serve
data for a window starting five weeks later.

Current window: 2022-07-05  (confirmed: all 5 MOSDAC products have
46–47 granules/day on this date — near half-hourly cadence).
"""

from datetime import datetime, timedelta

# ── Incident Window ────────────────────────────────────────────────────
EVENT_START = "2022-07-05"
EVENT_END = "2022-07-07"
EVENT_STEP_HOURS = 1
TOTAL_TIMESTEPS = 72

# ── Derived Constants ──────────────────────────────────────────────────
EVENT_CODE = "EVT-BOM-20220705-DELUGE"
EVENT_START_DT = datetime(2022, 7, 5, 0, 0, 0)
EVENT_END_DT = datetime(2022, 7, 7, 23, 0, 0)

# ── Helpers ────────────────────────────────────────────────────────────

def event_timestamp(timestep_1indexed: int) -> str:
    """Return the ISO timestamp for a given 1-indexed timestep.

    Timesteps beyond 24 hours use synthetic hour overflow
    (e.g. ``2022-07-05T36:00:00Z``) for backward compatibility with
    the replay engine's ``parse_iso_timestamp`` helper.
    """
    hour = timestep_1indexed - 1
    return f"{EVENT_START}T{hour:02d}:00:00Z"


def event_datetime(timestep_1indexed: int) -> datetime:
    """Return a proper datetime for a given 1-indexed timestep."""
    return EVENT_START_DT + timedelta(hours=timestep_1indexed - 1)
