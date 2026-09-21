"""
Alert Dispatch Service
======================
Fires a lightweight webhook notification whenever a HIGH or CRITICAL alert
is created by the replay engine or AI pipeline.

Configure via .env:
  ALERT_WEBHOOK_URL=https://your-endpoint.com/varuna-alert
  ALERT_WEBHOOK_SECRET=optional-secret-token

Falls back silently if ALERT_WEBHOOK_URL is not set — no crash, just a log warning.
"""

import json
import logging
import threading
from datetime import datetime
from typing import Optional

import requests
from app.core.config import settings

logger = logging.getLogger("VARUNA.AlertDispatch")


def _build_payload(alert) -> dict:
    """Construct a clean JSON payload from an Alert ORM object."""
    return {
        "source": "VARUNA",
        "event": "FLOOD_ALERT",
        "alert_id": str(alert.id),
        "severity": alert.severity,
        "alert_type": alert.alert_type,
        "locality": alert.locality_name or f"Cell {alert.cell_id}",
        "risk_score": round(alert.risk_score_total, 1),
        "flood_depth_cm": round(alert.flood_depth_estimate_cm, 1),
        "trust_score": round(alert.trust_score, 1),
        "lat": alert.cell_lat,
        "lon": alert.cell_lon,
        "timestep": alert.timestep,
        "reasoning": alert.reasoning_summary,
        "recommended_action": alert.recommended_action or "",
        "timestamp": alert.timestamp.isoformat() if isinstance(alert.timestamp, datetime) else str(alert.timestamp),
        "dispatched_at": datetime.utcnow().isoformat(),
    }


def _fire_webhook(payload: dict, url: str, secret: Optional[str]):
    """Send the webhook in a background thread — non-blocking."""
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-VARUNA-Secret"] = secret
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        if resp.ok:
            logger.info(f"Alert dispatched → {url} | {payload['severity']} | {payload['locality']}")
        else:
            logger.warning(f"Webhook returned {resp.status_code}: {resp.text[:200]}")
    except requests.RequestException as e:
        logger.warning(f"Webhook dispatch failed (non-critical): {e}")


def dispatch_alert(alert) -> bool:
    """
    Dispatch a flood alert via configured webhook.

    Only fires for HIGH and CRITICAL severities to avoid noise.
    Runs in a background thread so it never blocks the request cycle.

    Returns True if dispatch was attempted, False if not configured.
    """
    if alert.severity not in ("HIGH", "CRITICAL"):
        return False

    url = getattr(settings, "ALERT_WEBHOOK_URL", "")
    if not url:
        logger.debug("ALERT_WEBHOOK_URL not configured — skipping dispatch.")
        return False

    secret = getattr(settings, "ALERT_WEBHOOK_SECRET", "")
    payload = _build_payload(alert)

    # Fire in background thread — never block the API response
    t = threading.Thread(target=_fire_webhook, args=(payload, url, secret), daemon=True)
    t.start()
    return True
