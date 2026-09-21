"""
Member 2 Day 1 Endpoint Verification Test
Tests:
1. GET /api/v1/ai/inference/{timestep_id}
   - Verifies HTTP 200
   - Verifies presence of fused_features.source_contributions
   - Verifies shapes of storm_cells, risk_heatmap, nowcast, multi_hazard, flood_depth, trust_score, xai_explanation
2. POST /api/v1/ai/crowd-report
   - Verifies HTTP 200
   - Verifies confirm_score, deny_score, unrelated_score, actionable, detected_location
   - Tests both flood confirmation and denial cases
"""

import sys
import os
import json

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_inference_endpoint():
    print("Testing GET /api/v1/ai/inference/1 ...")
    resp = client.get("/api/v1/ai/inference/1")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    assert data["timestep_id"] == 1
    assert "timestamp" in data
    assert "storm_cells" in data
    assert "risk_heatmap" in data
    assert "nowcast" in data
    assert "multi_hazard" in data
    assert "fused_features" in data
    assert "flood_depth" in data
    assert "trust_score" in data
    assert "xai_explanation" in data

    fused = data["fused_features"]
    assert "source_contributions" in fused, "source_contributions missing from fused_features"
    sc = fused["source_contributions"]
    assert isinstance(sc, dict), f"source_contributions should be dict, got {type(sc)}"
    for src in ["insat3d_satellite", "imdaa_reanalysis", "srtm_dem", "qpe_rainfall"]:
        assert src in sc, f"Source '{src}' missing from source_contributions"
        assert isinstance(sc[src], (int, float)), f"Value for '{src}' should be numeric, got {sc[src]}"

    print(" [PASSED] GET /api/v1/ai/inference/1 -> HTTP 200")
    print(f"          fused_features.source_contributions = {json.dumps(sc, indent=2)}")
    return data


def test_crowd_report_endpoint():
    print("\nTesting POST /api/v1/ai/crowd-report ...")
    payload_confirm = {
        "report_text": "Heavy waterlogging and flood near Kurla station, water is waist deep! Urgent help required!",
        "predicted_flood_lat": 19.065,
        "predicted_flood_lon": 72.880,
    }
    resp1 = client.post("/api/v1/ai/crowd-report", json=payload_confirm)
    assert resp1.status_code == 200, f"Expected 200, got {resp1.status_code}: {resp1.text}"
    data1 = resp1.json()

    for field in ["report_text", "classification", "confidence", "confirm_score", "deny_score", "unrelated_score", "actionable"]:
        assert field in data1, f"Missing field '{field}' in response: {data1}"

    assert isinstance(data1["confirm_score"], int)
    assert isinstance(data1["deny_score"], int)
    assert isinstance(data1["unrelated_score"], int)
    assert isinstance(data1["actionable"], bool)
    assert data1["detected_location"] == "kurla"

    print(" [PASSED] POST /api/v1/ai/crowd-report (Confirm) -> HTTP 200")
    print(f"          Classification: {data1['classification']}, Conf: {data1['confidence']}, Actionable: {data1['actionable']}")
    print(f"          Scores: confirm={data1['confirm_score']}, deny={data1['deny_score']}, unrelated={data1['unrelated_score']}")
    print(f"          Location: {data1['detected_location']}")

    payload_deny = {
        "report_text": "Everything is clear and dry in Bandra, road is normal, no flood here.",
        "predicted_flood_lat": 19.055,
        "predicted_flood_lon": 72.835,
    }
    resp2 = client.post("/api/v1/ai/crowd-report", json=payload_deny)
    assert resp2.status_code == 200, f"Expected 200, got {resp2.status_code}: {resp2.text}"
    data2 = resp2.json()

    assert data2["detected_location"] == "bandra"
    assert data2["deny_score"] >= 1
    print(" [PASSED] POST /api/v1/ai/crowd-report (Deny) -> HTTP 200")
    print(f"          Classification: {data2['classification']}, Location: {data2['detected_location']}")

    return data1, data2


if __name__ == "__main__":
    print("=" * 60)
    print("MEMBER 2: DAY-1 ENDPOINT VERIFICATION")
    print("=" * 60)
    inf_data = test_inference_endpoint()
    cr1, cr2 = test_crowd_report_endpoint()
    print("\n" + "=" * 60)
    print("ALL DAY-1 ENDPOINT VERIFICATIONS SUCCEEDED (HTTP 200)")
    print("=" * 60)
