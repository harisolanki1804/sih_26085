"""
VARUNA Automated API & Scenario Verification Test
-------------------------------------------------
Validates:
1. Health endpoint & DB connectivity.
2. Regions list and hotspots retrieval.
3. Feature grid snapshot query.
4. Replay step advancement into heavy cloudburst phase.
5. Verification of explainable alert decomposition & trust score metrics.
6. Alert acknowledgement flow.
"""

import os
import sys
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200, f"Health failed: {response.text}"
    data = response.json()
    assert data["status"] == "ONLINE"
    assert data["database"] == "HEALTHY"
    assert data["datasets"]["dem_srtm_loaded"] is True
    print(" [PASSED] Health & Diagnostics Endpoint")


def test_regions():
    response = client.get("/api/v1/regions")
    assert response.status_code == 200
    regions = response.json()
    assert len(regions) >= 1
    region = regions[0]
    assert region["code"] == "IN-MH-BOM-01"

    # Test hotspots
    hotspot_resp = client.get(f"/api/v1/regions/{region['code']}/hotspots")
    assert hotspot_resp.status_code == 200
    hotspots = hotspot_resp.json()
    assert len(hotspots) >= 5
    print(f" [PASSED] Regions API (Found {len(regions)} region, {len(hotspots)} critical hotspots)")


def test_features():
    response = client.get("/api/v1/features/latest?region_code=IN-MH-BOM-01")
    assert response.status_code == 200
    feat_grid = response.json()
    assert feat_grid["total_cells"] == 90
    assert len(feat_grid["cells"]) == 90
    print(f" [PASSED] Features API (Verified 90-cell spatial grid)")


def test_replay_and_explainable_alert():
    # 1. Reset simulation
    client.post("/api/v1/replay/reset")

    # 2. Step to peak cloudburst timestep (e.g., timestep 36)
    step_resp = client.post("/api/v1/replay/step?step_to=36")
    assert step_resp.status_code == 200
    step_data = step_resp.json()
    assert step_data["timestep_id"] == 36
    assert step_data["new_alerts_count"] > 0
    print(f" [PASSED] Replay Step (Jumped to Timestep 36 Peak: {step_data['phase']}, Generated {step_data['new_alerts_count']} alerts)")

    # 3. Query alerts
    alerts_resp = client.get("/api/v1/alerts?is_active=true&severity=CRITICAL")
    assert alerts_resp.status_code == 200
    alerts = alerts_resp.json()
    if not alerts:
        # Fallback to HIGH
        alerts_resp = client.get("/api/v1/alerts?is_active=true")
        alerts = alerts_resp.json()

    assert len(alerts) > 0, "Expected active alerts at peak timestep 36"
    top_alert = alerts[0]
    assert top_alert["risk_score_total"] >= 60.0
    print(f" [PASSED] Active Alerts Query (Top Alert Score: {top_alert['risk_score_total']}, Severity: {top_alert['severity']})")

    # 4. Query explainability endpoint for top alert
    explain_resp = client.get(f"/api/v1/alerts/{top_alert['id']}/explain")
    assert explain_resp.status_code == 200
    exp = explain_resp.json()
    
    # Verify Additive Score Decomposition
    decomp = exp["score_breakdown"]
    total = decomp["rainfall_component"] + decomp["soil_saturation_component"] + decomp["topography_component"] + decomp["instability_component"]
    assert abs(total - decomp["total_score"]) < 0.2
    
    # Verify Trust Score
    trust = exp["trust_breakdown"]
    assert trust["trust_score"] > 75.0
    assert len(trust["key_drivers"]) >= 1

    print(" [PASSED] Explainability & Trust Decomposition Endpoint:")
    print(f"          - Reasoning: {exp['alert']['reasoning_summary'][:90]}...")
    print(f"          - Total Risk: {decomp['total_score']}/100 (Rain: {decomp['rainfall_component']}, Soil: {decomp['soil_saturation_component']}, Topo: {decomp['topography_component']}, CAPE: {decomp['instability_component']})")
    print(f"          - Trust Score: {trust['trust_score']}% ({trust['trust_level']}), Sensor Agreement: {trust['sensor_agreement']}%")

    # 5. Test alert acknowledgement
    ack_resp = client.post(f"/api/v1/alerts/{top_alert['id']}/acknowledge?operator_name=Rudra-EmergencyOps")
    assert ack_resp.status_code == 200
    ack_data = ack_resp.json()
    assert ack_data["acknowledged_by"] == "Rudra-EmergencyOps"
    assert ack_data["acknowledged_at"] is not None
    print(" [PASSED] Operator Alert Acknowledgment Workflow")


def main():
    print("\n--- Running VARUNA Automated API Test Suite ---")
    test_health()
    test_regions()
    test_features()
    test_replay_and_explainable_alert()
    print("\n ALL TEST SUITES PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
