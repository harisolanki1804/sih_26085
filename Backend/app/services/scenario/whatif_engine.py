"""
VARUNA What-If Scenario Engine
================================
Allows operators to simulate hypothetical extreme events:
- "What if rainfall doubles?"
- "What if tide is at maximum spring level?"
- "What if drainage fails in Kurla?"
- "What if 2 storms converge?"

This is a practical decision-support tool that enables:
- Infrastructure stress testing
- Emergency drill planning
- What-if analysis for disaster preparedness
- Sensitivity analysis showing which factor matters most

Uses the physics-informed risk model to ensure scenarios
are physically consistent (not just scaled numbers).
"""

import math
import random
import copy
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("VARUNA.Scenario")

# Pre-defined extreme scenarios based on historical events
PREDEFINED_SCENARIOS = [
    {
        "id": "EXTREME_RAINFALL_2X",
        "name": "Double Rainfall Intensity",
        "description": "What if rainfall intensity doubles across Mumbai? Simulates an unprecedented convective event.",
        "category": "rainfall",
        "modifications": {"rainfall_multiplier": 2.0},
        "reference_event": "26 July 2005 Mumbai Deluge (944mm in 24hrs)",
    },
    {
        "id": "EXTREME_RAINFALL_3X",
        "name": "Triple Rainfall Intensity",
        "description": "What-if catastrophic 3× rainfall — beyond any recorded event in Mumbai history.",
        "category": "rainfall",
        "modifications": {"rainfall_multiplier": 3.0},
        "reference_event": "Theoretical — no historical analog",
    },
    {
        "id": "HIGH_SPRING_TIDE",
        "name": "Maximum Spring Tide + Storm Surge",
        "description": "Combined high spring tide (4.8m) with 1m storm surge during peak rainfall.",
        "category": "tidal",
        "modifications": {"tide_height_override": 4.8, "storm_surge_m": 1.0},
        "reference_event": "Spring tide + cyclone combination",
    },
    {
        "id": "DRAINAGE_FAILURE_KURLA",
        "name": "Drainage System Failure at Kurla",
        "description": "Complete drainage blockage at Kurla junction — Mithi River overflow scenario.",
        "category": "infrastructure",
        "modifications": {"drainage_failure_cells": [36, 37, 38], "failure_severity": 1.0},
        "reference_event": "Kurla 2017 flooding",
    },
    {
        "id": "CYCLONE_ATTACHMENT",
        "name": "Cyclone Landfall near Mumbai",
        "description": "Cyclonic system making landfall with 120km/h winds and 200mm/hr rainfall.",
        "category": "cyclone",
        "modifications": {"wind_multiplier": 3.0, "rainfall_multiplier": 2.5, "cape_multiplier": 1.5},
        "reference_event": "Cyclone Tauktae 2021 trajectory",
    },
    {
        "id": "URBAN_HEAT_ISLAND",
        "name": "Urban Heat Island Enhancement",
        "description": "Enhanced convection over concrete jungle — BKC/Dadar heat island intensifies storm.",
        "category": "atmospheric",
        "modifications": {"cape_multiplier": 1.8, "ctt_drop_celsius": -15, "affected_cells": list(range(11, 30))},
        "reference_event": "Urban heat island effect research",
    },
    {
        "id": "MULTI_HAZARD_COMPOUND",
        "name": "Compound Event: Tide + Rain + Drainage Failure",
        "description": "Worst-case compound event — simultaneous tidal lock, extreme rain, and drainage failure.",
        "category": "compound",
        "modifications": {
            "tide_height_override": 4.6,
            "rainfall_multiplier": 1.8,
            "drainage_failure_cells": [27, 28, 29, 36, 37, 38],
            "failure_severity": 0.8,
        },
        "reference_event": "Compound event analysis",
    },
    {
        "id": "CLIMATE_2050",
        "name": "Climate Change 2050 Projection",
        "description": "Projected 2050 climate: +20% rainfall intensity, +0.3m sea level rise, more intense CAPE.",
        "category": "climate",
        "modifications": {
            "rainfall_multiplier": 1.2,
            "sea_level_rise_m": 0.3,
            "cape_multiplier": 1.3,
            "tide_offset_m": 0.3,
        },
        "reference_event": "IPCC AR6 SSP2-4.5 projection for Mumbai",
    },
]


class WhatIfScenarioEngine:
    """
    What-If Scenario Engine for operational decision support.

    Allows operators to simulate hypothetical modifications to
    current conditions and see how risk predictions change.
    """

    def __init__(self):
        self.predefined = {s["id"]: s for s in PREDEFINED_SCENARIOS}

    def list_scenarios(self) -> List[Dict]:
        """List all available predefined scenarios."""
        return [
            {
                "id": s["id"],
                "name": s["name"],
                "description": s["description"],
                "category": s["category"],
                "reference_event": s.get("reference_event", ""),
            }
            for s in PREDEFINED_SCENARIOS
        ]

    def run_scenario(
        self,
        scenario_id: str,
        timestep_data: Dict[str, Any],
        all_timesteps: List[Dict[str, Any]],
        timestep_idx: int,
        custom_modifications: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run a what-if scenario by modifying current conditions
        and re-running the risk model.

        Returns:
            - baseline risk (current conditions)
            - scenario risk (modified conditions)
            - delta analysis (what changed and why)
            - sensitivity analysis (which factor matters most)
        """
        # Get scenario modifications
        if scenario_id in self.predefined:
            mods = self.predefined[scenario_id]["modifications"].copy()
            scenario_name = self.predefined[scenario_id]["name"]
        elif scenario_id == "CUSTOM":
            mods = custom_modifications or {}
            scenario_name = "Custom Scenario"
        else:
            return {"error": f"Unknown scenario: {scenario_id}"}

        # Create modified timestep data
        modified_data = self._apply_modifications(timestep_data, mods)

        # Run baseline inference (current conditions)
        baseline = self._compute_risk(timestep_data)

        # Run scenario inference (modified conditions)
        scenario = self._compute_risk(modified_data)

        # Compute deltas
        delta_analysis = self._compute_deltas(baseline, scenario, mods)

        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(timestep_data)

        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "modifications_applied": mods,
            "baseline_risk": baseline,
            "scenario_risk": scenario,
            "delta_analysis": delta_analysis,
            "sensitivity": sensitivity,
            "infrastructure_impact": self._assess_infrastructure_impact(scenario, mods),
            "recommendation": self._generate_scenario_recommendation(
                baseline, scenario, delta_analysis, scenario_name
            ),
        }

    def sensitivity_analysis(
        self, timestep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze which input factor has the largest impact on risk.

        Tests each factor independently:
        - 2× rainfall
        - +2m tide
        - Drainage failure
        - +1000 J/kg CAPE
        - -10°C cloud top temperature
        """
        return self._sensitivity_analysis(timestep_data)

    # =================================================================
    # INTERNAL METHODS
    # =================================================================

    def _apply_modifications(
        self, timestep_data: Dict, mods: Dict
    ) -> Dict:
        """Apply scenario modifications to timestep data."""
        modified = copy.deepcopy(timestep_data)

        # Rainfall multiplier
        if "rainfall_multiplier" in mods:
            mult = mods["rainfall_multiplier"]
            for cell in modified["features"]:
                cell["rainfall_1h_mm"] = round(cell["rainfall_1h_mm"] * mult, 2)
                cell["rainfall_3h_mm"] = round(cell["rainfall_3h_mm"] * mult, 2)
                cell["rainfall_6h_mm"] = round(cell["rainfall_6h_mm"] * mult, 2)
                cell["rainfall_24h_mm"] = round(cell["rainfall_24h_mm"] * mult, 2)
            modified["avg_rainfall_1h_mm"] = round(modified.get("avg_rainfall_1h_mm", 0) * mult, 2)

        # Tide override
        if "tide_height_override" in mods:
            modified["tide_height_m"] = mods["tide_height_override"]
        if "storm_surge_m" in mods:
            modified["tide_height_m"] = modified.get("tide_height_m", 2.5) + mods["storm_surge_m"]
        if "tide_offset_m" in mods:
            modified["tide_height_m"] = modified.get("tide_height_m", 2.5) + mods["tide_offset_m"]

        # Sea level rise
        if "sea_level_rise_m" in mods:
            rise = mods["sea_level_rise_m"]
            for cell in modified["features"]:
                cell["elevation_m"] = max(0, cell["elevation_m"] - rise)

        # Wind multiplier
        if "wind_multiplier" in mods:
            mult = mods["wind_multiplier"]
            for cell in modified["features"]:
                cell["wind_speed_10m_kmh"] = round(cell["wind_speed_10m_kmh"] * mult, 1)
                cell["wind_gusts_kmh"] = round(cell.get("wind_gusts_kmh", 30) * mult, 1)

        # CAPE multiplier
        if "cape_multiplier" in mods:
            mult = mods["cape_multiplier"]
            for cell in modified["features"]:
                cell["cape_instability_jkg"] = round(cell["cape_instability_jkg"] * mult, 0)

        # CTT drop
        if "ctt_drop_celsius" in mods:
            drop = mods["ctt_drop_celsius"]
            for cell in modified["features"]:
                cell["cloud_top_temp_celsius"] = round(cell["cloud_top_temp_celsius"] + drop, 1)

        # Drainage failure
        if "drainage_failure_cells" in mods:
            failure_cells = mods["drainage_failure_cells"]
            severity = mods.get("failure_severity", 1.0)
            for cell in modified["features"]:
                if cell["cell_index"] in failure_cells:
                    cell["drainage_outfall_dist_m"] = round(
                        cell["drainage_outfall_dist_m"] * (1 + severity * 5), 0
                    )
                    cell["slope_deg"] = round(cell["slope_deg"] * 0.1, 2)  # flatten
                    cell["is_depression_bowl"] = True

        # Affected cells only
        if "affected_cells" in mods:
            affected = set(mods["affected_cells"])
            for cell in modified["features"]:
                if cell["cell_index"] not in affected:
                    # Reset to baseline for unaffected cells
                    pass  # keep as-is for simplicity

        return modified

    def _compute_risk(self, timestep_data: Dict) -> Dict:
        """Compute risk for all cells using the risk model."""
        from app.services.risk_model import risk_model_service

        features = timestep_data.get("features", [])
        is_tide = timestep_data.get("is_high_tide_locked", False)
        tide_h = timestep_data.get("tide_height_m", 2.5)

        cell_risks = []
        for cell in features:
            result = risk_model_service.calculate_cell_risk(
                rain_1h=cell.get("rainfall_1h_mm", 0),
                rain_3h=cell.get("rainfall_3h_mm", 0),
                rain_24h=cell.get("rainfall_24h_mm", 0),
                soil_moist_pct=cell.get("soil_moisture_pct", 50),
                elevation_m=cell.get("elevation_m", 10),
                slope_deg=cell.get("slope_deg", 2),
                cape_jkg=cell.get("cape_instability_jkg", 0),
                is_tide_locked=is_tide,
                tide_height_m=tide_h,
                drainage_outfall_dist_m=cell.get("drainage_outfall_dist_m", 5000),
                is_depression=cell.get("is_depression_bowl", False),
            )
            cell_risks.append({
                "cell_index": cell["cell_index"],
                "risk_score": result["total_risk_score"],
                "severity": result["severity"],
                "flood_depth_cm": result["flood_depth_estimate_cm"],
            })

        avg_risk = sum(c["risk_score"] for c in cell_risks) / max(1, len(cell_risks))
        max_risk = max(c["risk_score"] for c in cell_risks)
        critical = sum(1 for c in cell_risks if c["severity"] == "CRITICAL")
        high = sum(1 for c in cell_risks if c["severity"] == "HIGH")
        max_depth = max(c["flood_depth_cm"] for c in cell_risks)

        return {
            "avg_risk_score": round(avg_risk, 1),
            "max_risk_score": round(max_risk, 1),
            "critical_cells": critical,
            "high_risk_cells": high,
            "max_flood_depth_cm": round(max_depth, 1),
            "cell_risks": sorted(cell_risks, key=lambda c: -c["risk_score"]),
        }

    def _compute_deltas(self, baseline, scenario, mods) -> Dict:
        """Compute how risk changed between baseline and scenario."""
        avg_delta = scenario["avg_risk_score"] - baseline["avg_risk_score"]
        max_delta = scenario["max_risk_score"] - baseline["max_risk_score"]
        depth_delta = scenario["max_flood_depth_cm"] - baseline["max_flood_depth_cm"]
        critical_delta = scenario["critical_cells"] - baseline["critical_cells"]

        # Per-cell deltas
        base_by_idx = {c["cell_index"]: c for c in baseline["cell_risks"]}
        sc_by_idx = {c["cell_index"]: c for c in scenario["cell_risks"]}

        worst_affected = []
        for idx in sc_by_idx:
            if idx in base_by_idx:
                delta = sc_by_idx[idx]["risk_score"] - base_by_idx[idx]["risk_score"]
                if delta > 5:
                    worst_affected.append({
                        "cell_index": idx,
                        "baseline_risk": base_by_idx[idx]["risk_score"],
                        "scenario_risk": sc_by_idx[idx]["risk_score"],
                        "delta": round(delta, 1),
                    })
        worst_affected.sort(key=lambda c: -c["delta"])

        return {
            "avg_risk_change": round(avg_delta, 1),
            "max_risk_change": round(max_delta, 1),
            "max_depth_change_cm": round(depth_delta, 1),
            "additional_critical_cells": critical_delta,
            "severity_escalation": (
                "CATASTROPHIC" if avg_delta > 30
                else "SEVERE" if avg_delta > 20
                else "SIGNIFICANT" if avg_delta > 10
                else "MODERATE" if avg_delta > 5
                else "MINOR"
            ),
            "worst_affected_cells": worst_affected[:10],
        }

    def _sensitivity_analysis(self, timestep_data: Dict) -> Dict:
        """Test which factor has the biggest impact on risk."""
        factors = [
            {"name": "2× Rainfall", "mods": {"rainfall_multiplier": 2.0}},
            {"name": "High Tide (4.8m)", "mods": {"tide_height_override": 4.8}},
            {"name": "+1000 CAPE", "mods": {"cape_multiplier": 1.5}},
            {"name": "Drainage Failure (Dadar)", "mods": {"drainage_failure_cells": [27, 28, 29], "failure_severity": 1.0}},
            {"name": "3× Wind Speed", "mods": {"wind_multiplier": 3.0}},
            {"name": "CTT Drop -15°C", "mods": {"ctt_drop_celsius": -15}},
        ]

        baseline = self._compute_risk(timestep_data)
        results = []

        for factor in factors:
            modified = self._apply_modifications(timestep_data, factor["mods"])
            scenario = self._compute_risk(modified)
            delta = scenario["avg_risk_score"] - baseline["avg_risk_score"]
            results.append({
                "factor": factor["name"],
                "avg_risk_delta": round(delta, 1),
                "max_risk_delta": round(scenario["max_risk_score"] - baseline["max_risk_score"], 1),
                "critical_cells_delta": scenario["critical_cells"] - baseline["critical_cells"],
            })

        results.sort(key=lambda r: -r["avg_risk_delta"])

        return {
            "baseline_avg_risk": baseline["avg_risk_score"],
            "factors_tested": len(results),
            "results": results,
            "most_impactful": results[0]["factor"] if results else "N/A",
        }

    def _assess_infrastructure_impact(self, scenario_risk: Dict, mods: Dict) -> Dict:
        """Assess impact on critical infrastructure."""
        max_depth = scenario_risk["max_flood_depth_cm"]
        critical = scenario_risk["critical_cells"]

        # Critical infrastructure locations
        infrastructure = [
            {"name": "Mumbai Local Rail (Western Line)", "type": "rail", "flood_threshold_cm": 15, "cells": [3, 12, 21, 30, 39, 48, 57, 66, 75, 84]},
            {"name": "Mumbai Local Rail (Central Line)", "type": "rail", "flood_threshold_cm": 15, "cells": [5, 14, 23, 32, 41, 50, 59, 68, 77, 86]},
            {"name": "Western Express Highway", "type": "road", "flood_threshold_cm": 30, "cells": [48, 57, 66, 75, 84]},
            {"name": "Eastern Express Highway", "type": "road", "flood_threshold_cm": 30, "cells": [8, 17, 26, 35, 44, 53, 62, 71, 80, 89]},
            {"name": "Bandra-Kurla Complex", "type": "commercial", "flood_threshold_cm": 10, "cells": [32, 33, 41, 42]},
            {"name": "Mumbai Airport (CSIA)", "type": "airport", "flood_threshold_cm": 20, "cells": [50, 51, 59, 60]},
        ]

        impact_analysis = []
        for infra in infrastructure:
            # Check if any of its cells are at risk
            at_risk_cells = []
            for cell_idx in infra["cells"]:
                cell_risks = [c for c in scenario_risk["cell_risks"] if c["cell_index"] == cell_idx]
                if cell_risks and cell_risks[0]["flood_depth_cm"] > infra["flood_threshold_cm"]:
                    at_risk_cells.append(cell_risks[0])

            impact = "OPERATIONAL" if not at_risk_cells else (
                "IMPAIRED" if len(at_risk_cells) < len(infra["cells"]) // 2
                else "DISRUPTED"
            )
            impact_analysis.append({
                "name": infra["name"],
                "type": infra["type"],
                "impact_status": impact,
                "cells_at_risk": len(at_risk_cells),
                "total_cells": len(infra["cells"]),
            })

        return {
            "max_flood_depth_cm": max_depth,
            "critical_cells_count": critical,
            "infrastructure_impact": impact_analysis,
            "overall_infrastructure_status": (
                "ALL_OPERATIONAL" if all(i["impact_status"] == "OPERATIONAL" for i in impact_analysis)
                else "PARTIAL_DISRUPTION" if any(i["impact_status"] == "IMPAIRED" for i in impact_analysis)
                else "MAJOR_DISRUPTION"
            ),
        }

    def _generate_scenario_recommendation(
        self, baseline, scenario, delta, scenario_name
    ) -> str:
        """Generate human-readable recommendation for the scenario."""
        severity = delta["severity_escalation"]
        avg_risk = scenario["avg_risk_score"]
        critical = scenario["critical_cells"]

        if severity == "CATASTROPHIC":
            return (
                f"🚨 CATASTROPHIC: {scenario_name} would cause catastrophic flooding across Mumbai. "
                f"Average risk jumps by {delta['avg_risk_change']:.0f} points to {avg_risk:.0f}/100. "
                f"{critical} cells reach CRITICAL level. "
                f"IMMEDIATE citywide evacuation required. All rail/road infrastructure disrupted."
            )
        elif severity == "SEVERE":
            return (
                f"⚠️ SEVERE: {scenario_name} would significantly worsen conditions. "
                f"Average risk increases by {delta['avg_risk_change']:.0f} points. "
                f"{delta['additional_critical_cells']} additional cells become CRITICAL. "
                f"Evacuate low-lying areas immediately. Deploy emergency pumping at bottlenecks."
            )
        elif severity == "SIGNIFICANT":
            return (
                f"⚠️ SIGNIFICANT: {scenario_name} would notably increase flood risk. "
                f"Average risk rises {delta['avg_risk_change']:.0f} points. "
                f"Pre-position emergency teams in affected zones. Activate SMS alerts."
            )
        else:
            return (
                f"ℹ️ MINOR IMPACT: {scenario_name} shows manageable increase in risk. "
                f"Average risk changes by {delta['avg_risk_change']:.0f} points. "
                f"Monitor conditions. No immediate action required."
            )

    def run_custom_scenario(
        self,
        scenario_id: str,
        modifications: Dict[str, Any],
        timestep_idx: int = 35,
    ) -> Dict[str, Any]:
        """
        Run a custom scenario with arbitrary modifications.
        Used by the chatbot interface.
        """
        import os, json
        from app.core.config import settings

        path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        timesteps = data["timesteps"]
        idx = max(0, min(len(timesteps) - 1, timestep_idx))
        ts = timesteps[idx]

        return self.run_scenario(
            scenario_id="CUSTOM",
            timestep_data=ts,
            all_timesteps=timesteps,
            timestep_idx=idx,
            custom_modifications=modifications,
        )


# Singleton
scenario_engine = WhatIfScenarioEngine()
