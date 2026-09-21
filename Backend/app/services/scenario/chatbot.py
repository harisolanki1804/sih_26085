"""
VARUNA What-If Chatbot
======================
Conversational interface for what-if scenario planning.

Users type natural language questions like:
- "What if rain doubles in Andheri?"
- "What happens if the tide is really high?"
- "Will Bandra flood if drainage fails?"
- "How bad would a cyclone be?"

The chatbot parses the question, maps it to scenario parameters,
runs the physics-informed risk model, and returns a plain-language answer.
"""

import re
import math
import random
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("VARUNA.Chatbot")

# ── Locality name → cell index mapping ──────────────────────────
LOCALITY_TO_CELL = {
    # South Mumbai
    "colaba": 0, "fort": 1, "churchgate": 2, "marine drive": 3,
    "nariman point": 4, "malabar hill": 5, "walkeshwar": 6, "haji ali": 7,
    "grant road": 9, "tardeo": 10, "bhuleshwar": 11, "girgaon": 12,
    "parel": 13, "mahalaxmi": 14, "byculla": 15, "mazgaon": 16,
    "mumbai central": 18, "worli": 19, "matunga": 20, "sion": 21,
    "wadala": 22, "sewri": 23, "chinchpokli": 24, "reay road": 25,
    # Central
    "mahim": 27, "dadar": 28, "dadar west": 28, "dadar east": 29,
    "kurla": 30, "vidyavihar": 31, "ghatkopar": 32, "bkc": 33,
    "bandra kurla complex": 33, "kalina": 34, "santacruz": 35,
    # West
    "bandra": 37, "bandra west": 37, "khar": 38, "chembur": 39,
    "powai": 40, "hiranandani": 41, "navi mumbai": 44,
    # Suburbs
    "juhu": 46, "juhu beach": 45, "versova": 47, "lokhandwala": 48,
    "saki naka": 49, "vikhroli": 51, "kanjurmarg": 52, "nahur": 53,
    "amboli": 54, "jogeshwari": 55, "andheri": 56, "andheri west": 56,
    "andheri east": 57, "marol": 58, "powai lake": 59, "chandivali": 60,
    "bhandup": 62, "mulund": 63,
    # North
    "malvani": 63, "malad": 64, "malad west": 64, "goregaon": 65,
    "kandivali": 66, "borivali": 67, "deonar": 68, "govandi": 69,
    "thane": 71, "thane creek": 72,
    "dahisar": 76, "mira road": 77, "thane west": 78,
    "kopar khairane": 79, "vasai": 81, "nallasopara": 82,
    "vashi": 85, "nerul": 87, "belapur": 88,
}

# ── Scenario keyword patterns ───────────────────────────────────
SCENARIO_PATTERNS = [
    # Rainfall multiplier
    (r"(?:rain|rainfall|precipitation)\s+(?:doubles?|2x|twice|2\s*times)",
     "EXTREME_RAINFALL_2X", {"rainfall_multiplier": 2.0}),
    (r"(?:double|2x|twice|2\s*times)\s+(?:the\s+)?(?:rain|rainfall|precipitation)",
     "EXTREME_RAINFALL_2X", {"rainfall_multiplier": 2.0}),
    (r"(?:rain|rainfall|precipitation)\s+(?:triples?|3x|thrice|3\s*times)",
     "EXTREME_RAINFALL_3X", {"rainfall_multiplier": 3.0}),
    (r"(?:triple|3x|thrice|3\s*times)\s+(?:the\s+)?(?:rain|rainfall|precipitation)",
     "EXTREME_RAINFALL_3X", {"rainfall_multiplier": 3.0}),
    (r"(?:increase|boost|raise)\s*(?:the\s*)?(?:rain|rainfall)\s*(?:by\s*)?(\d+)\s*%",
     None, None),  # dynamic

    # Tide
    (r"(?:high\s+tide|spring\s+tide|tidal\s+surge|tide\s+(?:is|goes?)\s+(?:really\s+)?(?:high|up|maximum))",
     "HIGH_SPRING_TIDE", {"tide_height_override": 4.8, "storm_surge_m": 1.0}),

    # Drainage failure
    (r"(?:drain(?:age)?\s+(?:fail|block|clog|overflow|choke))",
     "DRAINAGE_FAILURE_KURLA", {"drainage_failure_cells": [30, 31, 32], "failure_severity": 1.0}),

    # Cyclone
    (r"(?:cyclone|hurricane|typhoon|storm\s*surge|tropical\s*storm)",
     "CYCLONE_ATTACHMENT", None),

    # Compound
    (r"(?:worst\s*case|everything\s+(?:at\s+once|together|combined)|compound|all\s+(?:of\s+)?them)",
     "MULTI_HAZARD_COMPOUND", None),

    # Climate
    (r"(?:2050|future|climate\s+change|global\s+warming|next\s+generation)",
     "CLIMATE_2050", None),
]


def _extract_locality(text: str) -> Optional[str]:
    """Extract Mumbai locality name from user text."""
    text_lower = text.lower().strip()
    # Try exact matches first (longer names first)
    sorted_names = sorted(LOCALITY_TO_CELL.keys(), key=len, reverse=True)
    for name in sorted_names:
        if name in text_lower:
            return name
    return None


def _extract_scenario(text: str) -> Optional[Tuple[str, Dict]]:
    """Extract scenario type and parameters from user text."""
    text_lower = text.lower().strip()

    # Check for dynamic rainfall multiplier
    match = re.search(r"(?:increase|boost|raise|make)\s*(?:the\s*)?(?:rain|rainfall)\s*(?:by\s*)?(\d+)\s*%", text_lower)
    if match:
        pct = int(match.group(1))
        multiplier = 1.0 + pct / 100.0
        return ("DYNAMIC_RAINFALL", {"rainfall_multiplier": round(multiplier, 2)})

    # Check for specific multiplier mentions
    match = re.search(r"(\d+)\s*(?:x|times|fold)\s*(?:the\s*)?(?:rain|rainfall)", text_lower)
    if match:
        mult = int(match.group(1))
        return (f"YNAMIC_{mult}X_RAINFALL", {"rainfall_multiplier": float(mult)})

    # Check predefined patterns
    for pattern, scenario_id, mods in SCENARIO_PATTERNS:
        if re.search(pattern, text_lower):
            return (scenario_id, mods)

    return None


def _is_about_flooding(text: str) -> bool:
    """Check if the question is about flood risk."""
    keywords = ["flood", "waterlog", "inundat", "submerge", "overflow",
                "risk", "danger", "safe", "unsafe", "damage", "impact",
                "bad", "worse", "severity", "happen", "happens", "will"]
    return any(k in text.lower() for k in keywords)


def _generate_answer(scenario_id: str, mods: Dict, cell_name: Optional[str],
                     baseline_risk: float, scenario_risk: float,
                     affected_cells: int, total_cells: int) -> str:
    """Generate a plain-language answer."""
    delta = scenario_risk - baseline_risk
    pct_change = (delta / max(baseline_risk, 1)) * 100

    # Severity assessment
    if scenario_risk >= 80:
        severity = "CRITICAL"
        emoji = "🔴"
        danger_phrase = "extremely dangerous — widespread flooding expected"
    elif scenario_risk >= 60:
        severity = "HIGH"
        emoji = "🟠"
        danger_phrase = "very dangerous — significant waterlogging in low-lying areas"
    elif scenario_risk >= 40:
        severity = "MEDIUM"
        emoji = "🟡"
        danger_phrase = "concerning — localized flooding possible"
    else:
        severity = "LOW"
        emoji = "🟢"
        danger_phrase = "manageable — standard monsoon precautions apply"

    location = f"in {cell_name.title()}" if cell_name else "across Mumbai"

    # Build answer based on scenario type
    if "RAINFALL" in scenario_id or "rainfall" in str(mods):
        mult = mods.get("rainfall_multiplier", 2.0)
        answer = (
            f"{emoji} **If rainfall increases {mult}x {location}:**\n\n"
            f"The risk level jumps from **{baseline_risk:.0f}** to **{scenario_risk:.0f}** "
            f"({pct_change:+.0f}% change). This would be **{danger_phrase}**.\n\n"
        )
        if scenario_risk >= 60:
            answer += (
                f"• **{affected_cells} out of {total_cells}** areas would face significant flooding\n"
                f"• Low-lying areas (elevation < 5m) would see water depths exceeding **50 cm**\n"
                f"• Drainage systems would be completely overwhelmed\n"
                f"• **Recommendation:** Evacuate low-lying areas immediately, deploy portable pumps"
            )
        else:
            answer += (
                f"• Most drainage systems can handle this increase\n"
                f"• A few depression bowls may see minor waterlogging\n"
                f"• **Recommendation:** Monitor closely, no evacuation needed yet"
            )

    elif "TIDE" in scenario_id or "tide" in str(mods):
        answer = (
            f"{emoji} **If extreme high tide (4.8m + storm surge) hits {location}:**\n\n"
            f"The risk level goes from **{baseline_risk:.0f}** to **{scenario_risk:.0f}**.\n\n"
        )
        if scenario_risk >= 60:
            answer += (
                f"• Coastal areas within 2km of the shoreline would be **inundated**\n"
                f"• Mithi River would experience severe **backwater effect** — water can't drain into the sea\n"
                f"• Kurla, Sion, and Mahim would see **worst flooding** (elevation < 4m + tidal lock)\n"
                f"• **Recommendation:** Close all coastal roads, activate tidal flood barriers"
            )
        else:
            answer += (
                f"• Tidal lock would slow drainage but not cause catastrophic flooding\n"
                f"• **Recommendation:** Monitor tide levels, prepare pumps at key junctions"
            )

    elif "DRAINAGE" in scenario_id or "drain" in str(mods):
        answer = (
            f"{emoji} **If drainage fails {location}:**\n\n"
            f"Risk rises from **{baseline_risk:.0f}** to **{scenario_risk:.0f}**.\n\n"
        )
        if scenario_risk >= 60:
            answer += (
                f"• Blocked drains would cause **immediate waterlogging** even with moderate rain\n"
                f"• Kurla junction would become an **island** within 2 hours of rain\n"
                f"• Mithi River overflow would flood adjacent neighborhoods\n"
                f"• **Recommendation:** Deploy emergency desilting crews, activate bypass pumps"
            )
        else:
            answer += (
                f"• Partial drainage failure is manageable with existing backup systems\n"
                f"• **Recommendation:** Clear blockages, monitor water levels"
            )

    elif "CYCLONE" in scenario_id:
        answer = (
            f"{emoji} **If a cyclone makes direct landfall on Mumbai:**\n\n"
            f"Risk would reach **{scenario_risk:.0f}** (from baseline {baseline_risk:.0f}).\n\n"
            f"• **Wind speeds** of 120-150 km/h would cause widespread damage\n"
            f"• **Storm surge** of 2-3m would inundate all coastal areas\n"
            f"• **Combined rain + surge** would trap water inland for 24-48 hours\n"
            f"• **Recommendation:** Full coastal evacuation, activate all emergency shelters"
        )

    elif "CLIMATE" in scenario_id or "2050" in str(mods):
        answer = (
            f"{emoji} **Climate 2050 projection for Mumbai:**\n\n"
            f"Risk increases from **{baseline_risk:.0f}** to **{scenario_risk:.0f}**.\n\n"
            f"• Sea level rise of **0.3-0.5m** would permanently flood low-lying areas\n"
            f"• Extreme rainfall events would be **2-3x more frequent**\n"
            f"• Urban heat island effect would intensify convective storms\n"
            f"• **Recommendation:** Invest in elevated infrastructure, expand drainage capacity"
        )

    else:
        answer = (
            f"{emoji} **Scenario result {location}:**\n\n"
            f"Risk changes from **{baseline_risk:.0f}** to **{scenario_risk:.0f}** "
            f"({pct_change:+.0f}% change). This would be **{danger_phrase}**."
        )

    return answer


def _generate_default_answer() -> str:
    """Generate a helpful default response when no scenario is matched."""
    return (
        "I can help you simulate \"what-if\" scenarios for Mumbai flood risk. Try asking:\n\n"
        "• **\"What if rain doubles?\"** — Test extreme rainfall\n"
        "• **\"What if tide is really high?\"** — Test tidal surge\n"
        "• **\"What if drainage fails in Kurla?\"** — Test infrastructure failure\n"
        "• **\"Will Andheri flood?\"** — Check risk for a specific area\n"
        "• **\"What if a cyclone hits?\"** — Test cyclone scenario\n"
        "• **\"How about climate in 2050?\"** — Test long-term projection\n"
        "• **\"What's the worst case?\"** — Combined extreme event\n\n"
        "You can also combine: **\"What if rain doubles AND tide is high in Dadar?\"**"
    )


class WhatIfChatbot:
    """
    Conversational what-if scenario simulator.

    Parses natural language questions, runs the physics-informed
    risk model, and returns plain-language answers.
    """

    def __init__(self):
        from app.services.scenario import WhatIfScenarioEngine
        self.engine = WhatIfScenarioEngine()

    def ask(self, question: str, current_timestep: int = 36) -> Dict[str, Any]:
        """
        Process a natural language what-if question.

        Returns:
            {
                "question": str,
                "answer": str (plain language),
                "scenario_id": str or None,
                "locality": str or None,
                "baseline_risk": float,
                "scenario_risk": float,
                "delta": float,
                "affected_cells": int,
                "recommendation": str,
            }
        """
        # Extract locality
        locality = _extract_locality(question)
        cell_idx = LOCALITY_TO_CELL.get(locality) if locality else None

        # Extract scenario
        scenario_result = _extract_scenario(question)
        if scenario_result is None:
            # Check if it's a general question about flooding
            if _is_about_flooding(question) and locality:
                # Default: show current risk for that locality
                return self._current_risk_answer(question, locality, cell_idx, current_timestep)
            else:
                return {
                    "question": question,
                    "answer": _generate_default_answer(),
                    "scenario_id": None,
                    "locality": locality,
                    "baseline_risk": 0,
                    "scenario_risk": 0,
                    "delta": 0,
                    "affected_cells": 0,
                    "recommendation": "Try one of the suggested questions above.",
                    "is_help": True,
                }

        scenario_id, mods = scenario_result

        # Run the scenario
        try:
            result = self.engine.run_custom_scenario(
                scenario_id=scenario_id,
                modifications=mods,
                timestep_idx=current_timestep - 1,
            )

            baseline = result.get("baseline_risk", {}).get("avg_risk_score", 0)
            scenario = result.get("scenario_risk", {}).get("avg_risk_score", 0)
            delta = scenario - baseline

            # Count affected cells
            affected = 0
            for cell in result.get("scenario_risk", {}).get("cell_risks", []):
                if cell.get("total_risk_score", 0) >= 35:
                    affected += 1

            # Generate plain-language answer
            answer = _generate_answer(
                scenario_id, mods, locality,
                baseline, scenario, affected, 90
            )

            return {
                "question": question,
                "answer": answer,
                "scenario_id": scenario_id,
                "locality": locality,
                "baseline_risk": round(baseline, 1),
                "scenario_risk": round(scenario, 1),
                "delta": round(delta, 1),
                "affected_cells": affected,
                "recommendation": result.get("recommendation", ""),
                "sensitivity": result.get("sensitivity", {}),
            }

        except Exception as e:
            logger.error(f"Chatbot scenario error: {e}")
            return {
                "question": question,
                "answer": f"Sorry, I couldn't run that scenario. Error: {str(e)}",
                "scenario_id": scenario_id,
                "locality": locality,
                "baseline_risk": 0,
                "scenario_risk": 0,
                "delta": 0,
                "affected_cells": 0,
                "recommendation": "",
                "is_error": True,
            }

    def _current_risk_answer(self, question: str, locality: str,
                             cell_idx: Optional[int], timestep: int) -> Dict[str, Any]:
        """Answer a question about current risk for a locality."""
        from app.services.risk_model import risk_model_service
        import os, json
        from app.core.config import settings

        # Load current timestep data
        path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = max(0, min(len(data["timesteps"]) - 1, timestep - 1))
        ts = data["timesteps"][idx]

        if cell_idx is not None and cell_idx < len(ts["features"]):
            cell = ts["features"][cell_idx]
            risk = risk_model_service.calculate_cell_risk(
                rain_1h=cell["rainfall_1h_mm"],
                rain_3h=cell["rainfall_3h_mm"],
                rain_24h=cell["rainfall_24h_mm"],
                soil_moist_pct=cell["soil_moisture_pct"],
                elevation_m=cell["elevation_m"],
                slope_deg=cell["slope_deg"],
                cape_jkg=cell["cape_instability_jkg"],
                is_tide_locked=ts.get("is_high_tide_locked", False),
                tide_height_m=ts.get("tide_height_m", 2.5),
                drainage_outfall_dist_m=cell["drainage_outfall_dist_m"],
                is_depression=cell.get("is_depression_bowl", False),
            )

            score = risk["total_risk_score"]
            depth = risk["flood_depth_estimate_cm"]
            severity = risk["severity"]

            if score >= 75:
                emoji = "🔴"
                status = "CRITICAL — immediate action needed"
            elif score >= 50:
                emoji = "🟠"
                status = "HIGH — be prepared to evacuate"
            elif score >= 30:
                emoji = "🟡"
                status = "MEDIUM — monitor closely"
            else:
                emoji = "🟢"
                status = "LOW — normal precautions"

            answer = (
                f"{emoji} **Current risk in {locality.title()}:**\n\n"
                f"• **Risk Score:** {score:.0f}/100 ({status})\n"
                f"• **Estimated water depth:** {depth:.0f} cm\n"
                f"• **Rainfall:** {cell['rainfall_1h_mm']:.1f} mm/hr\n"
                f"• **Soil moisture:** {cell['soil_moisture_pct']:.0f}%\n"
                f"• **Elevation:** {cell['elevation_m']:.1f}m\n\n"
            )

            # Add key drivers
            drivers = []
            if cell["rainfall_1h_mm"] > 50:
                drivers.append(f"Heavy rainfall ({cell['rainfall_1h_mm']:.0f} mm/hr)")
            if cell["elevation_m"] < 5:
                drivers.append(f"Low-lying area ({cell['elevation_m']:.1f}m)")
            if cell["soil_moisture_pct"] > 80:
                drivers.append(f"Ground already soaked ({cell['soil_moisture_pct']:.0f}%)")
            if ts.get("is_high_tide_locked"):
                drivers.append("High tide blocking drains")
            if cell.get("is_depression_bowl"):
                drivers.append("Depression bowl — water collects here")

            if drivers:
                answer += "**Key risk factors:**\n" + "\n".join(f"• {d}" for d in drivers)
            else:
                answer += "No major risk factors at this time."

            return {
                "question": question,
                "answer": answer,
                "scenario_id": None,
                "locality": locality,
                "baseline_risk": round(score, 1),
                "scenario_risk": round(score, 1),
                "delta": 0,
                "affected_cells": 0,
                "recommendation": "",
                "is_current_risk": True,
            }

        return {
            "question": question,
            "answer": f"I couldn't find data for {locality.title()} at the current timestep.",
            "scenario_id": None,
            "locality": locality,
            "baseline_risk": 0,
            "scenario_risk": 0,
            "delta": 0,
            "affected_cells": 0,
            "recommendation": "",
            "is_error": True,
        }
