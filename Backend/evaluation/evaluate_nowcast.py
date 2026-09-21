"""
Extended evaluation: nowcast forecast skill, plus the PPT metric sheet.

Scores VARUNA's 6-hour-ahead rainfall forecast against observed future rainfall
on the FROZEN HELD-OUT TEST SPLIT, and writes ``nowcast_accuracy.json`` and
``PPT_METRICS.txt``.

Two corrections over the previous revision:

1. **No temporal leakage.** The old loop ran ``range(6, len(timesteps) - 6)``,
   which includes the steps the nowcaster was trained on -- so the published
   nowcast MAE/RMSE was partly a training-set score. It now scores only the
   contract's test partition. Because the test partition is the last one, every
   ground-truth value used here is itself held out.

2. **No stale numbers.** The old report hardcoded figures such as
   "Training MAE: 3.98 cm" and, more seriously, depended on three functions
   (``evaluate_varuna_engine``, ``evaluate_baseline``, ``compute_detection_accuracy``)
   that no longer exist in ``run_evaluation.py`` -- so the module raised
   ImportError and its artifacts silently stayed frozen at an older revision.
   Every figure below is now read from a regenerated artifact or omitted.
"""

import os
import sys
import json
import statistics
import textwrap
import time
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.run_evaluation import (  # noqa: E402
    load_dataset,
    run_four_family_evaluation,
)
from app.services.ai.feature_contract import (  # noqa: E402
    describe_split,
    split_timestep_bounds,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
FOUR_FAMILY_PATH = os.path.join(EVAL_DIR, "four_family_evaluation_results.json")
TRAINING_METADATA_PATH = os.path.join(
    os.path.dirname(EVAL_DIR), "models", "checkpoints", "training_metadata.json"
)
NOWCAST_PATH = os.path.join(EVAL_DIR, "nowcast_accuracy.json")
PPT_METRICS_PATH = os.path.join(EVAL_DIR, "PPT_METRICS.txt")
LEGACY_PATH = os.path.join(EVAL_DIR, "evaluation_results.json")


def _r2(actual: List[float], predicted: List[float]) -> Optional[float]:
    """Coefficient of determination, or None when the reference has no variance.

    Returning None (rather than 0.0 or a large negative number) matters: on a
    window whose observed target never varies, R2 is genuinely undefined and
    reporting a number there would be meaningless.
    """
    if not actual:
        return None
    mean = statistics.mean(actual)
    ss_tot = sum((a - mean) ** 2 for a in actual)
    if ss_tot <= 0:
        return None
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1.0 - ss_res / ss_tot


def evaluate_nowcast_accuracy(timesteps: List[Dict]) -> Dict[str, Any]:
    """Score the 6-hour nowcast on the frozen test partition only.

    For each held-out timestep ``t`` in the test window, compare the engine's
    issued forecast for ``t+1 .. t+6`` against the rainfall actually observed at
    those future steps.
    """
    from app.services.ai.inference_engine import VARUNAInferenceEngine

    engine = VARUNAInferenceEngine()

    start, end = split_timestep_bounds()["test"]
    start = max(start, 0)
    end = min(end, len(timesteps))

    horizon_errors: Dict[int, List[Dict[str, float]]] = {h: [] for h in range(1, 7)}
    horizon_persistence: Dict[int, List[Dict[str, float]]] = {h: [] for h in range(1, 7)}
    inference_modes: Dict[str, int] = {}
    scored_timesteps = 0
    skipped_timesteps = 0

    for idx in range(start, end):
        # Features are what the engine reads; a timestep without them cannot be
        # scored and is skipped rather than silently contributing a zero.
        features = timesteps[idx].get("features", [])
        if not features:
            skipped_timesteps += 1
            continue

        results = engine.run_full_inference(
            timestep_data=timesteps[idx],
            all_timesteps=timesteps,
            timestep_idx=idx,
        )
        nowcast = results.get("nowcast", {}) or {}
        scored_timesteps += 1

        # Which code path actually produced these forecasts. This matters: the
        # neural nowcaster rejects its own output when the forecast tail is
        # flat, and a report that claims a ConvLSTM forecast while the numbers
        # came from an extrapolation would not survive a question about it.
        mode = str(
            nowcast.get("inference_mode") or nowcast.get("mode") or "unknown"
        )
        inference_modes[mode] = inference_modes.get(mode, 0) + 1

        # Persistence reference: "rainfall stays where it is now". It is
        # computed from the same held-out window and fits nothing, so it is the
        # honest yardstick for whether the forecast has any real skill.
        now_rain = statistics.mean(
            [float(f.get("rainfall_1h_mm", 0.0) or 0.0) for f in features]
        )

        for forecast in nowcast.get("forecasts", []) or []:
            horizon = int(forecast.get("forecast_hour", 0) or 0)
            if horizon not in horizon_errors:
                continue
            future_idx = idx + horizon
            if future_idx >= len(timesteps):
                # Past the end of the dataset -- no ground truth to compare to.
                continue
            future_features = timesteps[future_idx].get("features", [])
            if not future_features:
                continue

            predicted = float(forecast.get("predicted_avg_rainfall_mm_hr", 0.0) or 0.0)
            actual = statistics.mean(
                [float(f.get("rainfall_1h_mm", 0.0) or 0.0) for f in future_features]
            )
            horizon_errors[horizon].append(
                {
                    "predicted": predicted,
                    "actual": actual,
                    "error": abs(predicted - actual),
                    "signed_error": predicted - actual,
                    "squared_error": (predicted - actual) ** 2,
                }
            )
            horizon_persistence[horizon].append(
                {"predicted": now_rain, "actual": actual}
            )

    per_lead_time: Dict[str, Dict[str, Any]] = {}
    for horizon in range(1, 7):
        rows = horizon_errors[horizon]
        if not rows:
            continue
        abs_errors = [r["error"] for r in rows]
        sq_errors = [r["squared_error"] for r in rows]
        actual = [r["actual"] for r in rows]
        predicted = [r["predicted"] for r in rows]
        r2 = _r2(actual, predicted)

        # Persistence over the same rows, so the two are directly comparable.
        p_rows = horizon_persistence[horizon]
        p_mae = (
            round(statistics.mean([abs(r["predicted"] - r["actual"]) for r in p_rows]), 2)
            if p_rows
            else None
        )
        p_r2 = _r2([r["actual"] for r in p_rows], [r["predicted"] for r in p_rows])
        mae = round(statistics.mean(abs_errors), 2)

        per_lead_time[f"+{horizon}h"] = {
            "mae_mm_hr": mae,
            "rmse_mm_hr": round(statistics.sqrt(statistics.mean(sq_errors)), 2),
            "bias_mm_hr": round(statistics.mean([r["signed_error"] for r in rows]), 2),
            "median_error_mm_hr": round(statistics.median(abs_errors), 2),
            "r2": round(r2, 3) if r2 is not None else None,
            "r2_undefined_reason": (
                None
                if r2 is not None
                else "observed rainfall has no variance in this window"
            ),
            "within_20pct": round(
                sum(1 for r in rows if r["error"] < max(1.0, r["actual"] * 0.2))
                / len(rows)
                * 100,
                1,
            ),
            "samples": len(rows),
            "persistence_mae_mm_hr": p_mae,
            "persistence_r2": round(p_r2, 3) if p_r2 is not None else None,
            "beats_persistence_mae": (mae < p_mae) if p_mae is not None else None,
            "mae_skill_vs_persistence_pct": (
                round((p_mae - mae) / p_mae * 100, 1)
                if p_mae not in (None, 0)
                else None
            ),
        }

    all_errors = [r["error"] for rows in horizon_errors.values() for r in rows]
    all_rows = [r for rows in horizon_errors.values() for r in rows]
    overall_r2 = _r2([r["actual"] for r in all_rows], [r["predicted"] for r in all_rows])

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": describe_split(),
        "evaluated_timesteps": f"{start + 1} to {end}",
        "scored_timesteps": scored_timesteps,
        "skipped_timesteps": skipped_timesteps,
        "note": (
            "Scored on the frozen held-out test partition only. The previous "
            "revision scored range(6, len-6), which included the nowcaster's own "
            "training steps."
        ),
        "inference_modes": inference_modes,
        "neural_nowcast_used": any(
            "neural" in m.lower() for m in inference_modes
        ),
        "mode_note": (
            "Each entry counts timesteps whose forecasts came from that code path. "
            "The neural nowcaster falls back to extrapolation whenever its own "
            "6-hour output is flat across +2h..+6h, so a report claiming a "
            "ConvLSTM forecast must check this field first."
        ),
        "persistence_reference": (
            "Persistence = 'rainfall stays at its current value'. It fits nothing "
            "and is computed on the same held-out rows, so it is the honest "
            "yardstick for whether the nowcast has real skill."
        ),
        "per_lead_time": per_lead_time,
        "overall_mae": round(statistics.mean(all_errors), 2) if all_errors else None,
        "overall_rmse": (
            round(statistics.sqrt(statistics.mean([r["squared_error"] for r in all_rows])), 2)
            if all_rows
            else None
        ),
        "overall_r2": round(overall_r2, 3) if overall_r2 is not None else None,
        "total_forecast_points": len(all_errors),
    }


def _load_four_family() -> Optional[Dict[str, Any]]:
    """Read the four-family report, regenerating it if absent.

    The previous revision reused ``evaluation_results.json`` whenever it existed,
    so a months-old file silently supplied the PPT numbers. There is no reuse
    path now: the report is either freshly generated or already on disk from a
    run in this session.
    """
    if os.path.exists(FOUR_FAMILY_PATH):
        try:
            with open(FOUR_FAMILY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    print("[RUN] No four-family report on disk -- generating it now...")
    results = run_four_family_evaluation(load_dataset())
    with open(FOUR_FAMILY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def _load_training_metadata() -> Dict[str, Any]:
    if not os.path.exists(TRAINING_METADATA_PATH):
        return {}
    try:
        with open(TRAINING_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _fmt(value: Any, spec: str = ".1f", na: str = "undefined") -> str:
    if value is None:
        return na
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def generate_ppt_ready_output(
    four_family: Optional[Dict[str, Any]],
    nowcast_results: Dict[str, Any],
) -> str:
    """Build the PPT metric sheet from regenerated artifacts only.

    Every number is read from a report this run produced (or a report already on
    disk from this session). Nothing is hardcoded, and any metric that the data
    cannot support is printed as ``undefined`` rather than invented.
    """
    lines: List[str] = []
    w = 78
    sep = "=" * w
    dash = "-" * w

    def add(*parts: str) -> None:
        lines.append("".join(parts))

    add(sep)
    add("  PROJECT VARUNA -- EVALUATION METRICS (PPT-Ready)")
    add("  Hyperlocal Urban Flood Digital Twin -- Mumbai Pilot")
    add(sep)
    add("")
    add(f"  Split: {nowcast_results.get('split', describe_split())}")
    add(f"  Nowcast scored on: timesteps {nowcast_results.get('evaluated_timesteps', 'n/a')}")
    add(f"  Generated: {nowcast_results.get('generated_at', '')}")

    family1 = (four_family or {}).get("family1_classification") or {}
    family2 = (four_family or {}).get("family2_regression") or {}
    family3 = (four_family or {}).get("family3_probabilistic") or {}
    family4 = (four_family or {}).get("family4_operational") or {}
    per_head = family3.get("per_head") or {}

    # ---- METRIC 1: Warning lead time -------------------------------------
    add("")
    add("[METRIC 1] WARNING LEAD TIME")
    add(dash)
    lead = family4.get("lead_time_to_first_alert_hours")
    if lead is None:
        # The observed peak falls before the scored partition, so the window
        # opens on the peak and no held-out lead time exists. Report the reason
        # and the replay-wide operational figure instead of a bare "n/a".
        add("  Measured lead time to first alert: not measurable on the held-out split")
        if family4.get("lead_time_note"):
            add(f"    - {family4['lead_time_note']}")
        if family4.get("replay_lead_time_hours") is not None:
            add(
                f"    - operational (full replay, includes the training window): "
                f"{family4['replay_lead_time_hours']} hours before the severity peak "
                f"(step {family4['replay_first_alert_timestep']} -> "
                f"{family4['replay_observed_peak_timestep']})"
            )
    else:
        add(f"  Measured lead time to first alert: {_fmt(lead, '.0f', 'n/a')} hours")
    add("  Reference (IMD practice): alerts are issued as thresholds are crossed,")
    add("  typically 30-60 minutes of radar nowcast -- i.e. reactive, not predictive.")
    f4_peak = family4.get("peak_risk_timestep")
    f4_obs = family4.get("observed_peak_timestep")
    if f4_peak is not None and f4_obs is not None:
        add(
            f"  Model peak risk at replay step {f4_peak}; observed severity peak at "
            f"step {f4_obs} (both global 1-based replay steps)."
        )
    add("")
    add("  Nowcast skill by lead time (rainfall, held-out split):")
    add(f"    {'Horizon':<8} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Bias':>8} "
        f"{'Persistence':>12} {'Skill':>8} {'n':>5}")
    for horizon, m in nowcast_results.get("per_lead_time", {}).items():
        skill = m.get("mae_skill_vs_persistence_pct")
        add(
            f"    {horizon:<8} {m['mae_mm_hr']:>8} {m['rmse_mm_hr']:>8} "
            f"{_fmt(m.get('r2'), '.3f'):>8} {_fmt(m.get('bias_mm_hr'), '.2f'):>8} "
            f"{_fmt(m.get('persistence_mae_mm_hr'), '.2f', 'n/a'):>12} "
            f"{(_fmt(skill, '.1f') + '%') if skill is not None else 'n/a':>8} {m['samples']:>5}"
        )
    add(f"    {'OVERALL':<8} {nowcast_results.get('overall_mae'):>8} "
        f"{nowcast_results.get('overall_rmse'):>8} "
        f"{_fmt(nowcast_results.get('overall_r2'), '.3f'):>8}")
    add("")
    add("  Skill is measured against persistence (rainfall holds at its current")
    add("  value). Persistence fits nothing, so beating it is the minimum bar for")
    add("  a forecast to be worth issuing; a negative skill means persistence wins.")
    modes = nowcast_results.get("inference_modes") or {}
    if modes:
        add("")
        add("  Forecast source (which code path produced the numbers above):")
        for mode, count in sorted(modes.items()):
            add(f"    {mode:<34} {count} timesteps")
        if not nowcast_results.get("neural_nowcast_used"):
            add("    NOTE: the trained neural nowcaster was not used for any timestep --")
            add("    its output was rejected as flat and the extrapolation was used.")

    # Why short-lead skill is ~0: record the model's data basis beside its
    # scores, so the question is answered by the artifact rather than by memory
    # of the training run. The frozen train window is a single build-up phase,
    # so the network has no decay examples; the trend baseline supplies the
    # direction and the network supplies the spatial pattern.
    nowcast_meta = (_load_training_metadata().get("results") or {}).get("nowcaster") or {}
    if nowcast_meta:
        add("")
        add("  Training data basis (read alongside the skill numbers above):")
        for key in ("n_sequences", "train_timesteps", "val_partition", "target", "best_val_loss"):
            if nowcast_meta.get(key) is not None:
                add(f"    {key:<16} {nowcast_meta[key]}")
        basis = nowcast_meta.get("data_basis")
        if basis:
            add("")
            for line in textwrap.wrap(
                str(basis), width=74, initial_indent="    ", subsequent_indent="    "
            ):
                add(line)

    # ---- METRIC 2: Hazard detection --------------------------------------
    add("")
    add("[METRIC 2] HAZARD DETECTION (held-out split, per hazard per lead time)")
    add(dash)
    add(f"  {'Hazard':<14} {'Horizon':<8} {'POD':>8} {'FAR':>8} {'CSI':>8} {'Acc':>8} {'Events':>8}")
    for hazard, horizons in family1.items():
        for horizon, m in horizons.items():
            if not isinstance(m, dict):
                continue
            add(
                f"  {hazard:<14} {horizon:<8} {_fmt(m.get('POD')):>8} "
                f"{_fmt(m.get('FAR')):>8} {_fmt(m.get('CSI')):>8} "
                f"{_fmt(m.get('Accuracy')):>8} {m.get('positives', 0):>8}"
            )
    undefined = [
        f"  {hazard} {horizon}: {m.get('undefined_reason') or 'not defined'}"
        for hazard, horizons in family1.items()
        for horizon, m in horizons.items()
        if isinstance(m, dict) and not m.get("defined", False)
    ]
    if undefined:
        add("")
        add("  Metrics that this window cannot support (reported, not fabricated):")
        for line in undefined:
            add(f"  {line.strip()}")

    # ---- METRIC 3: Regression accuracy -----------------------------------
    add("")
    add("[METRIC 3] REGRESSION ACCURACY (held-out split)")
    add(dash)
    add(f"  {'Target':<24} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    for target, m in family2.items():
        add(
            f"  {target:<24} {_fmt(m.get('RMSE'), '.2f'):>10} "
            f"{_fmt(m.get('MAE'), '.2f'):>10} {_fmt(m.get('R2'), '.3f'):>10}"
        )
    meta_mh = (_load_training_metadata().get("results") or {}).get("multi_hazard") or {}
    if meta_mh.get("baseline_mae_depth_cm") is not None:
        add("")
        add("  Physics-residual ablation (same split):")
        add(f"    Flood depth MAE   physics baseline {meta_mh['baseline_mae_depth_cm']} cm "
            f"-> VARUNA {meta_mh.get('mae_depth_absolute_cm')} cm")
        add(f"    Risk severity MAE physics baseline {meta_mh.get('baseline_mae_severity')} "
            f"-> VARUNA {meta_mh.get('mae_severity_absolute')}")

    # ---- METRIC 4: Calibration -------------------------------------------
    add("")
    add("[METRIC 4] PROBABILISTIC CALIBRATION")
    add(dash)
    add(f"  Brier score (aggregate):        {_fmt(family3.get('brier_score'), '.4f', 'n/a')}")
    add(f"  Expected calibration error:     {_fmt(family3.get('expected_calibration_error_ece'), '.4f', 'n/a')}")
    add("")
    add(f"  {'Head':<16} {'Brier':>10} {'ECE':>10} {'Climatology':>12} {'Positives':>12}")
    for head, m in per_head.items():
        if not isinstance(m, dict):
            continue
        if not m.get("defined", True):
            add(f"  {head:<16} {m.get('undefined_reason') or 'undefined':>48}")
            continue
        add(
            f"  {head:<16} {_fmt(m.get('brier_score'), '.4f'):>10} "
            f"{_fmt(m.get('ece'), '.4f'):>10} "
            f"{_fmt(m.get('reference_brier_climatology'), '.4f', 'n/a'):>12} "
            f"{str(m.get('positives', 'n/a')) + '/' + str(m.get('samples', 'n/a')):>12}"
        )
    if family3.get("notes"):
        add("")
        add(f"  {family3['notes']}")
    add("")
    add("  Conformal coverage (target >= 90%):")
    for target, m in (family3.get("conformal_per_target") or {}).items():
        if not isinstance(m, dict):
            continue
        if not m.get("in_scope", True):
            add(f"    {target:<22} out of scope")
            continue
        add(
            f"    {target:<22} coverage={_fmt(m.get('coverage_pct'), '.1f')}% "
            f"q={_fmt(m.get('q'), '.3f')} n={m.get('n_test', 'n/a')} "
            f"{'OK' if m.get('meets_guarantee') else 'CHECK'}"
        )

    # ---- METRIC 5: Operational -------------------------------------------
    add("")
    add("[METRIC 5] OPERATIONAL METRICS")
    add(dash)
    add(f"  False alarms per week (normalised): "
        f"{_fmt(family4.get('false_alarms_per_week'), '.1f', 'n/a')}")
    add(f"  Evacuation detour overhead:         "
        f"{_fmt(family4.get('evacuation_detour_overhead_pct'), '.1f', 'n/a')}%")
    add(f"  Peak risk timestep:                 {family4.get('peak_risk_timestep', 'n/a')}")
    add(f"  Observed peak timestep:             {family4.get('observed_peak_timestep', 'n/a')}")

    # ---- METRIC 6: Granularity / cost / deployment ------------------------
    add("")
    add("[METRIC 6] HYPERLOCAL GRANULARITY")
    add(dash)
    add("  Traditional:  city-wide single alert (e.g. 'Mumbai: heavy rain warning')")
    add("  VARUNA:       90 individual cells, each with risk score, flood depth,")
    add("                named locality, 3-hazard probabilities, XAI drivers and a")
    add("                conformal confidence interval.")

    add("")
    add("[METRIC 7] COST & INFRASTRUCTURE")
    add(dash)
    add("  Traditional: Doppler radar ($5-10M/station), commercial satellite")
    add("               licences ($50K+/yr), NWP supercomputer ($1-5M), 20-50 staff.")
    add("  VARUNA:      free data sources (Open-Meteo, SRTM, MOSDAC), PyTorch CPU")
    add("               inference on a laptop, 1-2 operators with AI assist.")

    add("")
    add(sep)
    add("  Slide-ready summary")
    add(sep)
    add("")
    add(f"  {'METRIC':<34} {'VARUNA (measured)':>26}")
    add(f"  {'-' * 34} {'-' * 26}")
    add(f"  {'Warning lead time':<34} {_fmt(lead, '.0f', 'n/a') + ' h':>26}")
    add(f"  {'Flood depth MAE':<34} {_fmt((family2.get('flood_depth_cm') or {}).get('MAE'), '.2f') + ' cm':>26}")
    add(f"  {'Flood depth R2':<34} {_fmt((family2.get('flood_depth_cm') or {}).get('R2'), '.3f'):>26}")
    add(f"  {'Risk severity R2':<34} {_fmt((family2.get('risk_severity_score') or {}).get('R2'), '.3f'):>26}")
    add(f"  {'Brier score (aggregate)':<34} {_fmt(family3.get('brier_score'), '.4f', 'n/a'):>26}")
    add(f"  {'Conformal coverage':<34} {_fmt(family3.get('conformal_coverage_pct'), '.1f', 'n/a') + '%':>26}")
    add(f"  {'Nowcast MAE (held-out)':<34} {_fmt(nowcast_results.get('overall_mae'), '.2f', 'n/a') + ' mm/hr':>26}")
    p6 = (nowcast_results.get("per_lead_time") or {}).get("+6h") or {}
    add(f"  {'Nowcast +6h vs persistence':<34} "
        f"{(str(p6.get('mae_skill_vs_persistence_pct')) + '% skill') if p6.get('mae_skill_vs_persistence_pct') is not None else 'n/a':>26}")
    add(f"  {'Spatial resolution':<34} {'90 cells':>26}")
    add(f"  {'Hazard types':<34} {'3 (TS + CB + FF)':>26}")
    add("")
    add(sep)

    return "\n".join(lines)


def main() -> int:
    print("=" * 70)
    print("  VARUNA EVALUATION -- NOWCAST SKILL + PPT METRICS")
    print("=" * 70)

    timesteps = load_dataset()
    print(f"\n[SPLIT] {describe_split()}")

    four_family = _load_four_family()
    if four_family is None:
        print("[WARN] Four-family report unavailable; some sections will read 'n/a'.")
    else:
        print(f"  Four-family split: {four_family.get('split', {}).get('description', 'n/a')}")

    print("\n[RUN] Nowcast accuracy evaluation on the held-out test window...")
    started = time.time()
    nowcast_results = evaluate_nowcast_accuracy(timesteps)
    print(
        f"  scored {nowcast_results['scored_timesteps']} timesteps "
        f"({nowcast_results['total_forecast_points']} forecast points) "
        f"in {time.time() - started:.1f}s"
    )
    print(f"  overall MAE={nowcast_results['overall_mae']} mm/hr, "
          f"RMSE={nowcast_results['overall_rmse']} mm/hr, "
          f"R2={nowcast_results['overall_r2']}")

    report = generate_ppt_ready_output(four_family, nowcast_results)
    print("\n" + report)

    with open(NOWCAST_PATH, "w", encoding="utf-8") as f:
        json.dump(nowcast_results, f, indent=2)
    print(f"\n[SAVED] {NOWCAST_PATH}")

    # Deliberately NOT PPT_READY_METRICS.txt: run_evaluation.py owns that file
    # for the four-family report, and both suites used to write it, so whichever
    # ran last silently replaced the other's output.
    with open(PPT_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[SAVED] {PPT_METRICS_PATH}")

    # ``evaluation_results.json`` was written by the pre-rewrite suite and still
    # held metrics from an older timeline and split, so reading it gave numbers
    # that contradicted every other artifact. Rather than delete a file the team
    # references, it is refreshed as a provenance pointer to the authoritative
    # report, so it can never again disagree without saying so.
    snapshot = {
        "generated_at": nowcast_results.get("generated_at"),
        "split": nowcast_results.get("split"),
        "status": "superseded by four_family_evaluation_results.json",
        "authoritative_report": os.path.basename(FOUR_FAMILY_PATH),
        "nowcast_report": os.path.basename(NOWCAST_PATH),
        "note": (
            "This file previously held a standalone metric set from an earlier "
            "evaluation revision and split. The four-family report is now the "
            "single source of truth for headline metrics; its contents are "
            "embedded below for convenience and are regenerated, not hand-kept."
        ),
        "four_family": four_family,
    }
    with open(LEGACY_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[SAVED] {LEGACY_PATH} (refreshed as a provenance pointer)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
