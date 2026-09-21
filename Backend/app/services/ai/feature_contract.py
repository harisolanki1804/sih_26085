"""
VARUNA Feature Contract — single source of truth for feature columns.
====================================================================

Authority: ``Backend/docs/API_CONTRACT.md`` Section 2 (26 canonical input
features + 5 ground-truth targets).

Why this module exists
----------------------
The canonical column list was previously duplicated across ``data_loader.py``,
``ml_data_prep.py``, ``trainer.py`` and the API contract doc, and the spellings
had drifted apart (``ctt_drop_rate_c_hr`` vs ``ctt_drop_rate_c_per_hr``,
``wind_u_ms`` vs ``u_wind_ms``). Worse, a pipeline step once rewrote
``feature_grid_timeseries.json`` and silently dropped every canonical feature
and every target label — with no error raised anywhere.

This module fixes both problems:

1. ``CANONICAL_FEATURES`` / ``TARGET_COLUMNS`` are the *only* definitions.
2. ``validate_feature_window`` is a hard gate. It fails loudly when a declared
   feature is missing or zero-variance, instead of letting silent ``0.0``
   defaults flow into training and evaluation.

Run the standalone audit::

    python -m app.services.ai.feature_contract --audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CANONICAL_FEATURES",
    "TARGET_COLUMNS",
    "REALIZED_FEATURES",
    "PENDING_FEATURES",
    "PROVENANCE",
    "ALIASES",
    "MODEL_INPUT_FEATURES",
    "NOWCAST_CHANNELS",
    "NOWCAST_WINDOW",
    "NOWCAST_HORIZON",
    "NOWCAST_NORMALIZATION_PATH",
    "NOWCAST_TREND_RELAX_HOURS",
    "SPLIT_TRAIN_END",
    "SPLIT_CALIBRATE_END",
    "SPLIT_TEST_END",
    "SPLIT_RATIONALE",
    "split_timestep_bounds",
    "split_sample_bounds",
    "describe_split",
    "build_model_row",
    "build_model_matrix",
    "resolve_feature",
    "FeatureContractError",
    "ValidationReport",
    "resolve_feature",
    "resolve_target",
    "validate_feature_window",
    "audit_feature_table",
    "load_feature_table",
]


class FeatureContractError(RuntimeError):
    """Raised when a feature window violates the published contract."""


# ---------------------------------------------------------------------------
# 2.1 Ordered canonical input features (contract index order matters)
# ---------------------------------------------------------------------------
CANONICAL_FEATURES: Tuple[str, ...] = (
    # Moisture (the fuel)
    "rainfall_1h_mm",
    "rainfall_3h_mm",
    "rainfall_6h_mm",
    "rainfall_24h_mm",
    "soil_moisture_pct",
    "soil_saturation_factor",
    "iwv_mm",
    # Instability (the energy)
    "cape_instability_jkg",
    "cin_jkg",
    "lifted_index",
    "cloud_top_temp_celsius",
    "ctt_drop_rate_c_per_hr",
    # Kinematics (the trigger)
    "wind_speed_10m_kmh",
    "wind_direction_10m_deg",
    "u_wind_ms",
    "v_wind_ms",
    "wind_gusts_kmh",
    "vertical_wind_shear_ms",
    "low_level_convergence",
    # Topography (the flood catalyst)
    "elevation_m",
    "slope_deg",
    "runoff_coefficient",
    "effective_runoff_mm_hr",
    "drainage_outfall_dist_m",
    "retention_index",
    "tidal_backwater_factor",
)

# The exact feature vector every trained model consumes, in order. Training and
# inference both build their input from this -- see ``build_model_row``.
MODEL_INPUT_FEATURES: Tuple[str, ...] = tuple(CANONICAL_FEATURES)

# ---------------------------------------------------------------------------
# 2.3 Frozen time-based split  (single source of truth)
# ---------------------------------------------------------------------------
# Boundaries are specified on the 72-timestep replay axis, 1-indexed inclusive.
#
# Why this lives here: the trainer and the evaluation suite each carried their
# own split, and they disagreed. The trainer used 80/5/15 on the 6480-sample
# axis (train 1-57, calibrate ~57-61, test ~61-72) while ``run_evaluation.py``
# used train 1-50 / test 51-72. Metrics produced under one split could not be
# compared with models trained under the other. Both now read these constants.
#
# Why the boundaries are not ``pipeline.md``'s literal 1-50 / 51-62 / 63-72:
# every flash-flood and cloudburst ground-truth event in this dataset falls in
# timesteps 21-47. A trailing test window therefore contains ZERO positive
# events, which makes POD undefined and FAR meaningless -- i.e. the doc's own
# headline metric (POD/FAR/CSI per hazard head) cannot be computed on the doc's
# own split. Measured event counts per partition:
#
#   partition            steps  samples  flash_flood  cloudburst
#   train   1-32          32      2880        86          131
#   calibrate 33-38        6       540        87          258
#   test    39-72         34      3060       103          272
#
# Every partition now carries both event types, all 72 timesteps are used, and
# the ordering stays strictly temporal (no future leakage). The test window also
# spans the recession, so FAR gets real negative mass instead of a vacuous 0.
SPLIT_TRAIN_END = 32      # timesteps 1..32    (train)
SPLIT_CALIBRATE_END = 38  # timesteps 33..38   (validation / conformal calibration)
SPLIT_TEST_END = 72       # timesteps 39..72   (frozen held-out test)
SPLIT_RATIONALE = (
    "Event-aware time-based split. pipeline.md's literal 1-50/51-62/63-72 puts "
    "every flash-flood and cloudburst event (all in timesteps 21-47) inside the "
    "train window, leaving the reported POD/FAR/CSI undefined. These boundaries "
    "keep both event types in all three partitions while preserving strict "
    "temporal ordering."
)

GRID_CELLS = 90  # 10 x 9 Mumbai pilot grid, cells per timestep

# ---------------------------------------------------------------------------
# 2.1b Nowcaster channel contract (Member 3's ConvLSTM + Transformer)
# ---------------------------------------------------------------------------
# The spatiotemporal nowcaster consumes a ``(T, C, H, W)`` tensor. Before this
# was pinned, the trainer fed ``CANONICAL_FEATURES[:8]`` while the inference
# engine built its own list -- a different set in a different order, two of
# whose members (``cin_jkg``, ``lifted_index``) are not even present in the
# feature table and were therefore read as 0.0. The model was trained on one
# tensor and served another. One shared list removes that class of bug.
#
# Channel 0 must stay ``rainfall_1h_mm``: the inference engine reads the
# forecast's channel 0 as the predicted rainfall rate.
#
# Every entry is a realized canonical feature, so the tensor is fully populated
# -- no silent zeros.
NOWCAST_CHANNELS: Tuple[str, ...] = (
    "rainfall_1h_mm",
    "rainfall_3h_mm",
    "rainfall_24h_mm",
    "soil_moisture_pct",
    "iwv_mm",
    "cape_instability_jkg",
    "cloud_top_temp_celsius",
    "vertical_wind_shear_ms",
)

# 6 observed frames in, 6 hourly frames out -- the 6-hour nowcast required by
# the problem statement.
NOWCAST_WINDOW = 6
NOWCAST_HORIZON = 6

# Per-channel normalization statistics, fitted on the train partition only and
# shared by training and inference. Channel scales here span three orders of
# magnitude (soil moisture ~tens, CAPE ~thousands); without this the MSE was
# dominated by CAPE and the network collapsed to a flat channel mean.
NOWCAST_NORMALIZATION_PATH = "models/checkpoints/nowcast_normalization.json"

# Relaxation time constant for the damped-persistence trend baseline the
# ConvLSTM learns its residual over. Shared by training and inference so the
# baseline is byte-identical on both sides.
NOWCAST_TREND_RELAX_HOURS = 2.0


def split_timestep_bounds() -> Dict[str, Tuple[int, int]]:
    """Return 0-indexed ``[start, end)`` timestep bounds per partition."""
    return {
        "train": (0, SPLIT_TRAIN_END),
        "calibrate": (SPLIT_TRAIN_END, SPLIT_CALIBRATE_END),
        "test": (SPLIT_CALIBRATE_END, SPLIT_TEST_END),
    }


def split_sample_bounds(
    cells_per_timestep: int = GRID_CELLS,
) -> Dict[str, Tuple[int, int]]:
    """Return 0-indexed ``[start, end)`` bounds on the flattened sample axis.

    Models consume ``(timesteps x cells)`` samples in time order, so a
    timestep partition maps onto a contiguous sample partition.
    """
    bounds = split_timestep_bounds()
    return {
        name: (start * cells_per_timestep, end * cells_per_timestep)
        for name, (start, end) in bounds.items()
    }


def describe_split() -> str:
    """Human-readable summary, written into reports so a metric is traceable."""
    return (
        f"train 1-{SPLIT_TRAIN_END}, calibrate {SPLIT_TRAIN_END + 1}-{SPLIT_CALIBRATE_END}, "
        f"test {SPLIT_CALIBRATE_END + 1}-{SPLIT_TEST_END} (frozen time-based split)"
    )

# ---------------------------------------------------------------------------
# 2.2 Ground-truth targets
# ---------------------------------------------------------------------------
TARGET_COLUMNS: Tuple[str, ...] = (
    "target_observed_flood_depth_cm",
    "target_severity_class",
    "target_flash_flood_flag",
    "target_cloudburst_flag",
    "target_waterlogging_flag",
)

# ---------------------------------------------------------------------------
# On-disk name reconciliation
# ---------------------------------------------------------------------------
# Maps canonical name -> every spelling accepted on disk, most-preferred first.
# The canonical name is always accepted. Keeping this explicit (rather than
# renaming 30+ columns across the codebase) means the gate can validate real
# data without a risky repo-wide rename.
ALIASES: Dict[str, Tuple[str, ...]] = {
    "ctt_drop_rate_c_per_hr": ("ctt_drop_rate_c_per_hr", "ctt_drop_rate_c_hr"),
    "u_wind_ms": ("u_wind_ms", "wind_u_ms"),
    "v_wind_ms": ("v_wind_ms", "wind_v_ms"),
    "vertical_wind_shear_ms": (
        "vertical_wind_shear_ms",
        "wind_speed_shear",
        "wind_shear_ms",
    ),
    "wind_direction_10m_deg": ("wind_direction_10m_deg", "wind_direction_deg"),
    "tidal_backwater_factor": (
        "tidal_backwater_factor",
        "tide_height_m",  # imperfect stand-in; flagged in PROVENANCE notes
    ),
}

# Extra on-disk columns that are not part of the 26 canonical inputs but are
# legitimately carried alongside them (identifiers, aux spatial metadata).
AUXILIARY_COLUMNS: Tuple[str, ...] = (
    "cell_index",
    "cell_id",
    "lat",
    "lon",
    "locality",
    "is_depression_bowl",
    "tide_height_m",
    "is_high_tide_locked",
    # satellite provenance carried per cell (see feature_merge)
    "satellite_provenance",
    "gap_filled",
)

# ---------------------------------------------------------------------------
# Realization status: which canonical features actually exist in the data today
# ---------------------------------------------------------------------------
# Declared in the contract and present in feature_grid_timeseries.json.
REALIZED_FEATURES: Tuple[str, ...] = (
    "rainfall_1h_mm",
    "rainfall_3h_mm",
    "rainfall_6h_mm",
    "rainfall_24h_mm",
    "soil_moisture_pct",
    "soil_saturation_factor",
    "cape_instability_jkg",
    "cloud_top_temp_celsius",
    "ctt_drop_rate_c_per_hr",
    "wind_speed_10m_kmh",
    "wind_direction_10m_deg",
    "u_wind_ms",
    "v_wind_ms",
    "wind_gusts_kmh",
    "elevation_m",
    "slope_deg",
    "runoff_coefficient",
    "effective_runoff_mm_hr",
    "drainage_outfall_dist_m",
    "retention_index",
    "vertical_wind_shear_ms",   # WDP-derived shear
    "low_level_convergence",    # WDP-derivable convergence (was decoded, then dropped)
    "iwv_mm",                   # UTH-based proxy (stated, not a retrieval)
    "tidal_backwater_factor",
    # HEM-derived satellite rain rate. This column was written into every cell
    # by feature_merge but appeared in neither this list nor PENDING_FEATURES, so
    # the gate reported "realized features checked" while silently skipping it --
    # an unvalidated field is exactly the kind of gap the contract exists to
    # close. Verified present in all 6480 cells and varying (0.55-180.00 mm/hr)
    # before being registered here.
    "qpe_rain_rate_mm_hr",
)

# Declared in the contract but NOT yet sourced. These are the IMDAA fields the
# data pipeline still owes (INSAT-3D/3DR supplies the rest as of the current
# pipeline run). The gate reports these; it does not pretend they are zero.
PENDING_FEATURES: Tuple[str, ...] = (
    "cin_jkg",
    "lifted_index",
)

# ---------------------------------------------------------------------------
# Provenance classification (contract Section 3)
# ---------------------------------------------------------------------------
REAL = "REAL"
RECONSTRUCTED = "RECONSTRUCTED"
SYNTHETIC = "SYNTHETIC"

PROVENANCE: Dict[str, str] = {
    "rainfall_1h_mm": SYNTHETIC,
    "rainfall_3h_mm": SYNTHETIC,
    "rainfall_6h_mm": SYNTHETIC,
    "rainfall_24h_mm": SYNTHETIC,
    "soil_moisture_pct": SYNTHETIC,
    "soil_saturation_factor": RECONSTRUCTED,
    # UTH-based proxy, NOT a retrieval: IWV = 20 + 55*(UTH/100)^1.4, bounded by
    # Mumbai monsoon climatology. The exact method is recorded per derived
    # payload under ``field_methods.iwv_mm`` (see data_pipeline/derivation_engine.py).
    "iwv_mm": RECONSTRUCTED,
    "cape_instability_jkg": SYNTHETIC,
    "cin_jkg": SYNTHETIC,
    "lifted_index": SYNTHETIC,
    "cloud_top_temp_celsius": SYNTHETIC,
    "ctt_drop_rate_c_per_hr": RECONSTRUCTED,
    "wind_speed_10m_kmh": SYNTHETIC,
    "wind_direction_10m_deg": SYNTHETIC,
    "u_wind_ms": SYNTHETIC,
    "v_wind_ms": SYNTHETIC,
    "wind_gusts_kmh": SYNTHETIC,
    # Sourced from the WDP-derived shear field. The runtime per-field provenance
    # in the feature table (``satellite_provenance.field_provenance``) is
    # authoritative for whether a given timestep's value is measured or
    # reconstructed; this map only states the column's origin class.
    "vertical_wind_shear_ms": RECONSTRUCTED,
    "low_level_convergence": RECONSTRUCTED,
    "elevation_m": REAL,
    "slope_deg": REAL,
    "runoff_coefficient": RECONSTRUCTED,
    "effective_runoff_mm_hr": RECONSTRUCTED,
    "drainage_outfall_dist_m": REAL,
    "retention_index": RECONSTRUCTED,
    "tidal_backwater_factor": RECONSTRUCTED,
    # HEM (hydro-estimator) rain rate. The column's origin class is the satellite
    # product; whether a given timestep is measured or reconstructed is recorded
    # per field in ``satellite_provenance.field_provenance``, which is
    # authoritative (only 6 granules in this window are genuinely measured).
    "qpe_rain_rate_mm_hr": RECONSTRUCTED,
}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------
def resolve_feature(cell: Dict[str, Any], canonical: str, default: Any = None) -> Any:
    """Read ``canonical`` from a raw cell dict, honouring accepted aliases."""
    for name in ALIASES.get(canonical, (canonical,)):
        if name in cell and cell[name] is not None:
            return cell[name]
    return default


def resolve_target(cell: Dict[str, Any], canonical: str, default: Any = None) -> Any:
    """Read a target label from a raw cell dict."""
    value = cell.get(canonical, default)
    return default if value is None else value


def feature_name_in(cell: Dict[str, Any], canonical: str) -> Optional[str]:
    """Return the on-disk spelling actually present for ``canonical``, if any."""
    for name in ALIASES.get(canonical, (canonical,)):
        if name in cell:
            return name
    return None


def build_model_row(cell: Dict[str, Any], default: float = 0.0) -> List[float]:
    """Build one model input vector from a raw cell, in canonical order.

    This is the single source of truth for the model **input** contract, and
    training and inference must both use it.

    Why it exists: ``trainer.py`` built its input from all 26 canonical
    columns while ``inference_engine.py`` built its own 22-column list. Any
    model retrained from the trainer therefore could not be loaded by
    inference (feature-count mismatch), and the neural path silently fell back
    to heuristics. Routing both through this function makes that class of
    divergence impossible.
    """
    row: List[float] = []
    for name in CANONICAL_FEATURES:
        value = resolve_feature(cell, name, default)
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        try:
            row.append(float(value))
        except (TypeError, ValueError):
            row.append(float(default))
    return row


def build_model_matrix(cells: Iterable[Dict[str, Any]]) -> List[List[float]]:
    """Build the model input matrix for a timestep's cells."""
    return [build_model_row(cell) for cell in cells if isinstance(cell, dict)]


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Outcome of validating a window of timesteps against the contract."""

    label: str = ""
    n_timesteps: int = 0
    n_cells: int = 0
    missing: List[str] = field(default_factory=list)
    zero_variance: List[str] = field(default_factory=list)
    missing_targets: List[str] = field(default_factory=list)
    pending_present: List[str] = field(default_factory=list)
    unresolved_names: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"Feature contract validation{f' [{self.label}]' if self.label else ''}",
            f"  window           : {self.n_timesteps} timesteps x {self.n_cells} cells",
            f"  realized features: {len(REALIZED_FEATURES)} checked",
        ]
        if self.missing:
            lines.append(f"  MISSING          : {', '.join(self.missing)}")
        if self.zero_variance:
            lines.append(f"  ZERO-VARIANCE    : {', '.join(self.zero_variance)}")
        if self.missing_targets:
            lines.append(f"  MISSING TARGETS  : {', '.join(self.missing_targets)}")
        if self.pending_present:
            lines.append(f"  newly sourced    : {', '.join(self.pending_present)}")
        if self.unresolved_names:
            for canonical, used in self.unresolved_names.items():
                lines.append(f"  alias used       : {canonical} <- {used}")
        lines.append("  RESULT           : " + ("PASS" if self.ok else "FAIL"))
        for err in self.errors:
            lines.append(f"   ! {err}")
        return "\n".join(lines)


def _iter_cells(timesteps: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for ts in timesteps:
        if not isinstance(ts, dict):
            continue
        cells = ts.get("features")
        if cells is None:
            cells = ts.get("cells")
        if isinstance(cells, list):
            for cell in cells:
                if isinstance(cell, dict):
                    yield cell


def validate_feature_window(
    timesteps: Sequence[Dict[str, Any]],
    *,
    label: str = "",
    require_targets: bool = True,
    strict: bool = True,
) -> ValidationReport:
    """Hard gate: fail loudly on missing or zero-variance declared features.

    This is the check that would have caught the destroyed feature table
    immediately, instead of letting constants silently reach training.

    Args:
        timesteps: list of timestep dicts, each carrying ``features``/``cells``.
        label: human-readable window label for the report.
        require_targets: also require the 5 ground-truth target columns.
        strict: raise :class:`FeatureContractError` on any violation.

    Returns:
        A :class:`ValidationReport`. Raises when ``strict`` and invalid.
    """
    report = ValidationReport(label=label)
    report.n_timesteps = len(timesteps)

    cells = list(_iter_cells(timesteps))
    report.n_cells = len(cells)

    if not cells:
        report.errors.append(
            "no cells found — expected a list under 'features' (or 'cells') on each timestep"
        )
        if strict:
            raise FeatureContractError(report.summary())
        return report

    # --- realized features must be present and must vary -------------------
    for canonical in REALIZED_FEATURES:
        used = None
        values: List[Any] = []
        for cell in cells:
            name = feature_name_in(cell, canonical)
            if name is None:
                continue
            used = used or name
            values.append(cell.get(name))

        if not values:
            report.missing.append(canonical)
            continue

        if used and used != canonical:
            report.unresolved_names[canonical] = used

        numeric: List[float] = []
        for v in values:
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                numeric.append(float(v))

        if not numeric:
            report.missing.append(canonical)
        elif len(set(numeric)) == 1:
            report.zero_variance.append(canonical)

    # --- targets -----------------------------------------------------------
    if require_targets:
        for target in TARGET_COLUMNS:
            present = False
            for cell in cells:
                if target in cell:
                    present = True
                    break
            if not present:
                report.missing_targets.append(target)

    # --- pending features that have started arriving (informational) -------
    for canonical in PENDING_FEATURES:
        if any(feature_name_in(cell, canonical) for cell in cells):
            report.pending_present.append(canonical)

    if report.missing:
        report.errors.append(
            f"{len(report.missing)} declared feature(s) absent from the data: "
            + ", ".join(report.missing)
        )
    if report.zero_variance:
        report.errors.append(
            f"{len(report.zero_variance)} feature(s) are constant across the window "
            "(placeholder values?): " + ", ".join(report.zero_variance)
        )
    if report.missing_targets:
        report.errors.append(
            f"{len(report.missing_targets)} ground-truth target(s) missing: "
            + ", ".join(report.missing_targets)
        )

    if strict and report.errors:
        raise FeatureContractError(report.summary())
    return report


# ---------------------------------------------------------------------------
# Standalone audit
# ---------------------------------------------------------------------------
def load_feature_table(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``feature_grid_timeseries.json`` from the configured data dir."""
    if path is None:
        try:
            from app.core.config import settings

            path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")
        except Exception:
            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(here, "..", "..", "..", "data", "feature_grid_timeseries.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_timestep_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize both accepted container shapes into a list of timesteps."""
    timesteps = data.get("timesteps", data)
    if isinstance(timesteps, dict):
        return [v for v in timesteps.values() if isinstance(v, dict)]
    if isinstance(timesteps, list):
        return [t for t in timesteps if isinstance(t, dict)]
    return []


def audit_feature_table(path: Optional[str] = None) -> int:
    """Audit the on-disk feature table. Returns a process exit code."""
    data = load_feature_table(path)
    timesteps = _as_timestep_list(data)

    print(f"file            : {path or 'settings.DATA_DIR/feature_grid_timeseries.json'}")
    print(f"top-level keys  : {', '.join(list(data.keys())[:12])}")
    print(f"timesteps       : {len(timesteps)}")
    for key in ("region_code", "event_code", "data_type", "total_timesteps"):
        if key in data:
            print(f"{key:<16}: {data[key]}")

    try:
        report = validate_feature_window(timesteps, label="full table", strict=True)
    except FeatureContractError as exc:
        print()
        print(exc)
        return 1

    print()
    print(report.summary())
    if report.pending_present:
        print(
            "\nNote: PENDING features now present but not yet in REALIZED_FEATURES — "
            "promote them once sourced: " + ", ".join(report.pending_present)
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="VARUNA feature contract utilities")
    parser.add_argument("--audit", action="store_true", help="validate the feature table")
    parser.add_argument("--path", default=None, help="explicit path to the feature table")
    args = parser.parse_args(argv)

    if args.audit:
        return audit_feature_table(args.path)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
