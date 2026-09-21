"""
MOSDAC INSAT-3D/3DR L2 product decoder.
=======================================

Turns a downloaded MOSDAC ``.h5`` granule into per-cell values for the 90-cell
Mumbai pilot grid, with per-product handling verified against real granules
downloaded from the 2022-07-05 window:

=================  ============  ========================  =====================
Product            Primary var   Structure                  Notes
=================  ============  ========================  =====================
3DIMG_L2B_CTP      CTT, CTP      (1, 313, 312)             Kelvin -> degC
3DIMG_L2G_WDP      UCOMP, VCOMP  (1, 37, 201, 201)         1D 0.5 deg grid,
                                                           37 pressure levels
3DIMG_L2B_UTH      UTH           (1, 1408, 1402)           percent
3DIMG_L2B_HEM      HEM           (1, 2816, 2805)           mm/hr rain rate
3DIMG_L2B_CMK      CMK           (1, 2816, 2805)           0-3 cloud classes
=================  ============  ========================  =====================

Why explicit variable names, not auto-detection
-----------------------------------------------
The CTP granule contains *decoy* 2-D numeric arrays — ``CSBT_Latitude``,
``CSBT_Longitude`` (325x325 geolocation dupes), ``SAT_ZEN``, ``SOL_ZEN``,
``STDV_*`` and ``CLRFR_*``. A generic "first 2-D numeric array" heuristic
(one is still present in ``map_to_varuna_grid``) picks ``CLRFR_MIR``, a
clear-sky reflectance, as the primary data. That yields plausible-looking
numbers that are physically meaningless for cloud-top sampling — exactly the
silent-wrong-data class of bug that destroyed the feature table earlier.

This decoder therefore:

* keys each product by its REAL variable name (verified above), failing loudly
  when absent rather than guessing;
* applies ``_FillValue``, ``scale_factor`` and ``add_offset`` attributes
  (latitude/longitude are int16 with scale 0.01 and fill 31172);
* converts CTT Kelvin -> degC to match the canonical feature
  ``cloud_top_temp_celsius``;
* derives vertical wind shear and low-level convergence for WDP from the
  37 pressure levels (1000-900 vs 600-500 mb layers), since the canonical
  features ``vertical_wind_shear_ms`` and ``low_level_convergence`` are
  defined on exactly those layers;
* samples by nearest neighbour in geolocation space when 2-D lat/lon grids
  exist, and by regular index mapping on the WDP 1-D grid.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("VARUNA.ProductDecoder")

GRID_ROWS = 10
GRID_COLS = 9
GRID_CELLS = GRID_ROWS * GRID_COLS

# Mumbai pilot bounding box (must match map_to_varuna_grid defaults).
LAT_MIN, LAT_MAX = 18.88, 19.28
LON_MIN, LON_MAX = 72.75, 73.05

KELVIN_OFFSET = 273.15


def _grid_centres() -> List[Tuple[float, float]]:
    centres: List[Tuple[float, float]] = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            centres.append((
                LAT_MIN + (r + 0.5) * (LAT_MAX - LAT_MIN) / GRID_ROWS,
                LON_MIN + (c + 0.5) * (LON_MAX - LON_MIN) / GRID_COLS,
            ))
    return centres


def _read_scaled(dataset) -> np.ndarray:
    """Read a dataset applying _FillValue / scale_factor / add_offset.

    Returns a float array with fill values replaced by NaN.
    """
    arr = dataset[...]
    arr = np.asarray(arr, dtype=np.float64)

    fill = dataset.attrs.get("_FillValue")
    if fill is not None:
        fill_vals = np.atleast_1d(np.asarray(fill, dtype=np.float64))
        for fv in fill_vals:
            arr[arr == fv] = np.nan

    scale = dataset.attrs.get("scale_factor")
    offset = dataset.attrs.get("add_offset")
    if scale is not None:
        arr = arr * float(np.atleast_1d(scale)[0])
    if offset is not None:
        arr = arr + float(np.atleast_1d(offset)[0])
    return arr


def _lead_dims(arr: np.ndarray) -> np.ndarray:
    """Collapse leading singleton/time dimensions, keeping the 2-D field."""
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _find_dataset(h5file, names: Tuple[str, ...]):
    """Locate a dataset by candidate (case-insensitive) names, recursively."""
    wanted = {n.upper() for n in names}

    def _walk(group):
        for key in group:
            item = group[key]
            if isinstance(item, type(group)) or hasattr(item, "visititems"):
                if key.upper() in wanted:
                    return item
                found = _walk(item)
                if found is not None:
                    return found
            elif key.upper() in wanted:
                return item
        return None

    return _walk(h5file)


def _sample_geolocated(data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> List[float]:
    """Nearest-neighbour sampling on 2-D geolocation arrays.

    Geolocation must be sanitised BEFORE the reduction runs.

    The latitude/longitude arrays of an INSAT full-disk product carry
    ``_FillValue`` (32767) outside the Earth disk, which ``_read_scaled`` turns
    into NaN -- roughly a quarter of every CTP grid. ``np.argmin`` propagates
    NaN rather than skipping it, so a single NaN anywhere in the distance field
    makes argmin return *the NaN's own index*. The previous version therefore
    sampled pixel (0, 0) -- a corner fill cell -- for every one of the 90 grid
    points, and its fallback branch re-ran the same reduction on another
    NaN-carrying array. The values that came out looked plausible (finite,
    physically reasonable cloud-top temperatures) while describing a patch of
    the disk nowhere near Mumbai, which is the worst possible failure mode.

    Distances to non-finite geolocation are set to +inf so they can never win,
    and cells with no data are deprioritised behind cells that have some.
    """
    nan_mask = ~np.isfinite(data)
    geo_ok = np.isfinite(lat) & np.isfinite(lon)

    values: List[float] = []
    for clat, clon in _grid_centres():
        dist_sq = (lat - clat) ** 2 * (111.0 ** 2) + (lon - clon) ** 2 * (105.0 ** 2)
        # A cell we cannot geolocate is not a candidate, and a NaN must never
        # reach argmin.
        dist_sq = np.where(geo_ok & np.isfinite(dist_sq), dist_sq, np.inf)

        # Prefer the nearest geolocated cell that actually carries a value.
        dist_with_data = np.where(nan_mask, np.inf, dist_sq)
        if np.isfinite(dist_with_data).any():
            r0, c0 = np.unravel_index(np.argmin(dist_with_data), data.shape)
            values.append(round(float(data[r0, c0]), 3))
            continue

        # Nothing in the search grid has data -- fall back to the nearest
        # geolocated cell and report 0.0 rather than a NaN.
        if np.isfinite(dist_sq).any():
            r1, c1 = np.unravel_index(np.argmin(dist_sq), data.shape)
            v = float(data[r1, c1])
            values.append(round(v, 3) if np.isfinite(v) else 0.0)
        else:
            values.append(0.0)
    return values


def _nearest_index(axis: np.ndarray, target: float) -> int:
    """Index of the axis value closest to ``target``, ignoring non-finite ones.

    ``np.argmin`` propagates NaN, so an unfiltered ``abs(axis - target)`` would
    hand back the index of a NaN entry (see ``_sample_geolocated``).
    """
    delta = np.where(np.isfinite(axis), np.abs(axis - target), np.inf)
    if not np.isfinite(delta).any():
        return 0
    return int(np.argmin(delta))


def _sample_regular_1d(
    data_2d: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> List[float]:
    """Sample a regular lat/lon grid defined by 1-D coordinate arrays."""
    # For each cell pick nearest lat and lon indices on the 1-D axes.
    lat_targets = [LAT_MIN + (r + 0.5) * (LAT_MAX - LAT_MIN) / GRID_ROWS for r in range(GRID_ROWS)]
    lon_targets = [LON_MIN + (c + 0.5) * (LON_MAX - LON_MIN) / GRID_COLS for c in range(GRID_COLS)]
    lat_is = [_nearest_index(lats, t) for t in lat_targets]
    lon_is = [_nearest_index(lons, t) for t in lon_targets]

    values: List[float] = []
    for ri in lat_is:
        for ci in lon_is:
            v = float(data_2d[ri, ci])
            values.append(round(v, 3) if np.isfinite(v) else 0.0)
    return values


def _layer_mean(u: np.ndarray, v: np.ndarray, levels: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean u/v over the pressure layer between a and b (mb), order-agnostic."""
    lo, hi = min(a, b), max(a, b)
    mask = (levels >= lo) & (levels <= hi)
    if not mask.any():
        raise ValueError(f"no pressure levels in [{lo}, {hi}]")
    return u[mask].mean(axis=0), v[mask].mean(axis=0)


def decode_product(h5_file_path: str, product_id: str) -> Dict[str, Any]:
    """Decode one granule into per-cell values + provenance metadata.

    Returns dict with keys:
        values: List[float] length 90 (NaN-free; failed cells = 0.0 with note)
        units: str
        variable: str  (primary variable actually used)
        derived: dict  (optional extra fields, e.g. WDP shear/convergence)
    """
    import h5py

    product_id = product_id.upper()
    centres = _grid_centres()

    with h5py.File(h5_file_path, "r") as h5:
        # ------------------------------------------------------------------
        # CTP granule: cloud-top temperature (K -> degC) + pressure (hPa)
        # ------------------------------------------------------------------
        if product_id == "3DIMG_L2B_CTP":
            ctt_ds = _find_dataset(h5, ("CTT",))
            if ctt_ds is None:
                raise ValueError("CTT dataset missing from CTP granule")
            ctt = _lead_dims(_read_scaled(ctt_ds))
            if np.isfinite(ctt).any() and np.nanmax(ctt) > 150:
                # Kelvin -> Celsius
                ctt = ctt - KELVIN_OFFSET
            values = _sample_geolocated(ctt, *_geo2d(h5, ("Latitude", "Longitude")))
            return {"values": values, "units": "degC", "variable": "CTT", "derived": {}}

        # ------------------------------------------------------------------
        # WDP granule: 37-level u/v components on a 1-D 0.5 deg grid
        # ------------------------------------------------------------------
        if product_id == "3DIMG_L2G_WDP":
            u_ds = _find_dataset(h5, ("UCOMP",))
            v_ds = _find_dataset(h5, ("VCOMP",))
            lat_ds = _find_dataset(h5, ("latitude", "Latitude", "LAT"))
            lon_ds = _find_dataset(h5, ("longitude", "Longitude", "LON"))
            p_ds = _find_dataset(h5, ("plevels", "pressure", "PLEVELS"))
            if u_ds is None or v_ds is None:
                raise ValueError("UCOMP/VCOMP missing from WDP granule")

            u = np.asarray(u_ds[...], dtype=np.float64)
            v = np.asarray(v_ds[...], dtype=np.float64)
            # (1, 37, 201, 201) -> (37, 201, 201)
            if u.ndim == 4:
                u, v = u[0], v[0]

            lats = np.asarray(lat_ds[...], dtype=np.float64).ravel()
            lons = np.asarray(lon_ds[...], dtype=np.float64).ravel()
            levels = np.asarray(p_ds[...], dtype=np.float64).ravel() if p_ds is not None else None

            if levels is None or levels.size != u.shape[0]:
                raise ValueError("plevels missing or shape mismatch in WDP granule")

            # Bulk shear 1000-900 mb vs 600-500 mb (canonical definition:
            # 0-6 km ~= surface to ~500 mb; we take the deepest reliable pair)
            u_low, v_low = _layer_mean(u, v, levels, 1000, 900)
            u_high, v_high = _layer_mean(u, v, levels, 600, 500)
            shear = np.sqrt((u_high - u_low) ** 2 + (v_high - v_low) ** 2)  # m/s

            # 925 mb mean winds for the kinematics features
            u_925, v_925 = _layer_mean(u, v, levels, 950, 900)
            wind_speed = np.sqrt(u_925 ** 2 + v_925 ** 2)  # m/s

            # Low-level convergence: -d(u)/dx - d(v)/dy on the 925 mb layer.
            # Grid spacing must be in METRES (111 km per degree) so the
            # gradient comes out in s^-1; using km here inflates the result
            # by 1000x (a bug caught on real data: 3426e-5 vs ~3.4e-5 s^-1).
            dlon_m = float(np.abs(lons[1] - lons[0])) * 111_000.0 if lons.size > 1 else 55_500.0
            dlat_m = float(np.abs(lats[1] - lats[0])) * 111_000.0 if lats.size > 1 else 55_500.0
            du_dx = np.gradient(u_925, dlon_m, axis=1)
            dv_dy = np.gradient(v_925, dlat_m, axis=0)
            convergence = -(du_dx + dv_dy)  # s^-1

            shear_vals = _sample_regular_1d(shear, lats, lons)
            wind_vals = _sample_regular_1d(wind_speed, lats, lons)
            conv_vals = _sample_regular_1d(convergence * 1e5, lats, lons)  # -> 10^-5 s^-1
            u_vals = _sample_regular_1d(u_925, lats, lons)
            v_vals = _sample_regular_1d(v_925, lats, lons)

            return {
                "values": shear_vals,
                "units": "m/s",
                "variable": "UCOMP+VCOMP shear(1000-900/600-500mb)",
                "derived": {
                    "wind_speed_925ms": wind_vals,
                    "low_level_convergence_1e5": conv_vals,
                    "u_wind_925_ms": u_vals,
                    "v_wind_925_ms": v_vals,
                },
            }

        # ------------------------------------------------------------------
        # Simple single-variable L2 products with 2-D geolocation
        # ------------------------------------------------------------------
        spec = {
            "3DIMG_L2B_UTH": ("UTH", ("UTH",), "%", None),
            "3DIMG_L2B_HEM": ("HEM", ("HEM",), "mm/hr", None),
            "3DIMG_L2B_CMK": ("CMK", ("CMK",), "class", None),
        }
        if product_id in spec:
            var, names, units, transform = spec[product_id]
            ds = _find_dataset(h5, names)
            if ds is None:
                raise ValueError(f"{var} dataset missing from {product_id} granule")
            data = _lead_dims(_read_scaled(ds))
            if transform is not None:
                data = transform(data)
            lat_arr, lon_arr = _geo2d(h5, ("Latitude", "Longitude"), expected=data.shape)
            values = _sample_geolocated(data, lat_arr, lon_arr)
            return {"values": values, "units": units, "variable": var, "derived": {}}

    raise ValueError(f"unsupported product_id: {product_id}")


def _geo2d(h5file, latlon_names: Tuple[str, str], expected: Optional[Tuple[int, ...]] = None):
    """Load 2-D latitude/longitude arrays (applying scale), checking shape."""
    lat_ds = _find_dataset(h5file, (latlon_names[0], "LAT", "latitude"))
    lon_ds = _find_dataset(h5file, (latlon_names[1], "LON", "longitude"))
    if lat_ds is None or lon_ds is None:
        raise ValueError("geolocation datasets missing")
    lat = _read_scaled(lat_ds)
    lon = _read_scaled(lon_ds)
    if expected is not None and lat.shape != expected:
        # Try trimming to the data grid (some products pad geolocation)
        lat = lat[: expected[0], : expected[1]]
        lon = lon[: expected[0], : expected[1]]
    return lat, lon
