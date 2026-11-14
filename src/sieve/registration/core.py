"""Bounded native-WCS diagnostics; no identity, motion or difference-image verdicts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


@dataclass(frozen=True)
class Observation:
    """One native 2D FITS plane with explicit unqualified metadata."""

    path: Path
    sha256: str
    image_hdu: str
    wcs: WCS
    image: np.ndarray[Any, Any]
    variance: np.ndarray[Any, Any] | None
    flags: np.ndarray[Any, Any] | None
    plane_wcs_verified: bool
    header: fits.Header
    mjd_mid_utc: float | None
    band_um: tuple[float, float] | None
    detector: str | None
    exposure_id: str | None
    unit: str | None
    roundtrip_max_pixel: float
    pixel_scales_arcsec: tuple[float, float]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sample(shape: tuple[int, int], count: int = 25) -> np.ndarray[Any, Any]:
    y, x = np.meshgrid(
        np.linspace(0, shape[0] - 1, count), np.linspace(0, shape[1] - 1, count), indexing="ij"
    )
    return np.column_stack((x.ravel(), y.ravel()))


def load_observation(path: Path, image_hdu: str = "IMAGE") -> Observation:
    """Read celestial WCS, planes and metadata; refuse invalid celestial transforms."""
    path = path.resolve()
    sha = _sha256(path)
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[image_hdu]
        image = np.asarray(hdu.data, dtype=float)
        if image.ndim != 2 or min(image.shape) < 2:
            raise ValueError("IMAGE must be a 2D array with both dimensions >= 2")
        header = hdu.header.copy()
        if not header.get("CTYPE1") or not header.get("CTYPE2"):
            raise ValueError("INVALID_WCS: missing celestial axis definitions")
        wcs = WCS(header, fobj=hdul).celestial
        if not wcs.has_celestial or wcs.pixel_n_dim != 2:
            raise ValueError("INVALID_WCS: celestial WCS required")
        pixels = _sample(image.shape, 5)
        try:
            sky = wcs.pixel_to_world(pixels[:, 0], pixels[:, 1]).icrs
            x, y = wcs.world_to_pixel(sky)
            errors = np.hypot(x - pixels[:, 0], y - pixels[:, 1])
        except Exception as exc:
            raise ValueError("INVALID_WCS: transformation failed") from exc
        if not np.isfinite(errors).all() or float(np.max(errors)) > 0.01:
            raise ValueError("INVALID_WCS: round-trip error exceeds 0.01 pixel")
        scales = proj_plane_pixel_scales(wcs) * 3600
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("INVALID_WCS: invalid nominal pixel scales")

        def plane(name: str) -> np.ndarray[Any, Any] | None:
            if name not in hdul:
                return None
            value = np.asarray(hdul[name].data).copy()
            if value.shape != image.shape:
                raise ValueError(f"{name} shape differs from IMAGE")
            return value

        variance, flags = plane("VARIANCE"), plane("FLAGS")
        plane_wcs_verified = variance is not None and flags is not None
        for name in ("VARIANCE", "FLAGS"):
            if name not in hdul or not hdul[name].header.get("CTYPE1"):
                plane_wcs_verified = False
                continue
            try:
                plane_wcs = WCS(hdul[name].header, fobj=hdul).celestial
                plane_sky = plane_wcs.pixel_to_world(pixels[:, 0], pixels[:, 1]).icrs
                if np.max(sky.separation(plane_sky).arcsec) > 1e-5:
                    plane_wcs_verified = False
            except Exception:
                plane_wcs_verified = False
        band = None
        if "CWAVE" in hdul and "CBAND" in hdul:
            wave, width = plane("CWAVE"), plane("CBAND")
            if wave is not None and width is not None:
                try:
                    wave = (wave * u.Unit(hdul["CWAVE"].header["BUNIT"])).to_value(u.um)
                    width = (width * u.Unit(hdul["CBAND"].header["BUNIT"])).to_value(u.um)
                except (KeyError, ValueError, u.UnitConversionError):
                    wave = width = None
            if wave is not None and width is not None:
                good = np.isfinite(wave) & np.isfinite(width) & (wave > 0) & (width > 0)
                if np.any(good):
                    band = (float(np.median(wave[good])), float(np.median(width[good])))
        # A missing time scale is unknown; do not silently assume UTC.
        mjd = None
        if header.get("MJD-AVG") is not None and str(header.get("TIMESYS", "")).upper() == "UTC":
            mjd = float(Time(float(header["MJD-AVG"]), format="mjd", scale="utc").mjd)
    if _sha256(path) != sha:
        raise ValueError("input changed during read")
    return Observation(
        path,
        sha,
        image_hdu,
        wcs,
        image,
        variance,
        flags,
        plane_wcs_verified,
        header,
        mjd,
        band,
        str(header["DETECTOR"]) if "DETECTOR" in header else None,
        str(header["OBSID"]) if "OBSID" in header else None,
        str(header["BUNIT"]) if "BUNIT" in header else None,
        float(np.max(errors)),
        (float(scales[0]), float(scales[1])),
    )


def _inside(source: Observation, target: Observation, n: int = 41) -> float:
    """Fraction of sampled source pixel centers mapping inside target rectangle."""
    pixels = _sample(source.image.shape, n)
    sky = source.wcs.pixel_to_world(pixels[:, 0], pixels[:, 1]).icrs
    x, y = target.wcs.world_to_pixel(sky)
    return float(
        np.mean(
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= -0.5)
            & (y >= -0.5)
            & (x < target.image.shape[1] - 0.5)
            & (y < target.image.shape[0] - 0.5)
        )
    )


def associate(
    a: Observation,
    b: Observation,
    xy_a: np.ndarray[Any, Any],
    xy_b: np.ndarray[Any, Any],
    radius_arcsec: float,
) -> dict[str, Any]:
    """Enumerate gated WCS pairs; isolate ambiguity rather than choosing nearest."""
    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("positive finite association radius required")
    xa, xb = np.asarray(xy_a, dtype=float), np.asarray(xy_b, dtype=float)
    if xa.ndim != 2 or xb.ndim != 2 or xa.shape[1:] != (2,) or xb.shape[1:] != (2,):
        raise ValueError("positions must be Nx2 arrays in zero-based x,y pixels")
    if not np.isfinite(xa).all() or not np.isfinite(xb).all():
        raise ValueError("positions must be finite")
    for xy, observation in ((xa, a), (xb, b)):
        if len(xy) and np.any(
            (xy[:, 0] < -0.5)
            | (xy[:, 0] >= observation.image.shape[1] - 0.5)
            | (xy[:, 1] < -0.5)
            | (xy[:, 1] >= observation.image.shape[0] - 0.5)
        ):
            raise ValueError("position lies outside the native image")
    if not len(xa) or not len(xb):
        return {
            "candidate_pairs": [],
            "unambiguous_pairs": [],
            "ambiguous_pairs": [],
            "radius_arcsec": radius_arcsec,
        }
    ca: SkyCoord = a.wcs.pixel_to_world(xa[:, 0], xa[:, 1]).icrs
    cb: SkyCoord = b.wcs.pixel_to_world(xb[:, 0], xb[:, 1]).icrs
    i, j, separation, _ = search_around_sky(ca, cb, radius_arcsec * u.arcsec)
    counts_a = np.bincount(i, minlength=len(xa))
    counts_b = np.bincount(j, minlength=len(xb))
    pairs = [
        {
            "a": int(x),
            "b": int(y),
            "separation_arcsec": float(s),
            "ambiguous": bool(counts_a[x] > 1 or counts_b[y] > 1),
        }
        for x, y, s in zip(i, j, separation.arcsec, strict=True)
    ]
    return {
        "candidate_pairs": pairs,
        "unambiguous_pairs": [p for p in pairs if not p["ambiguous"]],
        "ambiguous_pairs": [p for p in pairs if p["ambiguous"]],
        "radius_arcsec": radius_arcsec,
        "limitations": [
            "Proximity is not identity; covariance and proper motion are not qualified.",
            "These are candidate pairings; unmatched detections are not anomalies.",
        ],
    }


def reference_residuals(
    a: Observation, b: Observation, xy_a: np.ndarray[Any, Any], xy_b: np.ndarray[Any, Any]
) -> dict[str, Any]:
    """Estimate a diagnostic shift on first half, evaluate second half, never assert identity.

    Callers must supply corresponding reference candidates in independent ordering.
    A simple constant tangent-plane shift is a diagnostic only and is not applied.
    """
    xa, xb = np.asarray(xy_a, dtype=float), np.asarray(xy_b, dtype=float)
    if xa.ndim != 2 or xa.shape[1:] != (2,) or xa.shape != xb.shape or len(xa) < 6:
        raise ValueError("at least six paired reference positions required")
    if not np.isfinite(xa).all() or not np.isfinite(xb).all():
        raise ValueError("reference positions must be finite")
    for xy, observation in ((xa, a), (xb, b)):
        if len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError("reference positions must be distinct")
        if np.any(
            (xy[:, 0] < -0.5)
            | (xy[:, 0] >= observation.image.shape[1] - 0.5)
            | (xy[:, 1] < -0.5)
            | (xy[:, 1] >= observation.image.shape[0] - 0.5)
        ):
            raise ValueError("reference lies outside the native image")
    ca = a.wcs.pixel_to_world(xa[:, 0], xa[:, 1]).icrs
    cb = b.wcs.pixel_to_world(xb[:, 0], xb[:, 1]).icrs
    east, north = ca.spherical_offsets_to(cb)
    offsets = np.column_stack((east.to_value(u.arcsec), north.to_value(u.arcsec)))
    split = len(offsets) // 2
    correction = np.median(offsets[:split], axis=0)
    heldout_before = np.linalg.norm(offsets[split:], axis=1)
    heldout_after = np.linalg.norm(offsets[split:] - correction, axis=1)
    return {
        "status": "REGISTRATION_UNQUALIFIED",
        "method": "median tangent-plane shift from first half; second-half residual diagnostic",
        "train_count": split,
        "heldout_count": len(offsets) - split,
        "shift_east_north_arcsec": correction.tolist(),
        "heldout_median_before_arcsec": float(np.median(heldout_before)),
        "heldout_median_after_arcsec": float(np.median(heldout_after)),
        "heldout_p95_after_arcsec": float(np.quantile(heldout_after, 0.95)),
        "limitations": [
            "No independent reference identities or calibrated position uncertainties are assumed.",
            "Constant offset is not a distortion model and is not applied to data.",
            "Held-out pairs can still share calibration and matching-selection biases.",
        ],
    }


def compare(
    a: Observation, b: Observation, matches: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Report sampling-based geometry and conservative task-specific scientific gates."""
    ab, ba = _inside(a, b), _inside(b, a)
    overlap = max(ab, ba)
    pairs = [] if matches is None else matches["unambiguous_pairs"]
    residuals = [x["separation_arcsec"] for x in pairs]
    same = bool(
        a.exposure_id
        and b.exposure_id
        and a.detector
        and b.detector
        and (a.exposure_id, a.detector) == (b.exposure_id, b.detector)
    )
    spatial = "NO_SAMPLED_COMMON_COVERAGE" if overlap == 0 else "REGISTRATION_UNQUALIFIED"
    same_instrument = bool(
        a.header.get("TELESCOP")
        and a.header.get("INSTRUME")
        and a.header.get("TELESCOP") == b.header.get("TELESCOP")
        and a.header.get("INSTRUME") == b.header.get("INSTRUME")
    )
    duplicate = a.sha256 == b.sha256 or (same and same_instrument)
    temporal = (
        "DUPLICATE_EXPOSURE"
        if duplicate
        else "POSSIBLE_DUPLICATE_EXPOSURE"
        if same
        else (
            "PARTIALLY_COMPARABLE"
            if a.exposure_id
            and b.exposure_id
            and a.detector
            and b.detector
            and a.mjd_mid_utc is not None
            and b.mjd_mid_utc is not None
            and a.mjd_mid_utc != b.mjd_mid_utc
            else "INSUFFICIENT_CALIBRATION"
        )
    )
    # Different central bands can be reported as mismatch only when their intervals do not overlap.
    spectral = "INSUFFICIENT_CALIBRATION"
    if a.band_um and b.band_um:
        lo = max(a.band_um[0] - a.band_um[1] / 2, b.band_um[0] - b.band_um[1] / 2)
        hi = min(a.band_um[0] + a.band_um[1] / 2, b.band_um[0] + b.band_um[1] / 2)
        spectral = "POSSIBLE_BANDPASS_MISMATCH" if lo >= hi else "PARTIALLY_COMPARABLE"
    return {
        "schema_version": "sieve.registration.diagnostics.v1",
        "inputs": [
            {
                "sha256": o.sha256,
                "source_name": o.path.name,
                "image_hdu": o.image_hdu,
                "telescope": o.header.get("TELESCOP"),
                "instrument": o.header.get("INSTRUME"),
                "date_start": o.header.get("DATE-OBS"),
                "date_end": o.header.get("DATE-END"),
                "time_scale": o.header.get("TIMESYS"),
                "reference_frame": o.header.get("RADESYS"),
                "equinox": o.header.get("EQUINOX"),
                "has_distortion": bool(o.wcs.has_distortion),
                "pixel_scale_matrix_deg": o.wcs.pixel_scale_matrix.tolist(),
                "exposure_id": o.exposure_id,
                "detector": o.detector,
                "mjd_mid_utc": o.mjd_mid_utc,
                "image_unit": o.unit,
                "variance_flag_wcs_verified": o.plane_wcs_verified,
                "band_median_um": o.band_um,
                "pixel_scales_arcsec": o.pixel_scales_arcsec,
                "roundtrip_max_pixel": o.roundtrip_max_pixel,
            }
            for o in (a, b)
        ],
        "sampled_overlap": {"a_in_b_fraction": ab, "b_in_a_fraction": ba, "grid": 41},
        "spatial_status": spatial,
        "temporal_status": temporal,
        "spectral_status": spectral,
        "flux_variability_status": "INSUFFICIENT_CALIBRATION",
        "source_motion_status": "REGISTRATION_UNQUALIFIED",
        "matched_unambiguous": len(pairs),
        "separation_median_arcsec": float(np.median(residuals)) if residuals else None,
        "separation_p95_arcsec": float(np.quantile(residuals, 0.95)) if residuals else None,
        "limitations": [
            "Overlap is sampled, not an exact footprint; tiny overlaps may be missed.",
            "Pair separations are in-sample, not independent astrometric registration residuals.",
            "Reference identity, covariance, proper motion, PSF and calibration are unqualified.",
            "Band medians omit response curves and source-local wavelength differences.",
        ],
    }


def reproject_visualization(
    source: Observation, target: Observation, bad_bits: int
) -> dict[str, Any]:
    """Reproject for display only; never interpolate bitfields or claim propagated variance."""
    from reproject import reproject_interp

    if source.flags is None or source.variance is None or not source.plane_wcs_verified:
        raise ValueError("visualization requires WCS-verified flag and variance planes")
    flags = np.asarray(source.flags)
    if not np.issubdtype(flags.dtype, np.integer) or np.any(flags < 0) or bad_bits < 0:
        raise ValueError("flags and bad-bit mask must be nonnegative integers")
    mask = (
        ((flags.astype(np.uint64) & np.uint64(bad_bits)) != 0)
        | ~np.isfinite(source.image)
        | ~np.isfinite(source.variance)
        | (source.variance <= 0)
    )
    data = np.where(mask, np.nan, source.image)
    image, footprint = reproject_interp(
        (data, source.wcs), target.wcs, shape_out=target.image.shape
    )
    good, mask_footprint = reproject_interp(
        ((~mask).astype(float), source.wcs),
        target.wcs,
        shape_out=target.image.shape,
        order="nearest-neighbor",
    )
    valid = np.isfinite(image) & (footprint > 0) & (mask_footprint > 0) & (good > 0.5)
    return {
        "image": np.where(valid, image, np.nan),
        "coverage": valid,
        "variance": None,
        "status": "VISUALIZATION_ONLY",
        "limitations": [
            "Interpolation changes noise covariance; variance is not propagated.",
            "Nearest-neighbor validity is not full mask propagation; no difference science.",
        ],
    }
