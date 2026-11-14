"""Held-out spatial-profile diagnostics with no calibrated identity verdict."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import lstsq

from sieve.photometry.qualified import QR3Frame


def compare_profiles(frame: QR3Frame, sky_positions: list[tuple[float, float]]) -> dict[str, Any]:
    """Freeze positions from two earlier exposures; fit held-out amplitudes only.

    One source is placed at the spherical midpoint, two at the supplied positions.
    Both models use the same held-out pixels/variance/constant-sky nuisance term.
    BIC is a model-conditional diagnostic, not a calibrated Bayes factor, posterior
    or proof of one/two identities. Signed amplitudes and bad fits are retained.
    """
    from astropy.coordinates import SkyCoord

    if len(sky_positions) != 2:
        raise ValueError("two frozen generation positions required")
    coordinates = SkyCoord(sky_positions, unit="deg")
    midpoint = coordinates[0].directional_offset_by(
        coordinates[0].position_angle(coordinates[1]),
        coordinates[0].separation(coordinates[1]) / 2,
    )
    x, y = frame.wcs.world_to_pixel(coordinates)
    mx, my = frame.wcs.world_to_pixel(midpoint)
    row: dict[str, Any] = {
        "status": "INCONCLUSIVE",
        "factor": "heldout_pixel_spatial_profile",
        "sky_positions_deg": [list(p) for p in sky_positions],
        "heldout_exposure_id": frame.header["OBSID"],
        "heldout_source_sha256": frame.sha256,
        "wavelength_model": "Amplitudes free at held-out wavelength; flux constancy not assumed",
        "delta_bic_two_minus_one": None,
        "limitations": [
            "positions not independently calibrated",
            "shared calibration covariance unknown",
            "finite ePSF profile and constant sky only",
            "neighbors may confound fits",
            "no reference object identities",
            "signed amplitudes retained",
            "BIC is not a calibrated probability or identity verdict",
            "constant-sky null does not model instrumental artifacts",
        ],
    }
    try:
        components = [frame.stamp(float(px), float(py)) for px, py in zip(x, y, strict=True)]
        single = frame.stamp(float(mx), float(my))
    except ValueError as exc:
        return {**row, "reason": str(exc), "models": {}}
    patches = [*components, single]
    ys = slice(min(s[0].start for _, s, _ in patches), max(s[0].stop for _, s, _ in patches))
    xs = slice(min(s[1].start for _, s, _ in patches), max(s[1].stop for _, s, _ in patches))
    shape = (ys.stop - ys.start, xs.stop - xs.start)
    rendered = []
    for psf, slices, _ in patches:
        image = np.zeros(shape)
        image[
            slices[0].start - ys.start : slices[0].stop - ys.start,
            slices[1].start - xs.start : slices[1].stop - xs.start,
        ] = psf
        rendered.append(image)
    valid = ~frame.bad[ys, xs] & np.isfinite(frame.image[ys, xs])
    n = int(valid.sum())
    if n < 25:
        return {**row, "reason": "INSUFFICIENT_COVERAGE", "models": {}}
    target = frame.image[ys, xs][valid]
    sigma = np.sqrt(frame.variance[ys, xs][valid])
    models: dict[str, dict[str, Any]] = {}
    for name, templates in (
        ("one", [rendered[2]]),
        ("multiple", rendered[:2]),
        ("constant_sky_null", []),
    ):
        columns = [p[valid] for p in templates] + [np.ones(n)]
        design = np.column_stack(columns) / sigma[:, None]
        amplitudes, _, rank, singular = lstsq(design, target / sigma)
        k = len(columns)
        if rank != k:
            models[name] = {"status": "FAILED", "reason": "SINGULAR_FIT", "bic": None}
            continue
        residual = target / sigma - design @ amplitudes
        chi2 = float(residual @ residual)
        models[name] = {
            "status": "DIAGNOSTIC_ONLY",
            "chi2": chi2,
            "degrees_of_freedom": n - k,
            "reduced_chi2": chi2 / (n - k),
            "bic": chi2 + k * float(np.log(n)),
            "condition_number": float(singular[0] / singular[-1]),
            "amplitudes_uJy": amplitudes[:-1].tolist(),
            "background_uJy_per_pixel": float(amplitudes[-1]),
        }
    delta = (
        models["multiple"]["bic"] - models["one"]["bic"]
        if models["multiple"]["bic"] is not None and models["one"]["bic"] is not None
        else None
    )
    return {
        **row,
        "models": models,
        "valid_pixels": n,
        "masked_pixels": int(valid.size - n),
        "delta_bic_two_minus_one": delta,
        "reason": "Measured conditional profile diagnostics; identity claim gate remains closed",
    }
