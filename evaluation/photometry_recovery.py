"""Evaluator-only injected-source recovery; excluded from the production wheel."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from sieve.photometry.qualified import QR3Frame, detect_positions, fit_flux


def match_positions(
    truth: np.ndarray[Any, Any], detections: np.ndarray[Any, Any], radius: float = 2.0
) -> tuple[list[dict[str, Any]], list[int]]:
    """Maximize gated one-to-one recovery, then minimize distance.

    Dedicated unmatched columns avoid forcing a truth source onto a remote peak.
    A detection can recover at most one blend component.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("positive finite match radius required")
    if len(truth) == 0:
        return [], list(range(len(detections)))
    distances = cdist(truth, detections)
    penalty = (len(truth) + 1) * radius
    cost = np.where(distances <= radius, distances, 2 * penalty)
    cost = np.column_stack([cost, np.full((len(truth), len(truth)), penalty)])
    rows, columns = linear_sum_assignment(cost)
    assignments = dict(zip(rows.tolist(), columns.tolist(), strict=True))
    matched = set()
    result = []
    for i in range(len(truth)):
        j = assignments[i]
        recovered = j < len(detections) and distances[i, j] <= radius
        if recovered:
            matched.add(j)
        result.append(
            {
                "truth_index": i,
                "recovered": bool(recovered),
                "detection_index": j if recovered else None,
                "distance_pixels": float(distances[i, j]) if recovered else None,
            }
        )
    return result, [i for i in range(len(detections)) if i not in matched]


def run_recovery(frame: QR3Frame, seed: int = 20261011, sites: int = 32) -> dict[str, Any]:
    """Frozen pilot: entire real background/variance/flag planes, finite released ePSF.

    No injected source position is passed to either detector. Forced photometry
    at evaluator positions is a separate measurement experiment. Real-image
    unmatched detections are *unlabelled*, not false detections.
    """
    rng = np.random.default_rng(seed)
    positions: list[tuple[float, float]] = []
    for _ in range(10000):
        p = (
            float(rng.uniform(12, frame.image.shape[1] - 12)),
            float(rng.uniform(12, frame.image.shape[0] - 12)),
        )
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 14 for q in positions):
            positions.append(p)
            if len(positions) == sites:
                break
    if len(positions) != sites:
        raise ValueError("not enough separated injection sites; do not silently reduce denominator")
    # Archive-variance Gaussian null has no true astrophysical sources. It is not real blank sky.
    blank = rng.normal(size=frame.image.shape) * np.sqrt(np.where(frame.bad, 0, frame.variance))
    baseline = []
    methods: tuple[Literal["DAO", "IRAF"], ...] = ("DAO", "IRAF")
    for method in methods:
        detected = detect_positions(frame, blank, method)
        baseline.append(
            {
                "method": method,
                "arm": "archive_variance_gaussian_null",
                "detections": detected.tolist(),
                "false_detections": len(detected),
            }
        )
    site_rows: list[dict[str, Any]] = []
    for i, (x, y) in enumerate(positions):
        psf, slices, zone = frame.stamp(x, y)
        fit = fit_flux(frame, blank, x, y)
        # Derive amplitude scale from the known variance/design even when the site is masked.
        # A failed coverage check does not exclude that site's injections or detections.
        noise = float(np.median(np.sqrt(frame.variance[slices][~frame.bad[slices]])))
        sigma = fit["flux_error_uJy"] or noise / float(np.sqrt(np.sum(psf**2)))
        site_rows.append(
            {
                "site_id": f"site:{i:03d}",
                "x_pixel": x,
                "y_pixel": y,
                "epsf_zone": zone,
                "nominal_sigma_uJy": sigma,
                "masked_pixels": int(frame.bad[slices].sum()),
            }
        )
    experiments = []
    for snr in (3.0, 5.0, 10.0, 20.0):
        for arm in (
            "real_image",
            "archive_variance_gaussian_blank",
            "two_source_blend",
            "masked_core",
            "one_pixel_artifact",
        ):
            base = frame.image.copy() if arm == "real_image" else blank.copy()
            data = base.copy()
            active_frame = frame
            truth = []
            if arm == "masked_core":
                bad, flags = frame.bad.copy(), frame.flags.copy()
                for x, y in positions:
                    ix, iy = int(round(x)), int(round(y))
                    bad[iy, ix] = True
                    flags[iy, ix] |= 1 << 6
                active_frame = replace(frame, bad=bad, flags=flags)
            for site, (x, y) in zip(site_rows, positions, strict=True):
                flux = snr * site["nominal_sigma_uJy"]
                psf, slices, _ = frame.stamp(x, y)
                if arm == "one_pixel_artifact":
                    data[int(round(y)), int(round(x))] += flux
                else:
                    data[slices] += flux * psf
                truth.append(
                    {
                        "site_id": site["site_id"],
                        "component": 0,
                        "x_pixel": x,
                        "y_pixel": y,
                        "injected_flux_uJy": flux,
                        "astrophysical_injection": arm != "one_pixel_artifact",
                    }
                )
                if arm == "two_source_blend":
                    companion, companion_slices, _ = frame.stamp(x + 1.5, y)
                    data[companion_slices] += flux * companion
                    truth.append(
                        {
                            "site_id": site["site_id"],
                            "component": 1,
                            "x_pixel": x + 1.5,
                            "y_pixel": y,
                            "injected_flux_uJy": flux,
                            "astrophysical_injection": True,
                        }
                    )
            detections = []
            truth_xy = np.array([[t["x_pixel"], t["y_pixel"]] for t in truth])
            for method in methods:
                detected = detect_positions(active_frame, data, method)
                associations, unmatched = match_positions(truth_xy, detected)
                recovered = sum(r["recovered"] for r in associations)
                detections.append(
                    {
                        "method": method,
                        "detections": detected.tolist(),
                        "associations": associations,
                        "recovered": recovered,
                        "denominator": len(truth),
                        "recovery_fraction": recovered / len(truth),
                        "unmatched_detection_indices": unmatched,
                        "false_detections": len(unmatched)
                        if arm not in ("real_image", "one_pixel_artifact")
                        else None,
                        "false_detection_reason": "pre-existing real sources unlabelled"
                        if arm == "real_image"
                        else "artifact targets counted separately"
                        if arm == "one_pixel_artifact"
                        else None,
                        "artifact_peak_recoveries": recovered
                        if arm == "one_pixel_artifact"
                        else None,
                    }
                )
            measurements = []
            for target in truth:
                x, y = target["x_pixel"], target["y_pixel"]
                fitted = fit_flux(active_frame, data, x, y)
                before = fit_flux(active_frame, base, x, y)
                delta = (
                    fitted["flux_uJy"] - before["flux_uJy"]
                    if fitted["flux_uJy"] is not None and before["flux_uJy"] is not None
                    else None
                )
                measurements.append(
                    {
                        **target,
                        "fit": fitted,
                        "baseline_fit": before,
                        "paired_delta_uJy": delta,
                        "fractional_error": (fitted["flux_uJy"] / target["injected_flux_uJy"] - 1)
                        if fitted["flux_uJy"] is not None
                        else None,
                        "paired_fractional_error": delta / target["injected_flux_uJy"] - 1
                        if delta is not None
                        else None,
                    }
                )
            experiments.append(
                {
                    "arm": arm,
                    "nominal_snr": snr,
                    "truth": truth,
                    "detection": detections,
                    "measurement": measurements,
                }
            )
    return {
        "seed": seed,
        "sites": site_rows,
        "null_controls": baseline,
        "experiments": experiments,
        "status": "MODEL_CONDITIONAL_RECOVERY_ONLY",
        "limitations": [
            "injection and fit share finite ePSF; not independent PSF validation",
            "paired deltas cancel existing noise; cannot calibrate uncertainty",
            "no added source Poisson noise or covariance",
            "real unmatched peaks have unknown identity",
            "no real blank-sky or moving-source truth cohort",
        ],
    }
