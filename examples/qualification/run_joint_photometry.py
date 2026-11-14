"""Frozen QR3 real-image blend photometry benchmark and boundary-fit audit.

Outputs retain paired injection increments, failures, timings, IRSA comparisons,
and an SVG figure. No source identity is inferred. See protocol-v2 before running.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table
from photutils.background import LocalBackground, MedianBackground
from photutils.psf import ImagePSF, PSFPhotometry, SourceGrouper

from sieve.photometry.qualified import (
    QR3Frame,
    detect_positions,
    file_digest,
    fit_flux,
    fit_flux_group,
    load_qr3_frame,
)
from sieve.provenance.runtime import source_identity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples" / "qualification"
METHODS = ("sieve_independent", "sieve_joint_constant", "sieve_joint_plane", "photutils_grouped")


def _render_profile(
    profile_frame: QR3Frame,
    zone: int,
    x: float,
    y: float,
    target_shape: tuple[int, int],
    size: int = 11,
) -> tuple[np.ndarray, tuple[slice, slice]]:
    half = (size - 1) // 2
    x0, y0 = int(np.floor(x - half)), int(np.floor(y - half))
    yy, xx = np.mgrid[y0 : y0 + size, x0 : x0 + size]
    model = ImagePSF(
        np.asarray(profile_frame.epsf["EPSF"][zone], dtype=float),
        oversampling=profile_frame.oversampling,
    )
    stamp = np.asarray(model.evaluate(xx, yy, flux=1.0, x_0=x, y_0=y), dtype=float)
    if not np.isfinite(stamp).all() or stamp.sum() <= 0:
        raise ValueError("invalid cross-exposure ePSF stamp")
    stamp /= stamp.sum()
    y_start, x_start = max(y0, 0), max(x0, 0)
    y_stop, x_stop = min(y0 + size, target_shape[0]), min(x0 + size, target_shape[1])
    slices = (slice(y_start, y_stop), slice(x_start, x_stop))
    cropped = stamp[y_start - y0 : y_stop - y0, x_start - x0 : x_stop - x0]
    return cropped, slices


def _cross_exposure_profile(
    science: QR3Frame, template: QR3Frame, x: float, y: float
) -> tuple[np.ndarray, tuple[slice, slice], int]:
    ra, dec = science.wcs.pixel_to_world_values(x, y)
    tx, ty = template.wcs.world_to_pixel_values(ra, dec)
    ox, oy = template.origin_xy
    distances = (template.epsf["XCENTER"] - float(tx) - ox) ** 2 + (
        template.epsf["YCENTER"] - float(ty) - oy
    ) ** 2
    zone = int(np.argmin(distances))
    psf, slices = _render_profile(template, zone, x, y, science.image.shape)
    return psf, slices, zone


def _choose_groups(frame: QR3Frame, seed: int, count: int = 8) -> list[dict[str, Any]]:
    """Choose repeatable real-image source-centered sites, two per sky quadrant."""
    rng = np.random.default_rng(seed)
    detections = detect_positions(frame, frame.image, "DAO")
    quadrants: dict[int, list[tuple[float, float]]] = {i: [] for i in range(4)}
    for x, y in detections:
        if not (18 < x < frame.image.shape[1] - 18 and 18 < y < frame.image.shape[0] - 18):
            continue
        q = int(x >= frame.image.shape[1] / 2) + 2 * int(y >= frame.image.shape[0] / 2)
        quadrants[q].append((float(x), float(y)))
    selected: list[tuple[int, float, float]] = []
    for quadrant, points in quadrants.items():
        if len(points) < count // 4:
            raise RuntimeError(f"not enough DAO positions in held-out quadrant {quadrant}")
        rng.shuffle(points)
        for x, y in points:
            if all(np.hypot(x - px, y - py) > 24 for _, px, py in selected):
                selected.append((quadrant, x, y))
                if sum(q == quadrant for q, _, _ in selected) == count // 4:
                    break
    if len(selected) != count:
        raise RuntimeError("could not select the frozen number of separated groups")
    separations = (1.0, 1.5, 2.5, 4.0)
    ratios = (1.0, 3.0)
    result = []
    for i, (quadrant, x, y) in enumerate(selected):
        angle = float(rng.uniform(0, 2 * np.pi))
        separation, ratio = separations[i % 4], ratios[(i // 4) % 2]
        dx, dy = np.cos(angle) * separation / 2, np.sin(angle) * separation / 2
        positions = [(x - dx, y - dy), (x + dx, y + dy)]
        if not all(
            12 < px < frame.image.shape[1] - 12 and 12 < py < frame.image.shape[0] - 12
            for px, py in positions
        ):
            raise RuntimeError("frozen selection unexpectedly placed a source near the cutout edge")
        result.append(
            {
                "group_id": f"{frame.header['OBSID']}:g{i:02d}",
                "center_x": x,
                "center_y": y,
                "quadrant": quadrant,
                "separation_pixels": separation,
                "bright_to_faint_flux_ratio": ratio,
                "positions": positions,
            }
        )
    return result


def _make_photutils(frame: QR3Frame, positions: list[tuple[float, float]]) -> PSFPhotometry:
    _, _, zone, _, _ = frame._stamp(*positions[0])
    # Photutils ImagePSF expects an oversampled ePSF sum of oversampling**2.
    model = ImagePSF(
        np.asarray(frame.epsf["EPSF"][zone], dtype=float) * np.prod(frame.oversampling),
        oversampling=frame.oversampling,
    )
    model.x_0.fixed = True
    model.y_0.fixed = True
    return PSFPhotometry(
        model,
        fit_shape=(11, 11),
        grouper=SourceGrouper(min_separation=5.0),
        aperture_radius=3.0,
        local_bkg_estimator=LocalBackground(7, 10, bkg_estimator=MedianBackground()),
    )


def _photutils_fit(
    frame: QR3Frame, data: np.ndarray, positions: list[tuple[float, float]]
) -> dict[str, Any]:
    fitter = _make_photutils(frame, positions)
    params = Table(
        {
            "x_init": [p[0] for p in positions],
            "y_init": [p[1] for p in positions],
        }
    )
    result = fitter(data, mask=frame.bad, error=np.sqrt(frame.variance), init_params=params)
    out: dict[str, Any] = {
        "fluxes": [float(v) if np.isfinite(v) else None for v in result["flux_fit"]],
        "errors": [float(v) if np.isfinite(v) and v > 0 else None for v in result["flux_err"]],
        "flags": [int(v) for v in result["flags"]],
        "local_background": [float(v) for v in result["local_bkg"]],
        "fit_pixels": [int(v) for v in result["n_pixels_fit"]],
    }
    return out


def _fit_method(
    method: str,
    frame: QR3Frame,
    data: np.ndarray,
    positions: list[tuple[float, float]],
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    if method == "sieve_independent":
        rows = [fit_flux(frame, data, *pos) for pos in positions]
        out = {
            "fluxes": [row["flux_uJy"] for row in rows],
            "errors": [row["flux_error_uJy"] for row in rows],
            "status": "MEASURED_CONDITIONAL"
            if all(row["status"] == "MEASURED_CONDITIONAL" for row in rows)
            else "FAILED",
            "reasons": [row.get("reason") for row in rows],
        }
    elif method in ("sieve_joint_constant", "sieve_joint_plane"):
        row = fit_flux_group(
            frame,
            data,
            positions,
            background_order=0 if method.endswith("constant") else 1,
        )
        out = {
            "fluxes": [source.get("flux_uJy") for source in row.get("sources", [])],
            "errors": [source.get("flux_error_uJy") for source in row.get("sources", [])],
            "status": row["status"],
            "reason": row.get("reason"),
            "condition_number": row.get("condition_number"),
            "reduced_chi2": row.get("reduced_chi2"),
            "covariance": row.get("amplitude_covariance_uJy2"),
        }
    else:
        out = _photutils_fit(frame, data, positions)
        out["status"] = (
            "MEASURED_CONDITIONAL" if all(v is not None for v in out["fluxes"]) else "FAILED"
        )
    out["wall_seconds"] = time.perf_counter() - started
    out["cpu_seconds"] = time.process_time() - cpu_started
    return out


def _boundary_audit(frames: list[QR3Frame]) -> dict[str, Any]:
    records = []
    reason_counts: dict[str, int] = {}
    for frame in frames:
        points = detect_positions(frame, frame.image, "DAO")
        for index, (x, y) in enumerate(points):
            row = fit_flux(frame, frame.image, float(x), float(y))
            edge_distance = float(
                min(x, y, frame.image.shape[1] - 1 - x, frame.image.shape[0] - 1 - y)
            )
            reason = row.get("reason", "MEASURED") if row["status"] == "FAILED" else "MEASURED"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            records.append(
                {
                    "obs_id": frame.header["OBSID"],
                    "source_id": f"{frame.header['OBSID']}:dao{index:04d}",
                    "x_pixel": float(x),
                    "y_pixel": float(y),
                    "edge_distance_pixels": edge_distance,
                    "epsf_zone": row.get("epsf_zone"),
                    "cwave_um": row.get("cwave_um"),
                    "cband_um": row.get("cband_um"),
                    "psf_edge_information_coverage": row.get("psf_edge_information_coverage"),
                    "psf_information_coverage": row.get("psf_information_coverage"),
                    "flux_uJy": row.get("flux_uJy"),
                    "flux_error_uJy": row.get("flux_error_uJy"),
                    "reduced_chi2": row.get("reduced_chi2"),
                    "status": row["status"],
                    "reason": reason,
                    "interpretation": (
                        "Real-scene fit is not a flux-truth or uncertainty-calibration test."
                    ),
                }
            )
    bins = []
    for lower, upper in ((0, 3), (3, 6), (6, 11), (11, 20), (20, 1000)):
        subset = [r for r in records if lower <= r["edge_distance_pixels"] < upper]
        bins.append(
            {
                "edge_distance_pixels": [lower, upper],
                "detections": len(subset),
                "measured": sum(r["status"] == "MEASURED_CONDITIONAL" for r in subset),
                "failed": sum(r["status"] == "FAILED" for r in subset),
                "median_edge_information_coverage": float(
                    np.median(
                        [
                            r["psf_edge_information_coverage"]
                            for r in subset
                            if r["psf_edge_information_coverage"] is not None
                        ]
                    )
                )
                if any(r["psf_edge_information_coverage"] is not None for r in subset)
                else None,
                "median_conditional_reduced_chi2": float(
                    np.median([r["reduced_chi2"] for r in subset if r["reduced_chi2"] is not None])
                )
                if any(r["reduced_chi2"] is not None for r in subset)
                else None,
            }
        )
    return {
        "status": "DESCRIPTIVE_REAL_SCENE_AUDIT_ONLY",
        "detections": len(records),
        "measured": sum(r["status"] == "MEASURED_CONDITIONAL" for r in records),
        "failed": sum(r["status"] == "FAILED" for r in records),
        "failure_reasons": reason_counts,
        "edge_bins": bins,
        "real_scene_bias": (
            "UNKNOWN; the unlabelled image contains unknown source blends and residuals."
        ),
        "uncertainty_calibration": "NOT_ESTIMABLE from these real-scene measurements alone.",
        "records": records,
    }


def _irsa_comparison(frames: list[QR3Frame]) -> dict[str, Any]:
    reference_path = DATA / "irsa-reference.json"
    reference = json.loads(reference_path.read_text())
    by_obs = {frame.header["OBSID"]: frame for frame in frames}
    rows = []
    for ref in reference["rows"]:
        frame = by_obs.get(ref["obs_id"])
        if frame is None:
            continue
        x, y = frame.wcs.world_to_pixel_values(ref["ra"], ref["dec"])
        fit = fit_flux(frame, frame.image, float(x), float(y))
        full_x, full_y = float(x) + frame.origin_xy[0], float(y) + frame.origin_xy[1]
        rows.append(
            {
                "obs_id": ref["obs_id"],
                "position_source": (
                    "official IRSA text-view RA/Dec transformed with pinned science WCS"
                ),
                "irsa_ra_deg": ref["ra"],
                "irsa_dec_deg": ref["dec"],
                "irsa_x_image_detector_pixels": ref["x_image"],
                "sieve_x_detector_pixels": full_x,
                "irsa_y_image_detector_pixels": ref["y_image"],
                "sieve_y_detector_pixels": full_y,
                "position_offset_pixels": float(
                    np.hypot(full_x - ref["x_image"], full_y - ref["y_image"])
                ),
                "irsa_flux_uJy": ref["flux_uJy"],
                "sieve_flux_uJy": fit["flux_uJy"],
                "fractional_flux_difference": (fit["flux_uJy"] / ref["flux_uJy"] - 1)
                if fit["flux_uJy"] is not None
                else None,
                "irsa_total_flux_error_uJy": ref["flux_err_uJy"],
                "irsa_tractor_fit_error_uJy": ref["flux_err_tractor_uJy"],
                "sieve_formal_conditional_error_uJy": fit["flux_error_uJy"],
                "irsa_lambda_um": ref["lambda_um"],
                "sieve_cwave_um": fit.get("cwave_um"),
                "irsa_source_flags": ref["flags"],
                "sieve_flags_at_position": fit.get("flag_bits"),
                "limits": (
                    "One selected bright source across two exposures; retained text-view "
                    "transcription; not a validation population or uncertainty comparison."
                ),
            }
        )
    return {
        "status": "NARROW_FORCED_FLUX_SCALE_CHECK",
        "source_reference_sha256": file_digest(reference_path),
        "selection": reference["selection"],
        "custody": reference["custody"],
        "rows": rows,
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method and r["fractional_errors"]]
        errors = np.asarray(
            [v for r in subset for v in r["fractional_errors"] if v is not None], dtype=float
        )
        by_sep: dict[str, Any] = {}
        for separation in (1.0, 1.5, 2.5, 4.0):
            samples = [
                abs(error)
                for row in subset
                if row["separation_pixels"] == separation
                for error in row["fractional_errors"]
                if error is not None
            ]
            by_sep[str(separation)] = {
                "n_component_draws": len(samples),
                "median_absolute_fractional_error": float(np.median(samples)) if samples else None,
            }
        summary[method] = {
            "group_replicate_draws": len(subset),
            "failed_group_draws": sum(
                r["status"] != "MEASURED_CONDITIONAL" for r in rows if r["method"] == method
            ),
            "component_draws": int(errors.size),
            "median_signed_fractional_error": float(np.median(errors)) if errors.size else None,
            "median_absolute_fractional_error": float(np.median(np.abs(errors)))
            if errors.size
            else None,
            "rms_fractional_error": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            "median_absolute_standardized_residual": float(
                np.median(
                    np.abs([v for r in subset for v in r["standardized_errors"] if v is not None])
                )
            )
            if any(v is not None for r in subset for v in r["standardized_errors"])
            else None,
            "formal_one_sigma_coverage": float(
                np.mean(
                    np.abs([v for r in subset for v in r["standardized_errors"] if v is not None])
                    <= 1
                )
            )
            if any(v is not None for r in subset for v in r["standardized_errors"])
            else None,
            "by_separation": by_sep,
            "median_fit_seconds_per_on_off_pair": float(
                np.median([r["wall_seconds"] for r in rows if r["method"] == method])
            ),
        }
    return summary


def _bootstrap_primary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    eligible = sorted({r["group_id"] for r in rows if r["separation_pixels"] in (1.0, 1.5, 2.5)})
    values: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    for group in eligible:
        for method in METHODS:
            parts = [
                r
                for r in rows
                if r["group_id"] == group and r["method"] == method and r["fractional_errors"]
            ]
            if parts:
                group_errors = [
                    abs(value)
                    for row in parts
                    for value in row["fractional_errors"]
                    if value is not None
                ]
                if group_errors:
                    values[method][group] = float(np.median(group_errors))
    complete = [
        g
        for g in eligible
        if all(g in values[m] for m in ("sieve_independent", "sieve_joint_plane"))
    ]
    if len(complete) < 3:
        return {
            "status": "INSUFFICIENT_GROUPS",
            "eligible_groups": len(eligible),
            "paired_groups": len(complete),
        }
    baseline = np.asarray([values["sieve_independent"][g] for g in complete])
    joint = np.asarray([values["sieve_joint_plane"][g] for g in complete])
    ratio = np.divide(joint, baseline, out=np.full_like(joint, np.nan), where=baseline > 0)
    observed = float(np.nanmedian(ratio))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(complete), size=(10000, len(complete)))
    boot = np.nanmedian(ratio[indices], axis=1)
    interval = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975]).tolist()
    status = (
        "IMPROVEMENT_SUPPORTED_BY_FROZEN_PILOT_GATE"
        if observed <= 0.8 and interval[1] < 1
        else "NO_DEMONSTRATED_IMPROVEMENT"
    )
    reference_groups = [
        group
        for group in eligible
        if group in values["sieve_joint_plane"] and group in values["photutils_grouped"]
    ]
    reference_comparison: dict[str, Any] = {
        "status": "EXPLORATORY_SECONDARY_COMPARISON",
        "prespecified_superiority_gate": False,
        "paired_groups": len(reference_groups),
        "interpretation": (
            "The bootstrap interval overlaps 1; this pilot does not establish superiority "
            "over grouped Photutils."
        ),
    }
    if len(reference_groups) >= 3:
        photutils = np.asarray([values["photutils_grouped"][g] for g in reference_groups])
        joint_reference = np.asarray([values["sieve_joint_plane"][g] for g in reference_groups])
        reference_ratio = np.divide(
            joint_reference,
            photutils,
            out=np.full_like(joint_reference, np.nan),
            where=photutils > 0,
        )
        reference_rng = np.random.default_rng(seed + 1)
        reference_indices = reference_rng.integers(
            0, len(reference_groups), size=(10000, len(reference_groups))
        )
        reference_boot = np.nanmedian(reference_ratio[reference_indices], axis=1)
        reference_interval = np.quantile(
            reference_boot[np.isfinite(reference_boot)], [0.025, 0.975]
        ).tolist()
        reference_comparison.update(
            joint_to_photutils_median_absolute_error_ratio=float(np.nanmedian(reference_ratio)),
            group_bootstrap_95_interval=reference_interval,
            joint_group_median_absolute_error=float(np.median(joint_reference)),
            photutils_group_median_absolute_error=float(np.median(photutils)),
            bootstrap_resamples=10000,
            seed=seed + 1,
        )
    return {
        "status": status,
        "eligible_groups": len(eligible),
        "paired_groups": len(complete),
        "joint_to_independent_median_absolute_error_ratio": observed,
        "group_bootstrap_95_interval": interval,
        "independent_group_median_absolute_error": float(np.median(baseline)),
        "joint_plane_group_median_absolute_error": float(np.median(joint)),
        "exploratory_photutils_reference": reference_comparison,
        "bootstrap_resamples": 10000,
        "seed": seed,
    }


def _write_figure(path: Path, summary: dict[str, Any]) -> None:
    width, height = 900, 560
    left, right, top, bottom = 95, 35, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    separations = (1.0, 1.5, 2.5, 4.0)
    methods = (
        "sieve_independent",
        "sieve_joint_constant",
        "sieve_joint_plane",
        "photutils_grouped",
    )
    colors = ("#64748b", "#e07a24", "#137c8b", "#803e75")
    series = []
    all_values = [
        summary[m]["by_separation"][str(s)]["median_absolute_fractional_error"]
        for m in methods
        for s in separations
        if summary[m]["by_separation"][str(s)]["median_absolute_fractional_error"] is not None
    ]
    ymax = max(0.1, max(all_values, default=0.1) * 1.12)
    for method, color in zip(methods, colors, strict=True):
        pts = []
        for i, separation in enumerate(separations):
            value = summary[method]["by_separation"][str(separation)][
                "median_absolute_fractional_error"
            ]
            if value is not None:
                x = left + i * plot_w / (len(separations) - 1)
                y = top + plot_h * (1 - value / ymax)
                pts.append((x, y, value))
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
        series.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for x, y, value in pts:
            series.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
                f"<title>{method}: {value:.4f}</title>"
            )
    axes = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" '
        f'x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
    ]
    for i, separation in enumerate(separations):
        x = left + i * plot_w / (len(separations) - 1)
        axes.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle">{separation:g}</text>'
        )
    for i in range(6):
        val = ymax * i / 5
        y = top + plot_h * (1 - i / 5)
        axes += [
            f'<line x1="{left - 5}" y1="{y:.1f}" '
            f'x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>',
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{val:.2f}</text>',
        ]
    legend = []
    for i, (method, color) in enumerate(zip(methods, colors, strict=True)):
        y = height - 35
        x = left + i * 190
        legend += [
            f'<line x1="{x}" y1="{y}" x2="{x + 25}" y2="{y}" stroke="{color}" stroke-width="4"/>',
            f'<text x="{x + 32}" y="{y + 4}">{method.replace("_", " ")}</text>',
        ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-size="18" font-family="sans-serif">
QR3 held-background blend injection: absolute fractional flux error</text>
<g font-family="sans-serif" font-size="12" fill="#111827">{"".join(axes)}
{"".join(series)}{"".join(legend)}
<text x="{width / 2}" y="{height - 8}" text-anchor="middle">
Component separation (native pixels)</text>
<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle">
Median absolute injected-flux fractional error</text>
<text x="{left}" y="{height - 60}" font-size="11">
Paired on/off increments on real QR3 backgrounds; fixed positions; unknown scene flux cancels.
</text></g></svg>'''
    path.write_text(svg)


def run(output: Path, replicates: int = 32) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    code_commit, source_tree_sha256, dirty, environment_id = source_identity()
    if replicates != 32:
        raise ValueError("protocol v2 freezes exactly 32 noise-pair replicates")
    output.mkdir(parents=True)
    manifest_path = DATA / "source.json"
    protocol_path = DATA / "photometry-method-protocol-v2.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if protocol["status"] != "FROZEN_BEFORE_METHOD_IMPLEMENTATION":
        raise ValueError("frozen protocol status required")
    frames = [load_qr3_frame(DATA / row["file"], row["sha256"]) for row in manifest["exposures"]]
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    boundary = _boundary_audit(frames)
    irsa = _irsa_comparison(frames)
    trial_rows: list[dict[str, Any]] = []
    all_groups = []
    base_seed = int(protocol["frozen_design"]["seed"])
    for exposure_index, frame in enumerate(frames):
        template = frames[(exposure_index + 1) % len(frames)]
        groups = _choose_groups(frame, base_seed + exposure_index)
        all_groups.extend(groups)
        for group_index, group in enumerate(groups):
            positions = group["positions"]
            profiles = [_cross_exposure_profile(frame, template, *p) for p in positions]
            for _, slices, _ in profiles:
                # Fail closed if the frozen 11x11 template is clipped.
                if (slices[0].stop - slices[0].start, slices[1].stop - slices[1].start) != (11, 11):
                    raise RuntimeError("injection site produced a clipped PSF stamp")
            zero = np.zeros_like(frame.image)
            noise_scales = []
            for pos in positions:
                noise_fit = fit_flux(frame, zero, *pos)
                if noise_fit["flux_error_uJy"] is None:
                    raise RuntimeError(
                        "selected injection source lacks valid formal scale: "
                        f"{noise_fit.get('reason')}"
                    )
                noise_scales.append(float(noise_fit["flux_error_uJy"]))
            faint_flux = protocol["frozen_design"]["nominal_faint_component_snr"] * max(
                noise_scales
            )
            fluxes = [faint_flux * group["bright_to_faint_flux_ratio"], faint_flux]
            injection = np.zeros_like(frame.image)
            injector_zones = []
            for flux, (profile, slices, zone) in zip(fluxes, profiles, strict=True):
                injection[slices] += flux * profile
                injector_zones.append(zone)
            group["science_obs_id"] = frame.header["OBSID"]
            group["injector_obs_id"] = template.header["OBSID"]
            group["injection_epsf_zones"] = injector_zones
            group["fitting_epsf_zones"] = [int(frame._stamp(*p)[2]) for p in positions]
            group["injected_fluxes_uJy"] = fluxes
            group["nominal_faint_snr"] = protocol["frozen_design"]["nominal_faint_component_snr"]
            for method in METHODS:
                for replicate in range(replicates):
                    rng_seed = (
                        base_seed + exposure_index * 1_000_000 + group_index * 10_000 + replicate
                    )
                    rng = np.random.default_rng(rng_seed)
                    sigma = np.sqrt(np.where(frame.bad, 0.0, frame.variance))
                    off = frame.image + rng.normal(size=frame.image.shape) * sigma
                    on_rng = np.random.default_rng(rng_seed + 5_000_000)
                    on = frame.image + on_rng.normal(size=frame.image.shape) * sigma + injection
                    off_fit = _fit_method(method, frame, off, positions)
                    on_fit = _fit_method(method, frame, on, positions)
                    deltas: list[float | None] = []
                    delta_errors: list[float | None] = []
                    standardized: list[float | None] = []
                    failures = []
                    for index, (truth, on_flux, off_flux, on_err, off_err) in enumerate(
                        zip(
                            fluxes,
                            on_fit["fluxes"],
                            off_fit["fluxes"],
                            on_fit["errors"],
                            off_fit["errors"],
                            strict=True,
                        )
                    ):
                        if on_flux is None or off_flux is None:
                            deltas.append(None)
                            delta_errors.append(None)
                            standardized.append(None)
                            failures.append(index)
                            continue
                        delta = float(on_flux - off_flux)
                        propagated = (
                            float(np.hypot(on_err, off_err))
                            if on_err is not None and off_err is not None
                            else None
                        )
                        deltas.append(delta)
                        delta_errors.append(propagated)
                        standardized.append(
                            (delta - truth) / propagated if propagated and propagated > 0 else None
                        )
                    errors = [
                        delta / truth - 1 if delta is not None else None
                        for delta, truth in zip(deltas, fluxes, strict=True)
                    ]
                    trial_rows.append(
                        {
                            "group_id": group["group_id"],
                            "science_obs_id": frame.header["OBSID"],
                            "method": method,
                            "replicate": replicate,
                            "seed": rng_seed,
                            "quadrant": group["quadrant"],
                            "separation_pixels": group["separation_pixels"],
                            "bright_to_faint_flux_ratio": group["bright_to_faint_flux_ratio"],
                            "injected_fluxes_uJy": fluxes,
                            "measured_deltas_uJy": deltas,
                            "fractional_errors": errors,
                            "delta_errors_uJy": delta_errors,
                            "standardized_errors": standardized,
                            "failed_component_indices": failures,
                            "status": "FAILED" if failures else "MEASURED_CONDITIONAL",
                            "wall_seconds": on_fit["wall_seconds"] + off_fit["wall_seconds"],
                            "cpu_seconds": on_fit["cpu_seconds"] + off_fit["cpu_seconds"],
                            "on_fit_diagnostics": {
                                k: v for k, v in on_fit.items() if k not in ("fluxes", "errors")
                            },
                            "off_fit_diagnostics": {
                                k: v for k, v in off_fit.items() if k not in ("fluxes", "errors")
                            },
                        }
                    )
    summary = _metric_summary(trial_rows)
    primary = _bootstrap_primary(trial_rows, base_seed)
    budgets = {
        "shallow": ["sieve_independent"],
        "intermediate": ["sieve_joint_constant"],
        "deep": ["sieve_joint_plane", "photutils_grouped"],
    }
    resource_rows = {}
    for budget, methods in budgets.items():
        subset = [r for r in trial_rows if r["method"] in methods]
        resource_rows[budget] = {
            "methods": methods,
            "method_fit_calls": len(subset) * 2,
            "group_replicate_rows": len(subset),
            "sum_fitter_wall_seconds": float(sum(r["wall_seconds"] for r in subset)),
            "group_replicates_per_fitter_wall_second": len(subset)
            / max(sum(r["wall_seconds"] for r in subset), 1e-12),
            "workers": 1,
            "cpu_seconds": float(sum(r["cpu_seconds"] for r in subset)),
            "wall_seconds_for_budget_stages": float(sum(r["wall_seconds"] for r in subset)),
            "process_peak_rss_bytes_cumulative": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if os.uname().sysname == "Darwin" else 1024)
            ),
            "input_files_read_bytes": int(
                sum((DATA / e["file"]).stat().st_size for e in manifest["exposures"])
            ),
            "note": (
                "Single-process pilot costs; method timing is per fit call; no parallel "
                "scaling evidence."
            ),
        }
    result = {
        "schema_version": "sieve.qr3.photometry-method-result.v1",
        "scientific_status": primary["status"],
        "source_commit": code_commit,
        "source_tree_sha256": source_tree_sha256,
        "source_tree_dirty": dirty,
        "environment_id": environment_id,
        "manifest_sha256": file_digest(manifest_path),
        "protocol_sha256": file_digest(protocol_path),
        "runner_sha256": file_digest(Path(__file__)),
        "fitter_sha256": file_digest(
            Path(__file__).resolve().parents[2] / "src/sieve/photometry/qualified.py"
        ),
        "protocol": protocol,
        "input_sha256": [frame.sha256 for frame in frames],
        "environment": {"numpy": np.__version__, "photutils": __import__("photutils").__version__},
        "photutils_grouped_configuration": {
            "fit_shape_pixels": [11, 11],
            "fixed_positions": True,
            "group_min_separation_pixels": 5.0,
            "initial_flux": (
                "Photutils aperture estimate with aperture_radius=3 pixels; no explicit flux_init"
            ),
            "background": "Photutils LocalBackground with 7-10 pixel annulus and median estimator",
            "image_psf_normalization": (
                "released oversampled ePSF multiplied by oversampling area for the "
                "Photutils native-pixel integrated-flux convention"
            ),
        },
        "boundary_fit_audit": boundary,
        "irsa_reference_comparison": irsa,
        "injection_groups": all_groups,
        "primary_comparison": primary,
        "method_metrics": summary,
        "resource_budgets": resource_rows,
        "cumulative_budget_cost": {
            "shallow_wall_seconds": resource_rows["shallow"]["sum_fitter_wall_seconds"],
            "intermediate_wall_seconds": resource_rows["shallow"]["sum_fitter_wall_seconds"]
            + resource_rows["intermediate"]["sum_fitter_wall_seconds"],
            "deep_wall_seconds": sum(
                row["sum_fitter_wall_seconds"] for row in resource_rows.values()
            ),
        },
        "timing": {
            "wall_seconds": time.perf_counter() - start_wall,
            "cpu_seconds": time.process_time() - start_cpu,
        },
        "interpretation": (
            "All identity results remain INCONCLUSIVE. The endpoint measures known additive "
            "increments on these real backgrounds; it does not validate real-scene total flux, "
            "sky sensitivity, FDR, or source identity."
        ),
    }
    with (output / "trials.csv").open("w", newline="") as stream:
        fields = [
            "group_id",
            "science_obs_id",
            "method",
            "replicate",
            "quadrant",
            "separation_pixels",
            "bright_to_faint_flux_ratio",
            "status",
            "injected_fluxes_uJy",
            "measured_deltas_uJy",
            "fractional_errors",
            "delta_errors_uJy",
            "standardized_errors",
            "wall_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(
                {
                    k: json.dumps(row[k]) if isinstance(row[k], (list, dict)) else row[k]
                    for k in fields
                }
            )
    with (output / "boundary-audit.csv").open("w", newline="") as stream:
        fields = [
            "obs_id",
            "source_id",
            "x_pixel",
            "y_pixel",
            "edge_distance_pixels",
            "epsf_zone",
            "cwave_um",
            "cband_um",
            "psf_edge_information_coverage",
            "psf_information_coverage",
            "flux_uJy",
            "flux_error_uJy",
            "reduced_chi2",
            "status",
            "reason",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k) for k in fields} for row in boundary["records"]])
    _write_figure(output / "blend-error-by-separation.svg", summary)
    result["artifacts_sha256"] = {
        name: file_digest(output / name)
        for name in ("trials.csv", "boundary-audit.csv", "blend-error-by-separation.svg")
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "scientific_status": result["scientific_status"],
                "primary_comparison": primary,
                "boundary": {
                    k: boundary[k]
                    for k in ("detections", "measured", "failed", "failure_reasons", "edge_bins")
                },
                "methods": summary,
                "output": str(output),
            },
            indent=2,
            allow_nan=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=32)
    args = parser.parse_args()
    run(args.output, args.replicates)


if __name__ == "__main__":
    main()
