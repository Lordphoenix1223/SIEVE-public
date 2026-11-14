"""Paired cross-exposure injection audit of the exact 61 recovered edge fits."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from examples.qualification.run_joint_photometry import (
    DATA,
    _boundary_audit,
    _cross_exposure_profile,
    _fit_method,
    file_digest,
    fit_flux,
    load_qr3_frame,
)
from sieve.provenance.runtime import source_identity


def run(output: Path, replicates: int = 32) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_commit, source_tree_sha256, dirty, environment_id = source_identity()
    if replicates != 32:
        raise ValueError("protocol v3 freezes exactly 32 paired perturbations")
    output.mkdir(parents=True)
    manifest_path = DATA / "source.json"
    protocol_path = DATA / "photometry-method-protocol-v3.json"
    protocol_v2_path = DATA / "photometry-method-protocol-v2.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    frames = [load_qr3_frame(DATA / row["file"], row["sha256"]) for row in manifest["exposures"]]
    audit = _boundary_audit(frames)
    obs_ids = [frame.header["OBSID"] for frame in frames]
    frame_index = {frame.header["OBSID"]: i for i, frame in enumerate(frames)}
    cohort = []
    failures = []
    for row in audit["records"]:
        x, y = row["x_pixel"], row["y_pixel"]
        ox, oy = frames[frame_index[row["obs_id"]]].origin_xy
        x0, y0 = int(np.floor(x - 5)), int(np.floor(y - 5))
        old_rejected = x0 < 0 or y0 < 0 or x0 + 11 > 256 or y0 + 11 > 256
        row["legacy_full_stamp_rejected"] = old_rejected
        row["global_detector_x"] = x + ox
        row["global_detector_y"] = y + oy
        if row["status"] == "FAILED":
            failures.append(row)
        if old_rejected:
            cohort.append(row)
    counts = {obs_id: sum(row["obs_id"] == obs_id for row in cohort) for obs_id in obs_ids}
    expected = dict(zip(obs_ids, (20, 21, 20), strict=True))
    if len(cohort) != 61 or counts != expected:
        raise RuntimeError(f"edge cohort differs from frozen selection: {len(cohort)} {counts}")

    seed = int(protocol.get("seed", 20261015))
    trials: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    methods = ("sieve_independent", "sieve_joint_plane")
    wall_start = time.perf_counter()
    for cohort_index, source in enumerate(cohort):
        index = frame_index[source["obs_id"]]
        frame = frames[index]
        template = frames[(index + 1) % len(frames)]
        position = (source["x_pixel"], source["y_pixel"])
        profile, slices, injection_zone = _cross_exposure_profile(frame, template, *position)
        baseline_scale = fit_flux(frame, np.zeros_like(frame.image), *position)
        if baseline_scale["flux_error_uJy"] is None:
            raise RuntimeError(
                f"recovered position has no current conditional error: {source['source_id']}"
            )
        injected_flux = 10 * float(baseline_scale["flux_error_uJy"])
        injection = np.zeros_like(frame.image)
        injection[slices] = injected_flux * profile
        positions.append(
            {
                "source_id": source["source_id"],
                "obs_id": source["obs_id"],
                "x_pixel": position[0],
                "y_pixel": position[1],
                "global_detector_x": source["global_detector_x"],
                "global_detector_y": source["global_detector_y"],
                "edge_distance_pixels": source["edge_distance_pixels"],
                "psf_edge_information_coverage": source["psf_edge_information_coverage"],
                "psf_information_coverage": source["psf_information_coverage"],
                "fitting_ePSF_zone": source.get("epsf_zone"),
                "injection_ePSF_zone": injection_zone,
                "injected_flux_uJy": injected_flux,
            }
        )
        sigma = np.sqrt(np.where(frame.bad, 0.0, frame.variance))
        for method in methods:
            for replicate in range(replicates):
                draw_seed = seed + index * 1_000_000 + cohort_index * 100 + replicate
                off_rng = np.random.default_rng(draw_seed)
                on_rng = np.random.default_rng(draw_seed + 5_000_000)
                off = frame.image + off_rng.normal(size=frame.image.shape) * sigma
                on = frame.image + on_rng.normal(size=frame.image.shape) * sigma + injection
                off_fit = _fit_method(method, frame, off, [position])
                on_fit = _fit_method(method, frame, on, [position])
                f0, f1 = off_fit["fluxes"][0], on_fit["fluxes"][0]
                e0, e1 = off_fit["errors"][0], on_fit["errors"][0]
                delta = f1 - f0 if f0 is not None and f1 is not None else None
                error = float(np.hypot(e0, e1)) if e0 is not None and e1 is not None else None
                z = (delta - injected_flux) / error if delta is not None and error else None
                trials.append(
                    {
                        "source_id": source["source_id"],
                        "method": method,
                        "replicate": replicate,
                        "seed": draw_seed,
                        "edge_distance_pixels": source["edge_distance_pixels"],
                        "injected_flux_uJy": injected_flux,
                        "measured_increment_uJy": delta,
                        "fractional_error": delta / injected_flux - 1
                        if delta is not None
                        else None,
                        "propagated_error_uJy": error,
                        "standardized_error": z,
                        "status": "MEASURED_CONDITIONAL" if delta is not None else "FAILED",
                        "wall_seconds": off_fit["wall_seconds"] + on_fit["wall_seconds"],
                        "cpu_seconds": off_fit["cpu_seconds"] + on_fit["cpu_seconds"],
                    }
                )

    method_summary: dict[str, Any] = {}
    for method in methods:
        rows = [row for row in trials if row["method"] == method]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["source_id"], []).append(row)
        per_position = {}
        for source_id, source_rows in grouped.items():
            errors = [
                row["fractional_error"]
                for row in source_rows
                if row["fractional_error"] is not None
            ]
            per_position[source_id] = {
                "n": len(errors),
                "median_signed_fractional_error": float(np.median(errors)) if errors else None,
                "median_absolute_fractional_error": float(np.median(np.abs(errors)))
                if errors
                else None,
                "noise_fractional_scatter": float(np.std(errors, ddof=1))
                if len(errors) > 1
                else None,
            }
        errors = [row["fractional_error"] for row in rows if row["fractional_error"] is not None]
        standardized = [
            row["standardized_error"] for row in rows if row["standardized_error"] is not None
        ]
        bins = []
        for low, high in ((0, 1), (1, 3), (3, 5.5)):
            selected = [
                row
                for row in rows
                if low <= row["edge_distance_pixels"] < high and row["fractional_error"] is not None
            ]
            vals = [row["fractional_error"] for row in selected]
            bins.append(
                {
                    "edge_distance_pixels": [low, high],
                    "positions": len({row["source_id"] for row in selected}),
                    "replicates": len(vals),
                    "median_signed_fractional_error": float(np.median(vals)) if vals else None,
                    "median_absolute_fractional_error": float(np.median(np.abs(vals)))
                    if vals
                    else None,
                }
            )
        method_summary[method] = {
            "positions": len(grouped),
            "failed_trials": sum(row["status"] == "FAILED" for row in rows),
            "measurements": len(errors),
            "median_signed_fractional_error": float(np.median(errors)) if errors else None,
            "median_absolute_fractional_error": float(np.median(np.abs(errors)))
            if errors
            else None,
            "rms_fractional_error": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "formal_one_sigma_coverage": float(np.mean(np.abs(standardized) <= 1))
            if standardized
            else None,
            "median_absolute_standardized_error": float(np.median(np.abs(standardized)))
            if standardized
            else None,
            "median_pair_seconds": float(np.median([row["wall_seconds"] for row in rows])),
            "per_position": per_position,
            "edge_distance_bins": bins,
        }
    result = {
        "schema_version": "sieve.qr3.edge-photometry-result.v1",
        "status": "PAIRED_INCREMENT_ACCURACY_ONLY",
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "source_tree_dirty": dirty,
        "environment_id": environment_id,
        "runner_sha256": file_digest(Path(__file__)),
        "protocol_v2_sha256": file_digest(protocol_v2_path),
        "protocol_v3_sha256": file_digest(protocol_path),
        "manifest_sha256": file_digest(manifest_path),
        "input_sha256": [frame.sha256 for frame in frames],
        "cohort_selection": protocol["edge_recovered_cohort"]["selection"],
        "selected_position_count": len(cohort),
        "counts_by_exposure": counts,
        "remaining_failures": [
            {
                "source_id": row["source_id"],
                "obs_id": row["obs_id"],
                "x_pixel": row["x_pixel"],
                "y_pixel": row["y_pixel"],
                "edge_distance_pixels": row["edge_distance_pixels"],
                "psf_information_coverage": row["psf_information_coverage"],
                "reason": row["reason"],
            }
            for row in failures
        ],
        "positions": positions,
        "methods": method_summary,
        "timing": {"wall_seconds": time.perf_counter() - wall_start},
        "limits": protocol["edge_recovered_cohort"]["limits"],
    }
    with (output / "edge-cohort-trials.csv").open("w", newline="") as stream:
        fields = [
            "source_id",
            "method",
            "replicate",
            "edge_distance_pixels",
            "status",
            "injected_flux_uJy",
            "measured_increment_uJy",
            "fractional_error",
            "propagated_error_uJy",
            "standardized_error",
            "wall_seconds",
            "cpu_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{name: row.get(name) for name in fields} for row in trials])
    _write_edge_figure(output / "edge-error-by-distance.svg", method_summary)
    result["artifacts_sha256"] = {
        name: file_digest(output / name)
        for name in ("edge-cohort-trials.csv", "edge-error-by-distance.svg")
    }
    (output / "edge-cohort-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "counts_by_exposure": counts,
                "remaining_failures": result["remaining_failures"],
                "methods": method_summary,
                "wall_seconds": result["timing"]["wall_seconds"],
            },
            indent=2,
            allow_nan=False,
        )
    )


def _write_edge_figure(path: Path, methods: dict[str, Any]) -> None:
    width, height = 820, 500
    left, top, plot_width, plot_height = 90, 55, 680, 320
    bins = ((0, 1), (1, 3), (3, 5.5))
    midpoints = [0.5, 2.0, 4.25]
    colors = {"sieve_independent": "#64748b", "sieve_joint_plane": "#137c8b"}
    values = [
        row["median_absolute_fractional_error"]
        for method in methods.values()
        for row in method["edge_distance_bins"]
        if row["median_absolute_fractional_error"] is not None
    ]
    ymax = max(0.1, max(values, default=0.1) * 1.15)
    elements = []
    for tick in range(6):
        value = ymax * tick / 5
        y = top + plot_height * (1 - tick / 5)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
                'y2="{:.1f}" stroke="#e5e7eb"/>'.format(y),
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}</text>',
            ]
        )
    for edge_bin, midpoint in zip(bins, midpoints, strict=True):
        x = left + plot_width * midpoint / 5.5
        elements.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle">'
            f"{edge_bin[0]}–{edge_bin[1]}</text>"
        )
    for method, color in colors.items():
        points = []
        for midpoint, row in zip(midpoints, methods[method]["edge_distance_bins"], strict=True):
            value = row["median_absolute_fractional_error"]
            if value is not None:
                x = left + plot_width * midpoint / 5.5
                y = top + plot_height * (1 - value / ymax)
                points.append((x, y, value))
        points_text = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        elements.append(
            f'<polyline points="{points_text}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for x, y, value in points:
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
                f"<title>{method}: {value:.4f}</title>"
            )
    legend = []
    for index, (method, color) in enumerate(colors.items()):
        x = left + index * 260
        y = height - 50
        legend.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" '
                f'stroke="{color}" stroke-width="4"/>',
                f'<text x="{x + 35}" y="{y + 4}">{method.replace("_", " ")}</text>',
            ]
        )
    figure = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="17">
Recovered QR3 edge cohort: injected-increment error by cutout edge distance</text>
<g font-family="sans-serif" font-size="12" fill="#111827">{"".join(elements)}{"".join(legend)}
<text x="{left + plot_width / 2}" y="{height - 8}" text-anchor="middle">
Cutout edge distance (native pixels)</text>
<text transform="translate(20 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">
Median absolute fractional increment error</text></g></svg>'''
    path.write_text(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=32)
    args = parser.parse_args()
    run(args.output, args.replicates)


if __name__ == "__main__":
    main()
