"""Compare shallow/deep resource budgets on the three pinned real QR3 cutouts.

The deeper stage only fits held-out one/multiple spatial-profile diagnostics for a
bounded subset of proximity proposals. These are experimental model diagnostics,
not identity evidence. The existing injection evaluator supplies matched DAO/IRAF
recovery controls on the same real image; it does not establish sky performance.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.units import arcsec

from evaluation.photometry_recovery import run_recovery
from sieve.execution import ResourceBudget, WorkUnit, run_work
from sieve.photometry.profiles import compare_profiles
from sieve.photometry.qualified import detect_positions, file_digest, fit_flux, load_qr3_frame
from sieve.provenance.runtime import source_identity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples" / "qualification"


def _worker(unit: WorkUnit) -> dict[str, Any]:
    payload = unit.payload
    frame = load_qr3_frame(ROOT / payload["file"], payload["sha256"])
    if unit.investigation == "detect_and_measure":
        positions = detect_positions(frame, frame.image, "DAO")
        rows = []
        for x, y in positions:
            fit = fit_flux(frame, frame.image, float(x), float(y))
            rows.append(
                {
                    "x_pixel": float(x),
                    "y_pixel": float(y),
                    "flux_uJy": fit["flux_uJy"],
                    "flux_error_uJy": fit["flux_error_uJy"],
                    "status": fit["status"],
                    "reason": fit.get("reason"),
                }
            )
        return {
            "obs_id": unit.exposure_id,
            "wavelength_median_um": float(np.nanmedian(frame.cwave)),
            "detections": rows,
            "failed_measurements": sum(row["status"] == "FAILED" for row in rows),
        }
    if unit.investigation == "heldout_profile_diagnostic":
        return compare_profiles(frame, [tuple(point) for point in payload["sky_positions"]])
    raise ValueError(f"unknown science investigation: {unit.investigation}")


def _unit(
    *,
    unit_id: str,
    exposure: dict[str, Any],
    investigation: str,
    wavelength: str,
    value: float,
    exploration: bool,
    source_hashes: tuple[str, ...],
    payload: dict[str, Any],
    estimated_cpu: float,
) -> WorkUnit:
    relative_file = exposure["file"]
    return WorkUnit(
        unit_id=unit_id,
        sky_region="qr3-d1-pinned-cutout",
        exposure_id=exposure["obs_id"],
        wavelength=wavelength,
        investigation=investigation,
        scientific_value=value,
        exploration=exploration,
        input_sha256=source_hashes + (exposure["sha256"],),
        estimated_input_bytes=(DATA / relative_file).stat().st_size,
        estimated_cpu_seconds=estimated_cpu,
        estimated_rss_bytes=512 * 1024**2,
        estimated_output_bytes=(1024**2 if investigation == "detect_and_measure" else 64**2),
        payload={
            "file": f"examples/qualification/{relative_file}",
            "sha256": exposure["sha256"],
            **payload,
        },
    )


def _candidate_units(
    detections: list[dict[str, Any]], manifest: dict[str, Any], hashes: tuple[str, ...], limit: int
) -> list[WorkUnit]:
    by_obs = {exp["obs_id"]: exp for exp in manifest["exposures"]}
    first, second = detections[:2]
    coordinates = []
    for output in (first, second):
        frame = load_qr3_frame(
            DATA / by_obs[output["obs_id"]]["file"], by_obs[output["obs_id"]]["sha256"]
        )
        rows = output["detections"]
        sky = [frame.wcs.pixel_to_world_values(row["x_pixel"], row["y_pixel"]) for row in rows]
        coordinates.append(SkyCoord([v[0] for v in sky], [v[1] for v in sky], unit="deg"))
    i_values, j_values, separation, _ = search_around_sky(
        coordinates[0], coordinates[1], 12 * arcsec
    )
    proposed = []
    left_rows, right_rows = first["detections"], second["detections"]
    heldout = manifest["exposures"][2]
    heldout_frame = load_qr3_frame(DATA / heldout["file"], heldout["sha256"])
    wavelength_edges_um = (0.75, 1.3, 2.0, 3.0, 5.0)
    for i, j, sep in zip(i_values, j_values, separation.arcsec, strict=True):
        i, j = int(i), int(j)
        points = [
            (float(coordinates[0][i].ra.deg), float(coordinates[0][i].dec.deg)),
            (float(coordinates[1][j].ra.deg), float(coordinates[1][j].dec.deg)),
        ]
        midpoint = coordinates[0][i].directional_offset_by(
            coordinates[0][i].position_angle(coordinates[1][j]), float(sep) * arcsec / 2
        )
        mid_x, mid_y = heldout_frame.wcs.world_to_pixel(midpoint)
        mid_x, mid_y = float(np.asarray(mid_x).item()), float(np.asarray(mid_y).item())
        px = min(max(int(round(mid_x)), 0), heldout_frame.image.shape[1] - 1)
        py = min(max(int(round(mid_y)), 0), heldout_frame.image.shape[0] - 1)
        midpoint_wavelength_um = float(heldout_frame.cwave[py, px])
        wavelength_bin = int(
            np.searchsorted(wavelength_edges_um, midpoint_wavelength_um, side="right") - 1
        )
        wavelength_bin = min(max(wavelength_bin, 0), len(wavelength_edges_um) - 2)
        wavelength_partition = (
            f"{wavelength_edges_um[wavelength_bin]:.2f}-"
            f"{wavelength_edges_um[wavelength_bin + 1]:.2f}um"
        )
        left, right = left_rows[i], right_rows[j]
        errors = (left["flux_error_uJy"], right["flux_error_uJy"])
        fluxes = (left["flux_uJy"], right["flux_uJy"])
        snrs = [abs(f / e) for f, e in zip(fluxes, errors, strict=True) if f is not None and e]
        min_snr = min(snrs) if len(snrs) == 2 else 0.0
        # Ranking only allocates compute. It does not estimate identity probability.
        score = min_snr + 1.0 / (1.0 + float(sep))
        proposed.append(
            (score, float(sep), i, j, points, min_snr, wavelength_partition, midpoint_wavelength_um)
        )
    proposed.sort(key=lambda row: (-row[0], row[2], row[3]))
    chosen = proposed[:limit]
    units = []
    for rank, (
        score,
        sep,
        i,
        j,
        points,
        min_snr,
        wavelength_partition,
        midpoint_wavelength_um,
    ) in enumerate(chosen):
        units.append(
            _unit(
                unit_id=f"profile-{i:05d}-{j:05d}",
                exposure=heldout,
                investigation="heldout_profile_diagnostic",
                wavelength=wavelength_partition,
                value=score,
                exploration=(rank % 5 == 4 or min_snr < 8),
                source_hashes=hashes,
                payload={
                    "sky_positions": points,
                    "proposal_separation_arcsec": sep,
                    "midpoint_wavelength_um": midpoint_wavelength_um,
                },
                estimated_cpu=1.0,
            )
        )
    return units


def _recovery_metrics(recovery: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for experiment in recovery["experiments"]:
        measurements = experiment["measurement"]
        fitted = [row for row in measurements if row["fit"]["flux_uJy"] is not None]
        fractional = [row["fractional_error"] for row in fitted]
        standardized = [
            (row["fit"]["flux_uJy"] - row["injected_flux_uJy"]) / row["fit"]["flux_error_uJy"]
            for row in fitted
            if row["fit"]["flux_error_uJy"]
        ]
        paired_fractional = [
            row["paired_fractional_error"]
            for row in measurements
            if row["paired_fractional_error"] is not None
        ]
        paired_standardized = [
            (row["paired_delta_uJy"] - row["injected_flux_uJy"]) / row["fit"]["flux_error_uJy"]
            for row in measurements
            if row["paired_delta_uJy"] is not None and row["fit"]["flux_error_uJy"]
        ]
        baseline_flux_fraction = [
            row["baseline_fit"]["flux_uJy"] / row["injected_flux_uJy"]
            for row in measurements
            if row["baseline_fit"]["flux_uJy"] is not None
        ]
        positions = [
            match["distance_pixels"]
            for output in experiment["detection"]
            for match in output["associations"]
            if match["recovered"]
        ]
        rows.append(
            {
                "arm": experiment["arm"],
                "nominal_snr": experiment["nominal_snr"],
                "flux_fit_count": len(fitted),
                "flux_fit_failures": len(measurements) - len(fitted),
                "median_fractional_flux_error": float(np.median(fractional))
                if fractional
                else None,
                "median_absolute_fractional_flux_error": (
                    float(np.median(np.abs(fractional))) if fractional else None
                ),
                "standardized_flux_residual_rms": (
                    float(np.sqrt(np.mean(np.square(standardized)))) if standardized else None
                ),
                "paired_injection_fractional_error_median_absolute": (
                    float(np.median(np.abs(paired_fractional))) if paired_fractional else None
                ),
                "paired_injection_standardized_residual_rms": (
                    float(np.sqrt(np.mean(np.square(paired_standardized))))
                    if paired_standardized
                    else None
                ),
                "baseline_scene_flux_fraction_median_absolute": (
                    float(np.median(np.abs(baseline_flux_fraction)))
                    if baseline_flux_fraction
                    else None
                ),
                "paired_measurement_interpretation": (
                    "Paired difference checks numerical response to the injected template; "
                    "it cancels scene flux and is not source-truth photometric accuracy."
                ),
                "formal_one_sigma_coverage": (
                    float(np.mean(np.abs(standardized) <= 1)) if standardized else None
                ),
                "median_recovered_position_error_pixels": (
                    float(np.median(positions)) if positions else None
                ),
                "position_match_count": len(positions),
                "limits": [
                    "Injected and fit profiles share the released finite ePSF.",
                    "Pixel noise is independent archive variance; source shot noise and "
                    "covariance are absent.",
                    "Association correctness is only known for these injected sites and "
                    "the one-to-one matching radius.",
                ],
            }
        )
    return {
        "experiments": rows,
        "status": recovery["status"],
        "limitations": recovery["limitations"],
    }


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in run["completed"]:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {
        "status": run["status"],
        "stop_reason": run["stop_reason"],
        "unit_order": run["unit_order"],
        "completed_unit_ids": [row["unit_id"] for row in run["completed"]],
        "completed_status_counts": statuses,
        "reused_checkpoints": run["reused_checkpoints"],
        "unstarted": run["unstarted"],
        "resources": run["resources"],
    }


def _run_arm(
    label: str,
    manifest: dict[str, Any],
    hashes: tuple[str, ...],
    output: Path,
    *,
    profile_limit: int,
    workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    detection_units = [
        _unit(
            unit_id=f"detect-{index}-{exposure['obs_id']}",
            exposure=exposure,
            investigation="detect_and_measure",
            wavelength="all-calibrated-cwave-pixels",
            value=1.0,
            exploration=(index == 2),
            source_hashes=hashes,
            payload={},
            estimated_cpu=30.0,
        )
        for index, exposure in enumerate(manifest["exposures"])
    ]
    detection_run = run_work(
        detection_units,
        _worker,
        checkpoint_dir=output / label / "checkpoints",
        budget=ResourceBudget(
            max_tasks=3,
            max_workers=workers,
            max_cpu_seconds=300,
            max_wall_seconds=600,
            max_rss_bytes=6 * 1024**3,
            max_input_bytes=256 * 1024**2,
            max_output_bytes=64 * 1024**2,
            exploration_fraction=1 / 3,
        ),
    )
    detect_rows = [
        row["result"] for row in detection_run["completed"] if row["status"] == "COMPLETED"
    ]
    if len(detect_rows) != 3:
        return {"label": label, "detection": detection_run, "scientific_status": "INCOMPLETE"}
    profile_run = None
    if profile_limit:
        profile_units = _candidate_units(detect_rows, manifest, hashes, profile_limit)
        profile_run = run_work(
            profile_units,
            _worker,
            checkpoint_dir=output / label / "checkpoints",
            budget=ResourceBudget(
                max_tasks=profile_limit,
                max_workers=workers,
                max_cpu_seconds=300,
                max_wall_seconds=600,
                max_rss_bytes=6 * 1024**3,
                max_input_bytes=256 * 1024**2,
                max_output_bytes=64 * 1024**2,
                exploration_fraction=0.2,
            ),
        )
    all_runs = [detection_run, *([profile_run] if profile_run else [])]
    profiles = [
        row["result"]
        for row in (profile_run or {}).get("completed", [])
        if row["status"] == "COMPLETED"
    ]
    wavelength_partitions = (
        {
            wavelength: sum(unit.wavelength == wavelength for unit in profile_units)
            for wavelength in sorted({unit.wavelength for unit in profile_units})
        }
        if profile_limit
        else {}
    )
    measured = [row for result in detect_rows for row in result["detections"]]
    resource_runs = [run["resources"] for run in all_runs]
    resource_totals = {
        "worker_cpu_seconds": sum(run["worker_cpu_seconds"] for run in resource_runs),
        "scheduler_process_cpu_seconds": sum(run["process_cpu_seconds"] for run in resource_runs),
        "scheduler_wall_seconds": sum(run["wall_seconds"] for run in resource_runs),
        "peak_process_rss_bytes": max(run["process_peak_rss_bytes"] for run in resource_runs),
        "logical_input_bytes_processed": sum(run["admitted_input_bytes"] for run in resource_runs),
        "checkpoint_bytes_written": sum(run["checkpoint_bytes"] for run in resource_runs),
        "resource_measurement_limits": [
            "RSS is the Python process high-water mark, not per-task incremental memory.",
            "Input bytes count pinned file sizes processed, not physical device reads; "
            "OS cache is uncontrolled.",
            "Checkpoint bytes are actual serialized bytes written; no other scratch "
            "outputs are generated.",
        ],
    }
    return {
        "label": label,
        "scientific_status": "INCONCLUSIVE_DIAGNOSTIC_ONLY",
        "detection_tasks": 3,
        "profile_tasks_requested": profile_limit,
        "work_unit_throughput_per_second": (3 + len(profiles)) / (time.perf_counter() - started),
        "resource_totals": resource_totals,
        "profiles_measured": len(profiles),
        "profile_wavelength_partitions": wavelength_partitions,
        "detections": len(measured),
        "measurement_failures": sum(row["failed_measurements"] for row in detect_rows),
        "photometry_status_counts": {
            status: sum(row["status"] == status for row in measured)
            for status in ("MEASURED_CONDITIONAL", "FAILED")
        },
        "median_profile_delta_bic_two_minus_one": float(
            np.median(
                [
                    p["delta_bic_two_minus_one"]
                    for p in profiles
                    if p.get("delta_bic_two_minus_one") is not None
                ]
            )
        )
        if any(p.get("delta_bic_two_minus_one") is not None for p in profiles)
        else None,
        "profile_status_counts": {
            status: sum(p["status"] == status for p in profiles)
            for status in ("INCONCLUSIVE", "SUPPORTS_MULTIPLE", "SUPPORTS_SINGLE")
        },
        "resource_runs": [_compact_run(run) for run in all_runs],
        "wall_seconds_including_setup": time.perf_counter() - started,
    }


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = DATA / "source.json"
    protocol_path = DATA / "protocol.json"
    manifest = json.loads(manifest_path.read_text())
    code_hashes = tuple(
        file_digest(path)
        for path in (
            manifest_path,
            protocol_path,
            ROOT / "src/sieve/photometry/qualified.py",
            ROOT / "src/sieve/photometry/profiles.py",
            ROOT / "src/sieve/execution/scheduler.py",
            ROOT / "evaluation/photometry_recovery.py",
            Path(__file__),
        )
    )
    code_commit, source_tree_sha256, dirty_source, environment_id = source_identity()
    recovery_start = time.perf_counter()
    recovery_cpu_start = time.process_time()
    frame_info = manifest["exposures"][0]
    frame = load_qr3_frame(DATA / frame_info["file"], frame_info["sha256"])
    recovery = run_recovery(frame)
    recovery_summary = [
        {
            "arm": experiment["arm"],
            "nominal_snr": experiment["nominal_snr"],
            "methods": [
                {
                    "name": method["method"],
                    "recovered": method["recovered"],
                    "denominator": method["denominator"],
                    "false_detections": method["false_detections"],
                    "artifact_peak_recoveries": method["artifact_peak_recoveries"],
                }
                for method in experiment["detection"]
            ],
        }
        for experiment in recovery["experiments"]
    ]
    recovery_seconds = time.perf_counter() - recovery_start
    recovery_cpu_seconds = time.process_time() - recovery_cpu_start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if os.uname().sysname == "Darwin" else peak_rss * 1024)
    shallow = _run_arm("shallow", manifest, code_hashes, output, profile_limit=0, workers=1)
    deep_one = _run_arm(
        "deep-one-worker", manifest, code_hashes, output, profile_limit=12, workers=1
    )
    deep_two = _run_arm(
        "deep-two-workers", manifest, code_hashes, output, profile_limit=12, workers=2
    )
    deep_speedup = (
        deep_one["wall_seconds_including_setup"] / deep_two["wall_seconds_including_setup"]
        if deep_two["wall_seconds_including_setup"]
        else None
    )
    result = {
        "schema_version": "sieve.resource-adaptive-qr3.result.v1",
        "status": "LOCAL_MEASURED_SCIENTIFIC_RESULT_INCONCLUSIVE",
        "code_commit": code_commit,
        "source_tree_sha256": source_tree_sha256,
        "dirty_source": dirty_source,
        "environment_id": environment_id,
        "pinned_code_sha256": {
            "manifest": code_hashes[0],
            "protocol": code_hashes[1],
            "qr3_photometry": code_hashes[2],
            "heldout_profiles": code_hashes[3],
            "scheduler": code_hashes[4],
            "injection_evaluator": code_hashes[5],
            "runner": code_hashes[6],
        },
        "source_manifest_sha256": file_digest(manifest_path),
        "protocol_sha256": file_digest(protocol_path),
        "source_data_sha256": [exposure["sha256"] for exposure in manifest["exposures"]],
        "baseline": {
            "name": (
                "Photutils DAOStarFinder plus fixed-position QR3 finite-ePSF weighted least squares"
            ),
            "detector_independent_reference_identities": "NOT_AVAILABLE",
            "injection_controls": recovery_summary,
            "photometry_and_position_metrics": _recovery_metrics(recovery),
            "injection_control_limit": (
                "Model-matched injections on one real image; not independent sky "
                "sensitivity or identity truth."
            ),
            "runtime_seconds": recovery_seconds,
            "process_cpu_seconds": recovery_cpu_seconds,
            "process_peak_rss_bytes": peak_rss_bytes,
            "input_cutout_bytes": (DATA / frame_info["file"]).stat().st_size,
        },
        "budgets": {"shallow": shallow, "deep_one_worker": deep_one, "deep_two_workers": deep_two},
        "measured_scaling": {
            "same_12_profile_units_worker_speedup": deep_speedup,
            "efficiency_at_two_workers": deep_speedup / 2 if deep_speedup is not None else None,
            "throughput_tasks_per_second": {
                "shallow": 3 / shallow["wall_seconds_including_setup"],
                "deep_one_worker": 15 / deep_one["wall_seconds_including_setup"],
                "deep_two_workers": 15 / deep_two["wall_seconds_including_setup"],
            },
            "limit": (
                "Single local host, one sequential pair of runs, filesystem cache "
                "warmed by first run; descriptive only."
            ),
        },
        "deeper_analysis_limit": (
            "Held-out one/multiple-profile delta-BIC is conditional diagnostic only; "
            "it cannot confirm source identity or discovery."
        ),
        "scientific_claim": (
            "All runs reuse the same three real QR3 D1 exposures. Deeper budget adds "
            "held-out profile diagnostics, not validated objects. Injection recovery "
            "reports conditional controls only."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory; must not exist"
    )
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "recovery_arms": len(result["baseline"]["injection_controls"]),
                "shallow_detections": result["budgets"]["shallow"]["detections"],
                "deep_profiles": result["budgets"]["deep_two_workers"]["profiles_measured"],
                "one_vs_two_worker_speedup": result["measured_scaling"][
                    "same_12_profile_units_worker_speedup"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
