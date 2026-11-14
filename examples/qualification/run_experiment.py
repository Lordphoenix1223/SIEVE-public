"""Offline real-QR3 qualification pilot. Refuses overwrite and qualified identity claims."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.units import arcsec

from evaluation.photometry_recovery import run_recovery
from sieve.identity.graph import IdentityBudget
from sieve.identity.workflow import run_identity
from sieve.photometry.profiles import compare_profiles
from sieve.photometry.provider import QR3ObservationProvider, build_competing_input
from sieve.photometry.qualified import detect_positions, file_digest, fit_flux, load_qr3_frame
from sieve.provenance.runtime import source_identity
from sieve.registration import associate, compare, load_observation


def write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def summarize_recovery(recovery: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    site_ids = [s["site_id"] for s in recovery["sites"]]
    bootstrap = np.random.default_rng(20261012).integers(0, len(site_ids), (2000, len(site_ids)))
    for experiment in recovery["experiments"]:
        fitted = [m for m in experiment["measurement"] if m["fit"]["flux_uJy"] is not None]
        errors = [m["fractional_error"] for m in fitted]
        deltas = [m["paired_fractional_error"] for m in fitted]
        z = [
            (m["fit"]["flux_uJy"] - m["injected_flux_uJy"]) / m["fit"]["flux_error_uJy"]
            for m in fitted
        ]
        site_recall = {}
        for detection in experiment["detection"]:
            grouped: dict[str, list[bool]] = {site: [] for site in site_ids}
            for target, association in zip(
                experiment["truth"], detection["associations"], strict=True
            ):
                grouped[target["site_id"]].append(association["recovered"])
            site_recall[detection["method"]] = np.array([np.mean(grouped[s]) for s in site_ids])
        paired = site_recall["DAO"] - site_recall["IRAF"]
        paired_interval = np.quantile(paired[bootstrap].mean(axis=1), [0.025, 0.975]).tolist()
        for detection in experiment["detection"]:
            n, k = detection["denominator"], detection["recovered"]
            p = k / n
            interval = np.quantile(
                site_recall[detection["method"]][bootstrap].mean(axis=1), [0.025, 0.975]
            ).tolist()
            summary.append(
                {
                    "arm": experiment["arm"],
                    "nominal_snr": experiment["nominal_snr"],
                    "method": detection["method"],
                    "recovered": k,
                    "denominator": n,
                    "recovery_fraction": p,
                    "site_bootstrap_95_conditional": interval,
                    "paired_DAO_minus_IRAF_95_conditional": paired_interval,
                    "interval_limit": (
                        "Post-pilot site bootstrap keeps blend components/arms together. "
                        "Conditional on this field/noise/background; not sky-performance coverage."
                    ),
                    "false_detections": detection["false_detections"],
                    "artifact_peak_recoveries": detection["artifact_peak_recoveries"],
                    "measurement_failures": len(experiment["measurement"]) - len(fitted),
                    "median_fractional_error": float(np.median(errors)) if errors else None,
                    "fractional_scatter_16_84": np.quantile(errors, [0.16, 0.84]).tolist()
                    if errors
                    else None,
                    "median_paired_fractional_error": float(np.median(deltas)) if deltas else None,
                    "standardized_residual_rms": float(np.sqrt(np.mean(np.square(z))))
                    if z
                    else None,
                    "formal_one_sigma_coverage": float(np.mean(np.abs(z) <= 1)) if z else None,
                }
            )
    return summary


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    base = Path(__file__).resolve().parent
    manifest_path, protocol_path = base / "source.json", base / "protocol.json"
    manifest, protocol = (
        json.loads(manifest_path.read_text()),
        json.loads(protocol_path.read_text()),
    )
    # Changing a prose protocol without changing the executable cannot create a new experiment.
    if (
        protocol["injection"]["seed"] != 20261011
        or protocol["injection"]["sites"] != 32
        or protocol["injection"]["snr_levels"] != [3, 5, 10, 20]
        or protocol["detection"]["threshold_sigma"] != 5
        or protocol["detection"]["fwhm_pixels"] != 2
        or protocol["detection"]["recovery_radius_pixels"] != 2
    ):
        raise ValueError(
            "unsupported protocol configuration; revise implementation and freeze a new pilot"
        )
    started, tic = datetime.now(UTC), time.perf_counter()
    commit, tree, dirty, environment = source_identity()
    frames = [load_qr3_frame(base / e["file"], e["sha256"]) for e in manifest["exposures"]]
    if len(frames) != 3 or len({f.header["OBSID"] for f in frames}) != 3:
        raise ValueError("three different real exposures required")
    # Guard the existing pilot before it generates hypotheses. A numeric WCS
    # round trip is not astrometric qualification, so science claims stay closed.
    native = [load_observation(base / e["file"]) for e in manifest["exposures"][:2]]
    registration_gate = compare(native[0], native[1])
    if registration_gate["spatial_status"] == "NO_SAMPLED_COMMON_COVERAGE" or registration_gate[
        "temporal_status"
    ] in {"DUPLICATE_EXPOSURE", "POSSIBLE_DUPLICATE_EXPOSURE"}:
        raise ValueError("registration gate refused overlap or exposure identity")
    exposures, generation = [], []
    for frame in frames[:2]:
        positions = detect_positions(frame, frame.image, "DAO")
        rows = []
        for i, (x, y) in enumerate(positions):
            row = fit_flux(frame, frame.image, float(x), float(y))
            row.update(
                measurement_id=f"measurement:{frame.header['OBSID']}:{i:05d}",
                raw_source_id=f"peak:{i:05d}",
            )
            rows.append(row)
        generation.append(positions)
        exposures.append(
            {
                "obs_id": frame.header["OBSID"],
                "source_sha256": frame.sha256,
                "mjd_avg_utc": frame.header["MJD-AVG"],
                "detector": 1,
                "detected_peaks": len(positions),
                "measurements": rows,
            }
        )
    measurement_report = {
        "schema_version": "sieve.qr3.measurements.v1",
        "manifest_sha256": file_digest(manifest_path),
        "protocol_sha256": file_digest(protocol_path),
        "code_commit": commit,
        "source_tree_sha256": tree,
        "runner_sha256": file_digest(Path(__file__)),
        "recovery_evaluator_sha256": file_digest(
            Path(__file__).resolve().parents[2] / "evaluation" / "photometry_recovery.py"
        ),
        "dirty_source": dirty,
        "environment_id": environment,
        "started_at": started.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "exposures": exposures,
        "error_status": "FORMAL_CONDITIONAL_ONLY_NOT_CALIBRATED",
    }
    write(output / "measurements.json", measurement_report)
    gated_pairs = associate(native[0], native[1], generation[0], generation[1], 12.0)
    registration_gate = compare(native[0], native[1], gated_pairs)
    registration_gate["motion_claim"] = "REFUSED"
    registration_gate["flux_variability_claim"] = "REFUSED"
    write(output / "registration-gate.json", registration_gate)
    # All radius candidates are retained, not a nearest-neighbor identity benchmark.
    coordinates = [
        SkyCoord(*frame.wcs.pixel_to_world_values(p[:, 0], p[:, 1]), unit="deg")
        for frame, p in zip(frames[:2], generation, strict=True)
    ]
    first, second, separations, _ = search_around_sky(*coordinates, 12 * arcsec)
    profiles = []
    for i, j, separation in sorted(zip(first, second, separations.arcsec, strict=True)):
        points = [
            (float(c.ra.deg), float(c.dec.deg)) for c in (coordinates[0][i], coordinates[1][j])
        ]
        result = compare_profiles(frames[2], points)
        result.update(
            case_id=f"proposal:{int(i):05d}:{int(j):05d}",
            separation_arcsec=float(separation),
            member_ids=[
                exposures[0]["measurements"][i]["measurement_id"],
                exposures[1]["measurements"][j]["measurement_id"],
            ],
        )
        profiles.append(result)
    write(
        output / "profiles.json",
        {
            "schema_version": "sieve.qr3.profiles.v1",
            "protocol_sha256": file_digest(protocol_path),
            "profiles": profiles,
            "generation_obs_ids": [f.header["OBSID"] for f in frames[:2]],
            "heldout_obs_id": frames[2].header["OBSID"],
            "shared_calibration_status": "UNKNOWN_COVARIANCE_NOT_DISJOINT_INPUTS",
        },
    )
    recovery = run_recovery(frames[0])
    write(output / "recovery.json", recovery)
    reference = json.loads((base / "irsa-reference.json").read_text())
    comparison = []
    for official, frame in zip(reference["rows"], frames[:2], strict=True):
        x, y = frame.wcs.world_to_pixel_values(
            reference["request"]["ra_deg"], reference["request"]["dec_deg"]
        )
        fit = fit_flux(frame, frame.image, float(x), float(y))
        fraction = (
            fit["flux_uJy"] / official["flux_uJy"] - 1 if fit["flux_uJy"] is not None else None
        )
        comparison.append(
            {
                "obs_id": official["obs_id"],
                "official": official,
                "sieve": fit,
                "fractional_flux_difference": fraction,
                "irsa_total_over_sieve_formal_error": official["flux_err_uJy"]
                / fit["flux_error_uJy"]
                if fit["flux_error_uJy"] is not None
                else None,
                "implied_irsa_extra_error_fraction": float(
                    np.sqrt(
                        max(
                            0, official["flux_err_uJy"] ** 2 - official["flux_err_tractor_uJy"] ** 2
                        )
                    )
                    / official["flux_uJy"]
                ),
                "independence_limit": "Shared image/ePSF calibration; not absolute validation",
                "population_limit": "One bright source; second exposure has an IRSA bad-pixel flag",
            }
        )
    write(
        output / "reference-comparison.json",
        {"source_sha256": file_digest(base / "irsa-reference.json"), "comparison": comparison},
    )
    provider = QR3ObservationProvider(manifest_path, output / "measurements.json")
    batch = provider.load({})
    write(output / "observation-batch.json", batch.model_dump(mode="json"))
    bundle = build_competing_input(
        batch, output / "profiles.json", protocol_path, measurement_report
    )
    write(output / "identity-input.json", bundle.model_dump(mode="json"))
    run_identity(output / "identity-input.json", output / "identity", IdentityBudget())
    summary = {
        "schema_version": "sieve.qr3.qualification.result.v1",
        "scientific_result": "INCONCLUSIVE",
        "code_commit": commit,
        "source_tree_sha256": tree,
        "dirty_source": dirty,
        "environment_id": environment,
        "manifest_sha256": file_digest(manifest_path),
        "protocol_sha256": file_digest(protocol_path),
        "runtime_seconds": time.perf_counter() - tic,
        "real_exposures": len(frames),
        "generation_peaks": [e["detected_peaks"] for e in exposures],
        "real_measurement_failures": [
            sum(r["status"] == "FAILED" for r in e["measurements"]) for e in exposures
        ],
        "real_proximity_pairs": len(profiles),
        "conditional_hypotheses": len(bundle.hypotheses),
        "heldout_profile_failures": sum(not p["models"] for p in profiles),
        "irsa_flux_comparison": comparison,
        "recovery_summary": summarize_recovery(recovery),
        "strong_baselines": {
            "Nway": "NOT_RUN: qualified positional covariance/reference cohort absent",
            "IRSA_SExtractor": "NOT_RUN: SExtractor executable absent",
            "IRSA_forced_photometry": "Two rows, one source; text transcription, export failed",
        },
        "identity_arms": {
            arm: {metric: None for metric in protocol["identity"]["metrics"]}
            for arm in ("A", "B", "C")
        },
        "identity_metric_reason": (
            "Independent identities, source-disjoint labels, qualified covariance and "
            "matched-recall thresholds absent. Proximity/profile diagnostics are not A/B/C scores."
        ),
        "chosen_factor": protocol["identity"]["chosen_factor"],
        "moving_source_control": "NOT_RUN: no moving-source truth in this fixed-position pilot",
        "real_blank_sky_control": "NOT_RUN: Gaussian variance null is not real blank sky",
        "claim": (
            "Finite-profile flux scale agrees for one real source. "
            "Errors and identity improvement remain unqualified."
        ),
    }
    write(output / "summary.json", summary)
    write(
        output / "SHA256SUMS.json",
        {
            str(p.relative_to(output)): file_digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "scientific_result",
                    "real_exposures",
                    "generation_peaks",
                    "real_measurement_failures",
                    "real_proximity_pairs",
                    "conditional_hypotheses",
                    "heldout_profile_failures",
                    "runtime_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
