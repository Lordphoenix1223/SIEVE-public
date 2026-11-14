"""Experimental QR3 comparison entry point with fail-closed scientific claims."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from sieve.photometry.qualified import detect_positions, fit_flux, load_qr3_frame
from sieve.provenance.runtime import source_identity
from sieve.registration.core import associate, compare, load_observation, reference_residuals


def run_qr3_pair(first: Path, second: Path, output: Path) -> dict[str, Any]:
    """Extract native peaks and write candidate-only comparison; never assert change."""
    started = time.perf_counter()
    if output.exists():
        raise ValueError("output already exists")
    observations = [load_observation(first), load_observation(second)]
    geometry = compare(observations[0], observations[1])
    blocked = geometry["spatial_status"] == "NO_SAMPLED_COMMON_COVERAGE" or geometry[
        "temporal_status"
    ] in {"DUPLICATE_EXPOSURE", "POSSIBLE_DUPLICATE_EXPOSURE"}
    positions: list[np.ndarray[Any, Any]] = []
    measurements: list[list[dict[str, Any]]] = []
    pairs: dict[str, Any] = {"candidate_pairs": [], "unambiguous_pairs": [], "ambiguous_pairs": []}
    reference: dict[str, Any] | None = None
    if not blocked:
        for observation in observations:
            frame = load_qr3_frame(observation.path, observation.sha256)
            peaks = detect_positions(frame, frame.image, "DAO")
            positions.append(peaks)
            measurements.append(
                [fit_flux(frame, frame.image, float(x), float(y)) for x, y in peaks]
            )
        pairs = associate(
            observations[0], observations[1], positions[0], positions[1], radius_arcsec=12.0
        )
        geometry = compare(observations[0], observations[1], pairs)
        unique = sorted(pairs["unambiguous_pairs"], key=lambda p: (p["a"], p["b"]))
        if len(unique) >= 6:
            reference = reference_residuals(
                observations[0],
                observations[1],
                np.asarray([positions[0][p["a"]] for p in unique]),
                np.asarray([positions[1][p["b"]] for p in unique]),
            )
    commit, tree, dirty, env = source_identity()
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        peak_rss = None
    result = {
        "schema_version": "sieve.qr3.comparison.v1",
        "mode": "EXPERIMENTAL_CANDIDATES_ONLY",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_native_units": peak_rss,
        "peak_rss_units": "bytes"
        if sys.platform == "darwin"
        else "KiB"
        if sys.platform.startswith("linux")
        else "unknown",
        "source_association": "REFUSED" if blocked else "CANDIDATES_ONLY",
        "motion_claim": "REFUSED",
        "flux_variability_claim": "REFUSED",
        "reasons": [
            "Unqualified astrometric covariance and reference identities; no motion conclusion.",
            "PSF, response, calibration and covariance not qualified; no variability conclusion.",
            "No sampled overlap or duplicate exposures refuse candidate pairing.",
        ],
        "geometry": geometry,
        "detections_per_exposure": [len(p) for p in positions],
        "pairings": pairs,
        "measurements": measurements,
        "reference_diagnostic": reference,
        "provenance": {
            "code_commit": commit,
            "source_tree_sha256": tree,
            "dirty_source": dirty,
            "environment_id": env,
            "input_sha256": [o.sha256 for o in observations],
            "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result
