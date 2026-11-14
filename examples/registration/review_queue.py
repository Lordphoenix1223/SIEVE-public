"""Build an evidence-completeness queue from a guarded QR3 comparison.

This is a review aid, not a classifier or a motion/variability gate for other APIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sieve.photometry.scanner import _BAD_FLAG_BITS


def build(comparison: dict[str, Any], comparison_sha256: str) -> dict[str, Any]:
    if comparison["schema_version"] != "sieve.qr3.comparison.v1":
        raise ValueError("unsupported comparison schema")
    geometry = comparison["geometry"]
    measurements = comparison["measurements"]
    records = []
    for pair in comparison["pairings"]["candidate_pairs"]:
        a, b = pair["a"], pair["b"]
        ms = [measurements[0][a], measurements[1][b]]
        usable = all(m["status"] == "MEASURED_CONDITIONAL" for m in ms)
        flags = [m.get("flag_bits") for m in ms]
        flags_known = all(isinstance(f, int) for f in flags)
        flagged = flags_known and any(f & _BAD_FLAG_BITS for f in flags)
        ambiguous = bool(pair["ambiguous"])
        reasons = []
        if not usable:
            reasons.append("measurement_not_usable")
        if not flags_known:
            reasons.append("measurement_flags_unavailable")
        elif flagged:
            reasons.append("excluded_pixel_flags_in_fit_window_not_artifact_classification")
        if ambiguous:
            reasons.append("multiple_positional_pairings")
        if not reasons:
            reasons.append("usable_no_excluded_pixel_flags_singleton_review_only")
        # Ordinal evidence triage, not a probability, artifact label or science verdict.
        priority = "deferred" if not usable or not flags_known else (
            "review_flags_or_ambiguity" if flagged or ambiguous else "review_measurements"
        )
        records.append({
            "pair": pair,
            "detection_ids": [f"measurement:{geometry['inputs'][0]['exposure_id']}:{a:05d}",
                              f"measurement:{geometry['inputs'][1]['exposure_id']}:{b:05d}"],
            "measurements": ms,
            "scientific_eligibility": {
                "motion": "INSUFFICIENT_EVIDENCE",
                "flux_variability": "INSUFFICIENT_EVIDENCE",
                "reasons": [geometry["source_motion_status"], geometry["flux_variability_status"]],
            },
            "review_priority": priority,
            "reasons": reasons,
            "catalog_association": "NOT_EVALUATED_FOR_FIELD",
            "independent_check": "NOT_RUN",
        })
    counts = Counter(r["review_priority"] for r in records)
    return {
        "schema_version": "sieve.qr3.review_queue.v1",
        "comparison_sha256": comparison_sha256,
        "input_sha256": comparison["provenance"]["input_sha256"],
        "counts": {
            "raw_peaks": comparison["detections_per_exposure"],
            "usable_measurements": [sum(m["status"] == "MEASURED_CONDITIONAL" for m in rows)
                                    for rows in measurements],
            "candidate_pairs": len(records),
            "usable_pairs": sum(all(m["status"] == "MEASURED_CONDITIONAL"
                                    for m in r["measurements"]) for r in records),
            "ambiguous_pairs": sum(bool(r["pair"]["ambiguous"]) for r in records),
            "catalog_associated_pairs": None,
            "science_eligible_pairs": 0,
            "review_priority": dict(counts),
        },
        "limitations": [
            "No trained or validated ranking; ordering groups evidence completeness only.",
            "SOURCE alone is not a bad pixel; excluded-pixel flags are not an artifact diagnosis.",
            "All pairs remain in the queue.",
            "Catalog matching is not evaluated across this field; null counts mean unknown.",
            "This queue is not wired into legacy identity or triage gates.",
            "No accuracy, discovery, motion, or variability claim is supported.",
        ],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.comparison.read_bytes()
    result = build(json.loads(raw), hashlib.sha256(raw).hexdigest())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
