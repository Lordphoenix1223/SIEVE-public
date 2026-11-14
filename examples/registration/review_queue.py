"""Build an evidence-completeness queue from a guarded QR3 comparison.

This is a review aid, not a classifier or a motion/variability gate for other APIs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from astropy.time import Time

from sieve.photometry.scanner import _BAD_FLAG_BITS


def field_identifications(comparison: dict[str, Any]) -> dict[str, Any]:
    """Match against the bounded, pinned Gaia cohort, never claim catalog completeness."""
    expected_inputs = [
        "13434d8c4615a7b788594d3065809c61dfa5c28d5fae6590cf5d7e8b952a4117",
        "0d6297702adcbb65786230b1ed7f9c4e6a58e1746885d8dd5ebbc46c8be24fc9",
    ]
    if comparison["provenance"]["input_sha256"] != expected_inputs:
        raise ValueError("Gaia cohort is only supported for its pinned QR3 inputs")
    source = Path(__file__).with_name("gaia-qr3-reference.csv")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "c6ca7cb28496f14e0146835018a7605130177b2dfc4be9a8fcb0572a72957517":
        raise ValueError("Gaia reference hash mismatch")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    catalog = [{"id": row["source_id"], "catalog": "Gaia DR3",
                "source_url": "https://gea.esac.esa.int/tap-server/tap/sync",
                "ra_deg": float(row["ra"]), "dec_deg": float(row["dec"]),
                "epoch_year": float(row["ref_epoch"]),
                "pmra_mas_yr": float(row["pmra"]), "pmdec_mas_yr": float(row["pmdec"])}
               for row in rows]
    detections = []
    for obs, measurements in zip(comparison["geometry"]["inputs"],
                                 comparison["measurements"], strict=True):
        epoch = float(Time(obs["mjd_mid_utc"], format="mjd", scale="utc").tt.jyear)
        for i, m in enumerate(measurements):
            if m["status"] != "MEASURED_CONDITIONAL":
                continue
            detections.append({"id": f"measurement:{obs['exposure_id']}:{i:05d}",
                               "ra_deg": m["ra_deg"], "dec_deg": m["dec_deg"],
                               "epoch_year": epoch,
                               "flags": [f"FITS_FLAG_BITS:{m['flag_bits']}"]
                               if m.get("flag_bits") else []})
    proc = subprocess.run(["node", str(Path(__file__).with_name("field_match.mjs"))],
                          input=json.dumps({"detections": detections, "catalog": catalog},
                                           allow_nan=False), text=True, capture_output=True,
                          check=True, timeout=60)
    out = json.loads(proc.stdout)
    out["catalog_sha256"] = digest
    out["catalog_scope"] = "bounded Gaia DR3 cone, G<18, proper motions required; incomplete"
    return out


def build(comparison: dict[str, Any], comparison_sha256: str,
          identities: dict[str, Any] | None = None) -> dict[str, Any]:
    if comparison["schema_version"] != "sieve.qr3.comparison.v1":
        raise ValueError("unsupported comparison schema")
    geometry = comparison["geometry"]
    measurements = comparison["measurements"]
    identity_by_id = {r["detection_id"]: r for r in identities["results"]} if identities else {}
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
    for record in records:
        matches = [identity_by_id.get(i) for i in record["detection_ids"]]
        if all(m is not None for m in matches):
            record["identify"] = matches
            ids = [{(c["catalog"], c["id"]) for c in m["candidates"]} for m in matches]
            record["catalog_association"] = (
                "SHARED_CANDIDATE_UNCONFIRMED" if ids[0] & ids[1]
                else "NO_SHARED_CANDIDATE_IN_BOUNDED_COHORT"
            )
    order = {"review_measurements": 0, "review_flags_or_ambiguity": 1, "deferred": 2}
    records.sort(key=lambda r: (order[r["review_priority"]], r["detection_ids"]))
    counts = Counter(r["review_priority"] for r in records)
    return {
        "schema_version": "sieve.qr3.review_queue.v1",
        "comparison_sha256": comparison_sha256,
        "input_sha256": comparison["provenance"]["input_sha256"],
        "catalog_provenance": ({"sha256": identities["catalog_sha256"],
                                "scope": identities["catalog_scope"]} if identities else None),
        "counts": {
            "raw_peaks": comparison["detections_per_exposure"],
            "usable_measurements": [sum(m["status"] == "MEASURED_CONDITIONAL" for m in rows)
                                    for rows in measurements],
            "candidate_pairs": len(records),
            "usable_pairs": sum(all(m["status"] == "MEASURED_CONDITIONAL"
                                    for m in r["measurements"]) for r in records),
            "ambiguous_pairs": sum(bool(r["pair"]["ambiguous"]) for r in records),
            "catalog_associated_pairs": (
                sum(r["catalog_association"] == "SHARED_CANDIDATE_UNCONFIRMED" for r in records)
                if identities is not None else None
            ),
            "science_eligible_pairs": 0,
            "review_priority": dict(counts),
        },
        "limitations": [
            "No trained or validated ranking; ordering groups evidence completeness only.",
            "SOURCE alone is not a bad pixel; excluded-pixel flags are not an artifact diagnosis.",
            "All pairs remain in the queue.",
            "Gaia cohort is bounded, magnitude-selected and requires proper motion.",
            "No match is not a discovery.",
            "Shared catalog candidates are not verified identities or independent labels.",
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
    comparison = json.loads(raw)
    identities = field_identifications(comparison)
    result = build(comparison, hashlib.sha256(raw).hexdigest(), identities)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
