"""Compare original QR3 proximity proposals with the guarded CLI output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True, help="original pilot output directory"
    )
    parser.add_argument("--upgraded", type=Path, required=True, help="compare-qr3 JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.baseline / "profiles.json"
    base = json.loads(base_path.read_text())["profiles"]
    upgraded = json.loads(args.upgraded.read_text())
    base_pairs = {tuple(map(int, x["case_id"].split(":")[1:])) for x in base}
    new_pairs = {(x["a"], x["b"]) for x in upgraded["pairings"]["candidate_pairs"]}
    bsummary = json.loads((args.baseline / "summary.json").read_text())
    out = {
        "schema_version": "sieve.registration.baseline-comparison.v1",
        "baseline_method": "existing pilot Astropy search_around_sky at 12 arcseconds",
        "reference_method": "Astropy search_around_sky is the existing method; not independent",
        "upgrade_method": "same WCS sky gate with ambiguity and scientific refusal states",
        "input_sha256": upgraded["provenance"]["input_sha256"],
        "baseline_profiles_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "upgraded_sha256": hashlib.sha256(args.upgraded.read_bytes()).hexdigest(),
        "baseline_detected_peaks": bsummary["generation_peaks"],
        "upgraded_detected_peaks": upgraded["detections_per_exposure"],
        "baseline_pairs": len(base_pairs),
        "upgraded_pairs": len(new_pairs),
        "common_pairs": len(base_pairs & new_pairs),
        "baseline_only": len(base_pairs - new_pairs),
        "upgraded_only": len(new_pairs - base_pairs),
        "ambiguous_pairs": len(upgraded["pairings"]["ambiguous_pairs"]),
        "baseline_scientific_result": bsummary["scientific_result"],
        "upgraded_motion_claim": upgraded["motion_claim"],
        "upgraded_flux_variability_claim": upgraded["flux_variability_claim"],
        "limitations": [
            "Both paths use Astropy matching; agreement is not independent validation.",
            "This field has no exhaustive reference identities or truth labels.",
            "Pilot includes injections and identity; its runtime is not comparable to the CLI.",
        ],
    }
    with args.output.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
