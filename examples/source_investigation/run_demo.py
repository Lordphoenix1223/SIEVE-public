"""Run the bounded real-QR3/Gaia investigation into a new output directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

import astropy
import numpy
import photutils
from pydantic import BaseModel, ConfigDict, Field

from examples.source_investigation import build_evidence, identify_bridge
from sieve.registration.core import compare, load_observation

HERE = Path(__file__).resolve().parent
CATALOG_SHA256 = "69395c68563b17afec52117c73f229a68f96b7840d92862efcdabc94198c42f0"
ARTIFACT_SHA256 = "c5ab6f3740197afc42d8ea2c642950f6d37d559b265414c166adf428ed8feab5"


class Marker(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["sieve.aladin.marker.v1"] = "sieve.aladin.marker.v1"
    id: str
    ra_deg: float = Field(ge=0, lt=360)
    dec_deg: float = Field(ge=-90, le=90)
    frame: Literal["ICRS"] = "ICRS"
    status: Literal["TENTATIVE_ASSOCIATION"] = "TENTATIVE_ASSOCIATION"
    inspector_case_id: Literal["A"] = "A"
    inspector_file: Literal["evidence.json"] = "evidence.json"
    identify_file: Literal["searchx-identifications.json"] = "searchx-identifications.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live_catalog(snapshot: Path) -> bytes:
    """Fetch Gaia and fail closed if the validated fixed-cohort result has changed."""
    reference = json.loads((HERE / "evidence.json").read_text())["catalog"]
    query = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": reference["adql"]}
    )
    with urllib.request.urlopen(reference["service"] + "?" + query, timeout=60) as response:
        content = response.read()
    live = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    with snapshot.open(newline="") as stream:
        retained = list(csv.DictReader(stream))
    if live != retained:
        raise ValueError("Live Gaia cone result differs from pinned cohort; review before using it")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument(
        "--live-gaia", action="store_true", help="query ESA Gaia; fail if the fixed cohort differs"
    )
    args = parser.parse_args()
    snapshot = HERE / "gaia-dr3-cone.csv"
    artifact = HERE / "artifact-results.json"
    if digest(snapshot) != CATALOG_SHA256 or digest(artifact) != ARTIFACT_SHA256:
        raise ValueError("Pinned Gaia or artifact reference hash mismatch")
    # Fetch before creating the output so network errors cannot masquerade as completion.
    catalog = live_catalog(snapshot) if args.live_gaia else snapshot.read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    catalog_path = args.output / "gaia-dr3-cone.csv"
    catalog_path.write_bytes(catalog)
    shutil.copyfile(artifact, args.output / "artifact-results.json")
    build_evidence.main(
        output_dir=args.output,
        catalog_path=catalog_path,
        catalog_mode="live_esa_gaia" if args.live_gaia else "retained_esa_gaia_snapshot",
    )
    package = build_evidence.Package.model_validate_json(
        (args.output / "evidence.json").read_text()
    )
    case = package.cases[0]
    exposures = json.loads((build_evidence.BASE / "source.json").read_text())["exposures"][:2]
    observations = [load_observation(build_evidence.BASE / item["file"]) for item in exposures]
    if [o.sha256 for o in observations] != [e["sha256"] for e in exposures]:
        raise ValueError("WCS inputs differ from the source-investigation inputs")
    geometry = compare(*observations)
    (args.output / "wcs-diagnostics.json").write_text(
        json.dumps(geometry, indent=2, allow_nan=False) + "\n"
    )
    identities = identify_bridge.build(case, package.catalog)
    for identity in identities:
        identity["wcs_diagnostics_file"] = "wcs-diagnostics.json"
        identity["scientific_comparability"] = {
            "motion": geometry["source_motion_status"],
            "flux_variability": geometry["flux_variability_status"],
            "motion_claim": "REFUSED", "flux_variability_claim": "REFUSED",
        }
    (args.output / "searchx-identifications.json").write_text(
        json.dumps(identities, indent=2, allow_nan=False) + "\n"
    )
    marker = Marker(
        id=case.measurements[0]["measurement_id"], ra_deg=case.ra_deg, dec_deg=case.dec_deg
    )
    (args.output / "aladin-marker.json").write_text(marker.model_dump_json(indent=2) + "\n")
    (args.output / "aladin-marker.schema.json").write_text(
        json.dumps(Marker.model_json_schema(), indent=2) + "\n"
    )
    (args.output / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sieve.source_demo.run.v1",
                "catalog_mode": package.catalog["access_mode"],
                "runtime": {
                    "python": platform.python_version(),
                    "astropy": astropy.__version__,
                    "numpy": numpy.__version__,
                    "photutils": photutils.__version__,
                },
                "source_sha256": {
                    "run_demo.py": digest(Path(__file__)),
                    "build_evidence.py": digest(HERE / "build_evidence.py"),
                    "identify_bridge.py": digest(HERE / "identify_bridge.py"),
                    "identify/match.mjs": digest(HERE / "identify" / "match.mjs"),
                    "identify/engine.mjs": digest(HERE / "identify" / "engine.mjs"),
                },
                "identify": "searchx-identifications.json",
                "harness": "NOT_RUN: no applicable moving-object backend executed",
                "artifact_case": "retained exploratory simulation, not re-executed by this command",
                "files_sha256": {
                    p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Validated evidence and Aladin marker: {args.output}")
    print(f"Gaia mode: {package.catalog['access_mode']}; Identify complete; Harness NOT_RUN")


if __name__ == "__main__":
    main()
