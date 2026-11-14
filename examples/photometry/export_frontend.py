"""Export a scan as conservative, validated Aladin marker evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sieve.photometry.scanner import PhotometryScan


class Marker(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str
    ra_deg: float = Field(ge=0, lt=360)
    dec_deg: float = Field(ge=-90, le=90)
    aperture_usable: bool
    aperture_sum: float
    aperture_sum_uncertainty: float = Field(ge=0)
    aperture_sum_unit: str
    centroid_flag_bits: int = Field(ge=0)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["sieve.frontend.markers.v1"] = "sieve.frontend.markers.v1"
    scientific_status: Literal["MEASUREMENTS_ONLY_NOT_IDENTITY"]
    exposure_id: str | None
    observation_mjd: float | None
    detector: int | None
    source_sha256: str
    source_name: str
    code_commit: str
    limitations: tuple[str, ...]
    markers: tuple[Marker, ...]


def export(scan: PhotometryScan) -> Evidence:
    markers = tuple(
        Marker(
            id=row["source_id"],
            ra_deg=row["ra_deg"],
            dec_deg=row["dec_deg"],
            aperture_usable=row["aperture_usable"],
            aperture_sum=row["aperture_sum"],
            aperture_sum_uncertainty=row["aperture_sum_uncertainty"],
            aperture_sum_unit=row["aperture_sum_unit"],
            centroid_flag_bits=row["centroid_flag_bits"],
        )
        for row in scan.detections
    )
    return Evidence(
        scientific_status=scan.scientific_status,
        exposure_id=scan.exposure_id,
        observation_mjd=scan.observation_mjd,
        detector=scan.detector,
        source_sha256=scan.source_sha256,
        source_name=scan.source_name,
        code_commit=scan.code_commit,
        limitations=scan.limitations,
        markers=markers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scan = PhotometryScan.model_validate_json(args.scan.read_text())
    evidence = export(scan)
    with args.output.open("x") as output:
        json.dump(evidence.model_dump(mode="json"), output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Exported {len(evidence.markers)} measurement-only markers to {args.output}")


if __name__ == "__main__":
    main()
