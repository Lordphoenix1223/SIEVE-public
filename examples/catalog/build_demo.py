"""Generate small synthetic archive-export examples; no evaluator truth in inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


def build(directory: Path, target_rows: int = 40) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for role, count, offset in (("reference", 64, 0), ("target", target_rows, 1000)):
        path = directory / f"{role}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                ["source_id", "ra", "dec", "flux_a", "flux_b", "sigma", "group", "quality"]
            )
            for i in range(count):
                writer.writerow(
                    [
                        f"source-{offset + i}",
                        (i * 17.1) % 360,
                        -60 + i % 120,
                        10 + 2 * math.sin(i) + (50 if role == "target" and i % 19 == 0 else 0),
                        8 + math.cos(i * 0.7),
                        0.5,
                        f"exposure-{role}-{i // 2}",
                        "bad" if role == "target" and i % 13 == 0 else "good",
                    ]
                )
        manifest = {
            "schema_version": "sieve.catalog.v1",
            "path": path.name,
            "format": "csv",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "provider": "synthetic-export",
            "dataset": "synthetic-two-flux-catalog",
            "release": "fixture-v1",
            "data_kind": "SYNTHETIC",
            "source_uri": "urn:sieve:synthetic:catalog-demo",
            "request_parameters": {"role": role, "rows": count},
            "citations": ["SIEVE synthetic engineering fixture; not astronomical data"],
            "identity_namespace": "synthetic-catalog-v1",
            "id_column": "source_id",
            "ra_column": "ra",
            "dec_column": "dec",
            "coordinate_frame": "icrs",
            "coordinate_unit": "deg",
            "features": [
                {
                    "name": name,
                    "column": name,
                    "unit": "Jy",
                    "description": f"Synthetic reported {name}",
                    "lower": 0,
                    "upper": 100,
                    "error_column": "sigma",
                    "uncertainty_kind": "reported_standard_deviation",
                }
                for name in ("flux_a", "flux_b")
            ],
            "split_groups": [{"namespace": "synthetic-exposure", "column": "group"}],
            "split_policy_ref": "explicit separate source and exposure groups in fixture",
            "support_domain": "synthetic finite two-flux feature exercise only",
            "selection_description": "deterministic trigonometric values; engineered offsets",
            "quality": {
                "column": "quality",
                "accepted_values": ["good"],
                "definition_ref": "synthetic declared good/bad flag; no instrument calibration",
            },
            "calibration_refs": [],
            "limitations": ["Synthetic catalog; no measured sources, sensitivity or probability"],
        }
        (directory / f"{role}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    build(Path(__file__).parent)
