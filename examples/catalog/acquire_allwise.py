"""Optional, capped public IRSA TAP example for the Fornax catalog workflow.

Run explicitly: python examples/catalog/acquire_allwise.py --output reports/allwise-input
This is two 64-row catalog queries, not a Fornax login or bulk archive download.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = "https://irsa.ipac.caltech.edu/TAP/sync"
SUPPLEMENT = "https://irsa.ipac.caltech.edu/data/WISE/docs/release/AllWISE/expsup/sec2_1a.html"


def acquire(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for role, low, high in (("reference", 202.40, 202.45), ("target", 202.45, 202.50)):
        query = (
            "SELECT TOP 64 designation,ra,dec,w1mpro,w2mpro,w1sigmpro,w2sigmpro,cc_flags "
            "FROM allwise_p3as_psd "
            f"WHERE ra >= {low} AND ra < {high} AND dec >= 47.20 AND dec < 47.30 "
            "AND ph_qual LIKE 'AA%' AND cc_flags = '0000' "
            "AND w1sigmpro > 0 AND w2sigmpro > 0 ORDER BY designation"
        )
        parameters = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "MAXREC": "64",
            "QUERY": query,
        }
        request = Request(
            ENDPOINT + "?" + urlencode(parameters),
            headers={"User-Agent": "SIEVE-capped-example/0.1"},
        )
        with urlopen(request, timeout=45) as response:
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError("IRSA response exceeded the one MB example cap")
        # Inspect the capped response before writing a manifest that calls it CSV.
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if not reader.fieldnames or "designation" not in reader.fieldnames:
            raise ValueError("IRSA did not return the expected catalog CSV")
        rows = list(reader)
        if not 8 <= len(rows) <= 64:
            raise ValueError(
                f"expected 8..64 rows, received {len(rows)}; revise the explicit sample"
            )
        path = output / f"{role}.csv"
        path.write_bytes(raw)
        manifest = {
            "schema_version": "sieve.catalog.v1",
            "path": path.name,
            "format": "csv",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "provider": "NASA-IRSA",
            "dataset": "allwise_p3as_psd",
            "release": "AllWISE-2013",
            "data_kind": "REAL_ARCHIVE",
            "source_uri": ENDPOINT,
            "request_parameters": {**parameters, "request_date": datetime.now(UTC).isoformat()},
            "citations": [SUPPLEMENT, "https://doi.org/10.1088/0004-6256/140/6/1868"],
            "identity_namespace": "irsa-allwise-2013-designation",
            "id_column": "designation",
            "ra_column": "ra",
            "dec_column": "dec",
            "coordinate_frame": "icrs",
            "coordinate_unit": "deg",
            "features": [
                {
                    "name": band + "mpro",
                    "column": band + "mpro",
                    "unit": "mag",
                    "description": f"AllWISE {band.upper()} reported profile-fit Vega magnitude",
                    "lower": -10,
                    "upper": 30,
                    "error_column": band + "sigmpro",
                    "uncertainty_kind": "reported_standard_deviation",
                }
                for band in ("w1", "w2")
            ],
            "split_groups": [{"namespace": "allwise-source", "column": "designation"}],
            "split_policy_ref": (
                "disjoint RA slices; object-identity check only, no exposure independence claim"
            ),
            "support_domain": (
                "capped AllWISE W1/W2 AA photometry and cc_flags 0000 catalog example"
            ),
            "selection_description": (
                "TOP 64 ordered by designation in two small adjacent RA rectangles; "
                "not representative"
            ),
            "quality": {
                "column": "cc_flags",
                "accepted_values": ["0000"],
                "definition_ref": SUPPLEMENT,
            },
            "calibration_refs": [SUPPLEMENT],
            "limitations": [
                "Static catalog example, not SPHEREx pixels or a moving-source search",
                "Catalog uncertainties/calibration are reported, not independently qualified",
                "Only source identity groups available; exposure dependence is unknown",
                "Magnitude/error ratio is not flux SNR; none is computed",
                "Small selected regions; no sensitivity, population or generalization claim",
            ],
        }
        (output / f"{role}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"role": role, "rows": len(rows), "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    acquire(parser.parse_args().output)
