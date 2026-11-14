"""Exercise the real FITS-to-catalog-to-inspector demonstration offline."""

import json
import os
import subprocess
import sys
from pathlib import Path

from examples.source_investigation.build_evidence import Package
from examples.source_investigation.run_demo import Marker

ROOT = Path(__file__).resolve().parents[2]


def test_demo_runs_real_pipeline_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "run"
    command = [
        sys.executable,
        "examples/source_investigation/run_demo.py",
        "--output",
        str(output),
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    package = Package.model_validate_json((output / "evidence.json").read_text())
    marker = Marker.model_validate_json((output / "aladin-marker.json").read_text())
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert package.catalog["access_mode"] == "retained_esa_gaia_snapshot"
    assert package.cases[0].status == "TENTATIVE_ASSOCIATION"
    assert package.cases[0].checks[0]["verdict"] == "INCONCLUSIVE"
    assert package.cases[0].catalog_candidates[0]["source_id"] == "6316705404065320832"
    assert marker.ra_deg == package.cases[0].ra_deg
    assert marker.dec_deg == package.cases[0].dec_deg
    assert marker.inspector_case_id == package.cases[0].id
    assert manifest["identify"].startswith("NOT_RUN")
    assert manifest["harness"].startswith("NOT_RUN")
    failed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "FileExistsError" in failed.stderr
