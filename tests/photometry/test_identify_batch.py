"""Batching must not change the original matcher's per-detection results."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples/source_investigation/identify/match.mjs"
ENGINE = (ROOT / "examples/source_investigation/identify/engine.mjs").as_uri()


def test_batch_preserves_original_results_and_unresolved_cases():
    detections = [
        {"id": "near", "ra_deg": 10.0, "dec_deg": 0.0, "epoch_year": 2026.0},
        {"id": "far", "ra_deg": 20.0, "dec_deg": 0.0, "epoch_year": 2026.0},
        {"id": "ambiguous", "ra_deg": 10.0, "dec_deg": 0.0, "epoch_year": 2026.0},
    ]
    cat = [{"id": "one", "catalog": "test", "source_url": "https://example.org",
            "ra_deg": 10.0, "dec_deg": 0.0, "epoch_year": 2026.0}]
    jobs = [{"detection": d, "catalog": cat if i < 2 else cat + [{**cat[0], "id": "two"}]}
            for i, d in enumerate(detections)]
    script = (f'import {{identify}} from "{ENGINE}"; import {{readFileSync}} from "node:fs";'
              'const j=JSON.parse(readFileSync(0,"utf8"));'
              'console.log(JSON.stringify(identify([j.detection],j.catalog).results[0]));')
    original = [json.loads(subprocess.run(
        ["node", "--input-type=module", "-e", script], input=json.dumps(job), text=True,
        capture_output=True, check=True).stdout) for job in jobs]
    batch = json.loads(subprocess.run(
        ["node", str(RUNNER)], input=json.dumps({"jobs": jobs}), text=True,
        capture_output=True, check=True).stdout)
    assert batch == original
    assert [item["status"] for item in batch] == ["inconclusive", "no_match", "ambiguous"]
    assert not any(item["discovery_claim"] for item in batch)
