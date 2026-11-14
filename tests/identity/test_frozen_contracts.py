"""Published schema/auxiliary pins are change boundaries, not scientific tests."""

import hashlib
import json
from pathlib import Path

from evaluation.identity import ComparisonProtocol, TruthSet
from evaluation.truth import TruthRecord
from sieve.identity.models import EvidenceResult, IdentityInput, PredictionTable
from sieve.interfaces import ObservationBatch
from sieve.provenance.models import ProvenanceGraph, RunReceipt

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = {
    "examples/identity/schemas/identity-input.schema.json": IdentityInput,
    "examples/identity/schemas/evidence-result.schema.json": EvidenceResult,
    "examples/identity/schemas/observation-batch.schema.json": ObservationBatch,
    "examples/identity/schemas/prediction-table.schema.json": PredictionTable,
    "examples/identity/schemas/provenance.schema.json": ProvenanceGraph,
    "examples/identity/schemas/run-receipt.schema.json": RunReceipt,
    "evaluation/schemas/identity-truth.schema.json": TruthSet,
    "evaluation/schemas/truth-record.schema.json": TruthRecord,
    "evaluation/schemas/identity-comparison.schema.json": ComparisonProtocol,
}


def test_published_contract_schemas_match_versioned_models():
    for path, model in SCHEMAS.items():
        assert json.loads((ROOT / path).read_text()) == model.model_json_schema()


def test_auxiliary_frozen_inventory_matches_bytes():
    inventory = json.loads((ROOT / "examples/auxiliary/kepler/frozen-inventory.json").read_text())
    assert inventory["scientific_identity_evidence"] is False
    for entry in inventory["files"]:
        path = ROOT / entry["path"]
        if entry["local_only"] and not path.exists():
            continue  # Original large run files remain on their original host.
        raw = path.read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]


def test_identity_core_has_no_evaluator_or_auxiliary_imports():
    import ast

    for path in (ROOT / "src/sieve/identity").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            assert not any(m.startswith(("evaluation", "sieve.specialist")) for m in modules)


def test_reference_fixture_raw_asset_pins_match_published_bytes():
    for kind in ("clean", "merge", "sparse", "artifact"):
        data = IdentityInput.model_validate_json(
            (ROOT / f"examples/identity/{kind}.json").read_text()
        )
        for entity in data.provenance.entities:
            if entity.uri and entity.uri.startswith("repo://SIEVE/"):
                path = ROOT / entity.uri.removeprefix("repo://SIEVE/")
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entity.sha256
