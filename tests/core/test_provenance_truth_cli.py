import ast
import builtins
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.truth import TruthRecord, TruthType, benchmark_eligible
from sieve.cli import read_bundle, run
from sieve.contracts.bundle import InferenceBundle
from sieve.pipeline import triage
from sieve.provenance.models import ProvenanceGraph
from sieve.triage.budget import CandidateBudget
from sieve.triage.dedup import DedupPolicy

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests/fixtures/frankenstein_case.json"


def test_lineage_closed_and_derivations_present():
    bundle, _ = read_bundle(FIXTURE)
    graph = bundle.provenance
    entities = {e.id for e in graph.entities}
    for record in graph.records:
        assert set(record.generated_entity_ids) == set(record.derivations)
        assert set(record.used_entity_ids) <= entities
    factor = next(
        f for r in bundle.evidence_results for f in r.factors if f.hypothesis_id == "h-clean"
    )
    record = next(r for r in graph.records if r.id == factor.provenance_ref)
    assert "d-a0" in record.derivations[factor.id]


def test_missing_lineage_entity_rejected():
    data = json.loads(FIXTURE.read_text())
    data["provenance"]["entities"] = data["provenance"]["entities"][1:]
    with pytest.raises(ValueError, match="unknown lineage entity"):
        InferenceBundle.model_validate(data)


def test_cycle_rejected():
    data = json.loads(FIXTURE.read_text())["provenance"]
    base = data["records"][0]
    records = []
    for key, other in (("a", "b"), ("b", "a")):
        record = copy.deepcopy(base)
        record.update(
            id="activity-" + key,
            used_entity_ids=[other],
            generated_entity_ids=[key],
            derivations={key: [other]},
        )
        records.append(record)
    data.update(
        entities=[
            {"id": key, "kind": "toy", "hash_missing_reason": "fixture"} for key in ("a", "b")
        ],
        records=records,
    )
    with pytest.raises(ValueError, match="cyclic"):
        ProvenanceGraph.model_validate(data)


def test_per_output_evidence_lineage_is_required():
    data = json.loads(FIXTURE.read_text())
    activity = next(r for r in data["provenance"]["records"] if r["id"] == "activity-evidence")
    # Inputs still exist in the activity, but not in this particular output's derivation.
    activity["derivations"]["factor-clean"] = ["request-clean"]
    with pytest.raises(ValueError, match="evidence inputs missing"):
        InferenceBundle.model_validate(data)


def test_request_hypothesis_lineage_is_required():
    data = json.loads(FIXTURE.read_text())
    activity = next(r for r in data["provenance"]["records"] if r["id"] == "activity-request")
    activity["derivations"]["request-clean"] = ["d-a0", "d-a1", "d-a2", "d-a3"]
    with pytest.raises(ValueError, match="requested hypothesis missing"):
        InferenceBundle.model_validate(data)


@pytest.mark.parametrize(
    "layer,key", [("detect", "d-a0"), ("propose", "p-clean"), ("hypothesize", "h-clean")]
)
def test_structural_parent_lineage_required(layer, key):
    data = json.loads(FIXTURE.read_text())
    activity = next(r for r in data["provenance"]["records"] if r["id"] == "activity-" + layer)
    activity["derivations"][key] = [activity["used_entity_ids"][-1]]
    with pytest.raises(ValueError, match="derivation"):
        InferenceBundle.model_validate(data)


def record(kind, **overrides):
    data = dict(
        id="truth",
        source_type=kind,
        source_ref="fixture",
        task_scope="membership",
        member_ids=("d-a0",),
        object_ids=("object-A",),
        frozen=True,
        complete_for_scope=True,
        limitations=("synthetic",),
        audit_ref="audit",
    )
    return TruthRecord(**{**data, **overrides})


@pytest.mark.parametrize("kind", [TruthType.HUMAN_ANNOTATION, TruthType.UNKNOWN])
def test_annotations_are_not_benchmark_truth(kind):
    with pytest.raises(ValueError, match="cannot declare"):
        record(kind)
    annotation = record(kind, complete_for_scope=False)
    assert not benchmark_eligible(annotation, "membership")


def test_benchmark_truth_scoped_and_audited():
    assert benchmark_eligible(record(TruthType.SIMULATED_TRUTH), "membership")
    assert not benchmark_eligible(record(TruthType.SIMULATED_TRUTH), "discovery")
    assert not benchmark_eligible(record(TruthType.BENCHMARK_TRUTH, audit_ref=None), "membership")
    assert not benchmark_eligible(record(TruthType.SIMULATED_TRUTH, frozen=False), "membership")
    assert not benchmark_eligible(record(TruthType.AUTHORITATIVE_EXTERNAL_RECORD), "membership")
    external = record(TruthType.AUTHORITATIVE_EXTERNAL_RECORD, task_scope="catalog_identity")
    assert benchmark_eligible(external, "catalog_identity")


def test_false_merge_is_established_only_in_evaluation():
    labels = [
        TruthRecord.model_validate(r)
        for r in json.loads((ROOT / "evaluation/fixtures/frankenstein_truth.json").read_text())
    ]
    bundle, _ = read_bundle(FIXTURE)
    mixed = next(p for p in bundle.proposals if p.id == "p-frankenstein")
    mapping = {
        d: object_id for label in labels for d in label.member_ids for object_id in label.object_ids
    }
    assert {mapping[d] for d in mixed.member_ids} == {"object-A", "object-B"}
    assert all(benchmark_eligible(label, "membership") for label in labels)


def test_production_no_evaluation_imports():
    for path in (ROOT / "src/sieve").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("evaluation")
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("evaluation") for alias in node.names)


def test_inference_does_not_load_truth(monkeypatch):
    bundle, _ = read_bundle(FIXTURE)
    original_import = builtins.__import__
    original_open = builtins.open

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("evaluation")
        return original_import(name, *args, **kwargs)

    def guarded_open(file, *args, **kwargs):
        assert "truth" not in str(file).lower()
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "open", guarded_open)
    first = triage(bundle, CandidateBudget(), DedupPolicy())
    second = triage(bundle, CandidateBudget(), DedupPolicy())
    assert first == second


def test_cli_receipt_and_deterministic_output_hash():
    first = run(FIXTURE, CandidateBudget(), DedupPolicy())
    second = run(FIXTURE, CandidateBudget(), DedupPolicy())
    assert first.receipt.input_sha256 == second.receipt.input_sha256
    assert first.receipt.output_sha256 == second.receipt.output_sha256
    assert first.receipt.code_tree_sha256 == second.receipt.code_tree_sha256
    assert first.receipt.provenance_ref in {r.id for r in first.provenance.records}
    assert first.statistical_filter["status"] == "FDR_NOT_APPLICABLE"


def test_cli_end_to_end():
    result = subprocess.run(
        [sys.executable, "-m", "sieve.cli", "triage", "--input", str(FIXTURE), "--budget", "2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["counts"]["cases_surfaced"] == 2
    assert len(output["cases"]) == 2
    assert output["receipt"]["schema_version"] == "sieve.v0"
    assert output["report_view"] == "compact"


def test_cli_full_report_projection():
    result = subprocess.run(
        [sys.executable, "-m", "sieve.cli", "triage", "--input", str(FIXTURE), "--full-report"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert "audit_decisions" in output
    assert "records" in output["provenance"]


@pytest.mark.parametrize("raw", ['{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_strict_json_input(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_bundle(path)


def test_cli_error_is_operational(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"hidden_truth": true}')
    result = subprocess.run(
        [sys.executable, "-m", "sieve.cli", "triage", "--input", str(bad)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["status"] == "INPUT_OR_EXECUTION_ERROR"
    assert not result.stdout
