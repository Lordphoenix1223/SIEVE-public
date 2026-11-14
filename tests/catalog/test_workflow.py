from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys

import pytest

pytest.importorskip("sklearn", reason="catalog optional dependencies not installed")

from examples.catalog.build_demo import build
from sieve.catalog.io import file_hash
from sieve.catalog.models import CatalogConfig, CatalogManifest
from sieve.catalog.workflow import run_catalog


@pytest.fixture
def inputs(tmp_path):
    build(tmp_path)
    return tmp_path


def manifest(directory, role="target"):
    return json.loads((directory / f"{role}.manifest.json").read_text())


def replace_manifest(directory, data, role="target"):
    (directory / f"{role}.manifest.json").write_text(json.dumps(data))


def edit_rows(directory, mutate, role="target"):
    path = directory / f"{role}.csv"
    with path.open() as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = reader.fieldnames
    mutate(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    data = manifest(directory, role)
    data["sha256"] = file_hash(path)
    replace_manifest(directory, data, role)


def run(directory, name="out", **options):
    return run_catalog(
        directory / "reference.manifest.json",
        directory / "target.manifest.json",
        directory / name,
        CatalogConfig(review_budget=5, trees=10, batch_rows=7, **options),
    )


def records(directory, name="out"):
    return [
        json.loads(line) for line in (directory / name / "records.jsonl").read_text().splitlines()
    ]


def test_full_workflow_accounts_every_row_and_keeps_claims_bounded(inputs):
    summary = run(inputs, export_json_array=True)
    output = records(inputs)
    assert len(output) == summary["counts"]["target"]["input_rows"] == 40
    assert summary["counts"]["output"] == {"EXCLUDED": 4, "SURFACED": 5, "DEFERRED": 31}
    assert len(json.loads((inputs / "out/cases.json").read_text())["cases"]) == 5
    assert json.loads((inputs / "out/records.json").read_text()) == output
    assert len({r["review_region"] for r in output if r["decision"] == "SURFACED"}) == 5
    assert {r["scientific_disposition"] for r in output} == {"INSUFFICIENT_EVIDENCE"}
    assert all(r["object_identity_status"] == "NOT_ESTABLISHED" for r in output)
    assert all(r["score"]["semantics"] == "RANKING_ONLY" for r in output if r["score"])
    assert summary["statistical_filter"]["status"] == "FDR_NOT_APPLICABLE"
    assert len(summary["rules_enforced"]) == 10
    receipt = json.loads((inputs / "out/receipt.json").read_text())
    for name, entry in receipt["artifacts"].items():
        assert file_hash(inputs / "out" / name) == entry["sha256"]
    assert receipt["resources"]["process_peak_rss_bytes"] > 0
    with sqlite3.connect(inputs / "out/inventory.sqlite") as db:
        assert db.execute("SELECT count(*) FROM inventory WHERE role='target'").fetchone()[0] == 40


def test_repeat_run_is_deterministic(inputs):
    run(inputs, "first")
    run(inputs, "second")
    for name in ("model.json", "records.jsonl", "cases.json", "summary.json"):
        assert (inputs / "first" / name).read_bytes() == (inputs / "second" / name).read_bytes()


def test_independent_output_verifier_and_trust_anchor(inputs):
    from sieve.catalog.verify import verify_run

    run(inputs)
    check = verify_run(inputs / "out")
    assert check["records_checked"] == 40
    assert check["receipt_trust"] == "UNANCHORED_LOCAL_RECEIPT"
    assert (
        verify_run(inputs / "out", check["receipt_sha256"])["receipt_trust"]
        == "MATCHES_SUPPLIED_HASH"
    )
    with pytest.raises(ValueError, match="trust anchor"):
        verify_run(inputs / "out", "0" * 64)


def test_independent_verifier_rejects_tampering(inputs):
    from sieve.catalog.verify import verify_run

    run(inputs)
    (inputs / "out/cases.json").write_text("{}")
    with pytest.raises(ValueError, match="bytes/hash mismatch"):
        verify_run(inputs / "out")


def test_existing_output_refuses_overwrite(inputs):
    run(inputs)
    with pytest.raises(ValueError, match="output already exists"):
        run(inputs)


def test_target_and_reference_same_bytes_refuse(inputs):
    data = manifest(inputs, "reference")
    replace_manifest(inputs, data)
    with pytest.raises(ValueError, match="assets must be distinct"):
        run(inputs)


def test_input_mutation_during_run_refuses_publication(inputs, monkeypatch):
    import sieve.catalog.workflow as workflow

    original = workflow.rank

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with (inputs / "target.csv").open("a") as stream:
            stream.write("changed after import\n")
        return result

    monkeypatch.setattr(workflow, "rank", changed)
    with pytest.raises(ValueError, match="changed during processing"):
        run(inputs)
    assert not (inputs / "out").exists()


def test_input_size_limit_refuses_before_import(inputs):
    with pytest.raises(ValueError, match="max_input_bytes exceeded"):
        run(inputs, max_input_bytes=100)
    assert not (inputs / "out").exists()


def test_hash_mismatch_refuses_before_publish(inputs):
    (inputs / "target.csv").write_text("changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run(inputs)
    assert not (inputs / "out").exists()


@pytest.mark.parametrize("field,value", [("unit", "s"), ("upper", 101), ("description", "changed")])
def test_incompatible_feature_contract_refuses(inputs, field, value):
    data = manifest(inputs)
    data["features"][0][field] = value
    replace_manifest(inputs, data)
    with pytest.raises(ValueError, match="feature semantics"):
        run(inputs)


@pytest.mark.parametrize("column", ["source_id", "group"])
def test_group_leakage_is_fatal(inputs, column):
    shared = "source-0" if column == "source_id" else "exposure-reference-0"
    edit_rows(inputs, lambda rows: rows[1].update({column: shared}))
    with pytest.raises(ValueError, match="group leakage"):
        run(inputs)
    assert not (inputs / "out").exists()
    failures = list(inputs.glob(".out.incomplete-*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "INCOMPLETE"


def test_exact_duplicate_kept_but_not_refitted_or_rescored(inputs):
    edit_rows(inputs, lambda rows: rows.append(dict(rows[1])))
    summary = run(inputs)
    assert summary["counts"]["target"]["DUPLICATE_EXACT"] == 1
    repeated = records(inputs)[-1]
    assert repeated["score"] is None
    assert repeated["duplicate_of_row"] == 2
    assert len(records(inputs)) == 41


def test_conflicting_duplicate_refuses(inputs):
    def mutate(rows):
        repeated = dict(rows[1])
        repeated["flux_a"] = "11"
        rows.append(repeated)

    edit_rows(inputs, mutate)
    with pytest.raises(ValueError, match="conflicting immutable source identity"):
        run(inputs)


@pytest.mark.parametrize(
    "field,value,status",
    [
        ("flux_a", "", "MISSING_FEATURES"),
        ("flux_a", "200", "UNSUPPORTED_REGIME"),
        ("flux_a", "NaN", "INVALID_ROW"),
        ("flux_a", "Infinity", "INVALID_ROW"),
        ("sigma", "-1", "INVALID_ROW"),
        ("ra", "360", "INVALID_ROW"),
        ("dec", "91", "INVALID_ROW"),
        ("group", "", "INVALID_ROW"),
    ],
)
def test_bad_or_unsupported_rows_are_retained(inputs, field, value, status):
    edit_rows(inputs, lambda rows: rows[1].update({field: value}))
    run(inputs)
    row = records(inputs)[1]
    assert row["import_status"] == status
    assert row["score"] is None
    assert len(records(inputs)) == 40


def test_reported_uncertainty_is_not_snr_or_calibration(inputs):
    edit_rows(inputs, lambda rows: rows[1].update(sigma=""))
    run(inputs)
    row = records(inputs)[1]
    assert row["features"]["flux_a"]["reported_uncertainty"] is None
    assert row["measurement_status"] == "IMPORTED_REPORTED_UNVERIFIED"
    assert "snr" not in row["features"]


@pytest.mark.parametrize("options", [{"max_rows_per_input": 10}, {"max_work_bytes": 1000}])
def test_limits_never_publish_truncated_complete_run(inputs, options):
    with pytest.raises(ValueError, match="exceeded"):
        run(inputs, **options)
    assert not (inputs / "out").exists()
    assert list(inputs.glob(".out.incomplete-*/failure.json"))


def test_constant_reference_feature_refuses(inputs):
    edit_rows(inputs, lambda rows: [r.update(flux_a="10") for r in rows], "reference")
    with pytest.raises(ValueError, match="constant training feature"):
        run(inputs)


def test_jsonl_invalid_lines_and_float_ids_are_audited(inputs):
    path = inputs / "target.jsonl"
    with (inputs / "target.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["source_id"] = 9007199254740992.0
    path.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"x":1,"x":2}\n')
    data = manifest(inputs)
    data.update(path=path.name, format="jsonl", sha256=file_hash(path))
    replace_manifest(inputs, data)
    summary = run(inputs)
    assert summary["counts"]["target"]["INVALID_ROW"] == 2
    assert len(records(inputs)) == 41


def test_multiline_csv_refuses_instead_of_miscounting_records(inputs):
    edit_rows(inputs, lambda rows: rows[1].update(source_id="contains\nnewline"))
    with pytest.raises(ValueError, match="multiline CSV"):
        run(inputs)


def test_parquet_import_streams_projected_batches(inputs):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    with (inputs / "target.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["unmapped_pixel_blob"] = "not imported"
    path = inputs / "target.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    data = manifest(inputs)
    data.update(path=path.name, format="parquet", sha256=file_hash(path))
    replace_manifest(inputs, data)
    run(inputs)
    assert len(records(inputs)) == 40
    assert "unmapped_pixel_blob" not in (inputs / "out/records.jsonl").read_text()


def test_model_fit_sees_only_reference_rows(inputs, monkeypatch):
    from sklearn.ensemble import IsolationForest

    original = IsolationForest.fit
    seen = []

    def guarded(self, x, *args, **kwargs):
        seen.append(x.copy())
        assert x[:, 0].max() < 13  # target contains engineered flux > 50
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(IsolationForest, "fit", guarded)
    run(inputs)
    assert len(seen) == 1


def test_catalog_inference_never_imports_or_opens_truth(inputs, monkeypatch):
    import builtins

    original_import = builtins.__import__
    original_open = builtins.open

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("evaluation")
        return original_import(name, *args, **kwargs)

    def guarded_open(file, *args, **kwargs):
        assert "evaluator_truth" not in str(file)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "open", guarded_open)
    run(inputs)


def test_known_truth_feature_refuses(inputs):
    data = manifest(inputs)
    data["features"][0]["name"] = "ground_truth"
    with pytest.raises(ValueError, match="truth/label"):
        CatalogManifest.model_validate(data)


def test_repeated_feature_column_refuses(inputs):
    data = manifest(inputs)
    data["features"][1]["column"] = data["features"][0]["column"]
    with pytest.raises(ValueError, match="duplicate source feature column"):
        CatalogManifest.model_validate(data)


def test_cross_calibration_transfer_refuses(inputs):
    data = manifest(inputs)
    data["calibration_refs"] = ["changed-zero-point-reference"]
    replace_manifest(inputs, data)
    with pytest.raises(ValueError, match="calibration references differ"):
        run(inputs)


def test_catalog_cli(inputs):
    command = [
        sys.executable,
        "-m",
        "sieve.cli",
        "catalog",
        "--reference",
        str(inputs / "reference.manifest.json"),
        "--input",
        str(inputs / "target.manifest.json"),
        "--output",
        str(inputs / "cli"),
        "--budget",
        "3",
        "--json-array",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counts"]["output"]["SURFACED"] == 3
    assert (inputs / "cli/records.json").exists()


def test_evaluator_feature_control_and_negative_control(tmp_path):
    from evaluation.catalog_controls import evaluate

    report = evaluate(tmp_path / "controls")
    assert report["status"] == "MEASURED_SYNTHETIC_FEATURE_CONTROL_ONLY"
    assert report["auc_on_scored_feature_control"] > 0.9
    assert report["shuffled_score_negative_control_auc"] < 0.8
    assert report["total_perturbed_rows"] == (
        report["excluded_perturbed_rows"]
        + report["surfaced_perturbed_rows"]
        + report["deferred_perturbed_rows"]
    )
