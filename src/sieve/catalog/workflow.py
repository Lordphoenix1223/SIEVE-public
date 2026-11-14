"""Disk-backed catalog inventory, reference-only model fitting and bounded review.

No external calls, pretrained pickle loading, physical-object identification or
science calibration occurs here. Published output directories are immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sqlite3
import sys
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from sieve.catalog.io import file_hash, load_manifest, table_rows
from sieve.catalog.models import CatalogConfig, CatalogManifest
from sieve.contracts.models import Score, ScoreSemantics
from sieve.provenance.models import content_id

RULE_IDS = tuple(f"M{i:02d}" for i in range(1, 11))
LIMITATIONS = (
    "Catalog measurements are imported reported values; calibration is unverified",
    "Ranking only; no physical identity, discovery, probability or case-level FDR",
    "Declared group disjointness does not prove absence of all scientific leakage",
    "Feature bounds are declared support, not measured sensitivity or completeness",
    "No pixel-level detection, orbit fitting or instrument-response qualification",
    "Model uses mapped values, not reported uncertainties, correlations or upper-limit likelihoods",
    "No million-row or distributed scalability qualification",
    "Per-region caps are a static review policy; there is no persistent aging scheduler",
)


def encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identifier(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("identifiers must be strings or integers, never floats or booleans")
    result = str(value).strip()
    if not result:
        raise ValueError("missing source/split-group identity")
    return result


def number(value: object) -> float:
    if value is None or isinstance(value, (bool, dict, list)):
        raise ValueError("missing or nonscalar numeric value")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError("nonfinite numeric value")
    # scikit-learn internally casts to float32; refuse overflow rather than clip.
    if abs(result) > np.finfo(np.float32).max:
        raise ValueError("numeric value exceeds model float32 representation")
    return result


def normalize(row: dict[str, Any], manifest: CatalogManifest) -> dict[str, Any]:
    source_id = identifier(row[manifest.id_column])
    ra, dec = number(row[manifest.ra_column]), number(row[manifest.dec_column])
    if not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise ValueError("invalid ICRS degree coordinate; no silent wrapping")
    groups = [(g.namespace, identifier(row[g.column])) for g in manifest.split_groups]
    groups.append(("identity:" + manifest.identity_namespace, source_id))
    values: dict[str, Any] = {}
    missing, outside = [], []
    for feature in manifest.features:
        raw = row[feature.column]
        if raw is None or raw == "":
            measured = None
            missing.append(feature.name)
        else:
            measured = number(raw)
            if not feature.lower <= measured <= feature.upper:
                outside.append(feature.name)
        error = None
        if feature.error_column:
            raw_error = row[feature.error_column]
            if raw_error is not None and raw_error != "":
                error = number(raw_error)
                if error < 0:
                    raise ValueError("negative reported standard deviation")
        values[feature.name] = {
            "value": measured,
            "unit": feature.unit,
            "reported_uncertainty": error,
            "uncertainty_kind": feature.uncertainty_kind,
            "missing_reason": "missing imported value" if measured is None else None,
            "uncertainty_missing_reason": "no reported uncertainty" if error is None else None,
        }
    quality = identifier(row[manifest.quality.column]) if manifest.quality else None
    if manifest.quality and quality not in manifest.quality.accepted_values:
        status, reasons = "QUALITY_EXCLUDED", ["quality value outside declared accepted set"]
    elif missing:
        status, reasons = "MISSING_FEATURES", ["missing features: " + ",".join(missing)]
    elif outside:
        status, reasons = "UNSUPPORTED_REGIME", ["outside declared support: " + ",".join(outside)]
    else:
        status, reasons = "ELIGIBLE", ["mapped values within declared support"]
    return {
        "schema_version": "sieve.catalog.v1",
        "record_id": content_id(
            "catalog-row",
            [manifest.identity_namespace, manifest.dataset, manifest.release, source_id],
        ),
        "source_id": source_id,
        "identity_namespace": manifest.identity_namespace,
        "position": {"ra_deg": ra, "dec_deg": dec, "frame": "icrs", "unit": "deg"},
        "features": values,
        "split_groups": groups,
        "quality_value": quality,
        "import_status": status,
        "reasons": reasons,
        "measurement_status": "IMPORTED_REPORTED_UNVERIFIED",
    }


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript(
        """
        PRAGMA cache_size=-8192;
        PRAGMA temp_store=FILE;
        CREATE TABLE inventory (
            role TEXT NOT NULL, ordinal INTEGER NOT NULL, record_id TEXT,
            status TEXT NOT NULL, payload TEXT NOT NULL, digest TEXT,
            region TEXT, score REAL, surfaced INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(role, ordinal));
        CREATE UNIQUE INDEX unique_identity ON inventory(role,record_id)
            WHERE status != 'DUPLICATE_EXACT' AND record_id IS NOT NULL;
        CREATE INDEX eligibility ON inventory(role,status,ordinal);
        CREATE TABLE groups (
            role TEXT NOT NULL, namespace TEXT NOT NULL, value TEXT NOT NULL,
            PRIMARY KEY(role,namespace,value));
        """
    )
    return db


def check_work_limit(work: Path, config: CatalogConfig) -> None:
    size = sum(p.stat().st_size for p in work.iterdir() if p.is_file())
    if size > config.max_work_bytes:
        raise ValueError("max_work_bytes exceeded; incomplete staging retained")


def ingest(
    db: sqlite3.Connection,
    role: str,
    manifest: CatalogManifest,
    asset: Path,
    work: Path,
    config: CatalogConfig,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ordinal, row, error in table_rows(
        asset, manifest, config.batch_rows, config.max_line_bytes
    ):
        if ordinal > config.max_rows_per_input:
            raise ValueError("max_rows_per_input exceeded; no complete output published")
        counts["input_rows"] += 1
        payload: dict[str, Any]
        try:
            if row is None:
                raise ValueError(error or "malformed row")
            payload = normalize(row, manifest)
        except (ValueError, TypeError, OverflowError, KeyError) as exc:
            payload = {
                "schema_version": "sieve.catalog.v1",
                "record_id": None,
                "import_status": "INVALID_ROW",
                "reasons": [str(exc)],
                "measurement_status": "NOT_IMPORTED",
            }
        status = payload["import_status"]
        rid = payload["record_id"]
        digest = hashlib.sha256(encode(payload).encode()).hexdigest()
        if rid is not None:
            existing = db.execute(
                "SELECT digest,ordinal FROM inventory WHERE role=? AND record_id=? "
                "AND status!='DUPLICATE_EXACT'",
                (role, rid),
            ).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError(
                        f"conflicting immutable source identity at {role} row {ordinal}"
                    )
                status = "DUPLICATE_EXACT"
                payload["import_status"] = status
                payload["duplicate_of_row"] = existing[1]
                payload["reasons"] = ["same source identity and identical mapped record"]
        if status == "ELIGIBLE":
            db.executemany(
                "INSERT OR IGNORE INTO groups VALUES(?,?,?)",
                [(role, namespace, value) for namespace, value in payload["split_groups"]],
            )
        region = None
        if "position" in payload:
            pos = payload["position"]
            width = config.region_width_deg
            region = (
                f"icrs-rect:{int(pos['ra_deg'] // width)}:{int((pos['dec_deg'] + 90) // width)}"
            )
        payload["source_locator"] = {"asset_sha256": manifest.sha256, "row_number": ordinal}
        db.execute(
            "INSERT INTO inventory(role,ordinal,record_id,status,payload,digest,region) "
            "VALUES(?,?,?,?,?,?,?)",
            (role, ordinal, rid, status, encode(payload), digest, region),
        )
        counts[status] += 1
        if ordinal % config.batch_rows == 0:
            db.commit()
            check_work_limit(work, config)
    db.commit()
    check_work_limit(work, config)
    return dict(sorted(counts.items()))


def feature_signature(manifest: CatalogManifest) -> list[tuple[object, ...]]:
    return [
        (f.name, f.unit, f.description, f.lower, f.upper, f.uncertainty_kind)
        for f in manifest.features
    ]


def validate_pair(reference: CatalogManifest, target: CatalogManifest) -> None:
    if reference.sha256 == target.sha256:
        raise ValueError("reference and target assets must be distinct; no fit on target")
    if feature_signature(reference) != feature_signature(target):
        raise ValueError("reference/target feature semantics, units or support differ")
    if (reference.provider, reference.dataset, reference.release, reference.support_domain) != (
        target.provider,
        target.dataset,
        target.release,
        target.support_domain,
    ):
        raise ValueError("reference and target must declare the same dataset/release/domain")
    if reference.identity_namespace != target.identity_namespace or {
        g.namespace for g in reference.split_groups
    } != {g.namespace for g in target.split_groups}:
        raise ValueError("reference/target identity and split-group namespaces differ")
    if reference.quality != target.quality:
        raise ValueError("reference/target quality policy differs")
    if reference.calibration_refs != target.calibration_refs:
        raise ValueError("reference/target calibration references differ")
    if reference.data_kind != target.data_kind:
        raise ValueError("reference/target data kind differs; cross-domain transfer is unqualified")


def matrix(payloads: list[str], manifest: CatalogManifest) -> np.ndarray[Any, Any]:
    rows = [json.loads(raw) for raw in payloads]
    return np.asarray(
        [[row["features"][f.name]["value"] for f in manifest.features] for row in rows],
        dtype=np.float64,
    )


def rank(
    db: sqlite3.Connection, reference: CatalogManifest, work: Path, config: CatalogConfig
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise ValueError("ranking requires uv sync --extra catalog") from exc
    overlap = db.execute(
        "SELECT a.namespace FROM groups a JOIN groups b "
        "ON a.namespace=b.namespace AND a.value=b.value "
        "WHERE a.role='reference' AND b.role='target' LIMIT 1"
    ).fetchone()
    if overlap:
        raise ValueError(f"reference/target group leakage in namespace {overlap[0]}")
    # Bottom-k SHA sample is independent of input ordering. Sort spills to disk;
    # only k rows enter Python memory. No target row influences fit selection.
    db.create_function(
        "sample_key", 1, lambda rid: hashlib.sha256(f"{config.seed}:{rid}".encode()).hexdigest()
    )
    selected = db.execute(
        "SELECT record_id,payload FROM inventory WHERE role='reference' AND status='ELIGIBLE' "
        "ORDER BY sample_key(record_id),record_id LIMIT ?",
        (config.training_rows,),
    ).fetchall()
    if len(selected) < 8:
        raise ValueError("at least eight distinct eligible reference rows are required")
    x = matrix([row[1] for row in selected], reference)
    if np.any(np.ptp(x.astype(np.float32).astype(np.float64), axis=0) == 0):
        raise ValueError("constant training feature; revise the frozen feature profile")
    model = IsolationForest(
        n_estimators=config.trees,
        max_samples=min(256, len(selected)),
        contamination="auto",
        random_state=config.seed,
        n_jobs=1,
    )
    model.fit(x)
    info = {
        "method": "sklearn.ensemble.IsolationForest",
        "version": version("scikit-learn"),
        "parameters": model.get_params(),
        "score_transform": "negative score_samples; higher means more unusual",
        "semantics": "RANKING_ONLY",
        "calibrated": False,
        "training_selection": "bottom-k SHA256(seed, stable record ID); reference only",
        "training_record_ids": [row[0] for row in selected],
        "training_matrix_sha256": hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),
        "feature_order": [f.name for f in reference.features],
        "reference_feature_summary": {
            f.name: {
                "unit": f.unit,
                "minimum": float(np.min(x[:, i])),
                "median": float(np.median(x[:, i])),
                "maximum": float(np.max(x[:, i])),
                "description": "descriptive fit-sample statistics; no significance interpretation",
            }
            for i, f in enumerate(reference.features)
        },
        "split_check": "PASS_DECLARED_ELIGIBLE_GROUPS",
        "limitations": list(LIMITATIONS),
    }
    (work / "model.json").write_text(encode(info) + "\n")
    cursor = db.execute(
        "SELECT ordinal,payload FROM inventory WHERE role='target' AND status='ELIGIBLE' "
        "ORDER BY ordinal"
    )
    while batch := cursor.fetchmany(config.batch_rows):
        scores = -model.score_samples(matrix([row[1] for row in batch], reference))
        if not np.isfinite(scores).all():
            raise ValueError("model produced nonfinite scores; no complete output published")
        db.executemany(
            "UPDATE inventory SET score=? WHERE role='target' AND ordinal=?",
            [(float(score), row[0]) for score, row in zip(scores, batch, strict=True)],
        )
        check_work_limit(work, config)
    db.commit()
    db.execute("CREATE INDEX ranking ON inventory(role,score DESC,record_id)")
    # Rank within each rectangle, then take one per rectangle per round. Scores
    # compare one model cohort only; cap + global budget bound the review packet.
    chosen = db.execute(
        "WITH ranked AS (SELECT ordinal,region,score,record_id,ROW_NUMBER() OVER "
        "(PARTITION BY region ORDER BY score DESC,record_id) AS turn "
        "FROM inventory WHERE role='target' AND score IS NOT NULL) "
        "SELECT ordinal FROM ranked WHERE turn<=? ORDER BY turn,score DESC,record_id LIMIT ?",
        (config.per_region, config.review_budget),
    ).fetchall()
    db.executemany("UPDATE inventory SET surfaced=1 WHERE role='target' AND ordinal=?", chosen)
    db.commit()
    return info


def output_record(row: tuple[Any, ...], domain: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(row[0])
    score, surfaced = row[1], row[2]
    status = payload["import_status"]
    payload["score"] = (
        Score(
            value=score,
            semantics=ScoreSemantics.RANKING_ONLY,
            direction="higher",
            domain=domain,
            event_or_null="relative feature-space isolation; no calibrated event or null",
        ).model_dump(mode="json")
        if score is not None
        else None
    )
    payload["decision"] = (
        "SURFACED" if surfaced else "DEFERRED" if score is not None else "EXCLUDED"
    )
    payload["scientific_disposition"] = (
        "UNSUPPORTED_REGIME" if status == "UNSUPPORTED_REGIME" else "INSUFFICIENT_EVIDENCE"
    )
    payload["object_identity_status"] = "NOT_ESTABLISHED"
    payload["review_region"] = row[3]
    payload["evidence_dependence"] = {
        "factor_family": "single_isolation_forest_rank",
        "independent_confirmations": 0,
        "probability_combination": "NOT_APPLICABLE",
    }
    payload["provenance_refs"] = ["receipt.json", "model.json"]
    return payload


def export(
    db: sqlite3.Connection, target: CatalogManifest, work: Path, config: CatalogConfig
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    cases = []
    array_stream = (work / "records.json").open("w") if config.export_json_array else None
    if array_stream:
        array_stream.write("[\n")
    try:
        with (
            (work / "records.jsonl").open("w") as records,
            (work / "rejected.jsonl").open("w") as rejected,
        ):
            cursor = db.execute(
                "SELECT payload,score,surfaced,region FROM inventory WHERE role='target' "
                "ORDER BY ordinal"
            )
            for i, row in enumerate(cursor):
                payload = output_record(row, target.support_domain)
                encoded = encode(payload)
                records.write(encoded + "\n")
                if array_stream:
                    array_stream.write((",\n" if i else "") + encoded)
                counts[payload["decision"]] += 1
                if payload["decision"] == "EXCLUDED":
                    rejected.write(encoded + "\n")
                if payload["decision"] == "SURFACED":
                    cases.append(payload)
                if (i + 1) % config.batch_rows == 0:
                    records.flush()
                    rejected.flush()
                    if array_stream:
                        array_stream.flush()
                    check_work_limit(work, config)
        if array_stream:
            array_stream.write("\n]\n")
    finally:
        if array_stream:
            array_stream.close()
    cases.sort(key=lambda row: (-row["score"]["value"], row["record_id"]))
    (work / "cases.json").write_text(encode({"schema_version": "sieve.catalog.v1", "cases": cases}))
    check_work_limit(work, config)
    return dict(sorted(counts.items()))


def run_catalog(
    reference_path: Path, target_path: Path, output: Path, config: CatalogConfig
) -> dict[str, Any]:
    from sieve.cli import source_identity

    started = datetime.now(UTC)
    timer = time.perf_counter()
    reference, ref_asset, ref_manifest_hash = load_manifest(reference_path)
    target, target_asset, target_manifest_hash = load_manifest(target_path)
    validate_pair(reference, target)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists; choose a new immutable run directory")
    for manifest, asset in ((reference, ref_asset), (target, target_asset)):
        if not asset.is_file() or asset.stat().st_size > config.max_input_bytes:
            raise ValueError("missing input or max_input_bytes exceeded")
        if file_hash(asset) != manifest.sha256:
            raise ValueError("input asset SHA256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_name(f".{output.name}.incomplete-{uuid.uuid4().hex}")
    work.mkdir()
    db: sqlite3.Connection | None = None
    try:
        for role, path in (("reference", reference_path), ("target", target_path)):
            (work / f"{role}.manifest.json").write_bytes(path.read_bytes())
        db = connect(work / "inventory.sqlite")
        ref_counts = ingest(db, "reference", reference, ref_asset, work, config)
        target_counts = ingest(db, "target", target, target_asset, work, config)
        model_info = rank(db, reference, work, config)
        output_counts = export(db, target, work, config)
        db.close()
        db = None
        # Detect changes during processing; nothing becomes COMPLETE beforehand.
        for manifest, asset, mpath, mhash in (
            (reference, ref_asset, reference_path, ref_manifest_hash),
            (target, target_asset, target_path, target_manifest_hash),
        ):
            if file_hash(asset) != manifest.sha256 or file_hash(mpath) != mhash:
                raise ValueError("input or manifest changed during processing")
        commit, tree, dirty, environment = source_identity()
        artifacts = {
            p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in sorted(work.iterdir())
            if p.is_file()
        }
        summary: dict[str, Any] = {
            "schema_version": "sieve.catalog.v1",
            "status": "COMPLETE_CATALOG_RANKING",
            "data_kind": {"reference": reference.data_kind, "target": target.data_kind},
            "counts": {"reference": ref_counts, "target": target_counts, "output": output_counts},
            "review_budget": config.review_budget,
            "model": {
                "method": model_info["method"],
                "training_rows": len(model_info["training_record_ids"]),
            },
            "statistical_filter": {
                "status": "FDR_NOT_APPLICABLE",
                "reason": "no calibrated null family",
            },
            "qualification": {
                "measurement_calibration": "NOT_VERIFIED",
                "real_sensitivity": "NOT_VERIFIED",
                "anomaly_probability": "NOT_VERIFIED",
                "case_level_fdr": "NOT_VERIFIED",
                "physical_identification": "NOT_ESTABLISHED",
                "million_detection_scale": "NOT_VERIFIED",
                "standards_compliance": "NOT_VERIFIED",
            },
            "rules_enforced": list(RULE_IDS),
            "selection_accounting": (
                "all target physical rows retained, including malformed, excluded, "
                "duplicate and deferred rows"
            ),
            "configuration": config.model_dump(mode="json"),
            "limitations": list(LIMITATIONS) + list(target.limitations),
        }
        receipt: dict[str, Any] = {
            "schema_version": "sieve.catalog.v1",
            "activity": "local_catalog_import_reference_fit_rank_export",
            "inputs": {
                "reference": {
                    "asset_sha256": reference.sha256,
                    "manifest_sha256": ref_manifest_hash,
                    "asset_uri": ref_asset.as_uri(),
                    "manifest_uri": reference_path.resolve().as_uri(),
                },
                "target": {
                    "asset_sha256": target.sha256,
                    "manifest_sha256": target_manifest_hash,
                    "asset_uri": target_asset.as_uri(),
                    "manifest_uri": target_path.resolve().as_uri(),
                },
            },
            "artifacts": artifacts,
            "derivations": {
                "model.json": [
                    "reference asset",
                    "reference manifest",
                    "configuration",
                    "code/environment",
                ],
                "records.jsonl": [
                    "target asset",
                    "target manifest",
                    "model.json",
                    "configuration",
                    "code/environment",
                ],
                "cases.json": ["records.jsonl", "configuration"],
            },
            "code_commit": commit,
            "code_tree_sha256": tree,
            "dirty": dirty,
            "environment_id": environment,
            "seed": config.seed,
            "started_at": started.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - timer,
            "resources": {
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024)
                ),
                "peak_rss_scope": "whole process including imports; macOS/Linux units",
                "artifact_bytes_before_receipt": sum(a["bytes"] for a in artifacts.values()),
                "storage_limit_scope": (
                    "staging files checked per batch; overshoot and SQLite temp sort files possible"
                ),
            },
            "configuration": config.model_dump(mode="json"),
            "rules_enforced": list(RULE_IDS),
            "limitations": list(LIMITATIONS),
        }
        (work / "summary.json").write_text(encode(summary) + "\n")
        receipt["artifacts"]["summary.json"] = {
            "sha256": file_hash(work / "summary.json"),
            "bytes": (work / "summary.json").stat().st_size,
        }
        (work / "receipt.json").write_text(encode(receipt) + "\n")
        check_work_limit(work, config)
        # Same-filesystem rename publishes the whole finished run at once. No
        # overwrite is allowed; create a reservation directory to defeat races.
        os.mkdir(output)
        try:
            os.rename(work, output)
        except OSError:
            output.rmdir()  # Only the empty directory created above belongs to us.
            raise
        return {**summary, "output_directory": str(output)}
    except Exception as exc:
        if db is not None:
            db.close()
        (work / "failure.json").write_text(
            encode({"status": "INCOMPLETE", "reason": str(exc), "rules": list(RULE_IDS)}) + "\n"
        )
        raise ValueError(f"{exc}; incomplete evidence retained at {work}") from exc
