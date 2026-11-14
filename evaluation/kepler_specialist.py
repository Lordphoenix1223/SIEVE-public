"""Preregistered archive-label retrieval benchmark. Excluded from inference wheel.

Train-only model selection; test labels are read only after selection is persisted.
Never select the reported winner from test results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sieve.specialist.features import FEATURE_VERSION, SERIES_LENGTH, features
from sieve.specialist.model import digest, environment, save_model, write_json
from sieve.specialist.workflow import ranking_scores, review_count

SEED = 42
TRAIN_SHA = "0cfe5d5d8fc4aa5f21262e0eb5a60cd997cb1c53314ffa07b59221158426e13c"
# The official published MD5 and byte pins are verified before this SHA is recorded.
TEST_MD5 = "eb9466af83a9de4469b3268c7e5e7fc7"
TEST_SHA = "119d63dddfe5f1fa2ea0399f91a3ca549c2fa1585551c59463cc56a332709b50"
CANDIDATES: tuple[dict[str, Any], ...] = (
    *({"id": f"logistic-C{c}", "kind": "logistic", "C": c} for c in (0.1, 1.0, 10.0)),
    *({"id": f"svc-C{c}", "kind": "svc", "C": c} for c in (0.1, 1.0, 10.0, 100.0)),
    *(
        {"id": f"extra-depth{d}-leaf{leaf}", "kind": "extra", "depth": d, "leaf": leaf}
        for d in (8, None)
        for leaf in (1, 3)
    ),
)
LIMITATIONS = (
    "Archive expert label retrieval; labels are not authoritative physical identity",
    "Preprocessed equal-length benchmark domain only; no raw Fornax transfer qualification",
    "Row-level holdout; original source/exposure groups unavailable",
    "Ranking scores are not calibrated anomaly probabilities or case-level FDR",
    "Small rare-class test denominator; no global best or discovery claim",
)


def build_candidate(spec: dict[str, Any]) -> Any:
    if spec["kind"] == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=spec["C"], class_weight="balanced", max_iter=3000, random_state=SEED
            ),
        )
    if spec["kind"] == "svc":
        return make_pipeline(
            StandardScaler(),
            SVC(C=spec["C"], gamma="scale", class_weight="balanced", random_state=SEED),
        )
    if spec["kind"] == "extra":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=spec["depth"],
            min_samples_leaf=spec["leaf"],
            class_weight="balanced",
            n_jobs=1,
            random_state=SEED,
        )
    raise ValueError("unknown candidate")


def read_ts(path: Path, expected_rows: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rows: list[NDArray[np.float64]] = []
    labels: list[int] = []
    data = False
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not data:
                if line.lower() == "@data":
                    data = True
                continue
            values, label = line.rsplit(":", 1)
            curve = np.fromstring(values, sep=",", dtype=np.float64)
            if len(curve) != SERIES_LENGTH or not np.isfinite(curve).all():
                raise ValueError("unexpected curve length or nonfinite archive values")
            if label not in {str(i) for i in range(1, 8)}:
                raise ValueError("unexpected archive label")
            rows.append(curve)
            labels.append(int(label))
    if len(rows) != expected_rows:
        raise ValueError("unexpected archive row count")
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def curve_hashes(curves: NDArray[np.float64]) -> NDArray[np.str_]:
    return np.asarray([hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in curves])


def metrics(
    y: NDArray[np.int64], scores: NDArray[np.float64], ids: NDArray[np.str_]
) -> dict[str, Any]:
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("invalid score vector")
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError("metrics require both classes")
    order = np.lexsort((ids, -scores))
    budget = review_count(len(y))
    recovered = int(y[order[:budget]].sum())
    # Wilson interval conditional on this test population; no transfer guarantee.
    p, z = recovered / positives, 1.959963984540054
    denom = 1 + z * z / positives
    center = (p + z * z / (2 * positives)) / denom
    radius = z * np.sqrt(p * (1 - p) / positives + z * z / (4 * positives**2)) / denom
    return dict(
        rows=len(y),
        positives=positives,
        budget=budget,
        recovered=recovered,
        recall_at_budget=p,
        precision_at_budget=recovered / budget,
        recall_wilson95=[float(center - radius), float(center + radius)],
        average_precision=float(average_precision_score(y, scores)),
        roc_auc=float(roc_auc_score(y, scores)),
    )


def select_on_train(
    x: NDArray[np.float64], y: NDArray[np.int64], ids: NDArray[np.str_]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(x, y, groups=ids)
    )
    results: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        started = time.perf_counter()
        fold_results = []
        for train, validation in folds:
            if set(ids[train]) & set(ids[validation]):
                raise ValueError("duplicate curves crossed training/validation")
            model = build_candidate(spec)
            model.fit(x[train], y[train])
            fold_results.append(
                metrics(y[validation], ranking_scores(model, x[validation]), ids[validation])
            )
        results.append(
            dict(
                candidate=spec,
                folds=fold_results,
                mean_recall=float(np.mean([f["recall_at_budget"] for f in fold_results])),
                mean_ap=float(np.mean([f["average_precision"] for f in fold_results])),
                seconds=time.perf_counter() - started,
            )
        )
        print(spec["id"], results[-1]["mean_recall"], results[-1]["mean_ap"], flush=True)
    winner = sorted(
        results, key=lambda r: (-r["mean_recall"], -r["mean_ap"], r["candidate"]["id"])
    )[0]["candidate"]
    return winner, results


def freeze_exposure(registry: Path, test_sha: str, protocol_sha: str) -> None:
    existing = json.loads(registry.read_text()) if registry.exists() else {}
    if test_sha in existing and existing[test_sha] != protocol_sha:
        raise ValueError("test already exposed to a different protocol; obtain a fresh holdout")
    existing[test_sha] = protocol_sha
    registry.parent.mkdir(parents=True, exist_ok=True)
    write_json(registry, existing)


def benchmark(inputs: Path, output: Path, registry: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("benchmark output already exists")
    train_path, test_path = (
        inputs / f"KeplerLightCurves_{split}.ts" for split in ("TRAIN", "TEST")
    )
    if digest(train_path) != TRAIN_SHA:
        raise ValueError("training pin mismatch")
    if test_path.stat().st_size != 40431577:
        raise ValueError("test size pin mismatch")
    with test_path.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if md5 != TEST_MD5:
        raise ValueError("official test pin mismatch")
    test_sha = digest(test_path)
    if test_sha != TEST_SHA:
        raise ValueError("test SHA-256 pin mismatch")
    protocol = dict(
        task="archive_label_7_RR_Lyrae_retrieval",
        review_fraction=0.05,
        seed=SEED,
        folds=3,
        splitter="StratifiedGroupKFold(exact-curve-hash)",
        primary="mean fold Recall@ceil(5%); tie mean AP; then candidate ID",
        feature_version=FEATURE_VERSION,
        features_sha256=digest(Path(__file__).parents[1] / "src/sieve/specialist/features.py"),
        evaluator_sha256=digest(Path(__file__)),
        train_sha256=TRAIN_SHA,
        test_sha256=test_sha,
        packages=environment(),
        candidates=CANDIDATES,
        baselines=[
            "seeded_random",
            "spectral_concentration",
            "IsolationForest_reference_only",
            "fixed_balanced_logistic_C1",
            "shuffled_training_labels_control",
        ],
    )
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    # No test values or labels have been parsed. Refuse changed protocols even before training.
    if registry.exists():
        previous = json.loads(registry.read_text()).get(test_sha)
        if previous is not None and previous != protocol_sha:
            raise ValueError("holdout already exposed to a different protocol")
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    started = time.perf_counter()
    train, labels = read_ts(train_path, 920)
    y = (labels == 7).astype(np.int64)
    train_ids = curve_hashes(train)
    x = features(train)
    winner, cv = select_on_train(x, y, train_ids)
    write_json(
        output / "selection.json",
        {
            "selected": winner,
            "cv": cv,
            "training_class_counts": {str(i): int((labels == i).sum()) for i in range(1, 8)},
            "test_labels_accessed": False,
            "protocol_sha256": protocol_sha,
        },
    )
    model = build_candidate(winner)
    model.fit(x, y)
    profile = save_model(
        model,
        output,
        training_sha256=TRAIN_SHA,
        protocol_sha256=protocol_sha,
        selected_candidate=winner["id"],
        limitations=LIMITATIONS,
    )
    # Conservative exposure marking precedes the first test parsing.
    freeze_exposure(registry, test_sha, protocol_sha)
    test, test_labels = read_ts(test_path, 399)
    test_ids = curve_hashes(test)
    if set(train_ids) & set(test_ids):
        raise ValueError("identical curve present in both official splits; qualification refused")
    test_x, test_y = features(test), (test_labels == 7).astype(np.int64)
    score = ranking_scores(model, test_x)
    results = {"selected_train_cv": metrics(test_y, score, test_ids)}
    baseline = build_candidate({"kind": "logistic", "C": 1.0})
    baseline.fit(x, y)
    results["fixed_balanced_logistic_C1"] = metrics(
        test_y, ranking_scores(baseline, test_x), test_ids
    )
    forest = IsolationForest(n_estimators=100, max_samples=256, random_state=SEED, n_jobs=1)
    forest.fit(x)
    results["IsolationForest_reference_only"] = metrics(
        test_y, -forest.score_samples(test_x), test_ids
    )
    results["spectral_concentration"] = metrics(test_y, test_x[:, 34], test_ids)
    random_scores = np.random.default_rng(SEED).random(len(test))
    results["seeded_random"] = metrics(test_y, random_scores, test_ids)
    # Training-label permutation control, not a new selected candidate.
    shuffled = build_candidate(winner)
    shuffled.fit(x, np.random.default_rng(SEED).permutation(y))
    results["shuffled_training_labels_control"] = metrics(
        test_y, ranking_scores(shuffled, test_x), test_ids
    )
    with (
        (output / "test.curves.jsonl").open("w") as curves,
        (output / "test.evaluator-labels.jsonl").open("w") as truth,
        (output / "test.scores.jsonl").open("w") as scores_file,
    ):
        for i, row in enumerate(test):
            identifier = "kepler-test:" + test_ids[i] + f":{i + 1:04}"
            curves.write(
                json.dumps({"id": identifier, "values": row.tolist()}, allow_nan=False) + "\n"
            )
            truth.write(
                json.dumps(
                    {
                        "id": identifier,
                        "archive_label": int(test_labels[i]),
                        "label_scope": "archive-expert-label-proxy",
                    }
                )
                + "\n"
            )
            scores_file.write(
                json.dumps(
                    {"id": identifier, "score": float(score[i]), "score_semantics": "RANKING_ONLY"}
                )
                + "\n"
            )
    write_json(
        output / "test.manifest.json",
        dict(
            schema_version="sieve.curves.v1",
            asset="test.curves.jsonl",
            sha256=digest(output / "test.curves.jsonl"),
            domain=profile.domain,
            preprocessing_ref="https://zenodo.org/records/11218015",
            source_uri="https://zenodo.org/records/11218015",
            limitations=list(LIMITATIONS),
        ),
    )
    from sieve.cli import source_identity

    commit, tree, dirty, env = source_identity()
    report = dict(
        protocol_sha256=protocol_sha,
        selected_candidate=winner,
        qualification="OFFICIAL_ARCHIVE_ROW_HOLDOUT_ONLY",
        test_exposed=True,
        metrics=results,
        seconds=time.perf_counter() - started,
        code_commit=commit,
        code_tree_sha256=tree,
        dirty=dirty,
        environment_id=env,
        profile_sha256=digest(output / "profile.json"),
        limitations=list(LIMITATIONS),
        remote_ci="UNRUN",
    )
    write_json(output / "benchmark.json", report)
    write_json(
        output / "receipt.json",
        {
            "artifacts": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
            "report": report,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=Path("reports/kepler-holdout-registry.json")
    )
    args = parser.parse_args()
    print(json.dumps(benchmark(args.inputs, args.output, args.registry), indent=2))


if __name__ == "__main__":
    main()
