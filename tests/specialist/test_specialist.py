"""Behavior and adversarial boundary tests; no network or official test labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.kepler_specialist import freeze_exposure, metrics, select_on_train
from sieve.specialist.features import DOMAIN, SERIES_LENGTH, features
from sieve.specialist.model import digest, load_model, save_model, write_json
from sieve.specialist.workflow import ranking_scores, run_curves


def curves(n: int = 12) -> np.ndarray:
    t = np.arange(SERIES_LENGTH)
    return np.stack(
        [np.sin(t * (0.02 + i * 0.012)) + 0.1 * np.sin(t * 0.003 * i) for i in range(n)]
    )


@pytest.fixture
def fitted(tmp_path: Path) -> tuple[Path, str, object]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features(curves()), np.array([0] * 6 + [1] * 6))
    save_model(
        model,
        model_dir,
        training_sha256="1" * 64,
        protocol_sha256="2" * 64,
        selected_candidate="test-only",
        limitations=("Test model only",),
    )
    return model_dir, digest(model_dir / "profile.json"), model


def manifest(tmp_path: Path, rows: list[object]) -> Path:
    asset = tmp_path / "curves.jsonl"
    asset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path = tmp_path / "input.json"
    write_json(
        path,
        dict(
            asset=asset.name,
            sha256=digest(asset),
            domain=DOMAIN,
            preprocessing_ref="test",
            source_uri="fixture:test",
            limitations=["Synthetic"],
        ),
    )
    return path


def test_features_are_independent_of_other_curves_and_affine_scale() -> None:
    x = curves()
    np.testing.assert_allclose(features(x[:1]), features(x)[:1], atol=1e-10)
    np.testing.assert_allclose(features(7 * x + 200), features(x), atol=1e-8)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((1, SERIES_LENGTH)),
        np.ones((1, SERIES_LENGTH - 1)),
        np.full((1, SERIES_LENGTH), np.nan),
    ],
)
def test_degenerate_and_missing_curves_refused(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        features(bad)


def test_safe_persistence_prediction_parity_and_anchor(fitted: tuple) -> None:
    path, sha, original = fitted
    loaded, _ = load_model(path, sha)
    np.testing.assert_array_equal(
        ranking_scores(loaded, features(curves())), ranking_scores(original, features(curves()))
    )
    with pytest.raises(ValueError, match="independently"):
        load_model(path, "0" * 64)
    (path / "model.skops").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model hash"):
        load_model(path, sha)


def test_wrong_environment_refused(fitted: tuple) -> None:
    path, _, _ = fitted
    profile = json.loads((path / "profile.json").read_text())
    profile["packages"]["scikit-learn"] = "0.0"
    write_json(path / "profile.json", profile)
    with pytest.raises(ValueError, match="environment"):
        load_model(path, digest(path / "profile.json"))


def test_complete_export_preserves_rejections_and_duplicate_ids(
    tmp_path: Path, fitted: tuple
) -> None:
    rows = [{"id": f"s:{i}", "values": row.tolist()} for i, row in enumerate(curves(5))]
    rows += [
        rows[0],
        {"id": "constant", "values": [0] * SERIES_LENGTH},
        {"id": "label-leak", "values": curves(1)[0].tolist(), "label": 7},
    ]
    input_path = manifest(tmp_path, rows)
    path, sha, _ = fitted
    output = tmp_path / "run"
    report = run_curves(input_path, path, output, sha, budget=2, json_array=True)
    assert report["counts"] == dict(rows=8, ranked=5, rejected=3, surfaced=2, deferred=3)
    records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert records == json.loads((output / "records.json").read_text())
    assert len(records) == 8
    assert all(r["object_identity"] == "NOT_ESTABLISHED" for r in records)
    assert all("label" not in r for r in records)
    assert len(json.loads((output / "cases.json").read_text())) == 2
    receipt = json.loads((output / "receipt.json").read_text())
    assert all(digest(output / name) == sha for name, sha in receipt["artifacts"].items())


def test_failed_cap_does_not_publish_output(tmp_path: Path, fitted: tuple) -> None:
    input_path = manifest(
        tmp_path, [{"id": str(i), "values": row.tolist()} for i, row in enumerate(curves(2))]
    )
    path, sha, _ = fitted
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="row cap"):
        run_curves(input_path, path, output, sha, max_rows=1)
    assert not output.exists()
    assert not list(tmp_path.glob(".curves-*"))


def test_tampered_input_refused(tmp_path: Path, fitted: tuple) -> None:
    input_path = manifest(tmp_path, [])
    (tmp_path / "curves.jsonl").write_text("{}\n")
    path, sha, _ = fitted
    with pytest.raises(ValueError, match="hash"):
        run_curves(input_path, path, tmp_path / "run", sha)


def test_domain_refused(tmp_path: Path, fitted: tuple) -> None:
    input_path = manifest(tmp_path, [])
    data = json.loads(input_path.read_text())
    data["domain"] = "Fornax-raw-not-qualified"
    write_json(input_path, data)
    path, sha, _ = fitted
    with pytest.raises(ValueError):
        run_curves(input_path, path, tmp_path / "run", sha)


def test_exact_budget_and_deterministic_ties() -> None:
    y = np.zeros(21, dtype=np.int64)
    y[0] = 1
    result = metrics(y, np.zeros(21), np.asarray([f"{i:03}" for i in range(21)]))
    assert result["budget"] == 2
    assert result["recovered"] == 1
    assert result["precision_at_budget"] == 0.5
    assert result["recall_wilson95"][0] < 0.3


def test_holdout_registry_rejects_changed_search(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    freeze_exposure(path, "test", "first")
    freeze_exposure(path, "test", "first")
    with pytest.raises(ValueError, match="fresh holdout"):
        freeze_exposure(path, "test", "different-search")


def test_selection_uses_only_fold_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.kepler_specialist as benchmark

    monkeypatch.setattr(
        benchmark,
        "CANDIDATES",
        (
            {"id": "a", "kind": "logistic", "C": 0.1},
            {"id": "b", "kind": "logistic", "C": 1.0},
        ),
    )
    x = features(curves(30))
    y = np.asarray([0, 1] * 15, dtype=np.int64)
    ids = np.asarray([str(i) for i in range(30)])
    winner, results = select_on_train(x, y, ids)
    expected = sorted(
        results, key=lambda r: (-r["mean_recall"], -r["mean_ap"], r["candidate"]["id"])
    )[0]["candidate"]
    assert winner == expected
    assert all(len(result["folds"]) == 3 for result in results)
    assert all(sum(f["rows"] for f in result["folds"]) == 30 for result in results)


def test_unreviewed_model_types_are_refused(fitted: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    import skops.io as sio

    path, sha, _ = fitted
    monkeypatch.setattr(sio, "get_untrusted_types", lambda **kwargs: ["unreviewed.CustomModel"])
    with pytest.raises(ValueError, match="unreviewed model types"):
        load_model(path, sha)


def test_changed_feature_implementation_is_refused(fitted: tuple) -> None:
    path, _, _ = fitted
    profile = json.loads((path / "profile.json").read_text())
    profile["features_sha256"] = "0" * 64
    write_json(path / "profile.json", profile)
    with pytest.raises(ValueError, match="feature implementation"):
        load_model(path, digest(path / "profile.json"))


@pytest.mark.parametrize("kind", ["extra", "svc"])
def test_other_search_families_persistence_parity(tmp_path: Path, kind: str) -> None:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import SVC

    model = (
        ExtraTreesClassifier(n_estimators=5, max_depth=2, random_state=42)
        if kind == "extra"
        else make_pipeline(StandardScaler(), SVC())
    )
    x = features(curves())
    model.fit(x, np.array([0] * 6 + [1] * 6))
    save_model(
        model,
        tmp_path,
        training_sha256="1" * 64,
        protocol_sha256="2" * 64,
        selected_candidate="test-only",
        limitations=("Test model",),
    )
    loaded, _ = load_model(tmp_path, digest(tmp_path / "profile.json"))
    np.testing.assert_array_equal(ranking_scores(model, x), ranking_scores(loaded, x))
