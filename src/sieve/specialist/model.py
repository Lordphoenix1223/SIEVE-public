"""Version-pinned, hash-checked estimator persistence. Never load pickle."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Reviewed scikit-learn Cython tree state used by ExtraTrees. Keep this fixed;
# never trust arbitrary types merely because a model requests them.
# https://skops.readthedocs.io/en/stable/persistence.html
REVIEWED_TYPES = frozenset({"sklearn.tree._tree.Tree"})


class SpecialistProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["sieve.specialist.v1"] = "sieve.specialist.v1"
    task: Literal["archive_label_7_RR_Lyrae_retrieval"] = "archive_label_7_RR_Lyrae_retrieval"
    domain: Literal["KeplerLightCurves-11218015/interpolated-equal-length-v1"] = (
        "KeplerLightCurves-11218015/interpolated-equal-length-v1"
    )
    feature_version: Literal["kepler-rrlyrae-fft-v1"] = "kepler-rrlyrae-fft-v1"
    series_length: Literal[4767] = 4767
    score_semantics: Literal["RANKING_ONLY"] = "RANKING_ONLY"
    model_file: Literal["model.skops"] = "model.skops"
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    features_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate: str
    packages: dict[str, str]
    limitations: tuple[str, ...]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict[str, str]:
    return {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "skops")}


def save_model(model: Any, path: Path, **metadata: Any) -> SpecialistProfile:
    import skops.io as sio

    sio.dump(model, path / "model.skops")
    profile = SpecialistProfile(
        model_sha256=digest(path / "model.skops"),
        features_sha256=digest(Path(__file__).with_name("features.py")),
        packages=environment(),
        **metadata,
    )
    (path / "profile.json").write_text(profile.model_dump_json(indent=2) + "\n")
    return profile


def load_model(path: Path, trusted_profile_sha256: str) -> tuple[Any, SpecialistProfile]:
    import skops.io as sio

    profile_path = path / "profile.json"
    if profile_path.stat().st_size > 64_000:
        raise ValueError("profile size cap exceeded")
    if digest(profile_path) != trusted_profile_sha256:
        raise ValueError("profile does not match independently supplied hash")
    profile = SpecialistProfile.model_validate_json(profile_path.read_bytes())
    if profile.packages != environment():
        raise ValueError("model environment differs from pinned training versions")
    if profile.features_sha256 != digest(Path(__file__).with_name("features.py")):
        raise ValueError("feature implementation differs from trained model")
    model_path = path / profile.model_file
    if model_path.stat().st_size > 100_000_000:
        raise ValueError("model size cap exceeded")
    if digest(model_path) != profile.model_sha256:
        raise ValueError("model hash mismatch")
    unknown = sio.get_untrusted_types(file=model_path)
    unreviewed = set(unknown) - REVIEWED_TYPES
    if unreviewed:
        raise ValueError(f"unreviewed model types refused: {sorted(unreviewed)}")
    return sio.load(model_path, trusted=sorted(REVIEWED_TYPES)), profile


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")
