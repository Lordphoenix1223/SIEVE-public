"""Shared source/environment identity; no benchmark or evaluator dependencies."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

from sieve.provenance.models import content_id


def source_identity() -> tuple[str, str, bool, str]:
    package = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes())
    root = package.parents[1]
    commit, dirty = "UNKNOWN_EXPORTED_SOURCE", True
    if (root / ".git").exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    installed = sorted((d.metadata["Name"], d.version) for d in distributions())
    lock = root / "uv.lock"
    environment = content_id(
        "env",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": installed,
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        },
    )
    return commit, digest.hexdigest(), dirty, environment
