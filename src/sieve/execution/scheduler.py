"""Small, resumable scheduler for source-pinned scientific work units.

The scheduler is intentionally local-first. It bounds concurrency and admits work
against declared resource estimates, then records measured wall/CPU/RSS/I/O/output
costs. Workers must be deterministic and side-effect-free; checkpoints are written
by the scheduler only after a worker returns successfully.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class WorkUnit:
    """One deterministic unit partitioned by sky, exposure, band and investigation."""

    unit_id: str
    sky_region: str
    exposure_id: str
    wavelength: str
    investigation: str
    scientific_value: float
    exploration: bool
    input_sha256: tuple[str, ...]
    estimated_input_bytes: int
    estimated_cpu_seconds: float
    estimated_rss_bytes: int
    estimated_output_bytes: int
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.unit_id or not self.investigation:
            raise ValueError("unit_id and investigation must be non-empty")
        if not math.isfinite(self.scientific_value) or not math.isfinite(
            self.estimated_cpu_seconds
        ):
            raise ValueError("value and resource estimates must be finite")
        if (
            min(
                self.scientific_value,
                self.estimated_input_bytes,
                self.estimated_cpu_seconds,
                self.estimated_rss_bytes,
                self.estimated_output_bytes,
            )
            < 0
        ):
            raise ValueError("value and resource estimates must be nonnegative")
        if any(len(digest) != 64 for digest in self.input_sha256):
            raise ValueError("each input reference must be a SHA-256 hex digest")
        try:
            for digest in self.input_sha256:
                int(digest, 16)
        except ValueError as exc:
            raise ValueError("each input reference must be a SHA-256 hex digest") from exc

    @property
    def definition_sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class ResourceBudget:
    """Run limits; estimates gate admission and measurements are retained per unit."""

    max_tasks: int
    max_workers: int = 1
    max_cpu_seconds: float = 3600.0
    max_wall_seconds: float = 3600.0
    max_rss_bytes: int = 8 * 1024**3
    max_input_bytes: int = 8 * 1024**3
    max_output_bytes: int = 1024**3
    exploration_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.max_tasks < 1 or self.max_workers < 1:
            raise ValueError("max_tasks and max_workers must be positive")
        if not 0 <= self.exploration_fraction <= 1:
            raise ValueError("exploration_fraction must be in [0, 1]")
        if (
            min(
                self.max_cpu_seconds,
                self.max_wall_seconds,
                self.max_rss_bytes,
                self.max_input_bytes,
                self.max_output_bytes,
            )
            < 0
        ):
            raise ValueError("resource limits must be nonnegative")
        if not math.isfinite(self.max_cpu_seconds) or not math.isfinite(self.max_wall_seconds):
            raise ValueError("time budgets must be finite")


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported Unix targets report KiB.
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def _ordered(units: tuple[WorkUnit, ...], reserve: float) -> list[WorkUnit]:
    ranked = sorted(units, key=lambda u: (-u.scientific_value, u.unit_id))
    explore = [u for u in ranked if u.exploration]
    exploit = [u for u in ranked if not u.exploration]
    if not explore or reserve == 0:
        return ranked
    if reserve == 1:
        return explore + exploit
    interval = max(1, round(1 / reserve))
    result: list[WorkUnit] = []
    while exploit or explore:
        for _ in range(interval - 1):
            if exploit:
                result.append(exploit.pop(0))
        if explore:
            result.append(explore.pop(0))
        elif exploit:
            result.append(exploit.pop(0))
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(encoded)


def _checkpoint_path(directory: Path, unit: WorkUnit) -> Path:
    return directory / f"{_digest(unit.unit_id)[:12]}-{unit.definition_sha256[:12]}.json"


def _read_checkpoint(path: Path, unit: WorkUnit) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        if record["unit_definition_sha256"] != unit.definition_sha256:
            return None
        result = record["result"]
        if record["result_sha256"] != _digest(result):
            return None
        if record["status"] != "COMPLETED":
            return None
        return cast(dict[str, Any], record)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _run_one(worker: Callable[[WorkUnit], dict[str, Any]], unit: WorkUnit) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.thread_time()
    try:
        result = worker(unit)
        if not isinstance(result, dict):
            raise TypeError("scientific workers must return a JSON object")
        json.dumps(result, sort_keys=True, allow_nan=False)
        return {
            "status": "COMPLETED",
            "result": result,
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.thread_time() - cpu_started,
            "process_peak_rss_bytes": _rss_bytes(),
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.thread_time() - cpu_started,
            "process_peak_rss_bytes": _rss_bytes(),
        }


def run_work(
    units: list[WorkUnit],
    worker: Callable[[WorkUnit], dict[str, Any]],
    *,
    checkpoint_dir: Path,
    budget: ResourceBudget,
) -> dict[str, Any]:
    """Execute an ordered prefix that fits estimates, checkpoint, and resume safely.

    Checkpoints are only reused when both the complete work-unit definition and
    result digest match. Actual limit overruns are retained and stop later admission.
    A worker must poll its own domain-specific deadline for hard interruption.
    """
    if len({u.unit_id for u in units}) != len(units):
        raise ValueError("unit_id values must be unique")
    started = time.perf_counter()
    process_cpu_started = time.process_time()
    ordered = _ordered(tuple(units), budget.exploration_fraction)
    completed: list[dict[str, Any]] = []
    skipped: list[str] = []
    deferred: list[str] = []
    admitted_input = admitted_cpu = admitted_output = output_bytes = actual_cpu = 0.0
    admitted_tasks = 0
    stop_reason: str | None = None
    next_index = 0
    pool = ThreadPoolExecutor(max_workers=budget.max_workers)
    pending: dict[Any, WorkUnit] = {}

    def submit(unit: WorkUnit) -> None:
        pending[pool.submit(_run_one, worker, unit)] = unit

    try:
        while next_index < len(ordered) or pending:
            while next_index < len(ordered) and len(pending) < budget.max_workers:
                unit = ordered[next_index]
                next_index += 1
                path = _checkpoint_path(checkpoint_dir, unit)
                cached = _read_checkpoint(path, unit)
                if cached is not None:
                    skipped.append(unit.unit_id)
                    completed.append(cached)
                    output_bytes += path.stat().st_size
                    continue
                if admitted_tasks >= budget.max_tasks:
                    next_index -= 1
                    stop_reason = "MAX_TASKS"
                    break
                if admitted_input + unit.estimated_input_bytes > budget.max_input_bytes:
                    next_index -= 1
                    stop_reason = "INPUT_BUDGET"
                    break
                if admitted_cpu + unit.estimated_cpu_seconds > budget.max_cpu_seconds:
                    next_index -= 1
                    stop_reason = "CPU_ESTIMATE_BUDGET"
                    break
                if admitted_output + unit.estimated_output_bytes > budget.max_output_bytes:
                    next_index -= 1
                    stop_reason = "STORAGE_ESTIMATE_BUDGET"
                    break
                reserved_rss = sum(u.estimated_rss_bytes for u in pending.values())
                if reserved_rss + unit.estimated_rss_bytes > budget.max_rss_bytes:
                    next_index -= 1
                    stop_reason = "RSS_ESTIMATE_BUDGET"
                    break
                if time.perf_counter() - started >= budget.max_wall_seconds:
                    next_index -= 1
                    stop_reason = "WALL_BUDGET"
                    break
                admitted_input += unit.estimated_input_bytes
                admitted_cpu += unit.estimated_cpu_seconds
                admitted_output += unit.estimated_output_bytes
                admitted_tasks += 1
                submit(unit)
            if not pending:
                break
            done, _ = wait_first(pending)
            for future in done:
                unit = pending.pop(future)
                metrics = future.result()
                result_bytes = len(json.dumps(metrics, sort_keys=True, allow_nan=False).encode())
                record = {
                    "schema_version": "sieve.work-unit-checkpoint.v1",
                    "unit_id": unit.unit_id,
                    "unit_definition_sha256": unit.definition_sha256,
                    "input_sha256": list(unit.input_sha256),
                    "partition": {
                        "sky_region": unit.sky_region,
                        "exposure_id": unit.exposure_id,
                        "wavelength": unit.wavelength,
                        "investigation": unit.investigation,
                    },
                    "status": metrics["status"],
                    "result": metrics.get("result"),
                    "error": metrics.get("error"),
                    "result_sha256": _digest(metrics.get("result", metrics.get("error"))),
                    "resources": {
                        "wall_seconds": metrics["wall_seconds"],
                        "cpu_seconds": metrics["cpu_seconds"],
                        "process_peak_rss_bytes": metrics["process_peak_rss_bytes"],
                        "input_bytes": unit.estimated_input_bytes,
                        "output_bytes": result_bytes,
                        "estimated_cpu_seconds": unit.estimated_cpu_seconds,
                        "estimated_rss_bytes": unit.estimated_rss_bytes,
                        "estimated_output_bytes": unit.estimated_output_bytes,
                    },
                }
                path = _checkpoint_path(checkpoint_dir, unit)
                checkpoint_bytes = _atomic_json(path, record)
                output_bytes += checkpoint_bytes
                actual_cpu += metrics["cpu_seconds"]
                completed.append(record)
                if metrics["status"] == "FAILED":
                    stop_reason = "WORKER_FAILED"
                elif output_bytes > budget.max_output_bytes:
                    stop_reason = "OUTPUT_BUDGET_OVERRUN"
                elif metrics["process_peak_rss_bytes"] > budget.max_rss_bytes:
                    stop_reason = "RSS_BUDGET_OVERRUN"
                elif actual_cpu > budget.max_cpu_seconds:
                    stop_reason = "CPU_BUDGET_OVERRUN"
                elif time.perf_counter() - started > budget.max_wall_seconds:
                    stop_reason = "WALL_BUDGET_OVERRUN"
                if stop_reason:
                    break
            if stop_reason:
                for future in pending:
                    if future.cancel():
                        deferred.append(pending[future].unit_id)
                # Running threads cannot be killed safely. Wait for them and preserve
                # their completed results rather than dropping finished science work.
                for future, unit in list(pending.items()):
                    if future.cancelled():
                        continue
                    metrics = future.result()
                    record = {
                        "schema_version": "sieve.work-unit-checkpoint.v1",
                        "unit_id": unit.unit_id,
                        "unit_definition_sha256": unit.definition_sha256,
                        "input_sha256": list(unit.input_sha256),
                        "partition": {
                            "sky_region": unit.sky_region,
                            "exposure_id": unit.exposure_id,
                            "wavelength": unit.wavelength,
                            "investigation": unit.investigation,
                        },
                        "status": metrics["status"],
                        "result": metrics.get("result"),
                        "error": metrics.get("error"),
                        "result_sha256": _digest(metrics.get("result", metrics.get("error"))),
                        "resources": {
                            "wall_seconds": metrics["wall_seconds"],
                            "cpu_seconds": metrics["cpu_seconds"],
                            "process_peak_rss_bytes": metrics["process_peak_rss_bytes"],
                            "input_bytes": unit.estimated_input_bytes,
                            "output_bytes": len(json.dumps(metrics, sort_keys=True).encode()),
                            "estimated_cpu_seconds": unit.estimated_cpu_seconds,
                            "estimated_rss_bytes": unit.estimated_rss_bytes,
                            "estimated_output_bytes": unit.estimated_output_bytes,
                        },
                    }
                    output_bytes += _atomic_json(_checkpoint_path(checkpoint_dir, unit), record)
                    completed.append(record)
                    actual_cpu += metrics["cpu_seconds"]
                pending.clear()
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    return {
        "schema_version": "sieve.execution.run.v1",
        "status": "BUDGET_LIMITED" if stop_reason else "COMPLETE",
        "stop_reason": stop_reason,
        "unit_order": [u.unit_id for u in ordered],
        "completed": sorted(completed, key=lambda row: row["unit_id"]),
        "reused_checkpoints": skipped,
        "unstarted": [u.unit_id for u in ordered[next_index:]] + deferred,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - process_cpu_started,
            "worker_cpu_seconds": actual_cpu,
            "process_peak_rss_bytes": _rss_bytes(),
            "admitted_input_bytes": admitted_input,
            "admitted_cpu_seconds": admitted_cpu,
            "admitted_output_bytes": admitted_output,
            "checkpoint_bytes": output_bytes,
            "max_workers": budget.max_workers,
            "budget": asdict(budget),
        },
    }


def wait_first(pending: dict[Any, WorkUnit]) -> tuple[set[Any], set[Any]]:
    """Small named wrapper to keep the completion policy explicit and testable."""
    return {next(as_completed([*pending]))}, set()
