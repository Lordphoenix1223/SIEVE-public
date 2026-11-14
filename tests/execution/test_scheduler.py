from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from sieve.execution import ResourceBudget, WorkUnit, run_work


def _unit(unit_id: str, *, exploration: bool = False, value: float = 1.0) -> WorkUnit:
    return WorkUnit(
        unit_id=unit_id,
        sky_region="nside8:4",
        exposure_id="exposure-1",
        wavelength="band-1",
        investigation="unit-test",
        scientific_value=value,
        exploration=exploration,
        input_sha256=("a" * 64,),
        estimated_input_bytes=20,
        estimated_cpu_seconds=0.1,
        estimated_rss_bytes=10,
        estimated_output_bytes=1024,
        payload={"value": unit_id},
    )


def test_budget_priority_preserves_exploration_and_resume(tmp_path: Path) -> None:
    units = [_unit("high", value=10), _unit("low", value=1), _unit("explore", exploration=True)]
    calls: list[str] = []

    def worker(unit: WorkUnit) -> dict[str, str]:
        calls.append(unit.unit_id)
        return {"value": unit.unit_id}

    checkpoint = tmp_path / "checkpoint"
    first = run_work(
        units,
        worker,
        checkpoint_dir=checkpoint,
        budget=ResourceBudget(max_tasks=2, exploration_fraction=0.5),
    )
    assert first["unit_order"] == ["high", "explore", "low"]
    assert {row["unit_id"] for row in first["completed"]} == {"high", "explore"}
    assert first["unstarted"] == ["low"]
    assert calls == ["high", "explore"]

    second = run_work(
        units,
        worker,
        checkpoint_dir=checkpoint,
        budget=ResourceBudget(max_tasks=2, exploration_fraction=0.5),
    )
    assert set(second["reused_checkpoints"]) == {"high", "explore"}
    assert sorted(calls) == ["explore", "high", "low"]
    assert second["status"] == "COMPLETE"
    assert all(row["resources"]["process_peak_rss_bytes"] > 0 for row in second["completed"])


def test_changed_unit_and_corrupt_checkpoint_are_recomputed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    unit = _unit("same")
    calls = 0

    def worker(_: WorkUnit) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"calls": calls}

    budget = ResourceBudget(max_tasks=1)
    run_work([unit], worker, checkpoint_dir=checkpoint, budget=budget)
    path = next(checkpoint.glob("*.json"))
    record = json.loads(path.read_text())
    record["result_sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    run_work([unit], worker, checkpoint_dir=checkpoint, budget=budget)
    assert calls == 2

    changed = WorkUnit(**{**unit.__dict__, "scientific_value": 9.0})
    run_work([changed], worker, checkpoint_dir=checkpoint, budget=budget)
    assert calls == 3
    assert len(list(checkpoint.glob("*.json"))) == 2


def test_worker_failure_is_retained_and_output_budget_stops(tmp_path: Path) -> None:
    def worker(unit: WorkUnit) -> dict[str, str]:
        if unit.unit_id == "bad":
            raise RuntimeError("fixture failure")
        time.sleep(0.01)
        return {"value": unit.unit_id}

    result = run_work(
        [_unit("bad", value=2), _unit("later")],
        worker,
        checkpoint_dir=tmp_path / "checkpoints",
        budget=ResourceBudget(max_tasks=2, max_workers=1),
    )
    assert result["status"] == "BUDGET_LIMITED"
    assert result["stop_reason"] == "WORKER_FAILED"
    failed = [row for row in result["completed"] if row["status"] == "FAILED"]
    assert failed and failed[0]["error"]["message"] == "fixture failure"


def test_resource_estimates_gate_storage_and_memory(tmp_path: Path) -> None:
    unit = _unit("large")
    storage = run_work(
        [unit],
        lambda _: {"ok": True},
        checkpoint_dir=tmp_path / "storage",
        budget=ResourceBudget(max_tasks=1, max_output_bytes=100),
    )
    assert storage["stop_reason"] == "STORAGE_ESTIMATE_BUDGET"
    assert storage["completed"] == []

    memory = run_work(
        [unit],
        lambda _: {"ok": True},
        checkpoint_dir=tmp_path / "memory",
        budget=ResourceBudget(max_tasks=1, max_rss_bytes=9),
    )
    assert memory["stop_reason"] == "RSS_ESTIMATE_BUDGET"
    assert memory["completed"] == []


def test_input_reference_requires_hex_sha256() -> None:
    unit = _unit("invalid")
    with pytest.raises(ValueError, match="SHA-256 hex"):
        WorkUnit(**{**unit.__dict__, "input_sha256": ("g" * 64,)})


def test_worker_count_bounds_parallel_local_execution(tmp_path: Path) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def worker(unit: WorkUnit) -> dict[str, str]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"value": unit.unit_id}

    result = run_work(
        [_unit(f"parallel-{index}") for index in range(4)],
        worker,
        checkpoint_dir=tmp_path / "parallel",
        budget=ResourceBudget(max_tasks=4, max_workers=2),
    )
    assert result["status"] == "COMPLETE"
    assert peak == 2
    assert len(result["completed"]) == 4
