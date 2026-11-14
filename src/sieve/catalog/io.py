from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from sieve.catalog.models import CatalogManifest


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def strict_json(raw: str | bytes) -> Any:
    return json.loads(raw, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[CatalogManifest, Path, str]:
    with path.open("rb") as stream:
        raw = stream.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("manifest exceeds the one MB metadata cap")
    manifest = CatalogManifest.model_validate(strict_json(raw))
    asset = Path(manifest.path)
    if not asset.is_absolute():
        asset = path.parent / asset
    return manifest, asset.resolve(), hashlib.sha256(raw).hexdigest()


def bounded_lines(stream: BinaryIO, limit: int) -> Iterator[bytes]:
    while line := stream.readline(limit + 1):
        if len(line) > limit:
            raise ValueError("input line exceeds max_line_bytes; no complete output published")
        yield line


def table_rows(
    path: Path, manifest: CatalogManifest, batch_rows: int, line_limit: int
) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    """Stream projected columns; malformed JSON lines become audit rows.

    CSV requires one physical line per record. Multiline cells refuse because
    they bypass the per-line memory bound. Parquet batches are projected first.
    """
    if manifest.format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("Parquet import requires uv sync --extra catalog") from exc
        table = pq.ParquetFile(path)
        missing = set(manifest.columns()) - set(table.schema_arrow.names)
        if missing:
            raise ValueError(f"missing mapped columns: {sorted(missing)}")
        ordinal = 0
        for batch in table.iter_batches(batch_size=batch_rows, columns=list(manifest.columns())):
            for row in batch.to_pylist():
                ordinal += 1
                yield ordinal, row, None
        return
    with path.open("rb") as stream:
        lines = bounded_lines(stream, line_limit)
        if manifest.format == "csv":
            try:
                header = next(csv.reader([next(lines).decode("utf-8-sig")], strict=True))
            except StopIteration as exc:
                raise ValueError("empty CSV input") from exc
            if len(header) != len(set(header)) or set(manifest.columns()) - set(header):
                raise ValueError("duplicate header or missing mapped columns")
            for ordinal, line in enumerate(lines, 1):
                try:
                    values = next(csv.reader(io.StringIO(line.decode("utf-8")), strict=True))
                    if len(values) != len(header):
                        raise ValueError("CSV row width differs from header")
                    row = dict(zip(header, values, strict=True))
                    yield ordinal, {k: row[k] for k in manifest.columns()}, None
                except (ValueError, csv.Error, UnicodeError) as exc:
                    if isinstance(exc, csv.Error) and "unexpected end of data" in str(exc):
                        raise ValueError(
                            "multiline CSV records are unsupported; export flat rows"
                        ) from exc
                    yield ordinal, None, str(exc)
        else:
            for ordinal, line in enumerate(lines, 1):
                try:
                    row = strict_json(line)
                    if not isinstance(row, dict):
                        raise ValueError("JSONL row must be an object")
                    if set(manifest.columns()) - set(row):
                        raise ValueError("missing mapped columns")
                    yield ordinal, {k: row[k] for k in manifest.columns()}, None
                except (ValueError, UnicodeError) as exc:
                    yield ordinal, None, str(exc)
