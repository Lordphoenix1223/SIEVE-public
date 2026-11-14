"""Bounded, pinned public benchmark acquisition (134 MB); no automatic inference."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from sieve.specialist.model import digest, write_json

PINS = {
    "TRAIN": (93213236, "0cfe5d5d8fc4aa5f21262e0eb5a60cd997cb1c53314ffa07b59221158426e13c"),
    "TEST": (40431577, "119d63dddfe5f1fa2ea0399f91a3ca549c2fa1585551c59463cc56a332709b50"),
}


def acquire(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for split, (size, sha) in PINS.items():
        name = f"KeplerLightCurves_{split}.ts"
        url = f"https://zenodo.org/api/records/11218015/files/{name}/content"
        dest = root / name
        if not dest.exists():
            partial = dest.with_suffix(".partial")
            try:
                total = 0
                with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > size:
                            raise ValueError("unexpected download size")
                        out.write(chunk)
                if total != size or digest(partial) != sha:
                    raise ValueError("download pin mismatch")
                partial.rename(dest)
            finally:
                partial.unlink(missing_ok=True)
        if dest.stat().st_size != size or digest(dest) != sha:
            raise ValueError("existing benchmark file differs from pin")
        with dest.open("rb") as stream:
            md5 = hashlib.file_digest(stream, "md5").hexdigest()
        items.append(dict(split=split, path=name, url=url, bytes=size, sha256=sha, md5=md5))
        print(split, "verified", sha, flush=True)
    write_json(
        root / "acquisition.json",
        dict(
            dataset="KeplerLightCurves",
            record="https://zenodo.org/records/11218015",
            license="CC-BY-4.0",
            files=items,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    acquire(parser.parse_args().output)
