"""Rebuild the deterministic spatial sample from the pinned public parent FITS.

Download the parent URL in source.json first. This command never downloads data,
overwrites a file, or claims to reconstruct the cutout's spectral WCS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(Path(__file__).with_name("source.json").read_text())
    if args.output.exists():
        raise ValueError("output exists; use a new destination")
    if (
        args.parent.stat().st_size != source["original_product_bytes"]
        or digest(args.parent) != source["original_product_sha256"]
    ):
        raise ValueError("parent size/hash differs from the pinned QR3 product")
    y0, y1 = source["cutout"]["numpy_y_slice"]
    x0, x1 = source["cutout"]["numpy_x_slice"]
    # Astropy's strict Cutout2D center convention matches the stored integer slices.
    position = ((x0 + x1) / 2, (y0 + y1) / 2)
    prefixes = (
        "WCSAXES",
        "CTYPE",
        "CUNIT",
        "CRPIX",
        "CRVAL",
        "CDELT",
        "PC",
        "CD",
        "PV",
        "PS",
        "A_",
        "B_",
        "AP_",
        "BP_",
        "WCSNAME",
        "LONPOLE",
        "LATPOLE",
        "RADESYS",
        "EQUINOX",
    )
    with fits.open(args.parent, memmap=True) as parent:
        spatial = WCS(parent["IMAGE"].header, fobj=parent).celestial
        cutout = Cutout2D(
            parent["IMAGE"].data, position, (y1 - y0, x1 - x0), wcs=spatial, mode="strict"
        )
        assert cutout.slices_original == (slice(y0, y1), slice(x0, x1))
        output = [parent[0].copy()]
        for name in ("IMAGE", "FLAGS", "VARIANCE", "ZODI"):
            header = parent[name].header.copy()
            for key in list(header):
                if key.startswith(prefixes):
                    del header[key]
            header.update(cutout.wcs.to_header(relax=True))
            data = parent[name].data[cutout.slices_original].copy()
            output.append(fits.ImageHDU(data=data, header=header, name=name))
        output.append(parent["WCS-WAVE"].copy())
        # Timestamped checksum comments would make repeated crops byte-different.
        # The sidecar SHA-256 pins the complete output bytes instead.
        for hdu in output:
            for key in ("CHECKSUM", "DATASUM"):
                hdu.header.remove(key, ignore_missing=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(output).writeto(args.output, checksum=False)
    actual = digest(args.output)
    print(f"sample_sha256={actual}")
    if actual != source["cutout"]["sample_sha256"]:
        raise ValueError("rebuilt sample hash differs from the pin; retained output for inspection")


if __name__ == "__main__":
    main()
