"""Verify pinned public parents and rebuild bounded science-profile cutouts.

Network access is opt-in. Existing bytes are verified, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def registered_header(
    original: fits.Header, celestial: fits.Header, x0: int, y0: int
) -> fits.Header:
    """Every cropped plane uses the same sky WCS and full-detector coordinate offset.

    Original WAVE-TAB references are removed because the visualization table is
    absent. CWAVE/CBAND pixel *values* retain the science wavelength information.
    """
    header = original.copy()
    wcs_key = re.compile(
        r"(WCSAXES|WCSNAME|CNAME\d|CRPIX\d|CUNIT\d|CTYPE\d|CRVAL\d|CDELT\d|"
        r"(?:PC|CD|PS|PV)\d+_\d+|LONPOLE|LATPOLE)[A-Z]?"
    )
    sip_key = re.compile(r"(A|B|AP|BP)_(ORDER|DMAX|\d+_\d+)")
    for key in list(header):
        if wcs_key.fullmatch(key) or sip_key.fullmatch(key):
            del header[key]
    header.update(celestial)
    header["XORIGIN"], header["YORIGIN"] = x0, y0
    for key, value in {
        "WCSAXESA": 2,
        "CRPIX1A": 1 - x0,
        "CRPIX2A": 1 - y0,
        "CRVAL1A": 0.0,
        "CRVAL2A": 0.0,
        "CDELT1A": 1.0,
        "CDELT2A": 1.0,
        "CUNIT1A": "pixel",
        "CUNIT2A": "pixel",
        "CTYPE1A": "LINEAR",
        "CTYPE2A": "LINEAR",
        "WCSNAMEA": "0-based active pixel coordinates",
    }.items():
        header[key] = value
    return header


def acquire(manifest_path: Path, raw_dir: Path, output_dir: Path, download: bool) -> None:
    manifest = json.loads(manifest_path.read_text())
    raw_dir.mkdir(parents=True, exist_ok=True)
    parents = {}
    for asset in manifest["parents"]:
        path = raw_dir / asset["url"].rsplit("/", 1)[1]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path}; use --download or supply pinned raw files")
            # A failed/partial transfer remains inspectable and will fail the next hash check.
            with urllib.request.urlopen(asset["url"], timeout=60) as response, path.open("xb") as f:
                while block := response.read(1024 * 1024):
                    f.write(block)
        if path.stat().st_size != asset["size"] or digest(path) != asset["sha256"]:
            raise ValueError(f"parent identity mismatch: {path}")
        parents[asset["id"]] = path
    output_dir.mkdir(parents=True, exist_ok=True)
    center = SkyCoord(*manifest["sky_center_deg"], unit="deg")
    with (
        fits.open(parents["solid_angle"]) as sapm,
        fits.open(parents["spectral_wcs"]) as swcs,
    ):
        for exposure in manifest["exposures"]:
            output = output_dir / exposure["file"]
            if output.exists():
                if digest(output) != exposure["sha256"]:
                    raise ValueError(f"existing cutout identity mismatch: {output}")
                continue
            with fits.open(parents[exposure["parent_id"]]) as raw:
                image = raw["IMAGE"]
                crop = Cutout2D(
                    image.data, center, 256, wcs=WCS(image.header).celestial, mode="strict"
                )
                slices = crop.slices_original
                y0, x0 = slices[0].start, slices[1].start
                if [x0, y0] != exposure["origin_xy"]:
                    raise ValueError("cutout coordinates differ from pinned protocol")
                celestial = crop.wcs.to_header(relax=True)
                header = registered_header(image.header, celestial, x0, y0)
                hdus = [
                    fits.PrimaryHDU(header=raw[0].header),
                    fits.ImageHDU(crop.data, header, name="IMAGE"),
                ]
                for name in ("FLAGS", "VARIANCE", "ZODI"):
                    hdr = registered_header(raw[name].header, celestial, x0, y0)
                    hdus.append(fits.ImageHDU(raw[name].data[slices], hdr, name=name))
                for parent, name, target in (
                    (sapm, "IMAGE", "SOLIDANGLE"),
                    (swcs, "CWAVE", "CWAVE"),
                    (swcs, "CBAND", "CBAND"),
                ):
                    hdus.append(
                        fits.ImageHDU(
                            np.asarray(parent[name].data[slices]),
                            registered_header(parent[name].header, celestial, x0, y0),
                            name=target,
                        )
                    )
                hdus.append(raw["EPSF"].copy())
                for hdu in hdus:
                    for key in ("CHECKSUM", "DATASUM"):
                        hdu.header.remove(key, ignore_missing=True)
                fits.HDUList(hdus).writeto(output, overwrite=False)
                if digest(output) != exposure["sha256"]:
                    raise ValueError(f"rebuilt cutout identity mismatch: {output}")
            print(f"VERIFIED {output} {digest(output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    acquire(Path(__file__).with_name("source.json"), args.raw_dir, args.output_dir, args.download)
