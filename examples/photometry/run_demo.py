"""Run the pinned public QR3 image-level detection example."""

from __future__ import annotations

import argparse
from pathlib import Path

from sieve.photometry import scan_fits, write_scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qr3-photometry/scan.json"),
        help="new JSON destination (existing outputs are never overwritten)",
    )
    args = parser.parse_args()
    source = Path(__file__).with_name("qr3-d1-cutout.fits")
    result = scan_fits(source, release="QR3", bandpass="D1", background_box_pixels=32)
    write_scan(result, args.output)
    print(
        f"{result.exposure_id} D1: detected={result.detected_peak_count}, "
        f"measured={len(result.detections)}, "
        f"edge_apertures_rejected={result.rejected_truncated_aperture_count}, "
        f"output={args.output}"
    )
    print("Status: measurements only; no candidate identity, motion, or discovery claim.")


if __name__ == "__main__":
    main()
