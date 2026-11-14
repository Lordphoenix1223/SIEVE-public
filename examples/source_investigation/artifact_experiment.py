"""Exploratory development/holdout morphology comparison; never a production veto."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.photometry_recovery import match_positions
from sieve.photometry.qualified import detect_positions, fit_flux, load_qr3_frame

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "examples/qualification"
OUT = Path(__file__).with_name("artifact-results.json")
meta = json.loads((BASE / "source.json").read_text())["exposures"][0]
f = load_qr3_frame(BASE / meta["file"], meta["sha256"])
rng = np.random.default_rng(741321)
positions = []
for _ in range(50000):
    x, y = rng.uniform(15, 241, 2)
    ix, iy = round(x), round(y)
    if f.bad[iy - 1 : iy + 2, ix - 1 : ix + 2].any():
        continue
    if all(np.hypot(x - a, y - b) > 12 for a, b in positions):
        positions.append((float(x), float(y)))
    if len(positions) == 64:
        break
if len(positions) != 64:
    raise RuntimeError("Cannot form two disjoint site sets")
noise = rng.normal(size=f.image.shape) * np.sqrt(np.where(f.bad, 0, f.variance))
results = []
for split, points in [("development", positions[:32]), ("heldout", positions[32:])]:
    for arm in ("point_blank", "pixel_blank", "blend_blank", "point_real"):
        data = (f.image if arm == "point_real" else noise).copy()
        for x, y in points:
            reference = fit_flux(f, noise, x, y)
            if reference["flux_error_uJy"] is None:
                raise RuntimeError("Insufficient reference coverage")
            amp = 20 * reference["flux_error_uJy"]
            if arm == "pixel_blank":
                data[round(y), round(x)] += amp
            else:
                psf, sl, _ = f.stamp(x, y)
                data[sl] += amp * psf
                if arm == "blend_blank":
                    psf, sl, _ = f.stamp(x + 1.5, y)
                    data[sl] += amp * psf
        background = f.background(data)
        for method in ("DAO", "IRAF"):
            found = detect_positions(f, data, method)
            associations, _ = match_positions(np.asarray(points), found, radius=2)
            rows = []
            for association in associations:
                if not association["recovered"]:
                    continue
                x, y = found[association["detection_index"]]
                ix, iy = round(x), round(y)
                patch = (data - background)[iy - 1 : iy + 2, ix - 1 : ix + 2]
                if patch.shape != (3, 3) or f.bad[iy - 1 : iy + 2, ix - 1 : ix + 2].any():
                    continue
                denom = float(np.sum(patch))
                if denom <= 0:
                    continue
                rows.append(
                    {
                        "site": association["truth_index"],
                        "x": float(x),
                        "y": float(y),
                        "concentration": float(patch[1, 1] / denom),
                    }
                )
            results.append(
                {
                    "split": split,
                    "arm": arm,
                    "method": method,
                    "sites": 32,
                    "detected": sum(x["recovered"] for x in associations),
                    "feature_available": len(rows),
                    "rows": rows,
                }
            )
for method in ("DAO", "IRAF"):
    dev = next(
        r
        for r in results
        if r["split"] == "development" and r["arm"] == "point_blank" and r["method"] == method
    )
    threshold = float(np.quantile([r["concentration"] for r in dev["rows"]], 0.95, method="higher"))
    for r in results:
        if r["method"] == method:
            r["development_threshold"] = threshold
            r["above_threshold"] = sum(x["concentration"] > threshold for x in r["rows"])
payload = {
    "schema_version": "sieve.artifact_exploration.v1",
    "seed": 741321,
    "input_sha256": meta["sha256"],
    "site_positions": positions,
    "method": "center pixel divided by background-subtracted 3x3 sum; threshold is 95th percentile of detected development blank ePSF sources",  # noqa: E501
    "limitations": [
        "model-matched ePSF injections and fit; no source shot noise or correlated noise",
        "injection sites differ but share one exposure and ePSF calibration",
        "real-field matching can pick pre-existing sources",
        "fixed amplitude level; not equal detected SNR",
        "exploratory threshold, not production validation",
    ],
    "results": results,
}
OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
for r in results:
    print(
        r["split"],
        r["method"],
        r["arm"],
        "detected",
        r["detected"],
        "feature",
        r["feature_available"],
        "above",
        r["above_threshold"],
        "threshold",
        round(r["development_threshold"], 3),
    )
