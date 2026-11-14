"""Bounded QR3 D1 finite-ePSF flux measurements, not calibrated identity evidence.

Pixels are converted using the released solid angle map. Photutils renders the
released effective PSF; SciPy solves a fixed-position amplitude and constant sky
jointly. Errors are conditional on independent archive pixel variances and an
exact PSF. Unknown real covariance/systematics are not fitted away.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground, StdBackgroundRMS
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import ImagePSF
from photutils.utils.exceptions import NoDetectionsWarning
from scipy.linalg import lstsq

from sieve.photometry.scanner import _BAD_FLAG_BITS

Array = np.ndarray[Any, Any]
SOURCE_BIT = 1 << 21


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class QR3Frame:
    """Verified locally rebuilt, registered science planes and original ePSF zones."""

    path: Path
    sha256: str
    header: fits.Header
    image: Array  # uJy per native pixel, including background
    variance: Array  # uJy squared per native pixel
    flags: Array
    bad: Array
    solid_angle: Array  # arcsec squared
    cwave: Array  # microns, full science calibration rather than WAVE-TAB
    cband: Array
    zodi: Array  # uJy per pixel, diagnostic only, not subtracted twice
    epsf: Array
    oversampling: tuple[int, int]
    origin_xy: tuple[int, int]

    @cached_property
    def wcs(self) -> Any:
        return WCS(self.header).celestial

    def background(self, data: Array | None = None) -> Array:
        """SOURCE is a sky-contamination mask only; bad mask is independent."""
        return np.asarray(
            Background2D(
                self.image if data is None else data,
                box_size=32,
                filter_size=3,
                mask=self.bad | ((self.flags & SOURCE_BIT) != 0),
                exclude_percentile=70,
                bkg_estimator=MedianBackground(),
                bkg_rms_estimator=StdBackgroundRMS(),
            ).background,
            dtype=float,
        )

    def _stamp(
        self, x: float, y: float, size: int = 11
    ) -> tuple[Array, tuple[slice, slice], int, float, float]:
        """Render the full finite ePSF, then retain its overlap with the cutout.

        The EPSF already describes effective pixel response: evaluate it at native
        centers, as in IRSA's tutorial; do not integrate the pixel response twice.
        Unit normalization defines flux within finite released ePSF support. Cropping
        never renormalizes the visible part: that would bias edge fluxes upward.
        """
        if size < 7 or size % 2 != 1 or not np.isfinite([x, y]).all():
            raise ValueError("finite positions and odd stamp size >=7 required")
        half = (size - 1) // 2
        x0, y0 = int(np.floor(x - half)), int(np.floor(y - half))
        ox, oy = self.origin_xy
        distances = (self.epsf["XCENTER"] - x - ox) ** 2 + (self.epsf["YCENTER"] - y - oy) ** 2
        zone = int(np.argmin(distances))
        model = ImagePSF(
            np.asarray(self.epsf["EPSF"][zone], dtype=float), oversampling=self.oversampling
        )
        yy, xx = np.mgrid[y0 : y0 + size, x0 : x0 + size]
        stamp = np.asarray(model.evaluate(xx, yy, flux=1.0, x_0=x, y_0=y), dtype=float)
        total = float(stamp.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("invalid released ePSF normalization")
        stamp /= total
        y_start, x_start = max(y0, 0), max(x0, 0)
        y_stop, x_stop = min(y0 + size, self.image.shape[0]), min(x0 + size, self.image.shape[1])
        if y_start >= y_stop or x_start >= x_stop:
            raise ValueError("PSF stamp does not overlap the cutout")
        cropped = stamp[y_start - y0 : y_stop - y0, x_start - x0 : x_stop - x0]
        local_variance = self.variance[y_start:y_stop, x_start:x_stop]
        finite_variance = np.isfinite(local_variance) & (local_variance > 0)
        if not finite_variance.any():
            raise ValueError("no finite variance overlaps the PSF stamp")
        reference_variance = float(np.median(local_variance[finite_variance]))
        full_variance = np.full(stamp.shape, reference_variance, dtype=float)
        full_variance[y_start - y0 : y_stop - y0, x_start - x0 : x_stop - x0] = np.where(
            finite_variance, local_variance, reference_variance
        )
        total_information = float(np.sum(np.square(stamp) / full_variance))
        visible_variance = np.where(finite_variance, local_variance, reference_variance)
        visible_information = float(
            np.sum(np.square(cropped) / visible_variance) / total_information
        )
        return (
            cropped,
            (slice(y_start, y_stop), slice(x_start, x_stop)),
            zone,
            total_information,
            visible_information,
        )

    def stamp(self, x: float, y: float, size: int = 11) -> tuple[Array, tuple[slice, slice], int]:
        """Render a normalized finite ePSF without renormalizing cutout-edge crops."""
        psf, slices, zone, _, _ = self._stamp(x, y, size)
        return psf, slices, zone


def load_qr3_frame(path: Path, expected_sha256: str) -> QR3Frame:
    """Fail closed on identity, units, registration, detector and invalid calibration."""
    actual = file_digest(path)
    if actual != expected_sha256:
        raise ValueError("pinned QR3 cutout SHA-256 mismatch")
    with fits.open(path, memmap=False) as hdul:
        header = hdul["IMAGE"].header.copy()
        if header.get("DETECTOR") != 1 or header.get("DETCOORD") != "sky":
            raise ValueError("only QR3 D1 sky-coordinate profile is supported")
        if header.get("TIMESYS") != "UTC" or header.get("RADESYS") != "ICRS":
            raise ValueError("UTC and ICRS metadata required")
        if not WCS(header).has_celestial:
            raise ValueError("celestial WCS required")
        image = np.asarray(hdul["IMAGE"].data, dtype=float)
        arrays = {
            n: np.asarray(hdul[n].data, dtype=float)
            for n in ("VARIANCE", "FLAGS", "SOLIDANGLE", "CWAVE", "CBAND", "ZODI")
        }
        if image.ndim != 2 or any(a.shape != image.shape for a in arrays.values()):
            raise ValueError("all science/calibration planes must be registered 2D arrays")
        for name, unit in (
            ("IMAGE", u.MJy / u.sr),
            ("VARIANCE", (u.MJy / u.sr) ** 2),
            ("ZODI", u.MJy / u.sr),
            ("SOLIDANGLE", u.arcsec**2),
            ("CWAVE", u.um),
            ("CBAND", u.um),
        ):
            if u.Unit(hdul[name].header["BUNIT"]) != unit:
                raise ValueError(f"unexpected {name} unit")
        for name in ("SOLIDANGLE", "CWAVE", "CBAND"):
            if not np.all(np.isfinite(arrays[name]) & (arrays[name] > 0)):
                raise ValueError(f"invalid {name} calibration")
            if int(hdul[name].header["DETECTOR"]) != 1:
                raise ValueError(f"{name} detector mismatch")
        points = np.array([0.0, image.shape[1] - 1.0]), np.array([0.0, image.shape[0] - 1.0])
        sky = WCS(header).celestial.pixel_to_world_values(*points)
        for name in arrays:
            plane = hdul[name].header
            if (plane.get("XORIGIN"), plane.get("YORIGIN")) != (
                header["XORIGIN"],
                header["YORIGIN"],
            ):
                raise ValueError(f"{name} detector-origin registration mismatch")
            if not np.allclose(
                WCS(plane).celestial.pixel_to_world_values(*points), sky, rtol=0, atol=1e-10
            ):
                raise ValueError(f"{name} sky WCS registration mismatch")
        flag_values = arrays["FLAGS"]
        if not np.all(np.isfinite(flag_values) & (flag_values >= 0)) or np.any(
            flag_values != np.floor(flag_values)
        ):
            raise ValueError("invalid detector bitmasks")
        flags = flag_values.astype(np.uint64)
        factor = (1 * u.MJy / u.sr).to_value(u.uJy / u.arcsec**2) * arrays["SOLIDANGLE"]
        variance = arrays["VARIANCE"] * factor**2
        flux_image = image * factor
        bad = (
            ~np.isfinite(flux_image)
            | ~np.isfinite(variance)
            | (variance <= 0)
            | ((flags & _BAD_FLAG_BITS) != 0)
        )
        epsf = hdul["EPSF"].data.copy()
        if int(hdul["EPSF"].header["DETECTOR"]) != 1:
            raise ValueError("ePSF detector mismatch")
        oversampling = (int(hdul["EPSF"].header["OVSMPY"]), int(hdul["EPSF"].header["OVSMPX"]))
        if oversampling != (5, 5) or not np.isfinite(epsf["EPSF"]).all():
            raise ValueError("unsupported or invalid ePSF")
    if file_digest(path) != actual:
        raise ValueError("input changed during load")
    return QR3Frame(
        path,
        actual,
        header,
        flux_image,
        variance,
        flags,
        bad,
        arrays["SOLIDANGLE"],
        arrays["CWAVE"],
        arrays["CBAND"],
        arrays["ZODI"] * factor,
        epsf,
        oversampling,
        (int(header["XORIGIN"]), int(header["YORIGIN"])),
    )


def detect_positions(frame: QR3Frame, data: Array, method: Literal["DAO", "IRAF"]) -> Array:
    """Fixed Photutils comparison; no supplied source positions enter detection."""
    threshold = 5 * np.sqrt(np.where(frame.bad, 0.0, frame.variance))
    if method == "DAO":
        finder = DAOStarFinder(
            threshold=threshold,
            fwhm=2.0,
            exclude_border=True,
            sharpness_range=(0.05, 2.0),
            roundness_range=(-2.0, 2.0),
        )
    elif method == "IRAF":
        finder = IRAFStarFinder(
            threshold=threshold,
            fwhm=2.0,
            exclude_border=True,
            sharpness_range=(0.05, 2.0),
            roundness_range=(0.0, 2.0),
        )
    else:
        raise ValueError("unknown detection method")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoDetectionsWarning)
        table = finder(data - frame.background(data), mask=frame.bad)
    if table is None:
        return np.empty((0, 2), dtype=float)
    return np.asarray([table["x_centroid"], table["y_centroid"]], dtype=float).T


def fit_flux(frame: QR3Frame, data: Array, x: float, y: float) -> dict[str, Any]:
    """Weighted amplitude + local constant sky with joint formal covariance.

    Negative amplitudes are retained. Masked-core, neighbor and high residual
    diagnostics do not become physical evidence or silently disappear.
    """
    if data.shape != frame.image.shape:
        raise ValueError("measurement image must match registered frame shape")
    row: dict[str, Any] = {
        "x_pixel": x,
        "y_pixel": y,
        "status": "FAILED",
        "flux_uJy": None,
        "flux_error_uJy": None,
        "limitations": [
            "independent archive variance only",
            "finite ePSF normalization; missing wings unqualified",
            "neighbors not jointly fitted",
            "position fixed; no astrometric covariance",
        ],
    }
    try:
        psf, slices, zone, total_information, visible_information = frame._stamp(x, y)
    except ValueError as exc:
        return {**row, "reason": str(exc)}
    valid = ~frame.bad[slices] & np.isfinite(data[slices])
    # PSF-information coverage rather than counting background pixels as signal.
    local_variance = frame.variance[slices]
    finite_variance = np.isfinite(local_variance) & (local_variance > 0)
    if not finite_variance.any():
        return {**row, "reason": "NO_FINITE_VARIANCE"}
    observed_information = float(np.sum(np.square(psf[valid]) / local_variance[valid]))
    coverage = observed_information / total_information
    row.update(
        epsf_zone=zone,
        valid_pixels=int(valid.sum()),
        masked_pixels=int((~valid).sum()),
        psf_edge_information_coverage=visible_information,
        psf_information_coverage=coverage,
        flag_bits=int(np.bitwise_or.reduce(frame.flags[slices].ravel())),
    )
    if valid.sum() < 25 or coverage < 0.8:
        return {**row, "reason": "INSUFFICIENT_PSF_COVERAGE"}
    error = np.sqrt(frame.variance[slices][valid])
    design = np.column_stack([psf[valid], np.ones(valid.sum())]) / error[:, None]
    target = data[slices][valid] / error
    solution, _, rank, _ = lstsq(design, target, lapack_driver="gelsd")
    if rank != 2:
        return {**row, "reason": "SINGULAR_FIT"}
    covariance = np.linalg.inv(design.T @ design)
    residual = target - design @ solution
    dof = int(valid.sum() - 2)
    ra, dec = frame.wcs.pixel_to_world_values(x, y)
    ix, iy = int(np.rint(x)), int(np.rint(y))
    row.update(
        status="MEASURED_CONDITIONAL",
        reason="fixed-position point-source profile",
        flux_uJy=float(solution[0]),
        flux_error_uJy=float(np.sqrt(covariance[0, 0])),
        background_uJy_per_pixel=float(solution[1]),
        background_error_uJy_per_pixel=float(np.sqrt(covariance[1, 1])),
        flux_background_covariance=float(covariance[0, 1]),
        chi2=float(residual @ residual),
        degrees_of_freedom=dof,
        reduced_chi2=float(residual @ residual / dof),
        cwave_um=float(frame.cwave[iy, ix]),
        cband_um=float(frame.cband[iy, ix]),
        zodi_uJy_per_pixel=float(frame.zodi[iy, ix]),
        ra_deg=float(ra),
        dec_deg=float(dec),
        source_flag_at_centroid=bool(frame.flags[iy, ix] & SOURCE_BIT),
    )
    return row


def fit_flux_group(
    frame: QR3Frame,
    data: Array,
    positions: list[tuple[float, float]],
    *,
    background_order: Literal[0, 1] = 1,
) -> dict[str, Any]:
    """Fit fixed-position ePSF amplitudes together with a shared local sky plane.

    This is a conditional weighted least-squares method. It exposes the full
    amplitude covariance and rank/conditioning diagnostics; it does not assign
    source identities or absorb unknown detector covariance. ``background_order=0``
    gives one shared constant and ``background_order=1`` adds two planar gradients.
    """
    if data.shape != frame.image.shape:
        raise ValueError("measurement image must match registered frame shape")
    if not positions or background_order not in (0, 1):
        raise ValueError("one or more positions and background order 0 or 1 required")
    if not np.isfinite(np.asarray(positions, dtype=float)).all():
        raise ValueError("finite source positions required")

    rendered: list[tuple[Array, tuple[slice, slice], int, float]] = []
    for x, y in positions:
        try:
            psf, slices, zone, total_information, _ = frame._stamp(x, y)
        except ValueError as exc:
            return {
                "status": "FAILED",
                "reason": str(exc),
                "sources": [{"x_pixel": px, "y_pixel": py} for px, py in positions],
                "background_order": background_order,
            }
        rendered.append((psf, slices, zone, total_information))

    y0 = min(slices[0].start for _, slices, _, _ in rendered)
    y1 = max(slices[0].stop for _, slices, _, _ in rendered)
    x0 = min(slices[1].start for _, slices, _, _ in rendered)
    x1 = max(slices[1].stop for _, slices, _, _ in rendered)
    ys, xs = slice(y0, y1), slice(x0, x1)
    shape = (y1 - y0, x1 - x0)
    templates = []
    coverages = []
    for psf, slices, _, total_information in rendered:
        image = np.zeros(shape, dtype=float)
        local_y = slice(slices[0].start - y0, slices[0].stop - y0)
        local_x = slice(slices[1].start - x0, slices[1].stop - x0)
        image[local_y, local_x] = psf
        templates.append(image)
        local_bad = frame.bad[slices] | ~np.isfinite(data[slices])
        local_variance = frame.variance[slices]
        valid_psf = ~local_bad & np.isfinite(local_variance) & (local_variance > 0)
        observed = float(np.sum(np.square(psf[valid_psf]) / local_variance[valid_psf]))
        coverages.append(observed / total_information)

    region_data = data[ys, xs]
    region_variance = frame.variance[ys, xs]
    valid = (
        ~frame.bad[ys, xs]
        & np.isfinite(region_data)
        & np.isfinite(region_variance)
        & (region_variance > 0)
    )
    n_pixels = int(valid.sum())
    n_background = 1 + 2 * background_order
    n_parameters = len(positions) + n_background
    source_rows: list[dict[str, Any]] = [
        {"x_pixel": x, "y_pixel": y, "epsf_zone": item[2], "psf_information_coverage": coverage}
        for (x, y), item, coverage in zip(positions, rendered, coverages, strict=True)
    ]
    base: dict[str, Any] = {
        "status": "FAILED",
        "sources": source_rows,
        "background_order": background_order,
        "valid_pixels": n_pixels,
        "masked_pixels": int(valid.size - n_pixels),
        "fit_window": [int(y0), int(y1), int(x0), int(x1)],
    }
    if n_pixels < max(25, n_parameters + 1):
        return {**base, "reason": "INSUFFICIENT_VALID_PIXELS"}
    if min(coverages) < 0.8:
        return {**base, "reason": "INSUFFICIENT_PSF_COVERAGE"}

    yy, xx = np.mgrid[y0:y1, x0:x1]
    x_center, y_center = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
    x_scale, y_scale = max((x1 - x0) / 2, 1.0), max((y1 - y0) / 2, 1.0)
    columns = [template[valid] for template in templates]
    columns.append(np.ones(n_pixels, dtype=float))
    if background_order == 1:
        columns.extend([((xx - x_center) / x_scale)[valid], ((yy - y_center) / y_scale)[valid]])
    sigma = np.sqrt(region_variance[valid])
    design = np.column_stack(columns) / sigma[:, None]
    target = region_data[valid] / sigma
    solution, _, rank, singular_values = lstsq(design, target, lapack_driver="gelsd")
    measured_condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0
        else None
    )
    if rank != n_parameters or measured_condition is None or measured_condition > 1e12:
        return {
            **base,
            "reason": "RANK_DEFICIENT_OR_ILL_CONDITIONED",
            "rank": int(rank),
            "parameter_count": n_parameters,
            "condition_number": measured_condition,
        }
    covariance = np.linalg.inv(design.T @ design)
    residual = target - design @ solution
    dof = n_pixels - n_parameters
    for index, row in enumerate(source_rows):
        row.update(
            flux_uJy=float(solution[index]),
            flux_error_uJy=float(np.sqrt(covariance[index, index])),
            status="MEASURED_CONDITIONAL",
        )
    background = {
        "intercept_uJy_per_pixel": float(solution[len(positions)]),
        "x_slope_scaled_uJy_per_pixel": 0.0,
        "y_slope_scaled_uJy_per_pixel": 0.0,
    }
    if background_order == 1:
        background["x_slope_scaled_uJy_per_pixel"] = float(solution[-2])
        background["y_slope_scaled_uJy_per_pixel"] = float(solution[-1])
    return {
        **base,
        "status": "MEASURED_CONDITIONAL",
        "reason": "fixed-position joint finite-ePSF amplitudes and shared local sky plane",
        "sources": source_rows,
        "background": background,
        "amplitude_covariance_uJy2": covariance[: len(positions), : len(positions)].tolist(),
        "chi2": float(residual @ residual),
        "degrees_of_freedom": dof,
        "reduced_chi2": float(residual @ residual / dof),
        "condition_number": measured_condition,
        "parameter_count": n_parameters,
    }
