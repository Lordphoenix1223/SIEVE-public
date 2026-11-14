"""A deliberately narrow SPHEREx MEF source-detection baseline.

This uses Photutils DAOFIND and circular aperture sums. It is useful for
reproducible engineering and inspection, but it is not the SPHEREx source
discovery or spectrophotometry pipeline.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import photutils
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground, StdBackgroundRMS
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sieve.provenance.runtime import source_identity


class PhotometryScan(BaseModel):
    """Portable output for one single-plane detection/measurement run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["sieve.photometry.scan.v0"] = "sieve.photometry.scan.v0"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_name: str
    release: str | None
    bandpass: str | None
    hdu: str
    data_unit: str
    exposure_id: str | None
    observation_mjd: float | None
    detector: int | None
    code_commit: str
    source_tree_sha256: str
    dirty_source: bool
    environment_id: str
    method: str = "photutils.DAOStarFinder+CircularAperture"
    method_version: str
    config: dict[str, int | float]
    background_median: float
    background_rms_median: float
    masked_fraction: float = Field(ge=0.0, le=1.0)
    background_masked_fraction: float = Field(ge=0.0, le=1.0)
    detected_peak_count: int = Field(ge=0)
    known_source_flagged_peak_count: int = Field(ge=0)
    unflagged_peak_count: int = Field(ge=0)
    known_source_flagged_measurement_count: int = Field(ge=0)
    rejected_truncated_aperture_count: int = Field(ge=0)
    detections: tuple[dict[str, Any], ...]
    scientific_status: Literal["MEASUREMENTS_ONLY_NOT_IDENTITY"] = "MEASUREMENTS_ONLY_NOT_IDENTITY"
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def finite_measurements(self) -> PhotometryScan:
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self


# These flags indicate pixels excluded from this exploratory measurement.
# SOURCE (bit 21) is deliberately not included: it denotes flux from known
# sources and does not mean a bad detector pixel. Unknown bits are retained.
_BAD_FLAG_BITS = (
    (1 << 0)  # TRANSIENT
    | (1 << 1)  # OVERFLOW
    | (1 << 2)  # SUR_ERROR
    | (1 << 6)  # NONFUNC
    | (1 << 7)  # DICHROIC
    | (1 << 9)  # MISSING_DATA
    | (1 << 10)  # HOT
    | (1 << 11)  # COLD
    | (1 << 14)  # PHANMISS
    | (1 << 15)  # NONLINEAR
    | (1 << 17)  # PERSIST
    | (1 << 19)  # OUTLIER
    | (1 << 20)  # CROSSTALK
    | (1 << 22)  # GHOST
    | (1 << 24)  # GHOST_EXT
    | (1 << 25)  # STREAK
    | (1 << 26)  # BLOOM
    | (1 << 27)  # SNOWBALL
    | (1 << 28)  # HALO
    | (1 << 29)  # SATELLITE_HALO
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image(hdul: fits.HDUList, name: str) -> np.ndarray[Any, Any]:
    hdu = hdul[name]
    if hdu.data is None or np.ndim(hdu.data) != 2:
        raise ValueError(f"HDU {name!r} must contain a two-dimensional image")
    return np.asarray(hdu.data, dtype=np.float64)


def scan_fits(
    path: Path,
    *,
    image_hdu: str = "IMAGE",
    flags_hdu: str = "FLAGS",
    variance_hdu: str = "VARIANCE",
    release: str | None = None,
    bandpass: str | None = None,
    fwhm_pixels: float = 2.0,
    threshold_sigma: float = 5.0,
    aperture_radius_pixels: float = 3.0,
    background_box_pixels: int = 64,
) -> PhotometryScan:
    """Detect point-like peaks and measure aperture sums in one 2D FITS HDU.

    The output aperture value has units ``<BUNIT> pixel``. For SPHEREx IMAGE
    planes this is a sum of MJy/sr pixel values, not source flux density: the
    detector-specific solid-angle calibration map is not applied here.
    """
    parameters = (fwhm_pixels, threshold_sigma, aperture_radius_pixels)
    if not all(np.isfinite(value) and value > 0 for value in parameters):
        raise ValueError("FWHM, threshold and aperture radius must be positive")
    if background_box_pixels < 8:
        raise ValueError("background boxes must be at least 8 pixels wide")
    source = path.resolve()
    source_sha256 = _sha256(source)
    with fits.open(source, memmap=True) as hdul:
        image_hdu_obj = hdul[image_hdu]
        image = _image(hdul, image_hdu)
        flag_values = _image(hdul, flags_hdu)
        if not np.all(np.isfinite(flag_values) & (flag_values >= 0)) or np.any(
            flag_values != np.floor(flag_values)
        ):
            raise ValueError("FLAGS must contain finite nonnegative integer bitmasks")
        flags = flag_values.astype(np.uint64)
        variance = _image(hdul, variance_hdu)
        if image.shape != flags.shape or image.shape != variance.shape:
            raise ValueError("IMAGE, FLAGS and VARIANCE shapes must match")
        if min(image.shape) < background_box_pixels:
            raise ValueError("image is smaller than the requested background box")
        bad = (
            ~np.isfinite(image)
            | ~np.isfinite(variance)
            | (variance <= 0)
            | ((flags & _BAD_FLAG_BITS) != 0)
        )
        # The SPHEREx SOURCE flag marks flux from known objects. Do not drop
        # those pixels from detection, but do keep them out of the sky estimate.
        background_mask = bad | ((flags & (1 << 21)) != 0)
        if np.count_nonzero(~bad) < max(100, image.size // 20):
            raise ValueError("fewer than 5% of image pixels remain usable after masks")

        background = Background2D(
            image,
            box_size=background_box_pixels,
            filter_size=(3, 3),
            exclude_percentile=70.0,
            mask=background_mask,
            bkg_estimator=MedianBackground(),
            bkg_rms_estimator=StdBackgroundRMS(),
        )
        residual = image - background.background
        error = np.sqrt(np.where(bad, 0.0, variance))
        finder = DAOStarFinder(
            fwhm=fwhm_pixels,
            threshold=threshold_sigma * error,
            exclude_border=True,
            sharpness_range=(0.05, 2.0),
            roundness_range=(-2.0, 2.0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            found = finder(residual, mask=bad)
            unflagged_found = finder(residual, mask=background_mask)
        wcs = WCS(image_hdu_obj.header, fobj=hdul).celestial
        if not wcs.has_celestial:
            raise ValueError("IMAGE must have a celestial WCS")
        detected_peak_count = 0 if found is None else len(found)
        known_source_flagged_peak_count = 0
        unflagged_peak_count = 0 if unflagged_found is None else len(unflagged_found)
        rejected_truncated_aperture_count = 0
        if found is None:
            rows: tuple[dict[str, Any], ...] = ()
        else:
            positions = np.column_stack((found["x_centroid"], found["y_centroid"]))
            peak_x = np.clip(np.rint(positions[:, 0]).astype(int), 0, image.shape[1] - 1)
            peak_y = np.clip(np.rint(positions[:, 1]).astype(int), 0, image.shape[0] - 1)
            known_source_flagged_peak_count = int(
                np.count_nonzero((flags[peak_y, peak_x] & (1 << 21)) != 0)
            )
            full_aperture = (
                (positions[:, 0] >= aperture_radius_pixels)
                & (positions[:, 0] < image.shape[1] - aperture_radius_pixels)
                & (positions[:, 1] >= aperture_radius_pixels)
                & (positions[:, 1] < image.shape[0] - aperture_radius_pixels)
            )
            rejected_truncated_aperture_count = int(np.count_nonzero(~full_aperture))
            positions = positions[full_aperture]
            found = found[full_aperture]
            if len(positions) == 0:
                rows = ()
            else:
                apertures = CircularAperture(positions, r=aperture_radius_pixels)
                phot = aperture_photometry(
                    residual,
                    apertures,
                    error=error,
                    mask=bad,
                    method="exact",
                )
                aperture_masks = apertures.to_mask(method="exact")
                coords = wcs.pixel_to_world(positions[:, 0], positions[:, 1])
                made: list[dict[str, Any]] = []
                for i, (detection, coordinate, aperture_mask) in enumerate(
                    zip(found, coords, aperture_masks, strict=True)
                ):
                    aperture_sum = float(phot["aperture_sum"][i])
                    aperture_error = float(phot["aperture_sum_err"][i])
                    x_pixel = float(detection["x_centroid"])
                    y_pixel = float(detection["y_centroid"])
                    x_index = min(max(int(round(x_pixel)), 0), image.shape[1] - 1)
                    y_index = min(max(int(round(y_pixel)), 0), image.shape[0] - 1)
                    row_flags = int(flags[y_index, x_index])
                    image_slices, mask_slices = aperture_mask.get_overlap_slices(image.shape)
                    if mask_slices is None or image_slices is None:
                        raise ValueError("full aperture unexpectedly has no image overlap")
                    aperture_weights = aperture_mask.data[mask_slices]
                    aperture_flag_values = flags[image_slices][aperture_weights > 0]
                    aperture_flag_bits = int(
                        np.bitwise_or.reduce(aperture_flag_values, initial=np.uint64(0))
                    )
                    aperture_bad_area = float(np.sum(aperture_weights[bad[image_slices]]))
                    aperture_bad_fraction = aperture_bad_area / float(np.sum(aperture_weights))
                    made.append(
                        {
                            "source_id": f"{source.name}:{i:05d}",
                            "x_pixel": x_pixel,
                            "y_pixel": y_pixel,
                            "ra_deg": float(coordinate.ra.deg),
                            "dec_deg": float(coordinate.dec.deg),
                            "peak_image_value": float(detection["peak"]),
                            "aperture_sum": aperture_sum,
                            "aperture_sum_unit": (
                                f"{image_hdu_obj.header.get('BUNIT', 'unknown')} pixel"
                            ),
                            "aperture_sum_uncertainty": aperture_error,
                            "aperture_snr_from_variance": aperture_sum / aperture_error
                            if aperture_error > 0
                            else None,
                            "centroid_flag_bits": row_flags,
                            "aperture_flag_bits": aperture_flag_bits,
                            "aperture_bad_pixel_fraction": aperture_bad_fraction,
                            "aperture_usable": aperture_bad_fraction == 0.0,
                            "centroid_masked": bool(bad[y_index, x_index]),
                        }
                    )
                rows = tuple(made)
        bunit = str(image_hdu_obj.header.get("BUNIT", "unknown"))

    if _sha256(source) != source_sha256:
        raise ValueError("input FITS bytes changed during the scan")
    commit, tree, dirty, environment = source_identity()
    return PhotometryScan(
        source_sha256=source_sha256,
        source_name=source.name,
        release=release,
        bandpass=bandpass,
        hdu=image_hdu,
        data_unit=bunit,
        exposure_id=(
            str(image_hdu_obj.header["OBSID"]) if "OBSID" in image_hdu_obj.header else None
        ),
        observation_mjd=(
            float(image_hdu_obj.header["MJD-OBS"]) if "MJD-OBS" in image_hdu_obj.header else None
        ),
        detector=(
            int(image_hdu_obj.header["DETECTOR"]) if "DETECTOR" in image_hdu_obj.header else None
        ),
        code_commit=commit,
        source_tree_sha256=tree,
        dirty_source=dirty,
        environment_id=environment,
        method_version=f"Photutils {photutils.__version__}",
        config={
            "fwhm_pixels": fwhm_pixels,
            "threshold_sigma": threshold_sigma,
            "aperture_radius_pixels": aperture_radius_pixels,
            "background_box_pixels": background_box_pixels,
        },
        background_median=float(np.nanmedian(background.background)),
        background_rms_median=float(np.nanmedian(background.background_rms)),
        masked_fraction=float(np.mean(bad)),
        background_masked_fraction=float(np.mean(background_mask)),
        detected_peak_count=detected_peak_count,
        known_source_flagged_peak_count=known_source_flagged_peak_count,
        unflagged_peak_count=unflagged_peak_count,
        known_source_flagged_measurement_count=sum(
            bool(int(row["centroid_flag_bits"]) & (1 << 21)) for row in rows
        ),
        rejected_truncated_aperture_count=rejected_truncated_aperture_count,
        detections=rows,
        limitations=(
            "Experimental point-source baseline, not SPHEREx pipeline extraction",
            "Aperture sums are not converted with the detector solid-angle calibration map",
            "No PSF fitting, crowded-field deblending, completeness or purity calibration",
            "Detection and measurements do not establish source identity or motion",
            "SOURCE mask pixels influence background estimation; no independent discovery test",
            "Aperture sums omit masked pixels; affected rows are retained and marked unusable",
            "Errors omit background-estimation, covariance, PSF and systematic uncertainty",
            "DAO uses archive pixel sigma with kernel scaling; thresholds are uncalibrated",
            "Aperture usability means mask completeness only, not scientific qualification",
        ),
    )


def write_scan(scan: PhotometryScan, output: Path) -> None:
    """Write a deterministic JSON export without overwriting an existing file."""
    if output.exists():
        raise ValueError("output exists; immutable measurement exports cannot be overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scan.model_dump(mode="json"), sort_keys=True, indent=2) + "\n")
