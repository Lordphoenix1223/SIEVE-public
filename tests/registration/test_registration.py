"""Geometry and conservative-gate checks use known world-coordinate truth."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS, Sip

from sieve.registration import associate, compare, load_observation, reproject_visualization


def make_image(
    path: Path,
    angle: float = 0,
    scale: float = 1,
    center: float = 30,
    crval: tuple[float, float] = (359.999, 3.0),
    reflect: bool = False,
    sip: bool = False,
) -> None:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"] if sip else ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = crval
    w.wcs.crpix = [center, center]
    theta = np.deg2rad(angle)
    w.wcs.cd = (
        scale / 3600 * np.array([[-np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    )
    if reflect:
        w.wcs.cd[:, 0] *= -1
    if sip:
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        a[2, 0] = 3e-5
        b[0, 2] = -2e-5
        w.sip = Sip(a, b, None, None, w.wcs.crpix)
    h = w.to_header(relax=True)
    h["OBSID"] = path.stem
    h["DETECTOR"] = 1
    h["MJD-AVG"] = 61241.0 + (0.01 if angle else 0)
    h["TIMESYS"] = "UTC"
    h["BUNIT"] = "MJy / sr"
    img = fits.ImageHDU(np.ones((60, 60)), header=h, name="IMAGE")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            img,
            fits.ImageHDU(np.ones((60, 60)), header=h.copy(), name="VARIANCE"),
            fits.ImageHDU(np.zeros((60, 60), dtype=np.int32), header=h.copy(), name="FLAGS"),
        ]
    ).writeto(path)


@pytest.mark.parametrize(
    "angle,scale,reflect,sip",
    [
        (30, 1, False, False),
        (90, 1, False, False),
        (180, 1, False, False),
        (30, 1.3, False, False),
        (30, 1, True, False),
        (30, 1, False, True),
    ],
)
def test_world_matching_handles_geometry_and_ra_wrap(tmp_path, angle, scale, reflect, sip):
    make_image(tmp_path / "a.fits")
    make_image(tmp_path / "b.fits", angle, scale, reflect=reflect, sip=sip)
    a, b = (load_observation(tmp_path / n) for n in ("a.fits", "b.fits"))
    truth_a = np.array([[22, 22], [29, 29], [36, 35], [20, 35]], dtype=float)
    sky = a.wcs.pixel_to_world(truth_a[:, 0], truth_a[:, 1])
    x, y = b.wcs.world_to_pixel(sky)
    truth_b = np.column_stack([x, y])
    matches = associate(a, b, truth_a, truth_b, 0.01)
    assert len(matches["unambiguous_pairs"]) == len(truth_a)
    assert {(r["a"], r["b"]) for r in matches["unambiguous_pairs"]} == set(
        zip(range(4), range(4), strict=True)
    )
    assert max(p["separation_arcsec"] for p in matches["candidate_pairs"]) < 0.001
    assert compare(a, b, matches)["spatial_status"] == "REGISTRATION_UNQUALIFIED"
    assert a.roundtrip_max_pixel < 0.01 and b.roundtrip_max_pixel < 0.01


def test_no_overlap_invalid_wcs_and_ambiguity(tmp_path):
    make_image(tmp_path / "a.fits")
    make_image(tmp_path / "b.fits", crval=(180, 3))
    a, b = (load_observation(tmp_path / n) for n in ("a.fits", "b.fits"))
    assert compare(a, b)["spatial_status"] == "NO_SAMPLED_COMMON_COVERAGE"
    duplicated = np.array([[20.0, 20.0], [20.001, 20.0]])
    matches = associate(a, a, duplicated, duplicated, 1)
    assert len(matches["ambiguous_pairs"]) == 4 and not matches["unambiguous_pairs"]
    assert compare(a, a)["temporal_status"] == "DUPLICATE_EXPOSURE"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((20, 20)), name="IMAGE")]).writeto(
        tmp_path / "bad.fits"
    )
    with pytest.raises(ValueError, match="INVALID_WCS"):
        load_observation(tmp_path / "bad.fits")


def test_visualization_reprojection_never_claims_variance(tmp_path):
    pytest.importorskip("reproject")
    make_image(tmp_path / "a.fits")
    make_image(tmp_path / "b.fits", 30)
    a, b = (load_observation(tmp_path / n) for n in ("a.fits", "b.fits"))
    result = reproject_visualization(a, b, bad_bits=1)
    assert result["status"] == "VISUALIZATION_ONLY" and result["variance"] is None
    assert result["coverage"].sum() > 0 and np.allclose(result["image"][result["coverage"]], 1)


def test_labeled_rotation_avoids_raw_pixel_false_motion(tmp_path):
    from scipy.spatial.distance import cdist

    make_image(tmp_path / "a.fits")
    make_image(tmp_path / "b.fits", 90)
    a, b = (load_observation(tmp_path / n) for n in ("a.fits", "b.fits"))
    xy = np.array([[12.0, 15.0], [18.0, 35.0], [41.0, 22.0], [46.0, 42.0]])
    sky = a.wcs.pixel_to_world(xy[:, 0], xy[:, 1])
    x, y = b.wcs.world_to_pixel(sky)
    other = np.column_stack([x, y])
    raw = cdist(xy, other)
    # A fixed one-pixel gate sees no stationary correspondences after 90 degree rotation.
    assert sum(raw[i, i] <= 1 for i in range(4)) == 0
    matches = associate(a, b, xy, other, 0.01)
    assert {(p["a"], p["b"]) for p in matches["unambiguous_pairs"]} == {(i, i) for i in range(4)}


def test_polar_partial_overlap_and_heldout_shift(tmp_path):
    from sieve.registration.core import reference_residuals

    make_image(tmp_path / "a.fits", crval=(359.999, 89.8))
    make_image(tmp_path / "b.fits", angle=30, center=50, crval=(359.999, 89.8))
    a, b = (load_observation(tmp_path / n) for n in ("a.fits", "b.fits"))
    result = compare(a, b)
    assert 0 < result["sampled_overlap"]["a_in_b_fraction"] < 1
    points = np.array([[15, 15], [20, 20], [25, 25], [30, 30], [35, 35], [40, 40]], float)
    sky = a.wcs.pixel_to_world(points[:, 0], points[:, 1])
    x, y = b.wcs.world_to_pixel(sky)
    other = np.column_stack((x, y))
    valid = (other[:, 0] >= 0) & (other[:, 0] < 60) & (other[:, 1] >= 0) & (other[:, 1] < 60)
    # Use the same frame for the distinct held-out-shift test; no false qualification.
    ref = reference_residuals(a, a, points, points + np.array([0.2, 0.1]))
    assert ref["heldout_count"] == 3 and ref["heldout_median_after_arcsec"] < 0.01
    assert ref["status"] == "REGISTRATION_UNQUALIFIED"
    assert valid.sum() > 0


def test_misaligned_quality_plane_refuses_reprojection(tmp_path):
    pytest.importorskip("reproject")
    path = tmp_path / "mismatch.fits"
    make_image(path)
    with fits.open(path, mode="update") as hdul:
        hdul["FLAGS"].header["CRVAL1"] += 0.01
    observation = load_observation(path)
    assert not observation.plane_wcs_verified
    with pytest.raises(ValueError, match="WCS-verified"):
        reproject_visualization(observation, observation, bad_bits=1)


def test_missing_time_and_invalid_band_units_remain_unqualified(tmp_path):
    path = tmp_path / "unknown.fits"
    make_image(path)
    with fits.open(path) as src:
        hdul = fits.HDUList([hdu.copy() for hdu in src])
    with hdul:
        del hdul["IMAGE"].header["TIMESYS"]
        wave = fits.ImageHDU(np.ones((60, 60)), name="CWAVE")
        wave.header["BUNIT"] = "second"
        width = fits.ImageHDU(np.ones((60, 60)), name="CBAND")
        width.header["BUNIT"] = "micron"
        hdul.extend([wave, width])
        hdul.writeto(path, overwrite=True)
    o = load_observation(path)
    assert o.mjd_mid_utc is None and o.band_um is None
    assert compare(o, o)["spectral_status"] == "INSUFFICIENT_CALIBRATION"
