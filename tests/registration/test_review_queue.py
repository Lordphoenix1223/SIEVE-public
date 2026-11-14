"""Ensure the review aid preserves uncertain and flagged evidence."""
from examples.registration.review_queue import build


def test_queue_preserves_source_flags_and_defers_missing_measurements():
    def measurement(status, flags):
        return {"status": status, "flag_bits": flags}

    comparison = {
        "schema_version": "sieve.qr3.comparison.v1",
        "geometry": {"inputs": [{"exposure_id": "a"}, {"exposure_id": "b"}],
                     "source_motion_status": "REGISTRATION_UNQUALIFIED",
                     "flux_variability_status": "INSUFFICIENT_CALIBRATION"},
        "measurements": [[measurement("MEASURED_CONDITIONAL", 1 << 21),
                          measurement("MEASURED_CONDITIONAL", 1)],
                         [measurement("MEASURED_CONDITIONAL", 1 << 21),
                          measurement("FAILED", 0)]],
        "pairings": {"candidate_pairs": [{"a": 0, "b": 0, "ambiguous": False},
                                         {"a": 1, "b": 1, "ambiguous": True}]},
        "detections_per_exposure": [2, 2],
        "provenance": {"input_sha256": ["a", "b"]},
    }
    queue = build(comparison, "hash")
    assert len(queue["records"]) == 2
    assert queue["records"][0]["review_priority"] == "review_measurements"
    assert queue["records"][1]["review_priority"] == "deferred"
    assert "multiple_positional_pairings" in queue["records"][1]["reasons"]
    assert queue["counts"]["catalog_associated_pairs"] is None
    assert all(r["scientific_eligibility"]["motion"] == "INSUFFICIENT_EVIDENCE"
               for r in queue["records"])


def test_shared_candidates_do_not_confirm_identity():
    def measurement():
        return {"status": "MEASURED_CONDITIONAL", "flag_bits": 1 << 21}

    comparison = {
        "schema_version": "sieve.qr3.comparison.v1",
        "geometry": {"inputs": [{"exposure_id": "a"}, {"exposure_id": "b"}],
                     "source_motion_status": "REGISTRATION_UNQUALIFIED",
                     "flux_variability_status": "INSUFFICIENT_CALIBRATION"},
        "measurements": [[measurement()], [measurement()]],
        "pairings": {"candidate_pairs": [{"a": 0, "b": 0, "ambiguous": False}]},
        "detections_per_exposure": [1, 1],
        "provenance": {"input_sha256": ["a", "b"]},
    }
    identities = {"catalog_sha256": "hash", "catalog_scope": "partial", "results": [
        {"detection_id": f"measurement:{obs}:00000", "status": "inconclusive",
         "candidates": [{"catalog": "Gaia", "id": "x"}]}
        for obs in ("a", "b")
    ]}
    result = build(comparison, "hash", identities)
    assert result["counts"]["catalog_associated_pairs"] == 1
    assert result["records"][0]["catalog_association"] == "SHARED_CANDIDATE_UNCONFIRMED"
    assert result["records"][0]["identify"][0]["status"] == "inconclusive"
    assert result["counts"]["science_eligible_pairs"] == 0
