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
