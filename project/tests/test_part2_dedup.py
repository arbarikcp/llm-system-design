"""Tests for the Chapter 6 dedup pipeline and Chapter 8 training health checks.

These were illustrations printed only in the book until this audit; they now live
in `llm_platform.training` so the chapters can quote real, tested code.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_platform.training import (  # noqa: E402
    estimated_jaccard,
    find_near_duplicates,
    health_check,
    lsh_bands,
    minhash_signature,
    shingles,
)

DOC_A = "the quick brown fox jumps over the lazy dog near the river bank today"
DOC_B = "the quick brown fox jumps over the lazy dog near the river bank tomorrow"
DOC_C = "completely unrelated text about distributed training and gpu interconnect"


def true_jaccard(x: str, y: str) -> float:
    a, b = shingles(x), shingles(y)
    return len(a & b) / len(a | b) if (a | b) else 0.0


# --------------------------------------------------------------------------- #
# Shingling
# --------------------------------------------------------------------------- #
def test_shingles_produce_overlapping_windows():
    out = shingles("a b c d e", k=3)
    assert out == {"a b c", "b c d", "c d e"}


def test_shingles_of_short_text_is_the_whole_text():
    assert shingles("hi there", k=5) == {"hi there"}


def test_shingles_of_empty_text_is_empty():
    assert shingles("") == set()


def test_shingles_are_case_insensitive():
    assert shingles("The Quick Brown Fox Jumps") == shingles("the quick brown fox jumps")


# --------------------------------------------------------------------------- #
# MinHash
# --------------------------------------------------------------------------- #
def test_signature_has_requested_length():
    assert len(minhash_signature(DOC_A, num_hashes=32)) == 32


def test_signature_is_deterministic():
    assert minhash_signature(DOC_A) == minhash_signature(DOC_A)


def test_identical_documents_have_identical_signatures():
    assert estimated_jaccard(minhash_signature(DOC_A), minhash_signature(DOC_A)) == 1.0


def test_empty_document_yields_zero_signature():
    assert minhash_signature("", num_hashes=8) == [0] * 8


def test_estimate_tracks_true_jaccard_for_near_duplicates():
    est = estimated_jaccard(minhash_signature(DOC_A, 256), minhash_signature(DOC_B, 256))
    assert est == pytest.approx(true_jaccard(DOC_A, DOC_B), abs=0.12)


def test_unrelated_documents_estimate_near_zero():
    est = estimated_jaccard(minhash_signature(DOC_A), minhash_signature(DOC_C))
    assert est < 0.05


def test_estimated_jaccard_handles_empty_signatures():
    assert estimated_jaccard([], [1, 2]) == 0.0


# --------------------------------------------------------------------------- #
# LSH banding
# --------------------------------------------------------------------------- #
def test_bands_partition_the_signature():
    sig = list(range(64))
    bands = lsh_bands(sig, bands=16)
    assert len(bands) == 16
    assert all(len(b) == 4 for b in bands)
    assert [x for band in bands for x in band] == sig


def test_near_duplicates_collide_in_at_least_one_band():
    a, b = minhash_signature(DOC_A), minhash_signature(DOC_B)
    assert set(lsh_bands(a)) & set(lsh_bands(b))


def test_unrelated_documents_do_not_collide():
    a, c = minhash_signature(DOC_A), minhash_signature(DOC_C)
    assert not (set(lsh_bands(a)) & set(lsh_bands(c)))


def test_lsh_bands_handles_degenerate_input():
    assert lsh_bands([], bands=4) == []
    assert lsh_bands([1, 2, 3], bands=0) == []


# --------------------------------------------------------------------------- #
# End-to-end dedup
# --------------------------------------------------------------------------- #
def test_find_near_duplicates_pairs_the_similar_documents_only():
    dupes = find_near_duplicates({"a": DOC_A, "b": DOC_B, "c": DOC_C}, threshold=0.5)
    assert [(x, y) for x, y, _ in dupes] == [("a", "b")]
    assert dupes[0][2] > 0.5


def test_threshold_filters_out_weak_matches():
    assert find_near_duplicates({"a": DOC_A, "b": DOC_B}, threshold=0.99) == []


def test_no_duplicates_in_a_distinct_corpus():
    assert find_near_duplicates({"a": DOC_A, "c": DOC_C}) == []


def test_results_are_sorted_by_descending_similarity():
    docs = {"a": DOC_A, "b": DOC_B, "d": DOC_A + " extra words here", "c": DOC_C}
    sims = [s for _, _, s in find_near_duplicates(docs, threshold=0.3)]
    assert sims == sorted(sims, reverse=True)


# --------------------------------------------------------------------------- #
# Chapter 8 health checks
# --------------------------------------------------------------------------- #
KW = dict(num_params=7e9, num_gpus=8, gpu_peak_flops=9.9e14)
# 6*N*tokens/s / (gpus*peak_flops) ~= 0.45 MFU, a realistic healthy step.
HEALTHY = {"tokens_per_sec": 85_000, "grad_norm": 1.0}


def test_healthy_fixture_is_actually_healthy():
    from llm_platform.training import mfu

    assert 0.4 < mfu(tokens_per_sec=HEALTHY["tokens_per_sec"], **KW) < 0.5


def test_healthy_step_raises_no_alerts():
    assert health_check(HEALTHY, **KW) == []


def test_low_mfu_is_alerted():
    alerts = health_check({"tokens_per_sec": 1e3, "grad_norm": 1.0}, **KW)
    assert any("MFU" in a for a in alerts)


def test_grad_norm_spike_is_alerted():
    alerts = health_check({**HEALTHY, "grad_norm": 50.0}, **KW)
    assert any("grad-norm" in a for a in alerts)


def test_ecc_errors_are_alerted():
    alerts = health_check({**HEALTHY, "ecc_errors": 2}, **KW)
    assert any("ECC" in a for a in alerts)


def test_all_three_failures_reported_together():
    alerts = health_check(
        {"tokens_per_sec": 1e3, "grad_norm": 50.0, "ecc_errors": 1}, **KW)
    assert len(alerts) == 3


def test_missing_optional_metrics_do_not_crash():
    assert health_check({"tokens_per_sec": HEALTHY["tokens_per_sec"]}, **KW) == []


def test_thresholds_are_configurable():
    assert health_check(HEALTHY, mfu_floor=0.99, **KW)
    assert health_check({**HEALTHY, "grad_norm": 2.0}, grad_norm_ceiling=1.0, **KW)
