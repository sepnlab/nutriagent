"""Regression tests for the batch-level safety timeout in main._run_queries."""

from src.main import _batch_timeout


def test_batch_timeout_scales_with_number_of_batches():
    # 54 queries, 4 workers -> 14 batches; each query may take query_timeout.
    assert _batch_timeout(54, 4, 150) == 150 * 14 + 60
    # 6 queries, 4 workers -> 2 batches.
    assert _batch_timeout(6, 4, 150) == 150 * 2 + 60
    # Exactly one batch.
    assert _batch_timeout(4, 4, 150) == 150 + 60


def test_batch_timeout_handles_empty_or_invalid_inputs():
    assert _batch_timeout(0, 4, 150) == 60
    assert _batch_timeout(1, 0, 150) == 60
