"""분산 평가 모듈 단위 테스트."""

import numpy as np
import pytest

from tts_auto_eval.distributed import _split_into_chunks, _merge_results


# ---------------------------------------------------------------------------
# _split_into_chunks tests
# ---------------------------------------------------------------------------


def test_split_into_chunks():
    """균등 분할 검증: 8 items / 4 workers → 4 chunks of 2."""
    items = list(range(8))
    chunks = _split_into_chunks(items, 4)
    assert len(chunks) == 4
    assert all(len(c) == 2 for c in chunks)
    # All items present
    flat = [x for c in chunks for x in c]
    assert sorted(flat) == items


def test_split_into_chunks_uneven():
    """비균등 분할 검증: 7 items / 3 workers."""
    items = list(range(7))
    chunks = _split_into_chunks(items, 3)
    assert len(chunks) == 3
    # All items present
    flat = [x for c in chunks for x in c]
    assert sorted(flat) == items
    assert sum(len(c) for c in chunks) == 7


def test_split_single_chunk():
    """단일 워커: 1 chunk."""
    items = list(range(5))
    chunks = _split_into_chunks(items, 1)
    assert len(chunks) == 1
    assert chunks[0] == items


# ---------------------------------------------------------------------------
# _merge_results tests
# ---------------------------------------------------------------------------


def test_merge_results():
    """2개 청크 결과 병합 검증."""
    chunk_a = {
        "metadata": {"model": "test", "total_samples": 2},
        "summary": {
            "wer": {"mean": 0.1, "count": 2},
        },
        "samples": [{"id": "a1"}, {"id": "a2"}],
    }
    chunk_b = {
        "metadata": {"model": "test", "total_samples": 3},
        "summary": {
            "wer": {"mean": 0.3, "count": 3},
        },
        "samples": [{"id": "b1"}, {"id": "b2"}, {"id": "b3"}],
    }

    merged = _merge_results([chunk_a, chunk_b])

    assert len(merged["samples"]) == 5
    assert merged["metadata"]["total_samples"] == 5
    # Mean of means: (0.1 + 0.3) / 2 = 0.2
    assert merged["summary"]["wer"]["mean"] == pytest.approx(0.2)
    assert merged["summary"]["wer"]["count"] == 5


def test_merge_empty():
    """빈 리스트 병합 → 빈 딕셔너리."""
    result = _merge_results([])
    assert result == {}
