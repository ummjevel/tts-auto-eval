"""프로파일링 메트릭 테스트."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tts_auto_eval.metrics.base import MetricResult
from tts_auto_eval.metrics.profiling import ProfilingMetric


def test_name():
    """name == 'profiling'."""
    metric = ProfilingMetric()
    assert metric.name == "profiling"


def test_no_model_fallback():
    """model kwarg 없으면 error가 details에 포함."""
    metric = ProfilingMetric()
    result = metric.evaluate_sample(
        audio=np.zeros(16000, dtype=np.float32),
        sr=16000,
        text="hello",
    )
    assert result.score == 0.0
    assert "error" in result.details
    assert result.details["error"] == "no model provided"
    assert "audio_duration_s" in result.details


def test_evaluate_with_mock_model():
    """mock model.synthesize로 RTF, wall_time, memory 확인."""
    metric = ProfilingMetric()

    mock_model = MagicMock()
    # Return 1 second of audio at 16000 Hz
    mock_model.synthesize.return_value = (np.zeros(16000, dtype=np.float32), 16000)

    result = metric.evaluate_sample(
        audio=np.zeros(16000, dtype=np.float32),
        sr=16000,
        text="hello world",
        model=mock_model,
    )

    assert result.metric_name == "profiling"
    assert "rtf" in result.details
    assert "wall_time_s" in result.details
    assert "peak_memory_mb" in result.details
    assert "audio_duration_s" in result.details
    assert "text_length" in result.details
    assert result.details["text_length"] == len("hello world")
    assert result.details["audio_duration_s"] == pytest.approx(1.0)
    assert result.score == result.details["rtf"]
    mock_model.synthesize.assert_called_once_with("hello world")


def test_aggregate():
    """mean_rtf, total_wall_time 등 집계 키 확인."""
    metric = ProfilingMetric()

    results = [
        MetricResult(
            sample_id=f"s{i}",
            metric_name="profiling",
            score=rtf,
            details={
                "rtf": rtf,
                "wall_time_s": wt,
                "peak_memory_mb": mem,
                "audio_duration_s": 1.0,
                "text_length": 10,
            },
        )
        for i, (rtf, wt, mem) in enumerate(
            [(0.5, 0.5, 100.0), (0.8, 0.8, 150.0), (0.3, 0.3, 80.0)]
        )
    ]

    agg = metric.aggregate(results)
    assert "mean_rtf" in agg
    assert "mean_wall_time_s" in agg
    assert "mean_peak_memory_mb" in agg
    assert "max_peak_memory_mb" in agg
    assert "total_wall_time_s" in agg
    assert agg["count"] == 3
    assert agg["mean_rtf"] == pytest.approx(np.mean([0.5, 0.8, 0.3]))
    assert agg["total_wall_time_s"] == pytest.approx(1.6)
    assert agg["max_peak_memory_mb"] == pytest.approx(150.0)
