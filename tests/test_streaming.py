"""StreamingMetric 테스트."""

import numpy as np
import pytest

from tts_auto_eval.metrics.streaming import StreamingMetric


@pytest.fixture
def metric():
    return StreamingMetric(chunk_duration_ms=500)


def _make_sine(duration_s: float = 2.0, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestStreamingMetric:
    def test_streaming_name(self, metric):
        assert metric.name == "streaming"

    def test_streaming_evaluate_silent_audio(self, metric):
        """Zeros audio should have seam_quality approximately 0."""
        audio = np.zeros(32000, dtype=np.float32)  # 2s at 16kHz
        result = metric.evaluate_sample(audio, 16000, "hello")
        assert result.details["seam_quality"] == pytest.approx(0.0, abs=1e-6)

    def test_streaming_evaluate_with_synthesis_time(self, metric):
        """Pass synthesis_time kwarg and verify RTF calculation."""
        audio = _make_sine(duration_s=2.0, sr=16000)
        result = metric.evaluate_sample(
            audio, 16000, "hello", synthesis_time=1.0
        )
        # RTF = synthesis_time / audio_duration = 1.0 / 2.0 = 0.5
        assert result.details["rtf"] == pytest.approx(0.5, abs=0.01)

    def test_seam_quality_smooth_audio(self, metric):
        """Smooth sine wave should have low seam_quality."""
        audio = _make_sine(duration_s=3.0, sr=16000)
        result = metric.evaluate_sample(audio, 16000, "hello")
        # Continuous sine wave has very uniform energy across chunk boundaries
        assert result.details["seam_quality"] < 0.1

    def test_aggregate(self, metric):
        """Verify custom aggregate keys."""
        from tts_auto_eval.metrics.base import MetricResult

        results = [
            MetricResult(
                sample_id="1",
                metric_name="streaming",
                score=0.9,
                details={"ttfb": 0.1, "seam_quality": 0.01, "rtf": 0.5},
            ),
            MetricResult(
                sample_id="2",
                metric_name="streaming",
                score=0.8,
                details={"ttfb": 0.2, "seam_quality": 0.02, "rtf": 0.6},
            ),
        ]
        agg = metric.aggregate(results)
        assert "mean_score" in agg
        assert "mean_ttfb" in agg
        assert "mean_rtf" in agg
        assert "mean_seam_quality" in agg
        assert agg["count"] == 2
        assert agg["mean_ttfb"] == pytest.approx(0.15, abs=0.01)
