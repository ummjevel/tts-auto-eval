"""LongformMetric 테스트."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_auto_eval.metrics.longform import LongformMetric


@pytest.fixture
def metric():
    return LongformMetric(window_duration_s=5.0)


def _make_sine(duration_s: float = 2.0, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestLongformMetric:
    def test_longform_name(self, metric):
        assert metric.name == "longform"

    def test_energy_decay_stable(self, metric):
        """Constant-amplitude audio should have low energy_decay."""
        audio = _make_sine(duration_s=3.0, sr=16000)
        decay = metric._compute_energy_decay(audio, 16000)
        assert decay < 0.001

    def test_breath_regularity_regular(self, metric):
        """Audio with evenly spaced silences should have high regularity."""
        sr = 16000
        # Build audio: 0.5s tone + 0.1s silence, repeated 5 times
        segments = []
        for _ in range(5):
            tone = _make_sine(duration_s=0.5, sr=sr)
            silence = np.zeros(int(sr * 0.1), dtype=np.float32)
            segments.extend([tone, silence])
        audio = np.concatenate(segments)
        regularity = metric._compute_breath_regularity(audio, sr)
        assert regularity > 0.8

    def test_evaluate_sample_returns_score(self, metric):
        """Verify 0 <= score <= 1."""
        audio = _make_sine(duration_s=3.0, sr=16000)
        result = metric.evaluate_sample(audio, 16000, "hello world")
        assert 0.0 <= result.score <= 1.0
        assert result.metric_name == "longform"
        assert "f0_drift" in result.details
        assert "energy_decay" in result.details
        assert "breath_regularity" in result.details

    def test_f0_drift_with_mock_parselmouth(self, metric):
        """Mock parselmouth and verify drift computation."""
        # Create mock pitch object with frequency data
        mock_pitch = MagicMock()
        # Simulate stable F0 around 200 Hz with slight drift
        f0_values = np.linspace(200, 210, 50)  # slight upward drift
        mock_pitch.selected_array = {"frequency": f0_values}

        mock_sound = MagicMock()
        mock_sound.to_pitch.return_value = mock_pitch

        mock_parselmouth = MagicMock()
        mock_parselmouth.Sound.return_value = mock_sound

        with patch.dict("sys.modules", {"parselmouth": mock_parselmouth}):
            audio = _make_sine(duration_s=3.0, sr=16000)
            drift = metric._compute_f0_drift(audio, 16000)
            # With linearly increasing F0 from 200 to 210 over 50 points,
            # slope should be approximately (210-200)/49 ≈ 0.204
            assert drift > 0.0
            assert drift < 1.0
