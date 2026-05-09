"""EmotionMetric 테스트."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tts_auto_eval.metrics.emotion import EmotionMetric


def test_emotion_name():
    """name 프로퍼티 확인."""
    metric = EmotionMetric()
    assert metric.name == "emotion"


def test_emotion_no_expected():
    """expected_emotion 없으면 score 0.0, error in details."""
    metric = EmotionMetric()
    audio = np.zeros(16000, dtype=np.float32)
    result = metric.evaluate_sample(audio=audio, sr=16000, text="hello")
    assert result.score == 0.0
    assert "error" in result.details


def test_emotion_match():
    """분류기가 일치하는 감정을 반환하면 score 1.0."""
    metric = EmotionMetric()
    metric._classifier = MagicMock()
    metric._classifier.return_value = [
        {"label": "happy", "score": 0.95},
        {"label": "sad", "score": 0.03},
    ]

    audio = np.zeros(16000, dtype=np.float32)
    result = metric.evaluate_sample(
        audio=audio, sr=16000, text="hello", expected_emotion="happy"
    )
    assert result.score == 1.0
    assert result.details["match"] is True
    assert result.details["predicted_emotion"] == "happy"
    assert result.details["confidence"] == pytest.approx(0.95)


def test_emotion_mismatch():
    """분류기가 다른 감정을 반환하면 score 0.0."""
    metric = EmotionMetric()
    metric._classifier = MagicMock()
    metric._classifier.return_value = [
        {"label": "angry", "score": 0.8},
        {"label": "happy", "score": 0.1},
    ]

    audio = np.zeros(16000, dtype=np.float32)
    result = metric.evaluate_sample(
        audio=audio, sr=16000, text="hello", expected_emotion="happy"
    )
    assert result.score == 0.0
    assert result.details["match"] is False
    assert result.details["predicted_emotion"] == "angry"


def test_emotion_aggregate():
    """aggregate에서 accuracy, correct, total 키 확인."""
    from tts_auto_eval.metrics.base import MetricResult

    metric = EmotionMetric()
    results = [
        MetricResult(sample_id="1", metric_name="emotion", score=1.0, details={}),
        MetricResult(sample_id="2", metric_name="emotion", score=0.0, details={}),
        MetricResult(sample_id="3", metric_name="emotion", score=1.0, details={}),
    ]
    agg = metric.aggregate(results)
    assert agg["accuracy"] == pytest.approx(2.0 / 3.0)
    assert agg["correct"] == 2
    assert agg["total"] == 3
    assert agg["count"] == 3
