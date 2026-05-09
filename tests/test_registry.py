"""메트릭 레지스트리 테스트."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_auto_eval.metrics.base import BaseMetric, MetricResult
from tts_auto_eval.metrics.registry import (
    _METRIC_REGISTRY,
    create_metric,
    discover_plugins,
    list_metrics,
    register_metric,
)


class DummyMetric(BaseMetric):
    """테스트용 더미 메트릭."""

    @property
    def name(self) -> str:
        return "dummy"

    def evaluate_sample(self, audio, sr, text, reference_audio=None, reference_sr=None):
        return MetricResult(sample_id="", metric_name=self.name, score=1.0)


def setup_function():
    """각 테스트 전 더미 등록 정리."""
    _METRIC_REGISTRY.pop("dummy", None)
    _METRIC_REGISTRY.pop("plugin_metric", None)


def test_register_and_create_metric():
    """더미 메트릭 등록 후 생성."""
    register_metric("dummy", "tests.test_registry", "DummyMetric")
    metric = create_metric("dummy")
    assert isinstance(metric, DummyMetric)
    assert metric.name == "dummy"


def test_create_unknown_metric_raises():
    """등록되지 않은 메트릭은 KeyError."""
    with pytest.raises(KeyError, match="Unknown metric"):
        create_metric("nonexistent_metric_xyz")


def test_list_metrics():
    """등록된 메트릭 목록이 정렬되어 반환."""
    register_metric("dummy", "tests.test_registry", "DummyMetric")
    names = list_metrics()
    assert "dummy" in names
    assert names == sorted(names)


def test_discover_plugins():
    """entry_points를 모킹하여 플러그인 등록 확인."""
    mock_ep = MagicMock()
    mock_ep.name = "plugin_metric"
    mock_ep.value = "tests.test_registry:DummyMetric"

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]

    import tts_auto_eval.metrics.registry as reg

    with patch("tts_auto_eval.metrics.registry.importlib.import_module"):
        with patch.object(reg, "entry_points", return_value=mock_eps):
            discover_plugins()

    assert "plugin_metric" in _METRIC_REGISTRY


def test_builtin_metrics_registered():
    """metrics 모듈 임포트 후 11개 빌트인 메트릭이 등록."""
    import tts_auto_eval.metrics  # noqa: F401

    expected = [
        "utmos", "wer", "pesq", "stoi",
        "speaker_similarity", "prosody", "fad", "itn", "phoneme",
        "mos_calibration", "profiling",
    ]
    registered = list_metrics()
    for name in expected:
        assert name in registered, f"{name} not registered"
