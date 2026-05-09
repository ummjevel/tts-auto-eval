"""MOS 캘리브레이션 메트릭 테스트."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_auto_eval.metrics.base import MetricResult
from tts_auto_eval.metrics.mos_calibration import MOSCalibrationMetric


def test_name():
    """name == 'mos_calibration'."""
    metric = MOSCalibrationMetric()
    assert metric.name == "mos_calibration"


def test_load_human_scores_json(tmp_path):
    """임시 JSON 파일에서 인간 MOS 점수 로드."""
    scores = {"sample_01": 4.2, "sample_02": 3.5, "sample_03": 2.8}
    scores_path = tmp_path / "human_scores.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")

    metric = MOSCalibrationMetric(human_scores_path=str(scores_path))
    assert len(metric._human_scores) == 3
    assert metric._human_scores["sample_01"] == 4.2


def test_evaluate_sample_with_human_score():
    """UTMOS 모킹 후 human_mos가 details에 포함되는지 확인."""
    metric = MOSCalibrationMetric()
    metric._human_scores = {"s1": 4.0}

    with patch.object(metric, "_predict_utmos", return_value=3.8):
        result = metric.evaluate_sample(
            audio=np.zeros(16000, dtype=np.float32),
            sr=16000,
            text="hello",
            sample_id="s1",
        )

    assert result.metric_name == "mos_calibration"
    assert result.score == 3.8
    assert result.details["predicted_mos"] == 3.8
    assert result.details["human_mos"] == 4.0
    assert result.details["has_human_score"] is True


def test_aggregate_correlation():
    """paired scores로 pearson/spearman 계산 확인."""
    metric = MOSCalibrationMetric()

    results = [
        MetricResult(
            sample_id=f"s{i}",
            metric_name="mos_calibration",
            score=pred,
            details={"predicted_mos": pred, "human_mos": human, "has_human_score": True},
        )
        for i, (pred, human) in enumerate(
            [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]
        )
    ]

    agg = metric.aggregate(results)
    assert agg["paired_samples"] == 5
    assert agg["total_samples"] == 5
    # Perfect correlation
    assert agg["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert agg["spearman_r"] == pytest.approx(1.0, abs=1e-6)


def test_aggregate_no_paired():
    """인간 점수가 없으면 correlation 키가 없어야 한다."""
    metric = MOSCalibrationMetric()

    results = [
        MetricResult(
            sample_id="s0",
            metric_name="mos_calibration",
            score=3.5,
            details={"predicted_mos": 3.5, "human_mos": None, "has_human_score": False},
        ),
    ]

    agg = metric.aggregate(results)
    assert agg["paired_samples"] == 0
    assert "pearson_r" not in agg
    assert "spearman_r" not in agg
