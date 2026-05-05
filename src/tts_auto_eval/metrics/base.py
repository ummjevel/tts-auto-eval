"""평가 지표 기본 클래스."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MetricResult:
    """단일 샘플에 대한 지표 결과."""

    sample_id: str
    metric_name: str
    score: float
    details: dict = field(default_factory=dict)


class BaseMetric(ABC):
    """평가 지표 기본 클래스.

    모든 지표는 이 클래스를 상속하여 evaluate_sample()을 구현한다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """지표 이름."""
        ...

    @abstractmethod
    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
    ) -> MetricResult:
        """단일 샘플 평가.

        Args:
            audio: 평가 대상 오디오
            sr: 샘플레이트
            text: 원본 텍스트
            reference_audio: 참조 오디오 (필요한 지표만)
            reference_sr: 참조 오디오 샘플레이트

        Returns:
            MetricResult
        """
        ...

    def evaluate_batch(
        self,
        samples: list[dict],
    ) -> list[MetricResult]:
        """배치 평가. 기본 구현은 순차 처리."""
        results = []
        for sample in samples:
            result = self.evaluate_sample(
                audio=sample["audio"],
                sr=sample["sr"],
                text=sample["text"],
                reference_audio=sample.get("reference_audio"),
                reference_sr=sample.get("reference_sr"),
            )
            result.sample_id = sample["id"]
            results.append(result)
        return results

    def aggregate(self, results: list[MetricResult]) -> dict:
        """결과 집계. 기본: 평균 + 표준편차."""
        scores = [r.score for r in results]
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "count": len(scores),
        }
