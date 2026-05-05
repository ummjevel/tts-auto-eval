"""분포 수준 평가 지표 - Frechet Audio Distance (FAD)."""

import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class FADMetric(BaseMetric):
    """Frechet Audio Distance (FAD).

    생성된 오디오 집합과 참조 오디오 집합 간의 분포 거리.
    개별 샘플이 아닌 집합 수준에서 동작하므로 aggregate에서 계산.
    """

    def __init__(self, model_name: str = "vggish"):
        """
        Args:
            model_name: 임베딩 모델 ('vggish' 또는 'encodec')
        """
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "fad"

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
    ) -> MetricResult:
        # FAD는 개별 샘플 평가가 아닌 집합 수준 평가
        # 여기서는 placeholder로 0 반환, aggregate에서 실제 계산
        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=0.0,
            details={"note": "FAD는 aggregate에서 계산됨"},
        )

    def compute_fad(
        self,
        generated_dir: str | Path,
        reference_dir: str | Path,
    ) -> float:
        """두 디렉토리의 오디오 파일 간 FAD 계산.

        Args:
            generated_dir: 생성된 오디오 파일 디렉토리
            reference_dir: 참조 오디오 파일 디렉토리

        Returns:
            FAD 점수 (낮을수록 좋음)
        """
        try:
            from fadtk import FAD

            fad = FAD(self._model_name)
            score = fad.score(str(reference_dir), str(generated_dir))
            return float(score)
        except ImportError:
            logger.warning("fadtk 미설치. `pip install fadtk`로 설치하세요.")
            return -1.0
        except Exception as e:
            logger.warning(f"FAD 계산 실패: {e}")
            return -1.0

    def aggregate(self, results: list[MetricResult]) -> dict:
        return {
            "note": "FAD는 compute_fad() 메서드로 별도 계산 필요",
            "count": len(results),
        }
