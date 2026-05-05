"""신호 품질 평가 지표 - PESQ, STOI.

참조 음성(reference_audio)이 필요한 지표들.
"""

import logging

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class PESQMetric(BaseMetric):
    """PESQ (Perceptual Evaluation of Speech Quality).

    ITU-T P.862 표준. 참조 음성 필요.
    점수 범위: -0.5 ~ 4.5 (wideband)
    """

    def __init__(self, mode: str = "wb"):
        """
        Args:
            mode: 'nb' (narrowband, 8kHz) 또는 'wb' (wideband, 16kHz)
        """
        self._mode = mode
        self._target_sr = 16000 if mode == "wb" else 8000

    @property
    def name(self) -> str:
        return "pesq"

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
    ) -> MetricResult:
        if reference_audio is None:
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=0.0,
                details={"error": "참조 음성 필요"},
            )

        from pesq import pesq as pesq_fn
        import librosa

        # 리샘플링
        if sr != self._target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._target_sr)
        ref_sr = reference_sr or sr
        if ref_sr != self._target_sr:
            reference_audio = librosa.resample(reference_audio, orig_sr=ref_sr, target_sr=self._target_sr)

        # 길이 맞춤
        min_len = min(len(audio), len(reference_audio))
        audio = audio[:min_len]
        reference_audio = reference_audio[:min_len]

        try:
            score = pesq_fn(self._target_sr, reference_audio, audio, self._mode)
        except Exception as e:
            logger.warning(f"PESQ 계산 실패: {e}")
            score = 0.0

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=float(score),
            details={"pesq_score": float(score), "mode": self._mode},
        )


class STOIMetric(BaseMetric):
    """STOI (Short-Time Objective Intelligibility).

    참조 음성 필요. 점수 범위: 0 ~ 1.
    """

    @property
    def name(self) -> str:
        return "stoi"

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
    ) -> MetricResult:
        if reference_audio is None:
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=0.0,
                details={"error": "참조 음성 필요"},
            )

        from pystoi import stoi as stoi_fn
        import librosa

        target_sr = 16000

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        ref_sr = reference_sr or sr
        if ref_sr != target_sr:
            reference_audio = librosa.resample(reference_audio, orig_sr=ref_sr, target_sr=target_sr)

        # 길이 맞춤
        min_len = min(len(audio), len(reference_audio))
        audio = audio[:min_len]
        reference_audio = reference_audio[:min_len]

        try:
            score = stoi_fn(reference_audio, audio, target_sr, extended=False)
        except Exception as e:
            logger.warning(f"STOI 계산 실패: {e}")
            score = 0.0

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=float(score),
            details={"stoi_score": float(score)},
        )
