"""스트리밍 TTS 평가 — TTFB, 청크 이음새 품질, RTF."""

import logging
import time

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class StreamingMetric(BaseMetric):
    """스트리밍 TTS 평가 지표.

    - TTFB (Time To First Byte): 첫 오디오 청크까지 지연시간
    - seam_quality: 청크 경계에서의 에너지 불연속성 (낮을수록 좋음)
    - rtf (Real-Time Factor): 합성 시간 / 오디오 길이 (낮을수록 빠름)
    """

    def __init__(self, chunk_duration_ms: int = 500):
        self._chunk_duration_ms = chunk_duration_ms

    @property
    def name(self) -> str:
        return "streaming"

    def _analyze_seam_quality(self, audio: np.ndarray, sr: int) -> float:
        """청크 경계에서의 에너지 불연속성을 측정.

        오디오를 chunk_duration_ms 간격으로 나누고,
        각 경계에서 인접 프레임 간 에너지 차이의 평균을 계산.
        값이 낮을수록 이음새가 자연스러움.
        """
        chunk_samples = int(sr * self._chunk_duration_ms / 1000)
        if len(audio) < chunk_samples * 2:
            return 0.0

        # Frame-level energy (short-time energy)
        frame_size = int(sr * 0.025)  # 25ms frames
        hop_size = int(sr * 0.010)    # 10ms hop

        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energies.append(float(np.sum(frame ** 2)))

        if not energies:
            return 0.0

        energies = np.array(energies)
        # Normalize
        max_e = np.max(energies)
        if max_e > 0:
            energies = energies / max_e

        # Boundary indices (in frames)
        frames_per_chunk = chunk_samples // hop_size
        boundary_diffs = []
        for boundary_frame in range(frames_per_chunk, len(energies) - 1, frames_per_chunk):
            if boundary_frame < len(energies):
                diff = abs(energies[boundary_frame] - energies[boundary_frame - 1])
                boundary_diffs.append(diff)

        if not boundary_diffs:
            return 0.0

        return float(np.mean(boundary_diffs))

    def _compute_rtf(self, audio: np.ndarray, sr: int, synthesis_time: float | None = None) -> float:
        """Real-Time Factor 계산. synthesis_time이 없으면 0."""
        if synthesis_time is None or synthesis_time <= 0:
            return 0.0
        duration = len(audio) / sr
        if duration <= 0:
            return 0.0
        return synthesis_time / duration

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
        **kwargs,
    ) -> MetricResult:
        # Analyze seam quality from pre-generated audio
        seam_quality = self._analyze_seam_quality(audio, sr)

        # RTF from kwargs if available
        synthesis_time = kwargs.get("synthesis_time")
        rtf = self._compute_rtf(audio, sr, synthesis_time)

        # TTFB from kwargs if available
        ttfb = kwargs.get("ttfb", 0.0)

        # Score: inverse of seam quality (lower seam discontinuity = higher score)
        # Normalized to 0-1 range (capped)
        score = max(0.0, 1.0 - seam_quality * 10)

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=score,
            details={
                "ttfb": ttfb,
                "seam_quality": seam_quality,
                "rtf": rtf,
                "chunk_duration_ms": self._chunk_duration_ms,
                "audio_duration_s": len(audio) / sr,
            },
        )

    def aggregate(self, results: list[MetricResult]) -> dict:
        scores = [r.score for r in results]
        ttfbs = [r.details["ttfb"] for r in results if r.details.get("ttfb")]
        rtfs = [r.details["rtf"] for r in results if r.details.get("rtf")]
        seams = [r.details["seam_quality"] for r in results]
        return {
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "mean_ttfb": float(np.mean(ttfbs)) if ttfbs else 0.0,
            "mean_rtf": float(np.mean(rtfs)) if rtfs else 0.0,
            "mean_seam_quality": float(np.mean(seams)) if seams else 0.0,
            "count": len(results),
        }
