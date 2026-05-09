"""모델 프로파일링 — RTF, 메모리, 추론 시간 측정."""

import logging
import time
import tracemalloc

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class ProfilingMetric(BaseMetric):
    """모델 프로파일링 지표.

    - rtf: Real-Time Factor (합성 시간 / 오디오 길이). 낮을수록 빠름.
    - wall_time: 합성 소요 시간 (초)
    - peak_memory_mb: 최대 메모리 사용량 (MB)
    """

    @property
    def name(self) -> str:
        return "profiling"

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
        **kwargs,
    ) -> MetricResult:
        model = kwargs.get("model")
        if model is None:
            # Fallback: just report audio stats
            duration = len(audio) / sr if sr > 0 else 0.0
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=0.0,
                details={"error": "no model provided", "audio_duration_s": duration},
            )

        # Profile the synthesis
        tracemalloc.start()
        t0 = time.perf_counter()
        synth_audio, synth_sr = model.synthesize(text)
        wall_time = time.perf_counter() - t0
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration = len(synth_audio) / synth_sr if synth_sr > 0 else 0.0
        rtf = wall_time / duration if duration > 0 else 0.0

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=rtf,
            details={
                "rtf": rtf,
                "wall_time_s": wall_time,
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "audio_duration_s": duration,
                "text_length": len(text),
            },
        )

    def aggregate(self, results: list[MetricResult]) -> dict:
        valid = [r for r in results if "rtf" in r.details]
        if not valid:
            return {"count": len(results), "error": "no profiling data"}

        rtfs = [r.details["rtf"] for r in valid]
        wall_times = [r.details["wall_time_s"] for r in valid]
        memories = [r.details["peak_memory_mb"] for r in valid]

        return {
            "mean_rtf": float(np.mean(rtfs)),
            "mean_wall_time_s": float(np.mean(wall_times)),
            "mean_peak_memory_mb": float(np.mean(memories)),
            "max_peak_memory_mb": float(np.max(memories)),
            "total_wall_time_s": float(np.sum(wall_times)),
            "count": len(valid),
        }
