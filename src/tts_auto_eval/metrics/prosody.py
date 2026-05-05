"""운율 평가 지표 - F0, Energy, Speaking Rate."""

import logging

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class ProsodyMetric(BaseMetric):
    """운율(Prosody) 분석.

    - F0 통계 (평균, 범위, 변동성)
    - 에너지 통계
    - 발화 속도 (초당 음절 수 추정)

    참조 음성 있을 시 F0 RMSE, 에너지 상관도 추가 계산.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "prosody"

    def _extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Parselmouth(Praat)로 F0 추출."""
        import parselmouth

        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01)
        f0 = pitch.selected_array["frequency"]
        return f0[f0 > 0]  # 유성음 구간만

    def _extract_energy(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """프레임별 RMS 에너지."""
        import librosa

        rms = librosa.feature.rms(y=audio, frame_length=int(0.025 * sr), hop_length=int(0.01 * sr))
        return rms.squeeze()

    def _estimate_speaking_rate(self, audio: np.ndarray, sr: int) -> float:
        """발화 속도 추정 (에너지 피크 기반, 초당 음절 수)."""
        import librosa

        duration = len(audio) / sr
        if duration < 0.1:
            return 0.0

        # onset detection으로 음절 수 추정
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        syllable_count = max(len(onsets), 1)
        return syllable_count / duration

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
    ) -> MetricResult:
        f0 = self._extract_f0(audio, sr)
        energy = self._extract_energy(audio, sr)
        speaking_rate = self._estimate_speaking_rate(audio, sr)

        details = {
            "f0_mean": float(np.mean(f0)) if len(f0) > 0 else 0.0,
            "f0_std": float(np.std(f0)) if len(f0) > 0 else 0.0,
            "f0_min": float(np.min(f0)) if len(f0) > 0 else 0.0,
            "f0_max": float(np.max(f0)) if len(f0) > 0 else 0.0,
            "energy_mean": float(np.mean(energy)),
            "energy_std": float(np.std(energy)),
            "speaking_rate": speaking_rate,
        }

        # 참조 음성이 있으면 비교 지표 추가
        if reference_audio is not None:
            ref_f0 = self._extract_f0(reference_audio, reference_sr or sr)
            ref_energy = self._extract_energy(reference_audio, reference_sr or sr)

            # F0 RMSE (길이 맞춤)
            if len(f0) > 0 and len(ref_f0) > 0:
                min_len = min(len(f0), len(ref_f0))
                f0_rmse = float(np.sqrt(np.mean((f0[:min_len] - ref_f0[:min_len]) ** 2)))
                details["f0_rmse"] = f0_rmse

            # 에너지 상관도
            if len(energy) > 0 and len(ref_energy) > 0:
                min_len = min(len(energy), len(ref_energy))
                corr = float(np.corrcoef(energy[:min_len], ref_energy[:min_len])[0, 1])
                details["energy_correlation"] = corr if not np.isnan(corr) else 0.0

        # 종합 점수: F0 변동성 (자연스러운 음성일수록 적절한 변동성을 가짐)
        score = details["f0_std"] / (details["f0_mean"] + 1e-8) if details["f0_mean"] > 0 else 0.0

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=score,
            details=details,
        )

    def aggregate(self, results: list[MetricResult]) -> dict:
        """운율 지표 집계."""
        agg = {}
        for key in ["f0_mean", "f0_std", "energy_mean", "speaking_rate"]:
            values = [r.details.get(key, 0) for r in results]
            agg[f"{key}_avg"] = float(np.mean(values))

        # F0 RMSE (참조 있는 경우만)
        f0_rmses = [r.details["f0_rmse"] for r in results if "f0_rmse" in r.details]
        if f0_rmses:
            agg["f0_rmse_avg"] = float(np.mean(f0_rmses))

        # 에너지 상관도
        energy_corrs = [r.details["energy_correlation"] for r in results if "energy_correlation" in r.details]
        if energy_corrs:
            agg["energy_correlation_avg"] = float(np.mean(energy_corrs))

        agg["count"] = len(results)
        return agg
