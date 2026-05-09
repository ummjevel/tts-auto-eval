"""긴 문장/문단 평가 — F0 drift, 에너지 감쇠, 호흡 위치 자연스러움."""

import logging

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class LongformMetric(BaseMetric):
    """장문 TTS 운율 일관성 평가.

    - f0_drift: F0의 시간에 따른 드리프트 (선형 회귀 기울기)
    - energy_decay: 에너지의 시간에 따른 감쇠율
    - breath_regularity: 묵음 구간(호흡 위치) 간격의 규칙성
    """

    def __init__(self, window_duration_s: float = 5.0):
        self._window_duration_s = window_duration_s

    @property
    def name(self) -> str:
        return "longform"

    def _compute_f0_drift(self, audio: np.ndarray, sr: int) -> float:
        """F0의 시간에 따른 드리프트 (기울기). 0에 가까울수록 안정적."""
        try:
            import parselmouth
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            pitch = snd.to_pitch(time_step=0.05)
            f0_values = pitch.selected_array["frequency"]
            # Remove unvoiced frames (0 Hz)
            voiced_mask = f0_values > 0
            f0_voiced = f0_values[voiced_mask]
            if len(f0_voiced) < 10:
                return 0.0
            # Linear regression: slope of F0 over time
            x = np.arange(len(f0_voiced))
            slope, _ = np.polyfit(x, f0_voiced, 1)
            return float(abs(slope))
        except ImportError:
            logger.warning("parselmouth not installed, skipping F0 drift")
            return 0.0

    def _compute_energy_decay(self, audio: np.ndarray, sr: int) -> float:
        """에너지의 시간에 따른 감쇠율. 0에 가까울수록 안정적."""
        frame_size = int(sr * 0.05)  # 50ms frames
        hop_size = int(sr * 0.025)   # 25ms hop

        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            energies.append(rms)

        if len(energies) < 10:
            return 0.0

        energies = np.array(energies)
        x = np.arange(len(energies))
        slope, _ = np.polyfit(x, energies, 1)
        return float(abs(slope))

    def _compute_breath_regularity(self, audio: np.ndarray, sr: int) -> float:
        """호흡/묵음 구간 간격의 규칙성. 1에 가까울수록 규칙적."""
        # Detect silence segments (energy below threshold)
        frame_size = int(sr * 0.025)
        hop_size = int(sr * 0.010)

        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energies.append(float(np.sqrt(np.mean(frame ** 2))))

        if not energies:
            return 1.0

        energies = np.array(energies)
        threshold = np.mean(energies) * 0.1  # 10% of mean energy

        # Find silence onsets
        is_silence = energies < threshold
        silence_onsets = []
        for i in range(1, len(is_silence)):
            if is_silence[i] and not is_silence[i - 1]:
                silence_onsets.append(i)

        if len(silence_onsets) < 2:
            return 1.0  # Not enough data

        # Intervals between silence onsets
        intervals = np.diff(silence_onsets)
        if len(intervals) == 0:
            return 1.0

        # Regularity = 1 - CV (coefficient of variation)
        cv = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0.0
        return max(0.0, 1.0 - cv)

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
        **kwargs,
    ) -> MetricResult:
        f0_drift = self._compute_f0_drift(audio, sr)
        energy_decay = self._compute_energy_decay(audio, sr)
        breath_regularity = self._compute_breath_regularity(audio, sr)

        # Composite score: weighted combination (higher = better)
        # Normalize drift and decay (lower is better) to 0-1 via sigmoid-like capping
        drift_score = max(0.0, 1.0 - f0_drift * 0.1)
        decay_score = max(0.0, 1.0 - energy_decay * 100)
        score = (drift_score * 0.3 + decay_score * 0.3 + breath_regularity * 0.4)

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=score,
            details={
                "f0_drift": f0_drift,
                "energy_decay": energy_decay,
                "breath_regularity": breath_regularity,
                "drift_score": drift_score,
                "decay_score": decay_score,
                "audio_duration_s": len(audio) / sr,
            },
        )
