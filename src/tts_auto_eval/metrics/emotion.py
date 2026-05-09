"""감정/스타일 평가 — SER 분류기로 의도한 감정 전달 정확도 측정."""

import logging

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class EmotionMetric(BaseMetric):
    """감정 전달 정확도 평가.

    TTS 출력 음성의 감정을 SER(Speech Emotion Recognition)로 분류하고,
    의도한 감정과의 일치 여부를 측정.
    """

    def __init__(self, device: str = "cpu", cache_dir: str = ".cache"):
        self._device = device
        self._cache_dir = cache_dir
        self._classifier = None

    @property
    def name(self) -> str:
        return "emotion"

    def _load_classifier(self):
        if self._classifier is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            self._classifier = hf_pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=0 if self._device.startswith("cuda") else -1,
            )
            logger.info("Emotion classifier loaded")
        except ImportError:
            logger.warning("transformers not installed, emotion classification unavailable")
            self._classifier = None

    def _classify(self, audio: np.ndarray, sr: int) -> tuple[str, float]:
        """감정 분류. Returns (predicted_emotion, confidence)."""
        self._load_classifier()
        if self._classifier is None:
            return "unknown", 0.0

        import librosa

        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        results = self._classifier(audio, sampling_rate=16000, top_k=5)
        top = results[0]
        return top["label"].lower(), top["score"]

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
        **kwargs,
    ) -> MetricResult:
        expected_emotion = kwargs.get("expected_emotion")
        if not expected_emotion:
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=0.0,
                details={"error": "no expected_emotion provided"},
            )

        predicted, confidence = self._classify(audio, sr)
        match = 1.0 if predicted == expected_emotion.lower() else 0.0

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=match,
            details={
                "expected_emotion": expected_emotion,
                "predicted_emotion": predicted,
                "confidence": confidence,
                "match": bool(match),
            },
        )

    def aggregate(self, results: list[MetricResult]) -> dict:
        scores = [r.score for r in results]
        return {
            "accuracy": float(np.mean(scores)) if scores else 0.0,
            "total": len(results),
            "correct": sum(1 for s in scores if s == 1.0),
            "count": len(results),
        }
