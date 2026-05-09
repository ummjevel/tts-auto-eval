"""발음 사전 기반 평가 — G2P 변환 후 음소 단위 정확도 측정."""

import logging

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class PhonemeMetric(BaseMetric):
    """Phoneme Error Rate (PER) 평가.

    G2P로 원본 텍스트와 ASR 결과를 음소로 변환한 뒤 비교.
    """

    def __init__(
        self,
        whisper_model: str = "large-v3",
        language: str = "ko",
        device: str = "cpu",
        cache_dir: str = ".cache",
        whisper_backend: str = "openai-whisper",
    ):
        self._language = language
        self._asr = None
        self._g2p = None
        self._asr_config = {
            "backend": whisper_backend,
            "model_name": whisper_model,
            "device": device,
            "cache_dir": cache_dir,
        }

    @property
    def name(self) -> str:
        return "phoneme"

    def _load_g2p(self):
        if self._g2p is not None:
            return
        if self._language == "ko":
            from g2pk import G2p

            self._g2p = G2p()
        else:
            from g2p_en import G2p

            self._g2p = G2p()
        logger.info(f"G2P 로드 완료 (language={self._language})")

    def _to_phonemes(self, text: str) -> str:
        """텍스트를 음소 시퀀스로 변환."""
        self._load_g2p()
        return self._g2p(text)

    def _transcribe(self, audio: np.ndarray, sr: int) -> str:
        if self._asr is None:
            from tts_auto_eval.asr import create_asr

            self._asr = create_asr(**self._asr_config)
        return self._asr.transcribe(audio, sr, self._language)

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
        **kwargs,
    ) -> MetricResult:
        from jiwer import wer as compute_wer

        # ASR -> 인식 텍스트
        transcription = self._transcribe(audio, sr)

        # G2P 변환
        ref_phonemes = self._to_phonemes(text)
        hyp_phonemes = self._to_phonemes(transcription)

        if not ref_phonemes:
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=1.0,
                details={"per": 0.0, "ref_phonemes": "", "hyp_phonemes": ""},
            )

        # PER 계산 (jiwer의 wer을 음소 시퀀스에 적용)
        per = compute_wer(ref_phonemes, hyp_phonemes)
        accuracy = max(0.0, 1.0 - per)

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=accuracy,
            details={
                "per": per,
                "accuracy": accuracy,
                "transcription": transcription,
                "ref_phonemes": ref_phonemes,
                "hyp_phonemes": hyp_phonemes,
            },
        )
