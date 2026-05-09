"""명료도 평가 지표 - WER, CER."""

import logging
import re

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


def normalize_text_ko(text: str) -> str:
    """한국어 텍스트 정규화.

    구두점 제거, 소문자 변환, 다중 공백 제거.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣa-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_en(text: str) -> str:
    """영어 텍스트 정규화.

    ZipVoice seedtts.py 참고: 구두점 제거, 소문자 변환, 다중 공백 제거.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str, language: str = "ko") -> str:
    """언어에 따른 텍스트 정규화."""
    if language == "en":
        return normalize_text_en(text)
    return normalize_text_ko(text)


class WERMetric(BaseMetric):
    """Word Error Rate (WER) via Whisper ASR.

    Seed-TTS 프로토콜 참고:
    - sentence_wer: 문장별 WER 평균 (모든 문장에 동일 가중치)
    - global_wer: 전체 substitutions+deletions+insertions / 전체 단어 수
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
        self._asr_config = {
            "backend": whisper_backend,
            "model_name": whisper_model,
            "device": device,
            "cache_dir": cache_dir,
        }

    @property
    def name(self) -> str:
        return "wer"

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
    ) -> MetricResult:
        from jiwer import wer, cer

        transcription = self._transcribe(audio, sr)

        ref_normalized = normalize_text(text, self._language)
        hyp_normalized = normalize_text(transcription, self._language)

        if not ref_normalized:
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=0.0,
                details={
                    "transcription": transcription,
                    "reference": text,
                    "wer": 0.0,
                    "cer": 0.0,
                },
            )

        wer_score = wer(ref_normalized, hyp_normalized)
        cer_score = cer(ref_normalized, hyp_normalized)

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=wer_score,
            details={
                "transcription": transcription,
                "reference": text,
                "ref_normalized": ref_normalized,
                "hyp_normalized": hyp_normalized,
                "wer": wer_score,
                "cer": cer_score,
            },
        )

    def aggregate(self, results: list[MetricResult]) -> dict:
        """Seed-TTS 프로토콜: Sentence-level WER과 Global WER 모두 제공."""
        from jiwer import compute_measures

        wer_scores = [r.details["wer"] for r in results]
        cer_scores = [r.details["cer"] for r in results]

        # Sentence-level WER (문장별 평균)
        sentence_wer = float(np.mean(wer_scores))
        sentence_cer = float(np.mean(cer_scores))

        # Global WER (전체 집계)
        all_refs = [r.details["ref_normalized"] for r in results]
        all_hyps = [r.details["hyp_normalized"] for r in results]
        global_measures = compute_measures(all_refs, all_hyps)

        return {
            "sentence_wer": sentence_wer,
            "sentence_cer": sentence_cer,
            "global_wer": float(global_measures["wer"]),
            "global_cer": float(np.mean(cer_scores)),
            "wer_std": float(np.std(wer_scores)),
            "count": len(results),
        }
