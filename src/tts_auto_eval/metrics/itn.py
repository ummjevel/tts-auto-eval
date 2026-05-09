"""ITN (Inverse Text Normalization) 평가 지표.

숫자, 날짜, 통화 등 비표준 텍스트의 발음 정확도를 카테고리별로 측정.
"""

import logging
import re

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class ITNMetric(BaseMetric):
    """ITN 능력 평가.

    written form → TTS → Whisper STT → expected spoken form과 비교.
    카테고리별 WER을 산출.
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
        return "itn"

    def _transcribe(self, audio: np.ndarray, sr: int) -> str:
        if self._asr is None:
            from tts_auto_eval.asr import create_asr

            self._asr = create_asr(**self._asr_config)
        return self._asr.transcribe(audio, sr, self._language)

    def _normalize(self, text: str) -> str:
        """비교를 위한 텍스트 정규화."""
        from tts_auto_eval.metrics.intelligibility import normalize_text

        return normalize_text(text, self._language)

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
        expected_spoken: str | None = None,
        tags: list[str] | None = None,
    ) -> MetricResult:
        """ITN 샘플 평가.

        Args:
            text: 원본 written form
            expected_spoken: 기대되는 발음 형태 (있을 경우 이것과 비교)
        """
        from jiwer import wer, cer

        transcription = self._transcribe(audio, sr)

        # 비교 대상: expected_spoken이 있으면 그것과, 없으면 원본 text와 비교
        compare_text = expected_spoken if expected_spoken else text
        ref_normalized = self._normalize(compare_text)
        hyp_normalized = self._normalize(transcription)

        if not ref_normalized:
            wer_score = 0.0
            cer_score = 0.0
        else:
            wer_score = wer(ref_normalized, hyp_normalized)
            cer_score = cer(ref_normalized, hyp_normalized)

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=wer_score,
            details={
                "written_form": text,
                "expected_spoken": expected_spoken or "",
                "transcription": transcription,
                "ref_normalized": ref_normalized,
                "hyp_normalized": hyp_normalized,
                "wer": wer_score,
                "cer": cer_score,
                "tags": tags or [],
            },
        )

    def aggregate(self, results: list[MetricResult]) -> dict:
        """ITN 카테고리별 집계."""
        # 전체 WER/CER
        wer_scores = [r.details["wer"] for r in results]
        cer_scores = [r.details["cer"] for r in results]

        agg = {
            "overall_wer": float(np.mean(wer_scores)) if wer_scores else 0.0,
            "overall_cer": float(np.mean(cer_scores)) if cer_scores else 0.0,
            "count": len(results),
        }

        # 카테고리별 집계
        category_results: dict[str, list[float]] = {}
        for r in results:
            for tag in r.details.get("tags", []):
                if tag not in category_results:
                    category_results[tag] = []
                category_results[tag].append(r.details["wer"])

        categories = {}
        for tag, scores in category_results.items():
            categories[tag] = {
                "wer": float(np.mean(scores)),
                "count": len(scores),
            }

        agg["categories"] = categories
        return agg
