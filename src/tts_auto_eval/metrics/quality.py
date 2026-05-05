"""음질 평가 지표 - UTMOS."""

import logging

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class UTMOSMetric(BaseMetric):
    """UTMOS (Universal Text-to-speech MOS) 예측.

    Wav2Vec2 기반 MOS 예측 모델. 1-5 스케일.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return "utmos"

    def _load_model(self):
        if self._model is not None:
            return

        import torch

        torch.set_num_threads(1)

        logger.info("UTMOS 모델 로딩 중...")
        self._model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("UTMOS 모델 로딩 완료")

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
    ) -> MetricResult:
        import torch
        import librosa

        self._load_model()

        # UTMOS는 16kHz 입력 필요
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        waveform = torch.from_numpy(audio).unsqueeze(0).float().to(self._device)

        with torch.no_grad():
            score = self._model(waveform, sr=16000).item()

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=score,
            details={"mos_score": score},
        )
