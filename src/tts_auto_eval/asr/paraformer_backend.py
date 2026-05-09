"""Paraformer ASR 백엔드 — FunASR 기반, 중국어 최적화."""

import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.asr import BaseASR

logger = logging.getLogger(__name__)


class ParaformerASR(BaseASR):
    """FunASR Paraformer 기반 ASR."""

    def __init__(self, model_name: str = "paraformer-zh", device: str = "cpu", cache_dir: str = ".cache"):
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from funasr import AutoModel

        cache_path = Path(self._cache_dir) / "paraformer"
        cache_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Paraformer {self._model_name} 모델 로딩 중...")
        self._model = AutoModel(
            model=self._model_name,
            device=self._device,
        )
        logger.info("Paraformer 모델 로딩 완료")

    def transcribe(self, audio: np.ndarray, sr: int, language: str) -> str:
        import librosa

        self._load_model()

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        result = self._model.generate(input=audio, batch_size=1)
        if result and len(result) > 0:
            return result[0].get("text", "").strip()
        return ""
