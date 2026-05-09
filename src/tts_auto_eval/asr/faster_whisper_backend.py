"""faster-whisper ASR 백엔드.

CTranslate2 기반으로 openai-whisper 대비 3-4배 빠른 추론.
"""

import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.asr import BaseASR

logger = logging.getLogger(__name__)


class FasterWhisperASR(BaseASR):
    """faster-whisper 기반 ASR."""

    def __init__(self, model_name: str = "large-v3", device: str = "cpu", cache_dir: str = ".cache"):
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        compute_type = "float16" if self._device.startswith("cuda") else "int8"
        download_root = Path(self._cache_dir) / "faster_whisper"
        download_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"faster-whisper {self._model_name} 모델 로딩 중... (cache: {download_root})")
        self._model = WhisperModel(
            self._model_name,
            device=self._device.split(":")[0],  # "cuda:0" → "cuda"
            compute_type=compute_type,
            download_root=str(download_root),
        )
        logger.info("faster-whisper 모델 로딩 완료")

    def transcribe(self, audio: np.ndarray, sr: int, language: str) -> str:
        import librosa

        self._load_model()

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        segments, _ = self._model.transcribe(audio, language=language)
        return " ".join(seg.text for seg in segments).strip()
