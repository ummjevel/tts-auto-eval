"""openai-whisper ASR 백엔드."""

import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.asr import BaseASR

logger = logging.getLogger(__name__)


class OpenAIWhisperASR(BaseASR):
    """openai-whisper 기반 ASR."""

    def __init__(self, model_name: str = "large-v3", device: str = "cpu", cache_dir: str = ".cache"):
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import whisper

        download_root = Path(self._cache_dir) / "whisper"
        download_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Whisper {self._model_name} 모델 로딩 중... (cache: {download_root})")
        self._model = whisper.load_model(
            self._model_name,
            device=self._device,
            download_root=str(download_root),
        )
        logger.info("Whisper 모델 로딩 완료")

    def transcribe(self, audio: np.ndarray, sr: int, language: str) -> str:
        import librosa

        self._load_model()

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        result = self._model.transcribe(audio, language=language, task="transcribe")
        return result["text"]
