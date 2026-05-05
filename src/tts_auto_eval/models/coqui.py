"""Coqui TTS / XTTS 백엔드 어댑터.

설치: pip install TTS
참고: https://github.com/coqui-ai/TTS
"""

import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.models.base import BaseTTSModel

logger = logging.getLogger(__name__)


class CoquiTTSModel(BaseTTSModel):
    """Coqui TTS 모델 어댑터.

    다양한 TTS 모델 지원. XTTS v2는 zero-shot 음성 복제 가능.
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        speaker_wav: str | None = None,
        language: str = "ko",
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._speaker_wav = speaker_wav
        self._language = language
        self._device = device
        self._tts = None

    @property
    def name(self) -> str:
        short = self._model_name.split("/")[-1] if "/" in self._model_name else self._model_name
        return f"coqui-{short}"

    def _load_model(self):
        if self._tts is not None:
            return

        try:
            from TTS.api import TTS

            logger.info(f"Coqui TTS 로딩 중 ({self._model_name})...")
            self._tts = TTS(model_name=self._model_name).to(self._device)
            logger.info("Coqui TTS 로딩 완료")
        except ImportError:
            raise ImportError(
                "Coqui TTS가 설치되지 않았습니다. `pip install TTS`로 설치하세요."
            )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self._load_model()

        # XTTS v2는 speaker_wav와 language가 필요
        if self._speaker_wav:
            wav = self._tts.tts(
                text=text,
                speaker_wav=self._speaker_wav,
                language=self._language,
            )
        else:
            wav = self._tts.tts(text=text)

        audio = np.array(wav, dtype=np.float32)
        sr = self._tts.synthesizer.output_sample_rate if hasattr(self._tts, "synthesizer") else 22050

        return audio, sr
