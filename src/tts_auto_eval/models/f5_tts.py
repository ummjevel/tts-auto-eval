"""F5-TTS 백엔드 어댑터.

설치: pip install f5-tts
참고: https://github.com/SWivid/F5-TTS
"""

import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.models.base import BaseTTSModel

logger = logging.getLogger(__name__)


class F5TTSModel(BaseTTSModel):
    """F5-TTS 모델 어댑터.

    zero-shot 음성 복제 TTS. 참조 음성(ref_audio)과 참조 텍스트(ref_text)가 필요.
    """

    def __init__(
        self,
        model_type: str = "F5-TTS",
        ref_audio: str | None = None,
        ref_text: str = "",
        device: str | None = None,
    ):
        self._model_type = model_type
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._device = device
        self._client = None

    @property
    def name(self) -> str:
        return f"f5-tts-{self._model_type.lower()}"

    def _load_model(self):
        if self._client is not None:
            return

        try:
            from f5_tts.api import F5TTS

            logger.info(f"F5-TTS 로딩 중 (model_type={self._model_type})...")
            self._client = F5TTS(model_type=self._model_type, device=self._device)
            logger.info("F5-TTS 로딩 완료")
        except ImportError:
            raise ImportError(
                "f5-tts가 설치되지 않았습니다. `pip install f5-tts`로 설치하세요."
            )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self._load_model()

        audio, sr, _ = self._client.infer(
            ref_file=self._ref_audio or "",
            ref_text=self._ref_text,
            gen_text=text,
        )

        if isinstance(audio, np.ndarray):
            return audio, sr

        # torch.Tensor인 경우
        return audio.cpu().numpy(), sr
